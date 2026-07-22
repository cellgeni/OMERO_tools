#!/usr/bin/env python
"""
duplicate.py

Duplicate an OMERO Project / Dataset / Image (and everything beneath it) from one OMERO group into another group, leaving the originals untouched.

How it works
------------
OMERO's server-side ``Duplicate`` request always creates the copy *inside the same group* as the source object. To end up with a copy in a *different* group we therefore run two server-side requests in sequence:

    1. omero.cmd.Duplicate  - deep-copies the object + its children (Datasets, Images, Pixels, ...) within the SOURCE group.
       Ref: https://docs.openmicroscopy.org/omero-blitz/5.6.2/slice2html/omero/cmd/Duplicate.html
    2. omero.cmd.Chgrp2     - moves that fresh copy into the TARGET group.
       Ref: https://docs.openmicroscopy.org/omero-blitz/5.6.2/slice2html/omero/cmd/Chgrp2.html

Because Chgrp2 moves the *duplicate*, the original data stays exactly where it was. If instead you want to MOVE the original (no copy left behind), pass ``--move`` and only the Chgrp2 step runs, on the original id.

Groups may be given by numeric id OR by name (see --source-group / --target-group). Purely-numeric values are treated as ids; anything else is looked up by name.

Requirements
------------
* omero-py installed (``pip install omero-py``) with a working Ice runtime.
* The connecting user must be a member of BOTH the source and target groups (or be an administrator), and have permission to read the source data and write into the target group. The target group must allow the copy to be moved into it (i.e. its permission level must accept the move).

Usage
-----
    python duplicate.py \
        --host omero.example.org --user alice --password secret \
        --type Project --id 123 \
        --source-group 3 --target-group 7

    # Groups by name instead of id:
    python duplicate.py ... --source-group "Lab A" --target-group "Lab B"

    # Multiple objects of the same type in one run (comma-separated ids):
    python duplicate.py ... --type Dataset --id 1,14,28,50

    # Move the originals instead of duplicating:
    python duplicate.py ... --move

Notes
-----
* Duplicating Images copies their pixel data, which can be large and slow.
  Tune --loops / --poll-ms if big copies time out.
* Duplicating a Project also duplicates its Datasets and Images; duplicating a
  Dataset also duplicates its Images. You only ever pass the top-level object.
* Progress: each request prints server-reported step events plus a heartbeat
  every few seconds so you can tell it's still working during long copies.
"""
import argparse
import sys

from omero.gateway import BlitzGateway
from omero.cmd import Duplicate, Chgrp2, ERR
from omero.callbacks import CmdCallbackI


VALID_TYPES = ("Project", "Dataset", "Image")


def id_list(value):
    """argparse type: parse '1,14,28,50' into an ordered list of unique ints."""
    ids, seen = [], set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            i = int(part)
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid id: {part}")
        if i not in seen:
            seen.add(i)
            ids.append(i)
    if not ids:
        raise argparse.ArgumentTypeError("no ids provided")
    return ids


def resolve_group(conn, value):
    """Resolve a group given by numeric id OR by name -> (id, name).

    Purely-numeric values are treated as group ids; anything else is looked up
    by name via the admin service. Raises RuntimeError if not found.
    """
    admin = conn.getAdminService()
    if value.isdigit(): 
        # treat as id
        gid = int(value)
        try:
            grp = admin.getGroup(gid)
        except Exception:
            raise RuntimeError(f"No group found with id {gid}")
    else:                                      
        try:
            # treat as name
            grp = admin.lookupGroup(value)
        except Exception:
            raise RuntimeError(f"No group found with name '{value}'")
    return grp.getId().getValue(), grp.getName().getValue()


class _ProgressCB(CmdCallbackI):
    """CmdCallbackI that prints server-reported progress.

    ``step()`` is invoked remotely by the server (on an Ice thread) after each
    processing step of a graph operation, so it reports real progress when the
    server emits step events.
    """

    def step(self, complete, total, current=None):
        pct = (100.0 * complete / total) if total else 0.0
        print(f"    progress: step {complete}/{total} ({pct:.0f}%)", flush=True)


def submit(conn, request, loops=500, poll_ms=1000):
    """Submit a server-side request and block until it finishes.

    Prints per-step progress (via _ProgressCB.step) plus a heartbeat every few
    seconds so you can tell it's alive during long, quiet stretches such as
    pixel copies. Returns the response, or raises RuntimeError on error/timeout.
    Total wait is roughly ``loops * poll_ms`` milliseconds.
    """
    handle = conn.c.getSession().submit(request)
    cb = _ProgressCB(conn.c, handle)
    max_wait = loops * poll_ms / 1000.0
    try:
        waited = 0.0
        last_beat = 0.0
        # block(ms) returns True once finished, False on each timeout
        while not cb.block(poll_ms):
            waited += poll_ms / 1000.0
            if waited - last_beat >= 5:
                # heartbeat at most every 5s
                #print(f"           ({waited:.0f}s elapsed)", flush=True)
                last_beat = waited
            if waited >= max_wait:
                raise RuntimeError("Request timed out before completing (increase --loops / --poll-ms)")
        rsp = cb.getResponse()
    finally:
        cb.close(True)

    if rsp is None:
        raise RuntimeError("Request finished but returned no response")
    if isinstance(rsp, ERR):
        raise RuntimeError(f"Server request failed: {rsp.category} / {rsp.name} / {rsp.parameters}")
    return rsp


def find_new_id(duplicate_response, model_type):
    """Extract the new object id of the given type from a DuplicateResponse.

    The response maps fully-qualified model class names -> [new ids], e.g.
    {"ome.model.containers.Project": [456]}.
    """
    for class_name, ids in duplicate_response.duplicates.items():
        if class_name.split(".")[-1].lower() == model_type.lower() and ids:
            return ids[0]
    return None


def duplicate_across_groups(conn, obj_type, obj_id, source_group, target_group, move=False, loops=200, poll_ms=1000):
    conn.setGroupForSession(source_group)
    conn.SERVICE_OPTS.setOmeroGroup(source_group)

    # sanity-check the object exists and is visible in the source group.
    obj = conn.getObject(obj_type, obj_id)
    if obj is None:
        raise RuntimeError(f"{obj_type} {obj_id} not found in group {source_group} (check id, type and permissions)")
    print(f"Found {obj_type}: '{obj.getName()}' (id={obj_id}) in group {source_group}")

    if move:
        print(f"*MOVING ORIGINAL* {obj_type} id={obj_id} -> group {target_group} ...")
        chgrp = Chgrp2()
        chgrp.targetObjects = {obj_type: [obj_id]}
        chgrp.groupId = target_group
        submit(conn, chgrp, loops, poll_ms)
        print("Move complete.")
        return obj_id

    print(f"Duplicating {obj_type} {obj_id} within group {source_group} (recursively) ...")
    dup = Duplicate()
    dup.targetObjects = {obj_type: [obj_id]}
    dup_rsp = submit(conn, dup, loops, poll_ms)

    new_id = find_new_id(dup_rsp, obj_type)
    if new_id is None:
        raise RuntimeError(f"Duplicate succeeded but no new {obj_type} id was returned")
    print("---")
    print(f"Created duplicate {obj_type} id={new_id} (still in group {source_group})")
    print("---")
    print(f"Moving duplicate {obj_type} {new_id} -> group {target_group} (recursively) ...")
    chgrp = Chgrp2()
    chgrp.targetObjects = {obj_type: [new_id]}
    chgrp.groupId = target_group
    submit(conn, chgrp, loops, poll_ms)

    print(f"Done. Duplicate {obj_type} id={new_id} now lives in group {target_group}; original untouched.")
    return new_id


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Duplicate an OMERO Project/Dataset/Image from one Group to another.")
    p.add_argument("--host", type=str, default="wsi-omero-prod-02.internal.sanger.ac.uk", help="OMERO server hostname")
    p.add_argument("--port", type=int, default=4064,help="OMERO server port (default 4064)")
    p.add_argument("--user", required=True, help="OMERO username")
    p.add_argument("--password", required=True, help="OMERO password")
    p.add_argument("--type", required=True, choices=VALID_TYPES, help="Type of the top-level object to copy")
    p.add_argument("--id", type=id_list, required=True, metavar="ID[,ID...]", help="Id of the object to copy, or a comma-separated list of ids of the same --type (e.g. 1,14,28,50)")
    p.add_argument("--source-group", required=True, metavar="GROUP", help="Group the object currently lives in (id or name)")
    p.add_argument("--target-group", required=True, metavar="GROUP", help="Group to copy (or move) the object into (id or name)")
    p.add_argument("--move", action="store_true", help="Move the original instead of duplicating (no copy left behind)")
    p.add_argument("--stop-on-error", action="store_true", help="Abort immediately if any id fails (default: continue and report a summary)")
    p.add_argument("--loops", type=int, default=200, help="Max poll iterations to wait per request (default 200)")
    p.add_argument("--poll-ms", type=int, default=1000, help="Milliseconds between polls (default 1000)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    conn = BlitzGateway(args.user, args.password, host=args.host, port=args.port, secure=True)
    if not conn.connect():
        print(f"ERROR: could not connect to OMERO server (check host/credentials).", file=sys.stderr)
        return 1

    # Keep the session alive during long-running pixel copies.
    conn.c.enableKeepAlive(60)

    try:
        # Resolve group id-or-name arguments to numeric ids.
        try:
            src_gid, src_gname = resolve_group(conn, args.source_group)
            tgt_gid, tgt_gname = resolve_group(conn, args.target_group)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"Source group: '{src_gname}' (id={src_gid})")
        print(f"Target group: '{tgt_gname}' (id={tgt_gid})")

        # Confirm the user can actually work in both groups.
        my_group_ids = {g.getId() for g in conn.getGroupsMemberOf()}
        is_admin = conn.isAdmin()
        for gid in (src_gid, tgt_gid):
            if gid not in my_group_ids and not is_admin:
                print(f"ERROR: connecting user is not a member of group {gid} (and is not an admin).", file=sys.stderr)
                return 1

        results, failures = [], []
        for obj_id in args.id:
            print("===")
            print(f"{args.type} {obj_id} ")
            try:
                new_id = duplicate_across_groups(
                    conn,
                    obj_type=args.type,
                    obj_id=obj_id,
                    source_group=src_gid,
                    target_group=tgt_gid,
                    move=args.move,
                    loops=args.loops,
                    poll_ms=args.poll_ms,
                )
                results.append((obj_id, new_id))
            except Exception as exc:
                print(f"ERROR processing {args.type} {obj_id}: {exc}", file=sys.stderr)
                failures.append(obj_id)
                if args.stop_on_error:
                    break
            print("===")

        print("Summary")
        verb = "moved" if args.move else "duplicated -> new id"
        for src_id, new_id in results:
            print(f"  OK  {args.type} {src_id} {verb} {new_id}")
        for src_id in failures:
            print(f"  FAIL {args.type} {src_id}")
        print(f"=== {len(results)} succeeded, {len(failures)} failed.")
        if failures:
            return 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())