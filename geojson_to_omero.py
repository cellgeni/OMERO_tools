#!/usr/bin/env python3
"""
Upload polygon annotations from a GeoJSON file (e.g. a QuPath export) to an
OMERO image as Polygon ROIs, one ROI per GeoJSON feature, named after the
feature's "name" property (falling back to its classification name).

Coordinates can be given in pixels (default) or in microns. For microns the
pixel size is taken from --pixel-size, or read from the image metadata in OMERO.

Credentials are asked for interactively after the script starts. For batch use
they can instead be provided via the environment variables
OMERO_HOST, OMERO_PORT, OMERO_USER and OMERO_PASSWORD.

Examples:
    python geojson_to_omero.py annotations.geojson 123456
    python geojson_to_omero.py annotations.geojson 123456 --units microns
    python geojson_to_omero.py annotations.geojson 123456 --units microns --pixel-size 0.5
    python geojson_to_omero.py annotations.geojson 123456 --dry-run
"""

import argparse
import getpass
import json
import os
import sys


# --------------------------------------------------------------------------- #
# GeoJSON parsing
# --------------------------------------------------------------------------- #

def read_features(path):
    """Return a list of GeoJSON features from a FeatureCollection, a single
    Feature, a bare geometry or a plain list of features."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, list):
        return data
    if data.get("type") == "FeatureCollection":
        return data.get("features", [])
    if data.get("type") == "Feature":
        return [data]
    if "coordinates" in data or data.get("type") == "GeometryCollection":
        return [{"type": "Feature", "geometry": data, "properties": {}}]
    raise ValueError(f"Unrecognised GeoJSON structure in {path}")


def polygons_of(geometry):
    """Return a list of polygons, each a list of rings ([exterior, *holes])."""
    if not geometry:
        return []
    gtype = geometry.get("type")
    if gtype == "Polygon":
        return [geometry["coordinates"]]
    if gtype == "MultiPolygon":
        return list(geometry["coordinates"])
    if gtype == "GeometryCollection":
        out = []
        for geom in geometry.get("geometries", []):
            out.extend(polygons_of(geom))
        return out
    return []


def feature_name(props, index):
    name = props.get("name")
    if not name:
        name = (props.get("classification") or {}).get("name")
    return str(name) if name else f"Polygon_{index}"


def feature_color(props):
    color = props.get("color") or (props.get("classification") or {}).get("color")
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        return [int(c) for c in color[:3]]
    return None


def parse_annotations(features):
    """Turn GeoJSON features into simple dicts with raw (unscaled) rings."""
    annotations = []
    for i, feat in enumerate(features, start=1):
        geometry = feat.get("geometry") or {}
        props = feat.get("properties") or {}
        name = feature_name(props, i)

        polygons = polygons_of(geometry)
        if not polygons:
            print(f"  WARNING: feature '{name}' has geometry type "
                  f"'{geometry.get('type')}', which is not a polygon - skipped.")
            continue

        rings = []
        for poly in polygons:
            if len(poly) > 1:
                print(f"  WARNING: feature '{name}' has {len(poly) - 1} hole(s); "
                      "OMERO polygons cannot have holes, only the outer "
                      "boundary is uploaded.")
            exterior = [(float(p[0]), float(p[1])) for p in poly[0]]
            # GeoJSON rings repeat the first point at the end; OMERO doesn't need it
            if len(exterior) > 1 and exterior[0] == exterior[-1]:
                exterior = exterior[:-1]
            if len(exterior) < 3:
                print(f"  WARNING: a polygon of '{name}' has < 3 points - skipped.")
                continue
            rings.append(exterior)

        if not rings:
            continue

        plane = geometry.get("plane") or {}
        annotations.append({
            "name": name,
            "color": feature_color(props),
            "z": plane.get("z"),
            "t": plane.get("t"),
            "rings": rings,
        })
    return annotations


# --------------------------------------------------------------------------- #
# OMERO helpers
# --------------------------------------------------------------------------- #

def rgba_to_int(r, g, b, a=255):
    """OMERO stores colours as signed 32-bit RGBA integers."""
    value = (r << 24) | (g << 16) | (b << 8) | a
    return value - (1 << 32) if value >= (1 << 31) else value


def ask_credentials(args):
    host = args.host or os.environ.get("OMERO_HOST") or input("OMERO host: ").strip()
    port = args.port or int(os.environ.get("OMERO_PORT") or 4064)
    user = args.user or os.environ.get("OMERO_USER") or input("OMERO username: ").strip()
    password = os.environ.get("OMERO_PASSWORD") or getpass.getpass("OMERO password: ")
    return host, port, user, password


def image_pixel_size(image):
    """Return (x, y) pixel size of an OMERO image in microns (None if unset)."""
    sizes = []
    for getter in (image.getPixelSizeX, image.getPixelSizeY):
        size = getter(units="MICROMETER")
        sizes.append(size.getValue() if size is not None else None)
    return tuple(sizes)


def make_roi(ann, image_id, add_label, add_color):
    from omero.model import ImageI, PolygonI, RoiI
    from omero.rtypes import rint, rstring

    roi = RoiI()
    roi.setName(rstring(ann["name"]))
    roi.setImage(ImageI(image_id, False))

    for ring in ann["rings"]:
        poly = PolygonI()
        poly.setPoints(rstring(" ".join(f"{x:.3f},{y:.3f}" for x, y in ring)))
        if add_label:
            poly.setTextValue(rstring(ann["name"]))
        if ann["z"] is not None:
            poly.setTheZ(rint(int(ann["z"])))
        if ann["t"] is not None:
            poly.setTheT(rint(int(ann["t"])))
        if add_color and ann["color"]:
            r, g, b = ann["color"]
            poly.setStrokeColor(rint(rgba_to_int(r, g, b, 255)))
            poly.setFillColor(rint(rgba_to_int(r, g, b, 0)))
        roi.addShape(poly)
    return roi


def scale_annotations(annotations, sx, sy):
    """Convert microns -> pixels by dividing by the pixel size."""
    for ann in annotations:
        ann["rings"] = [[(x / sx, y / sy) for x, y in ring] for ring in ann["rings"]]


def check_bounds(annotations, size_x, size_y):
    xs = [x for a in annotations for r in a["rings"] for x, _ in r]
    ys = [y for a in annotations for r in a["rings"] for _, y in r]
    if not xs:
        return
    print(f"  Annotation extent (px): x {min(xs):.0f}..{max(xs):.0f}, "
          f"y {min(ys):.0f}..{max(ys):.0f}; image size {size_x} x {size_y}")
    tol = 0.05  # allow annotations to stick out a little
    if (min(xs) < -tol * size_x or max(xs) > (1 + tol) * size_x
            or min(ys) < -tol * size_y or max(ys) > (1 + tol) * size_y):
        print("  WARNING: annotations extend well beyond the image - "
              "check --units / --pixel-size.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Upload GeoJSON polygons to an OMERO image as named ROIs.")
    p.add_argument("geojson", help="Path to the GeoJSON file")
    p.add_argument("image_id", type=int, help="OMERO Image ID")
    p.add_argument("--units", choices=["pixels", "microns"], default="pixels",
                   help="Units of the GeoJSON coordinates (default: pixels)")
    p.add_argument("--pixel-size", type=float,
                   help="Pixel size in microns/pixel (X, and Y unless "
                        "--pixel-size-y is given). If omitted with --units "
                        "microns, it is read from OMERO.")
    p.add_argument("--pixel-size-y", type=float,
                   help="Pixel size in Y in microns/pixel, if different from X")
    p.add_argument("--host", help="OMERO host (else $OMERO_HOST or prompt)")
    p.add_argument("--port", type=int, help="OMERO port (else $OMERO_PORT or 4064)")
    p.add_argument("--user", help="OMERO username (else $OMERO_USER or prompt)")
    p.add_argument("--replace", action="store_true",
                   help="Delete ALL existing ROIs on the image before uploading")
    p.add_argument("--no-label", action="store_true",
                   help="Do not show the name as text label on the shape")
    p.add_argument("--no-color", action="store_true",
                   help="Ignore colours from the GeoJSON")
    p.add_argument("--dry-run", action="store_true",
                   help="Parse and report only; do not connect to OMERO")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print(f"Reading {args.geojson}")
    annotations = parse_annotations(read_features(args.geojson))
    if not annotations:
        print("No polygon annotations found - nothing to upload.")
        return 1
    n_shapes = sum(len(a["rings"]) for a in annotations)
    print(f"  Found {len(annotations)} annotation(s) / {n_shapes} polygon(s): "
          + ", ".join(a["name"] for a in annotations))

    pixel_size = None
    if args.units == "microns" and args.pixel_size:
        pixel_size = (args.pixel_size, args.pixel_size_y or args.pixel_size)

    if args.dry_run:
        if args.units == "microns" and pixel_size is None:
            print("  (pixel size would be read from OMERO; coordinates shown unscaled)")
        elif pixel_size:
            scale_annotations(annotations, *pixel_size)
        for a in annotations:
            pts = sum(len(r) for r in a["rings"])
            print(f"  - {a['name']}: {len(a['rings'])} polygon(s), {pts} points, "
                  f"z={a['z']}, color={a['color']}")
        print("Dry run - nothing uploaded.")
        return 0

    from omero.gateway import BlitzGateway

    host, port, user, password = ask_credentials(args)
    conn = BlitzGateway(user, password, host=host, port=port, secure=True)
    print(f"Connecting to {host}:{port} as {user} ...")
    if not conn.connect():
        print("ERROR: could not log in to OMERO (check host / credentials).")
        return 1

    try:
        # Look the image up across all groups, then work in the image's group
        conn.SERVICE_OPTS.setOmeroGroup("-1")
        image = conn.getObject("Image", args.image_id)
        if image is None:
            print(f"ERROR: image {args.image_id} not found or not accessible.")
            return 1
        conn.SERVICE_OPTS.setOmeroGroup(image.getDetails().getGroup().getId())
        print(f"  Image {args.image_id}: {image.getName()}")

        if args.units == "microns":
            if pixel_size is None:
                psx, psy = image_pixel_size(image)
                if psx is None:
                    print("ERROR: image has no pixel size in OMERO; "
                          "please provide --pixel-size.")
                    return 1
                pixel_size = (psx, psy or psx)
                print(f"  Pixel size from OMERO: {pixel_size[0]} x {pixel_size[1]} um")
            scale_annotations(annotations, *pixel_size)

        check_bounds(annotations, image.getSizeX(), image.getSizeY())

        if args.replace:
            result = conn.getRoiService().findByImage(args.image_id, None, conn.SERVICE_OPTS)
            roi_ids = [roi.getId().getValue() for roi in result.rois]
            if roi_ids:
                print(f"  Deleting {len(roi_ids)} existing ROI(s) ...")
                conn.deleteObjects("Roi", roi_ids, wait=True)

        update = conn.getUpdateService()
        for ann in annotations:
            roi = make_roi(ann, args.image_id, not args.no_label, not args.no_color)
            saved = update.saveAndReturnObject(roi, conn.SERVICE_OPTS)
            print(f"  Uploaded '{ann['name']}' -> ROI {saved.getId().getValue()}")

        print(f"Done: {len(annotations)} ROI(s) added to image {args.image_id}.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
