import os
import io
import json
import logging
from getpass import getpass
from numpy import random
from datetime import datetime, timezone
import omero
import numpy as np
import scanpy as sc
import pandas as pd
from tqdm import tqdm
from PIL import Image
from omero.gateway import BlitzGateway
from shapely import affinity, plotting
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt
import squidpy as sq
import fire
import yaml
import hashlib
from datetime import datetime

log = logging.getLogger(__name__)


def _seed_from_time(known_time):
    if isinstance(known_time, datetime):
        s = known_time.isoformat()
    else:
        s = str(known_time)
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little", signed=False)

def _parse_omero_points(points_obj):
    """Parse OMERO points string -> (M,2) array of (x,y)."""
    if points_obj is None:
        return np.zeros((0,2), float)
    # try common access patterns
    try:
        val = points_obj.getValue() if hasattr(points_obj, "getValue") else points_obj
    except Exception:
        val = points_obj
    s = str(val).strip()
    if not s:
        return np.zeros((0,2), float)
    toks = s.replace("\n"," ").replace("\t"," ").split()
    pts = []
    for tok in toks:
        if "," in tok:
            a,b = tok.split(",",1)
        else:
            parts = tok.split()
            if len(parts)!=2:
                continue
            a,b = parts
        try:
            pts.append((float(a), float(b)))
        except ValueError:
            continue
    return np.asarray(pts, dtype=float)

def _densify_along_path(points_xy, n, closed=False):
    """
    Return n points sampled sequentially along the path defined by points_xy (x,y).
    If closed=True, path is closed (last->first edge included).
    Points returned in path order as (y,x).
    """
    if points_xy.shape[0] == 0 or n <= 0:
        return np.zeros((0,2), float)
    pts = points_xy.copy()
    if closed:
        # repeat first point at end for edge calculation
        pts_ext = np.vstack([pts, pts[0:1]])
    else:
        pts_ext = pts
    segs = pts_ext[1:] - pts_ext[:-1]   # (M-1, 2)
    seg_len = np.hypot(segs[:,0], segs[:,1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total == 0:
        # degenerate -> repeat single point
        xy = np.tile(pts[0], (n,1))
        return xy[:, ::-1]  # to (y,x)
    # positions along path equally spaced from 0 .. total (excluded total if closed to avoid duplicate)
    if closed:
        dists = np.linspace(0, total, n, endpoint=False)
    else:
        dists = np.linspace(0, total, n, endpoint=True)
    out_x = np.empty(n, dtype=float)
    out_y = np.empty(n, dtype=float)
    for i, d in enumerate(dists):
        # find segment index
        idx = np.searchsorted(cum, d, side='right') - 1
        idx = min(max(idx, 0), len(seg_len)-1)
        seg_d = d - cum[idx]
        frac = seg_d / (seg_len[idx] + 1e-12)
        sx, sy = pts_ext[idx]
        ex, ey = pts_ext[idx+1]
        xpt = sx + frac * (ex - sx)
        ypt = sy + frac * (ey - sy)
        out_x[i] = xpt
        out_y[i] = ypt
    return np.column_stack([out_y, out_x])  # (y,x)

def roi_to_points(
    roi_or_shapes,
    known_time,
    spots_per_shape=200,
    total_spots=None,
    clip_to_image_shape=None,
    return_xy=False,
):
    """Return sequential (y,x) points per-shape suitable for constructing polygons/lines."""
    # We keep deterministic seeding for any incidental randomness (not used here)
    rng = np.random.default_rng(_seed_from_time(known_time))

    # grab shapes
    if hasattr(roi_or_shapes, "copyShapes"):
        shapes = list(roi_or_shapes.copyShapes())
    elif hasattr(roi_or_shapes, "iterateShapes"):
        shapes = list(roi_or_shapes.iterateShapes())
    else:
        shapes = list(roi_or_shapes)

    supported = [sh for sh in shapes if any(k in type(sh).__name__.lower() for k in ("rectangle","ellipse","polygon","polyline"))]
    if not supported:
        return np.zeros((0,2), float)

    # allocate counts
    if total_spots is not None:
        base = total_spots // len(supported)
        rem = total_spots % len(supported)
        n_list = [base + (1 if i < rem else 0) for i in range(len(supported))]
    else:
        n_list = [int(spots_per_shape)] * len(supported)

    out_chunks = []
    for sh, n in zip(supported, n_list):
        if n <= 0:
            continue
        nm = type(sh).__name__.lower()

        # Rectangle -> build 4 corner vertices (x,y), then densify along closed perimeter
        if "rectangle" in nm:
            x = float(sh.getX().getValue()); y = float(sh.getY().getValue())
            w = float(sh.getWidth().getValue()); h = float(sh.getHeight().getValue())
            # vertices in clockwise order starting top-left
            verts = np.array([
                [x,     y    ],
                [x + w, y    ],
                [x + w, y + h],
                [x,     y + h],
            ], dtype=float)
            pts = _densify_along_path(verts, n, closed=True)

        # Ellipse -> parametric perimeter points in order
        elif "ellipse" in nm:
            cx = float(sh.getX().getValue()); cy = float(sh.getY().getValue())
            rx = float(sh.getRadiusX().getValue()); ry = float(sh.getRadiusY().getValue())
            thetas = np.linspace(0, 2*np.pi, n, endpoint=False)
            xs = cx + rx * np.cos(thetas)
            ys = cy + ry * np.sin(thetas)
            pts = np.column_stack([ys, xs])

        # Polygon -> parse vertices and densify along closed polygon
        elif "polygon" in nm:
            poly_xy = _parse_omero_points(getattr(sh, "getPoints", lambda: sh.getPoints())())
            if poly_xy.shape[0] < 3:
                continue
            pts = _densify_along_path(poly_xy, n, closed=True)

        # Polyline -> parse vertices and densify along open path
        elif "polyline" in nm:
            poly_xy = _parse_omero_points(getattr(sh, "getPoints", lambda: sh.getPoints())())
            if poly_xy.shape[0] < 2:
                continue
            pts = _densify_along_path(poly_xy, n, closed=False)

        else:
            continue

        out_chunks.append(pts)

    if not out_chunks:
        return np.zeros((0,2), float)

    pts = np.vstack(out_chunks)

    if clip_to_image_shape is not None:
        H, W = clip_to_image_shape
        pts[:,0] = np.clip(pts[:,0], 0, H-1)
        pts[:,1] = np.clip(pts[:,1], 0, W-1)

    if return_xy:
        pts = pts[:, [1,0]]
    pts[:, [0, 1]] = pts[:, [1, 0]]
    return pts


def ReadConfFile(FilePath):
    with open(FilePath, 'r') as file:
        data = yaml.safe_load(file)
    omero_image_id = data['omero_image_id']
    column_name_x = data['column_name_x']
    column_name_y = data['column_name_y']
    rot_angle = data['rot_angle']
    flipX = data['flipX']
    flipY = data['flipY']
    pixelsize = data['pixelsize']
    out_folder = data['output_folder']
    save_images = data['save_images']
    # NEW: optional flag to save ROI positions
    save_rois_positions = data.get('save_rois_positions', False)

    segmentation_csv_path_list = []
    for p in data['segmentation_csv']:
        segmentation_csv_path_list.append(p)

    return (
        omero_image_id,
        column_name_x,
        column_name_y,
        segmentation_csv_path_list,
        rot_angle,
        flipX,
        flipY,
        pixelsize,
        out_folder,
        save_images,
        save_rois_positions,
    )


# -------------------------------------------------------------------------
# Helpers to extract coordinates from various OMERO shape types
# -------------------------------------------------------------------------

def _rdouble_to_float(rdouble_obj):
    """Safely convert an OMERO RDouble-like object (with .val) to float."""
    try:
        return float(rdouble_obj.val)
    except Exception:
        return None


def _extract_shape_coordinates(shape):
    """
    Extract coordinates and metadata from a single OMERO shape or an OMERO Roi container.

    If 'shape' is a Roi container (RoiI), this will try to obtain the primary shape
    (shape = roi.getPrimaryShape()) or the first child from roi.copyShapes() and parse that.

    Returns a dict with:
      - shape_id
      - shape_type   (raw OMERO class name)
      - type         (one of: 'polygon', 'line', 'rectangle', 'circle' or None)
      - text         (label, if available)
      - coords       (geometry)
    """
    # If a Roi container was accidentally passed in, try to get a concrete shape out of it
    try:
        st_name = shape.__class__.__name__.lower()
    except Exception:
        st_name = ""

    if 'roii' in st_name:
        # it's a Roi container — prefer its primary shape, otherwise the first child shape
        try:
            primary = None
            if hasattr(shape, "getPrimaryShape"):
                primary = shape.getPrimaryShape()
            if primary is None and hasattr(shape, "copyShapes"):
                shapes = shape.copyShapes()
                # depending on the OMERO wrapper, copyShapes may return a list-like object or generator
                if shapes:
                    # shapes may be an OMERO wrapper object; try indexing first, else iterate
                    try:
                        primary = shapes[0]
                    except Exception:
                        # iterate to get first
                        for s in shapes:
                            primary = s
                            break
            if primary is not None:
                # recurse on the concrete shape
                return _extract_shape_coordinates(primary)
            # If we couldn't find a child shape, return a minimal empty record
            return {
                'shape_id': None,
                'shape_type': 'RoiI',
                'type': None,
                'text': None,
                'coords': None,
            }
        except Exception:
            log.exception("Failed to extract primary shape from RoiI")
            return {
                'shape_id': None,
                'shape_type': 'RoiI',
                'type': None,
                'text': None,
                'coords': None,
            }

    # From here on, 'shape' should be a concrete shape (PolygonI, RectangleI, EllipseI, etc.)
    shape_type = shape.__class__.__name__
    st_lower = shape_type.lower()

    # Try to get shape id
    shape_id = None
    try:
        if hasattr(shape, 'id') and shape.id is not None:
            shape_id = int(shape.id.val)
    except Exception:
        shape_id = None

    # Try to get text / label
    text_label = None
    try:
        if hasattr(shape, "getTextValue"):
            tv = shape.getTextValue()
            if tv is not None:
                text_label = tv.val
    except Exception:
        # some shapes don't implement getTextValue
        pass

    coords = None
    geom_type = None  # 'polygon', 'line', 'rectangle', 'circle'

    # Polygon / Polyline / curve-like (points field)
    if 'polygon' in st_lower or 'polyline' in st_lower:
        try:
            if hasattr(shape, "getPoints"):
                pts_obj = shape.getPoints()
            else:
                pts_obj = None
            if pts_obj is not None:
                # pts_obj may be an OMERO wrapper where .val is the string of "x,y x,y ..."
                pts_str = getattr(pts_obj, "val", None)
                if pts_str is None and isinstance(pts_obj, (str, bytes)):
                    pts_str = pts_obj
                if pts_str:
                    pts_str = pts_str.strip()
                    pts = [
                        list(map(float, xy.split(',')))
                        for xy in pts_str.split(' ')
                        if xy
                    ]
                    coords = pts
                    # PolygonI -> polygon; PolylineI -> line/curve
                    if 'polygon' in st_lower:
                        geom_type = 'polygon'
                    else:
                        geom_type = 'line'
        except Exception:
            log.exception("Failed to parse points for Polygon/Polyline shape")

    # Rectangle
    elif 'rectangle' in st_lower:
        try:
            x = _rdouble_to_float(shape.getX())
            y = _rdouble_to_float(shape.getY())
            w = _rdouble_to_float(shape.getWidth())
            h = _rdouble_to_float(shape.getHeight())
            coords = {'x': x, 'y': y, 'width': w, 'height': h}
            geom_type = 'rectangle'
        except Exception:
            log.exception("Failed to parse Rectangle shape")
    
    # Ellipse / Circle
    elif 'ellipse' in st_lower:
        try:
            x = _rdouble_to_float(shape.getX())
            y = _rdouble_to_float(shape.getY())
            rx = _rdouble_to_float(shape.getRadiusX())
            ry = _rdouble_to_float(shape.getRadiusY())
            coords = {'x': x, 'y': y, 'radius_x': rx, 'radius_y': ry}
            geom_type = 'circle'
        except Exception:
            log.exception("Failed to parse Ellipse shape")

    # Line (straight line between 2 points)
    elif 'line' in st_lower:
        try:
            # simple LineI uses getX1/getY1/getX2/getY2
            x1 = _rdouble_to_float(shape.getX1())
            y1 = _rdouble_to_float(shape.getY1())
            x2 = _rdouble_to_float(shape.getX2())
            y2 = _rdouble_to_float(shape.getY2())
            coords = [[x1, y1], [x2, y2]]
            geom_type = 'line'
        except Exception:
            log.exception("Failed to parse Line shape")

    # Fallback: if we have points via getPoints(), use them
    if coords is None and hasattr(shape, 'getPoints'):
        try:
            pts_obj = shape.getPoints()
            pts_str = getattr(pts_obj, "val", None)
            if pts_str is None and isinstance(pts_obj, (str, bytes)):
                pts_str = pts_obj
            if pts_str:
                pts_str = pts_str.strip()
                pts = [
                    list(map(float, xy.split(',')))
                    for xy in pts_str.split(' ')
                    if xy
                ]
                coords = pts
                if geom_type is None:
                    geom_type = 'polygon'
        except Exception:
            # silently ignore fallback errors
            pass

    return {
        'shape_id': shape_id,
        'shape_type': shape_type,
        'type': geom_type,   # 'polygon', 'line', 'rectangle', 'circle' or None
        'text': text_label,
        'coords': coords,
    }


# -------------------------------------------------------------------------
# Collect ROIs from OMERO, with both polygon points and full shape geometry
# -------------------------------------------------------------------------

def collect_ROIs_from_OMERO(omero_username, omero_password, omero_host, omero_image_id):
    ROIs = []
    all_annotations = []  # will store full geometry for all shapes
    log.info(f"Connecting to OMERO at {omero_host}")

    with BlitzGateway(omero_username, omero_password, host=omero_host, secure=True) as conn:
        if not conn.connect():
            raise RuntimeError("Could not connect to OMERO. Check host/user/password.")

        log.info(f"Connected: isConnected={conn.isConnected()}")

        # make sure ID is int
        try:
            image_id_int = int(omero_image_id)
        except ValueError:
            raise ValueError(f"omero_image_id '{omero_image_id}' is not an integer")

        # search image
        log.info(f"Looking for ImageId {image_id_int}")
        conn.SERVICE_OPTS.setOmeroGroup('-1')   # all groups

        image = conn.getObject("Image", image_id_int)
        print("IMAGE OBJECT:", image)

        if image is None:
            raise ValueError(
                f"Could not find Image with ID {image_id_int}. "
                "Possible reasons:\n"
                "  * Wrong image ID\n"
                "  * You don’t have permission / not in the right group\n"
                "  * Wrong OMERO host or login details"
            )

        log.info(f"Found image id={image.id} name='{image.name}'")
        log.info(
            f"Found image in group id={image.details.group.id.val} "
            f"name='{image.details.group.name.val}'"
        )
        
        log.info("Storing rendered thumbnail in memory for QC")
        img_data = image.getThumbnail()  # tiny preview image
        rendered_thumb = Image.open(io.BytesIO(img_data))
        
        group_id = image.details.group.id
        conn.setGroupForSession(group_id.val)

        # get image ROIs
        roi_service = conn.getRoiService()
        log.info("Retrieving ROIs")
        result = roi_service.findByImage(image.id, None)

        for roi in result.rois:
            roi_id = int(roi.id.val)
            shapes_coords = []
            roi_name = None

            try:
                primary_shape = roi.getPrimaryShape()

                # Try to get ROI name from primary shape textValue
                try:
                    tv = primary_shape.getTextValue()
                    if tv is not None and tv.val:
                        roi_name = tv.val
                    else:
                        roi_name = f"ROI_{roi_id}"
                except Exception:
                    roi_name = f"ROI_{roi_id}"

                # ORIGINAL behaviour: parse polygon points from primary shape
                try:
                    print(roi_name)
                    #pts_obj = primary_shape.getPoints()
                    known_time = datetime.now(timezone.utc).isoformat()
                    points = roi_to_points(roi, known_time)
                    print(points[:10])
                    if points is not None:
                        
                        ROIs.append({
                            "name": roi_name,
                            "points": points
                        })
                        log.debug(
                            f"Found ROI id={roi_id} name='{roi_name}' "
                            f"type={primary_shape.__class__.__name__}"
                        )
                except Exception:
                    log.exception(
                        "Failed to parse ROI primary shape for polygon points "
                        "(used for cell assignment), skipping for ROIs list"
                    )

                # NEW: collect coordinates for all shapes in this ROI
                try:
                    for s in roi.copyShapes():
                        sc_dict = _extract_shape_coordinates(s)
                        shapes_coords.append(sc_dict)
                except Exception:
                    log.exception("Failed to extract coordinates for shapes in ROI")

            except Exception:
                log.exception("Failed to parse ROI, skipping")
                continue

            all_annotations.append(
                {
                    'roi_id': roi_id,
                    'roi_name': roi_name,
                    'shapes': shapes_coords,
                }
            )

    log.info(f"Found {len(ROIs)} ROIs in total (with polygon points for cell assignment)")
    log.info(f"Collected full coordinates for {len(all_annotations)} ROIs (all shapes)")
    return ROIs, image, all_annotations


def rotate_flip_polygon(polygon, img_center, rot_angle, flipX=False, flipY=False):
    angle = np.deg2rad(rot_angle)
    move1_matrix = np.array([[1, 0, img_center[0]], [0, 1, img_center[1]], [0, 0, 1]])
    rot_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    flip_matrix = np.eye(3, 3)
    if flipX:
        flip_matrix[0, 0] = -1
    if flipY:
        flip_matrix[1, 1] = -1

    if np.abs(rot_angle) == 90 or np.abs(rot_angle) == 270:
        move2_matrix = np.array([[1, 0, -img_center[1]], [0, 1, -img_center[0]], [0, 0, 1]])
    else:
        move2_matrix = np.array([[1, 0, -img_center[0]], [0, 1, -img_center[1]], [0, 0, 1]])
    
    tr_mat = np.dot(move1_matrix, rot_matrix)
    tr_mat = np.dot(tr_mat, flip_matrix)
    tr_mat = np.dot(tr_mat, move2_matrix)

    matrix_elements = [
        tr_mat[0, 0], tr_mat[0, 1],
        tr_mat[1, 0], tr_mat[1, 1],
        tr_mat[0, 2], tr_mat[1, 2],
    ]
    return affinity.affine_transform(polygon, matrix_elements)


def rotate_flip_all_polygons(ROIs, image, rot_angle, flipX, flipY):
    New_polygons = []
    for roi in ROIs:
        polygon = rotate_flip_polygon(
            Polygon(roi['points']),
            [image.getSizeX() / 2, image.getSizeY() / 2],
            rot_angle,
            flipX,
            flipY,  # fixed typo: was FlipY
        )
        New_polygons.append(polygon)
    return New_polygons


def assign_cell_to_annotation(segmentation_table, ROIs, New_polygons, column_name_x, column_name_y, pixelsize):
    annotations = []
    log.info("Calling cell positions inside ROIs.")
    for barcode, row in tqdm(segmentation_table.iterrows(), total=segmentation_table.shape[0]):
        roi_annoation = []
        for roi, polygon in zip(ROIs, New_polygons):
            point = Point([row[column_name_x] / pixelsize, row[column_name_y] / pixelsize])
            if polygon.contains(point):
                roi_annoation.append(roi['name'])
        if not roi_annoation:
            roi_annoation = ["N-A"]
        annotations.append([barcode, "; ".join(set(roi_annoation))])
    return annotations


def plot_small_image(segmentation_table, out_folder, sample_name, column_name_x, column_name_y):
    ann_list_unique = list(set(segmentation_table['annotation']))
    fig, axs = plt.subplots(1, 1, figsize=(20, 20))
    for name_roi in ann_list_unique:
        sub_segm_table = segmentation_table[segmentation_table['annotation'] == name_roi]
        random_color = [
            random.randint(100) / 100,
            random.randint(100) / 100,
            random.randint(100) / 100,
        ]
        plt.scatter(sub_segm_table[column_name_x], sub_segm_table[column_name_y],
                    color=random_color, s=1)
    path_fig = out_folder + '/' + sample_name[:-2] + '.png'
    plt.axis('equal')
    axs.legend(ann_list_unique, fontsize=15, markerscale=5)
    plt.savefig(path_fig, format="jpg", dpi=300)


def add_annotations_to_table(segmentation_table, annotations):
    ann_list = []
    for a in annotations:
        ann_list.append(a[1])
    segmentation_table['annotation'] = ann_list
    return segmentation_table


# -------------------------------------------------------------------------
# NEW: save OMERO annotations to separate JSON file
# -------------------------------------------------------------------------

def _strip_suffixes(name):
    """Strip common suffixes like .csv, .csv.gz, .tsv, .tsv.gz."""
    base = name
    for ext in ('.csv.gz', '.tsv.gz', '.csv', '.tsv', '.gz'):
        if base.endswith(ext):
            base = base[: -len(ext)]
    return base


def save_omero_annotations_to_json(
    all_annotations,
    out_folder,
    sample_name,
    image,
):
    """
    Save OMERO ROI geometries into a JSON file:

    {
      "image_id": ...,
      "image_name": ...,
      "image_size_x": ...,
      "image_size_y": ...,
      "coordinate_units": "pixels",
      "rois": [ {roi_id, roi_name, shapes: [ {shape_id, shape_type, type, text, coords}, ...]}, ... ]
    }
    """
    payload = {
        'image_id': int(image.id),
        'image_name': image.getName(),
        'image_size_x': int(image.getSizeX()),
        'image_size_y': int(image.getSizeY()),
        'coordinate_units': 'pixels',
        'rois': all_annotations,
    }

    base = _strip_suffixes(sample_name)
    out_path = os.path.join(out_folder, f"{base}_omero_annotations.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info(f"Saved OMERO ROI annotations JSON to: {out_path}")

def get_OMERO_credentials():
    logging.basicConfig(format='%(asctime)s %(message)s')    
    log.setLevel(logging.DEBUG)
    omero_host = "wsi-omero-prod-02.internal.sanger.ac.uk"
    omero_username = input("Username$")
    omero_password = getpass("Password$")
    return omero_host, omero_username, omero_password

def main(ConfFilePath):
    # here we assume one segmentation csv per one omero id!!
    omero_host, omero_username, omero_password = get_OMERO_credentials()
    (
        omero_image_id,
        column_name_x,
        column_name_y,
        segmentation_csv_path_list,
        rot_angle,
        flipX,
        flipY,
        pixelsize,
        out_folder,
        save_images,
        save_rois_positions,   # NEW flag
    ) = ReadConfFile(ConfFilePath)

    (
        omero_image_id,
        column_name_x,
        column_name_y,
        segmentation_csv_path_list,
        rot_angle,
        flipX,
        flipY,
        pixelsize,
        out_folder,
        save_images,
        save_rois_positions,   # NEW flag
    ) = ReadConfFile(ConfFilePath)
    
    # Collect polygons for cell assignment AND full geometry for all shapes
    ROIs, image, all_annotations = collect_ROIs_from_OMERO(
        omero_username,
        omero_password,
        omero_host,
        omero_image_id,
    )

    if len(ROIs) == 0:
        print('0 ROIs were found! Check whether you have any annotations in OMERO or are you owner of dataset?')
    else:
        for segmentation_csv_path in segmentation_csv_path_list:
            sample_name = os.path.basename(segmentation_csv_path)
            print('Working on segnmentation for: ' + sample_name)
            segmentation_table = pd.read_csv(segmentation_csv_path)

            if rot_angle != 0 and flipX is not False and flipY is not False:
                New_polygons = rotate_flip_all_polygons(ROIs, image, rot_angle, flipX, flipY)
            else:
                New_polygons = []
                for roi in ROIs:
                    New_polygons.append(Polygon(roi['points']))

            annotations = assign_cell_to_annotation(
                segmentation_table,
                ROIs,
                New_polygons,
                column_name_x,
                column_name_y,
                pixelsize,
            )
            segmentation_table = add_annotations_to_table(segmentation_table, annotations)
            path_csv = out_folder + '/' + sample_name
            segmentation_table.to_csv(path_csv, index=False)

            if save_images:
                plot_small_image(segmentation_table, out_folder, sample_name, column_name_x, column_name_y)

            # NEW: optionally save ROI positions to JSON
            if save_rois_positions:
                save_omero_annotations_to_json(
                    all_annotations=all_annotations,
                    out_folder=out_folder,
                    sample_name=sample_name,
                    image=image,
                )


if __name__ == "__main__":
    fire.Fire(main)
