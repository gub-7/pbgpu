"""
Fiducial marker rendering for multi-view reconstruction.

Adds synthetic reference geometry (2 orange squares + 1 blue circle)
to each view image so that DUSt3R / MASt3R have strong, unambiguous
multi-view correspondences beyond the subject itself.

3D layout
---------
The markers live in world space above the subject:

    Blue circle:       (  0,   -D,  Z_h )   -- centred, behind object
    Orange square L:   ( -S,   +D,  Z_h )   -- left,  in front of object
    Orange square R:   ( +S,   +D,  Z_h )   -- right, in front of object

World convention:  +X = right, +Y = forward, +Z = up.

The circle is placed behind the object and the squares in front so
that from the side camera the circle and squares appear at different
horizontal positions (good separation for correspondences).

Rendering per view
------------------
FRONT  (camera on -Y axis, sees X and Z):
    All three markers appear in a row above the subject.  Due to
    perspective (different Y depths) the raw projections differ
    slightly in height and size.  We snap all three to the same
    vertical pixel and normalise sizes so the front view shows a
    clean horizontal row.

SIDE  (camera on +X axis, sees Y and Z):
    The circle (Y = -D) appears on the LEFT of the image.
    Both squares (Y = +D, different X) project to nearby positions
    and are merged into a single square on the RIGHT.  The circle
    is rendered larger than the square.

TOP  (camera on +Z axis, sees X and Y):
    The circle (Y = -D) appears at the TOP of the image.
    The two squares (Y = +D) appear at the BOTTOM, separated
    horizontally.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from api.models import ViewLabel, ViewSpec
from pipelines.preprocess import BACKGROUND_GRAY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Marker colours
# ---------------------------------------------------------------------------

ORANGE = (255, 140, 0)
BLUE = (30, 100, 255)


# ---------------------------------------------------------------------------
# 3D marker positions (world space)
# ---------------------------------------------------------------------------

# Vertical offset above object centre (world +Z)
_Z_HEIGHT = 0.45

# Forward/backward offset (world +Y / -Y).
# The circle sits behind the object, the squares sit in front.
# This gives clean separation in the side and top views.
_DEPTH_OFFSET = 0.25

# Horizontal spread of the two squares (world +/-X)
_X_SPREAD = 0.22

# World-space radius for each marker type.
# The circle is physically larger so it appears bigger from the side,
# matching the user's requirement ("blue circle should be large and
# the square should be a touch smaller").
_CIRCLE_RADIUS_3D = 0.042
_SQUARE_RADIUS_3D = 0.030

CIRCLE_POS_3D = np.array([0.0, -_DEPTH_OFFSET, _Z_HEIGHT])
SQUARE_L_POS_3D = np.array([-_X_SPREAD, _DEPTH_OFFSET, _Z_HEIGHT])
SQUARE_R_POS_3D = np.array([_X_SPREAD, _DEPTH_OFFSET, _Z_HEIGHT])


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------


def _project_point(
    point_world: np.ndarray,
    R_w2c: np.ndarray,
    t_w2c: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[float, float, float]:
    """
    Project a 3D world point to 2D pixel coordinates.

    Returns (u, v, depth_cam) where depth_cam is the Z coordinate in
    camera space (positive = in front of camera).
    """
    pt_cam = R_w2c @ point_world + t_w2c
    z = pt_cam[2]
    if z <= 0:
        return -1.0, -1.0, z  # behind camera

    u = fx * pt_cam[0] / z + cx
    v = fy * pt_cam[1] / z + cy
    return float(u), float(v), float(z)


def _apparent_radius(
    world_radius: float,
    depth: float,
    focal: float,
) -> float:
    """Compute the pixel radius of a sphere at a given depth."""
    if depth <= 0:
        return 0.0
    return focal * world_radius / depth


# ---------------------------------------------------------------------------
# Per-view rendering
# ---------------------------------------------------------------------------


def _get_camera_matrices(pose) -> tuple[np.ndarray, np.ndarray]:
    """Extract R_w2c and t_w2c from a SphericalPose."""
    from pipelines.camera_init import pose_to_extrinsics

    ext = pose_to_extrinsics(pose)
    R_w2c = np.array(ext.R_w2c).reshape(3, 3)
    t_w2c = np.array(ext.t_w2c)
    return R_w2c, t_w2c


def _draw_square(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    half_side: float,
    color: tuple[int, int, int],
) -> None:
    """Draw a filled square centred at (cx, cy)."""
    x0 = cx - half_side
    y0 = cy - half_side
    x1 = cx + half_side
    y1 = cy + half_side
    draw.rectangle([x0, y0, x1, y1], fill=color, outline=color)


def _draw_circle(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    radius: float,
    color: tuple[int, int, int],
) -> None:
    """Draw a filled circle centred at (cx, cy)."""
    x0 = cx - radius
    y0 = cy - radius
    x1 = cx + radius
    y1 = cy + radius
    draw.ellipse([x0, y0, x1, y1], fill=color, outline=color)


def render_markers_on_image(
    image: Image.Image,
    view_label: ViewLabel,
    pose,
    intrinsics,
) -> Image.Image:
    """
    Render fiducial markers onto an image for the given view.

    Parameters
    ----------
    image : the preprocessed image (RGB, square, gray background)
    view_label : which canonical view (FRONT, SIDE, TOP)
    pose : SphericalPose for this view
    intrinsics : CameraIntrinsics

    Returns
    -------
    A copy of the image with fiducial markers rendered on top.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    R_w2c, t_w2c = _get_camera_matrices(pose)
    fx, fy = intrinsics.fx, intrinsics.fy
    cx_img, cy_img = intrinsics.cx, intrinsics.cy

    # Project all three markers
    circle_uv = _project_point(CIRCLE_POS_3D, R_w2c, t_w2c, fx, fy, cx_img, cy_img)
    sq_l_uv = _project_point(SQUARE_L_POS_3D, R_w2c, t_w2c, fx, fy, cx_img, cy_img)
    sq_r_uv = _project_point(SQUARE_R_POS_3D, R_w2c, t_w2c, fx, fy, cx_img, cy_img)

    # Compute apparent sizes using per-marker 3D radii
    circle_r = _apparent_radius(_CIRCLE_RADIUS_3D, circle_uv[2], fx)
    sq_l_r = _apparent_radius(_SQUARE_RADIUS_3D, sq_l_uv[2], fx)
    sq_r_r = _apparent_radius(_SQUARE_RADIUS_3D, sq_r_uv[2], fx)

    if view_label == ViewLabel.FRONT:
        _render_front(draw, circle_uv, sq_l_uv, sq_r_uv,
                       circle_r, sq_l_r, sq_r_r, intrinsics)
    elif view_label == ViewLabel.SIDE:
        _render_side(draw, circle_uv, sq_l_uv, sq_r_uv,
                      circle_r, sq_l_r, sq_r_r, intrinsics)
    elif view_label == ViewLabel.TOP:
        _render_top(draw, circle_uv, sq_l_uv, sq_r_uv,
                     circle_r, sq_l_r, sq_r_r, intrinsics)

    return img


def _render_front(
    draw: ImageDraw.ImageDraw,
    circle_uv: tuple[float, float, float],
    sq_l_uv: tuple[float, float, float],
    sq_r_uv: tuple[float, float, float],
    circle_r: float,
    sq_l_r: float,
    sq_r_r: float,
    intrinsics,
) -> None:
    """
    FRONT view: render all three markers in a clean horizontal row.

    The circle and squares are at slightly different depths, so their
    projected v-coordinates and sizes differ.  We snap them to a
    uniform row: same vertical position, same marker size.
    """
    # Use the average v-coordinate for the row
    avg_v = (circle_uv[1] + sq_l_uv[1] + sq_r_uv[1]) / 3.0

    # Use the average radius for uniform sizing in the front view
    avg_r = (circle_r + sq_l_r + sq_r_r) / 3.0

    # Clamp to reasonable range
    min_r = 25.0
    max_r = 80.0
    avg_r = max(min_r, min(max_r, avg_r))

    # Ensure the row is above the subject (upper portion of image)
    max_v = intrinsics.height * 0.40
    row_v = min(avg_v, max_v)

    # Ensure markers don't go off the top of the image
    row_v = max(avg_r + 10, row_v)

    # Horizontal positions from projection (these are already correct)
    _draw_square(draw, sq_l_uv[0], row_v, avg_r, ORANGE)
    _draw_circle(draw, circle_uv[0], row_v, avg_r, BLUE)
    _draw_square(draw, sq_r_uv[0], row_v, avg_r, ORANGE)

    logger.debug(
        "FRONT markers: sq_l=(%.0f,%.0f) circle=(%.0f,%.0f) sq_r=(%.0f,%.0f) r=%.0f",
        sq_l_uv[0], row_v, circle_uv[0], row_v, sq_r_uv[0], row_v, avg_r,
    )


def _render_side(
    draw: ImageDraw.ImageDraw,
    circle_uv: tuple[float, float, float],
    sq_l_uv: tuple[float, float, float],
    sq_r_uv: tuple[float, float, float],
    circle_r: float,
    sq_l_r: float,
    sq_r_r: float,
    intrinsics,
) -> None:
    """
    SIDE view: circle on the left (large), merged square on the right (smaller).

    From the side camera, the two squares are at different X depths but
    the same Y, so they project to nearby positions.  We merge them into
    a single square at their average position.  The circle has a larger
    3D radius so it naturally appears bigger.
    """
    # Circle - use projected position directly
    c_u, c_v = circle_uv[0], circle_uv[1]

    # Merge the two squares into one at their average position/size
    merged_u = (sq_l_uv[0] + sq_r_uv[0]) / 2.0
    merged_v = (sq_l_uv[1] + sq_r_uv[1]) / 2.0
    merged_r = (sq_l_r + sq_r_r) / 2.0

    # Clamp sizes
    min_r = 20.0
    max_r = 90.0
    circle_r = max(min_r, min(max_r, circle_r))
    merged_r = max(min_r, min(max_r, merged_r))

    # Snap both markers to the same vertical position (same height)
    avg_v = (c_v + merged_v) / 2.0
    max_v = intrinsics.height * 0.40
    row_v = min(avg_v, max_v)
    row_v = max(max(circle_r, merged_r) + 10, row_v)

    _draw_circle(draw, c_u, row_v, circle_r, BLUE)
    _draw_square(draw, merged_u, row_v, merged_r, ORANGE)

    logger.debug(
        "SIDE markers: circle=(%.0f,%.0f,r=%.0f) square=(%.0f,%.0f,r=%.0f)",
        c_u, row_v, circle_r, merged_u, row_v, merged_r,
    )


def _render_top(
    draw: ImageDraw.ImageDraw,
    circle_uv: tuple[float, float, float],
    sq_l_uv: tuple[float, float, float],
    sq_r_uv: tuple[float, float, float],
    circle_r: float,
    sq_l_r: float,
    sq_r_r: float,
    intrinsics,
) -> None:
    """
    TOP view: circle at the top of the image, two squares at the bottom.

    The top-down camera sees the X-Y plane.  The circle (at Y = -D)
    appears near the top of the image, and the squares (at Y = +D)
    appear near the bottom, with horizontal separation.
    """
    # Clamp sizes
    min_r = 20.0
    max_r = 80.0
    circle_r = max(min_r, min(max_r, circle_r))
    sq_l_r = max(min_r, min(max_r, sq_l_r))
    sq_r_r = max(min_r, min(max_r, sq_r_r))

    # Use average size for the two squares (they're at the same depth)
    sq_avg_r = (sq_l_r + sq_r_r) / 2.0

    # Ensure markers are within image bounds
    W, H = intrinsics.width, intrinsics.height
    c_u = max(circle_r + 5, min(W - circle_r - 5, circle_uv[0]))
    c_v = max(circle_r + 5, min(H - circle_r - 5, circle_uv[1]))
    sl_u = max(sq_avg_r + 5, min(W - sq_avg_r - 5, sq_l_uv[0]))
    sl_v = max(sq_avg_r + 5, min(H - sq_avg_r - 5, sq_l_uv[1]))
    sr_u = max(sq_avg_r + 5, min(W - sq_avg_r - 5, sq_r_uv[0]))
    sr_v = max(sq_avg_r + 5, min(H - sq_avg_r - 5, sq_r_uv[1]))

    _draw_circle(draw, c_u, c_v, circle_r, BLUE)
    _draw_square(draw, sl_u, sl_v, sq_avg_r, ORANGE)
    _draw_square(draw, sr_u, sr_v, sq_avg_r, ORANGE)

    logger.debug(
        "TOP markers: circle=(%.0f,%.0f) sq_l=(%.0f,%.0f) sq_r=(%.0f,%.0f) r=%.0f",
        c_u, c_v, sl_u, sl_v, sr_u, sr_v, sq_avg_r,
    )


# ---------------------------------------------------------------------------
# Batch rendering for all views
# ---------------------------------------------------------------------------


def add_fiducial_markers(
    view_specs: list[ViewSpec],
    image_dir: Path,
    output_dir: Path,
    intrinsics=None,
) -> list[ViewSpec]:
    """
    Add fiducial markers to all view images for DUSt3R / MASt3R.

    Reads clean (background-removed, normalised) images from *image_dir*,
    renders perspective-correct fiducial markers onto each, and writes
    the marked images to *output_dir*.

    The original clean images in *image_dir* are preserved for Trellis
    and other downstream stages that don't need the markers.

    Parameters
    ----------
    view_specs : list of ViewSpecs with poses
    image_dir : directory containing clean preprocessed images
    output_dir : directory to write images with markers
    intrinsics : CameraIntrinsics (defaults to pipeline default)

    Returns
    -------
    Updated list of ViewSpec (filenames unchanged, images in output_dir).
    """
    if intrinsics is None:
        from pipelines.config import get_default_intrinsics
        intrinsics = get_default_intrinsics()

    output_dir.mkdir(parents=True, exist_ok=True)

    for vs in view_specs:
        src_path = image_dir / vs.image_filename
        if not src_path.exists():
            logger.warning("Image not found for markers: %s", src_path)
            continue

        with Image.open(src_path) as img:
            img = img.convert("RGB")
            marked = render_markers_on_image(img, vs.label, vs.pose, intrinsics)

        dst_path = output_dir / vs.image_filename
        marked.save(dst_path, format="PNG")

        logger.info(
            "Added fiducial markers to %s -> %s",
            vs.label.value, dst_path,
        )

    logger.info(
        "Fiducial markers added to %d views -> %s",
        len(view_specs), output_dir,
    )
    return view_specs


def strip_markers_from_pointcloud(
    points: np.ndarray,
    colors: Optional[np.ndarray],
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Remove fiducial marker geometry from a point cloud.

    Points near the known 3D marker positions are removed.
    This should be called after coarse reconstruction to clean up
    the point cloud before Trellis or export.

    Parameters
    ----------
    points : (N, 3) world-space points
    colors : optional (N, 3) colours in [0, 1]

    Returns
    -------
    filtered_points : (M, 3)
    filtered_colors : (M, 3) or None
    """
    # Generous radius around each marker to catch all reconstructed
    # marker geometry (markers may be slightly smeared in reconstruction)
    removal_radius_circle = _CIRCLE_RADIUS_3D * 3.0
    removal_radius_square = _SQUARE_RADIUS_3D * 3.0

    keep = np.ones(len(points), dtype=bool)

    # Remove points near the circle
    dists = np.linalg.norm(points - CIRCLE_POS_3D, axis=1)
    keep &= dists > removal_radius_circle

    # Remove points near each square
    for sq_pos in [SQUARE_L_POS_3D, SQUARE_R_POS_3D]:
        dists = np.linalg.norm(points - sq_pos, axis=1)
        keep &= dists > removal_radius_square

    n_removed = len(points) - keep.sum()
    if n_removed > 0:
        logger.info(
            "Removed %d marker points from point cloud (%d -> %d)",
            n_removed, len(points), keep.sum(),
        )

    filtered_points = points[keep]
    filtered_colors = colors[keep] if colors is not None else None
    return filtered_points, filtered_colors

