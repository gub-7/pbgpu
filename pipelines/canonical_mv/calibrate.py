"""
Camera calibration preview module.

Provides a fast path to re-run camera initialization and coarse
visual-hull reconstruction with overridden camera parameters, returning
a preview image without persisting full artifacts.  Used by the
calibration UI so the user can iterate quickly on camera angles.
"""

import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from api.storage import StorageManager

from .camera_init import (
    CameraRig,
    CameraSpec,
    build_canonical_rig,
    build_intrinsic_matrix,
    camera_position_from_angles,
    look_at,
    DEFAULT_SENSOR_WIDTH_MM,
)
from .coarse_recon import (
    compute_visual_hull,
    threshold_occupancy,
    render_visual_hull_preview,
    compute_depth_maps,
    DEFAULT_GRID_HALF_EXTENT,
    DEFAULT_CONSENSUS_RATIO,
    DEFAULT_MASK_DILATION,
)
from .config import CanonicalMVConfig, CANONICAL_VIEW_ORDER

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_calibration_preview(
    job_id: str,
    sm: StorageManager,
    camera_overrides: Dict[str, Dict[str, float]],
    top_up_hint: Optional[List[float]] = None,
    grid_resolution: int = 64,
    consensus_ratio: float = DEFAULT_CONSENSUS_RATIO,
    mask_dilation: int = DEFAULT_MASK_DILATION,
) -> Dict[str, Any]:
    """
    Run a fast camera calibration preview.

    Loads existing segmented masks, builds a camera rig from the
    provided overrides, computes a visual hull, and renders preview
    images — all without persisting artifacts.

    Args:
        job_id: Job ID (must have completed at least preprocess_views).
        sm: Storage manager for loading existing artifacts.
        camera_overrides: Per-view camera params, e.g.::

            {
                "front": {"yaw_deg": 0, "pitch_deg": 0, "distance": 2.5, "focal_length": 50},
                "side":  {"yaw_deg": 90, "pitch_deg": 0, "distance": 2.5, "focal_length": 50},
                "top":   {"yaw_deg": 0, "pitch_deg": -90, "distance": 2.5, "focal_length": 50},
            }

        top_up_hint: 3-element list [x, y, z] for the top camera's
            look-at up vector.  Controls the in-plane rotation of the
            top-down view.  Default: [-1, 0, 0].
        grid_resolution: Voxel grid resolution per axis (lower = faster).
        consensus_ratio: Fraction of views that must agree for a voxel
            to be occupied.
        mask_dilation: Pixels to dilate masks before projection.

    Returns:
        Dict with:
            - ``preview_png``: bytes of the 3/4-angle preview PNG.
            - ``depth_pngs``: dict of view_name → depth map PNG bytes.
            - ``n_occupied``: number of occupied voxels.
            - ``occupancy_pct``: fraction of grid that is occupied.
            - ``rig``: serialized camera rig dict (for inspection).

    Raises:
        ValueError: if required artifacts are missing.
    """
    logger.info(f"[{job_id}] calibration: starting (grid={grid_resolution})")

    # ------------------------------------------------------------------
    # 1. Load preprocess metrics to determine canvas size
    # ------------------------------------------------------------------
    preprocess_metrics = sm.load_artifact_json(job_id, "preprocess_metrics.json")
    if preprocess_metrics is None:
        raise ValueError(
            "preprocess_metrics.json not found — preprocess_views must "
            "have completed before calibration."
        )
    canvas_size = preprocess_metrics.get("canvas_size", 1024)
    image_size = (canvas_size, canvas_size)

    # ------------------------------------------------------------------
    # 2. Build camera rig from overrides
    # ------------------------------------------------------------------
    up_hint = (
        np.array(top_up_hint, dtype=np.float64)
        if top_up_hint is not None
        else np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    )

    rig = _build_rig_from_overrides(camera_overrides, image_size, up_hint)
    logger.info(f"[{job_id}] calibration: rig built")

    # ------------------------------------------------------------------
    # 3. Load segmented masks + images
    # ------------------------------------------------------------------
    masks, images = _load_segmented_views(job_id, sm, target_size=image_size)
    if len(masks) < 2:
        raise ValueError(
            f"Only {len(masks)} segmented views found — need at least 2."
        )
    logger.info(f"[{job_id}] calibration: loaded {len(masks)} views")

    # ------------------------------------------------------------------
    # 4. Compute visual hull
    # ------------------------------------------------------------------
    occupancy, grid_origin, voxel_size = compute_visual_hull(
        masks=masks,
        rig=rig,
        grid_resolution=grid_resolution,
        grid_half_extent=DEFAULT_GRID_HALF_EXTENT,
        consensus_ratio=consensus_ratio,
        mask_dilation=mask_dilation,
    )

    binary = threshold_occupancy(occupancy, consensus_ratio)
    n_occupied = int(binary.sum())
    occupancy_pct = float(n_occupied / binary.size) if binary.size > 0 else 0.0

    logger.info(
        f"[{job_id}] calibration: {n_occupied} occupied voxels "
        f"({occupancy_pct * 100:.1f}%)"
    )

    # ------------------------------------------------------------------
    # 5. Render 3/4-angle preview
    # ------------------------------------------------------------------
    preview_img = render_visual_hull_preview(
        occupancy,
        grid_origin,
        voxel_size,
        level=consensus_ratio,
        image_size=(512, 512),
    )

    # Add overlay text
    _annotate_preview(preview_img, n_occupied, grid_resolution, consensus_ratio)

    preview_buf = io.BytesIO()
    Image.fromarray(preview_img).save(preview_buf, format="PNG")
    preview_png = preview_buf.getvalue()

    # ------------------------------------------------------------------
    # 6. Render per-view depth maps (small, for quick feedback)
    # ------------------------------------------------------------------
    depth_size = (256, 256)
    # Build a mini-rig at depth_size for depth map rendering
    mini_rig = _build_rig_from_overrides(camera_overrides, depth_size, up_hint)

    depth_maps = compute_depth_maps(
        occupancy, mini_rig, grid_origin, voxel_size, depth_size,
        level=consensus_ratio,
    )

    # Find global max depth for consistent normalization
    max_depth = 0.0
    for dm in depth_maps.values():
        if np.any(dm > 0):
            max_depth = max(max_depth, float(dm.max()))
    if max_depth == 0:
        max_depth = 1.0

    depth_pngs: Dict[str, bytes] = {}
    for vn, dm in depth_maps.items():
        depth_pngs[vn] = _depth_to_colored_png(dm, max_depth)

    # ------------------------------------------------------------------
    # 7. Render per-view mask overlay (show where voxels project)
    # ------------------------------------------------------------------
    overlay_pngs: Dict[str, bytes] = {}
    for vn in CANONICAL_VIEW_ORDER:
        if vn in masks and vn in images:
            overlay = _render_mask_projection_overlay(
                occupancy, rig, grid_origin, voxel_size,
                vn, images[vn], masks[vn],
                consensus_ratio,
            )
            buf = io.BytesIO()
            Image.fromarray(overlay).save(buf, format="PNG")
            overlay_pngs[vn] = buf.getvalue()

    return {
        "preview_png": preview_png,
        "depth_pngs": depth_pngs,
        "overlay_pngs": overlay_pngs,
        "n_occupied": n_occupied,
        "occupancy_pct": occupancy_pct,
        "rig": rig.to_dict(),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_rig_from_overrides(
    camera_overrides: Dict[str, Dict[str, float]],
    image_size: Tuple[int, int],
    top_up_hint: np.ndarray,
) -> CameraRig:
    """Build a CameraRig from per-view parameter overrides."""
    from .camera_init import (
        build_intrinsic_matrix,
        camera_position_from_angles,
        look_at,
        focal_length_pixels,
        DEFAULT_NEAR_PLANE,
        DEFAULT_FAR_PLANE,
        DEFAULT_SENSOR_WIDTH_MM,
    )
    import math

    image_width, image_height = image_size
    cameras: Dict[str, Dict[str, Any]] = {}

    for vn in CANONICAL_VIEW_ORDER:
        overrides = camera_overrides.get(vn, {})

        # Defaults matching CANONICAL_CAMERAS
        defaults = {
            "front": {"yaw_deg": 0.0, "pitch_deg": 0.0, "distance": 2.5, "focal_length": 50.0},
            "side": {"yaw_deg": 90.0, "pitch_deg": 0.0, "distance": 2.5, "focal_length": 50.0},
            "top": {"yaw_deg": 0.0, "pitch_deg": -90.0, "distance": 2.5, "focal_length": 50.0},
        }
        d = defaults.get(vn, defaults["front"])

        yaw_deg = float(overrides.get("yaw_deg", d["yaw_deg"]))
        pitch_deg = float(overrides.get("pitch_deg", d["pitch_deg"]))
        distance = float(overrides.get("distance", d["distance"]))
        focal_mm = float(overrides.get("focal_length", d["focal_length"]))

        yaw_rad = math.radians(yaw_deg)
        pitch_rad = math.radians(pitch_deg)

        position = camera_position_from_angles(yaw_rad, pitch_rad, distance)
        target = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Use custom up-hint for near-vertical cameras (top view)
        if abs(pitch_deg) > 80.0:
            extrinsic = look_at(position, target, up=top_up_hint)
        else:
            extrinsic = look_at(position, target)

        intrinsic = build_intrinsic_matrix(
            focal_mm, DEFAULT_SENSOR_WIDTH_MM, image_width, image_height
        )

        cameras[vn] = {
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
            "position": position,
            "yaw_deg": yaw_deg,
            "pitch_deg": pitch_deg,
            "distance": distance,
            "focal_length_mm": focal_mm,
            "image_width": image_width,
            "image_height": image_height,
        }

    shared_params = {
        "focal_length_mm": cameras["front"]["focal_length_mm"],
        "focal_length_px": focal_length_pixels(
            cameras["front"]["focal_length_mm"],
            DEFAULT_SENSOR_WIDTH_MM,
            image_width,
        ),
        "sensor_width_mm": DEFAULT_SENSOR_WIDTH_MM,
        "image_size": [image_width, image_height],
        "distance": cameras["front"]["distance"],
        "near_plane": DEFAULT_NEAR_PLANE,
        "far_plane": DEFAULT_FAR_PLANE,
    }

    return CameraRig(
        cameras=cameras,
        shared_params=shared_params,
        refinement={"applied": False, "method": "calibration_override"},
    )


def _load_segmented_views(
    job_id: str,
    sm: StorageManager,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load segmented view masks and RGB images (reuses coarse_recon logic)."""
    from .coarse_recon import _load_segmented_views as _load
    return _load(job_id, sm, target_size=target_size)


def _annotate_preview(
    img: np.ndarray,
    n_occupied: int,
    grid_res: int,
    consensus: float,
) -> None:
    """Add overlay text to the preview image (modifies in-place)."""
    h, w = img.shape[:2]

    # Dark bar at bottom
    cv2.rectangle(img, (0, h - 44), (w, h), (0, 0, 0), -1)

    text = f"Voxels: {n_occupied:,}  |  Grid: {grid_res}  |  Consensus: {consensus:.0%}"
    cv2.putText(
        img, text, (10, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 255), 1,
    )


def _depth_to_colored_png(depth_map: np.ndarray, max_depth: float) -> bytes:
    """Convert a float32 depth map to a colored PNG (turbo colormap)."""
    h, w = depth_map.shape
    valid = depth_map > 0

    # Normalize to 0-255
    norm = np.zeros((h, w), dtype=np.uint8)
    if np.any(valid) and max_depth > 0:
        norm[valid] = (
            (1.0 - depth_map[valid] / max_depth) * 240
        ).clip(0, 240).astype(np.uint8)

    # Apply colormap
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    # Set background to dark
    colored[~valid] = (30, 30, 40)

    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)).save(
        buf, format="PNG"
    )
    return buf.getvalue()


def _render_mask_projection_overlay(
    occupancy: np.ndarray,
    rig: CameraRig,
    grid_origin: np.ndarray,
    voxel_size: float,
    view_name: str,
    image: np.ndarray,
    mask: np.ndarray,
    level: float,
) -> np.ndarray:
    """
    Render an overlay showing where the visual hull projects onto a view.

    The input image is dimmed, the mask outline is drawn in green,
    and the projected visual hull silhouette is drawn in red.
    This makes it easy to see if the camera projection aligns with
    the actual mask.
    """
    h, w = image.shape[:2]

    # Dim the input image
    overlay = (image.astype(np.float32) * 0.4).astype(np.uint8)

    # Draw mask outline in green
    binary_mask = (mask > 127).astype(np.uint8)
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    # Project occupied voxels into this view
    binary = occupancy >= level
    surface_coords = np.argwhere(binary)
    if len(surface_coords) == 0:
        return overlay

    surface_world = (
        surface_coords.astype(np.float64) * voxel_size
        + grid_origin
        + voxel_size / 2
    )

    n_points = len(surface_world)
    ones = np.ones((n_points, 1), dtype=np.float64)
    homo = np.hstack([surface_world, ones])

    ext = rig.get_extrinsic(view_name)
    intr = rig.get_intrinsic(view_name)

    cam_coords = (ext @ homo.T).T
    z_cam = cam_coords[:, 2]
    p_img = (intr @ cam_coords[:, :3].T).T

    with np.errstate(divide="ignore", invalid="ignore"):
        px = p_img[:, 0] / p_img[:, 2]
        py = p_img[:, 1] / p_img[:, 2]

    valid = (z_cam > 0) & (px >= 0) & (px < w) & (py >= 0) & (py < h)
    valid_idx = np.where(valid)[0]

    # Draw projected voxels as red dots
    projected_mask = np.zeros((h, w), dtype=np.uint8)
    if len(valid_idx) > 0:
        px_int = px[valid_idx].astype(np.int32)
        py_int = py[valid_idx].astype(np.int32)
        projected_mask[py_int, px_int] = 255

    # Dilate to make dots visible
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    projected_mask = cv2.dilate(projected_mask, kernel, iterations=1)

    # Draw projected silhouette outline in red
    proj_contours, _ = cv2.findContours(
        projected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, proj_contours, -1, (255, 60, 60), 2)

    # Fill projected area with semi-transparent red
    red_fill = overlay.copy()
    red_fill[projected_mask > 0] = (
        red_fill[projected_mask > 0].astype(np.float32) * 0.5
        + np.array([255, 40, 40], dtype=np.float32) * 0.5
    ).astype(np.uint8)
    overlay = red_fill

    # Label
    cv2.putText(
        overlay,
        f"{view_name} — green=mask, red=hull projection",
        (8, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
    )

    return overlay

