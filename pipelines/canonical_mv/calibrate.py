"""
Camera calibration preview module.

Provides a fast path to re-run camera initialization and coarse
visual-hull reconstruction with overridden camera parameters, returning
a preview image without persisting full artifacts.  Used by the
calibration UI so the user can iterate quickly on camera angles.

Every variable that affects alignment is exposed:
    - Per-view camera placement: yaw, pitch, distance, focal_length
    - Per-view up-hint vector (controls in-plane roll)
    - Per-view image transforms: rotation (0/90/180/270), flip_h, flip_v
    - Shared sensor width (affects FOV)
    - Grid half-extent (world-space size of reconstruction volume)
    - Consensus ratio, mask dilation
"""

import io
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from api.storage import StorageManager

from .camera_init import (
    CameraRig,
    build_intrinsic_matrix,
    camera_position_from_angles,
    focal_length_pixels,
    look_at,
    DEFAULT_SENSOR_WIDTH_MM,
    DEFAULT_NEAR_PLANE,
    DEFAULT_FAR_PLANE,
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
from .config import CANONICAL_VIEW_ORDER

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default per-view settings (matches CANONICAL_CAMERAS in config.py)
# ---------------------------------------------------------------------------

DEFAULT_VIEW_PARAMS: Dict[str, Dict[str, Any]] = {
    "front": {
        "yaw_deg": 0.0,
        "pitch_deg": 0.0,
        "distance": 2.5,
        "focal_length": 50.0,
        "up_hint": [0.0, 1.0, 0.0],
        "rotation_deg": 0,
        "flip_h": False,
        "flip_v": False,
    },
    "side": {
        "yaw_deg": 90.0,
        "pitch_deg": 0.0,
        "distance": 2.5,
        "focal_length": 50.0,
        "up_hint": [0.0, 1.0, 0.0],
        "rotation_deg": 0,
        "flip_h": False,
        "flip_v": False,
    },
    "top": {
        "yaw_deg": 0.0,
        "pitch_deg": -90.0,
        "distance": 2.5,
        "focal_length": 50.0,
        "up_hint": [-1.0, 0.0, 0.0],
        "rotation_deg": 0,
        "flip_h": False,
        "flip_v": False,
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_calibration_preview(
    job_id: str,
    sm: StorageManager,
    camera_overrides: Dict[str, Dict[str, Any]],
    grid_resolution: int = 64,
    grid_half_extent: float = DEFAULT_GRID_HALF_EXTENT,
    sensor_width_mm: float = DEFAULT_SENSOR_WIDTH_MM,
    consensus_ratio: float = DEFAULT_CONSENSUS_RATIO,
    mask_dilation: int = DEFAULT_MASK_DILATION,
) -> Dict[str, Any]:
    """
    Run a fast camera calibration preview.

    Loads existing segmented masks, builds a camera rig from the
    provided overrides (including per-view image transforms), computes
    a visual hull, and renders preview images.

    Args:
        job_id: Job ID (must have completed at least preprocess_views).
        sm: Storage manager for loading existing artifacts.
        camera_overrides: Per-view camera + image params.  Each view
            can specify any combination of::

                {
                    "yaw_deg": float,
                    "pitch_deg": float,
                    "distance": float,
                    "focal_length": float,
                    "up_hint": [x, y, z],
                    "rotation_deg": 0 | 90 | 180 | 270,
                    "flip_h": bool,
                    "flip_v": bool,
                }

            Missing keys fall back to defaults.
        grid_resolution: Voxel grid resolution per axis.
        grid_half_extent: World-space half-extent of the voxel grid.
        sensor_width_mm: Camera sensor width in mm (affects FOV).
        consensus_ratio: Fraction of views that must agree.
        mask_dilation: Pixels to dilate masks before projection.

    Returns:
        Dict with preview_png, depth_pngs, overlay_pngs, n_occupied,
        occupancy_pct, and the serialized rig.
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
    # 2. Load segmented masks + images (raw, before transforms)
    # ------------------------------------------------------------------
    raw_masks, raw_images = _load_segmented_views(job_id, sm, target_size=image_size)
    if len(raw_masks) < 2:
        raise ValueError(
            f"Only {len(raw_masks)} segmented views found — need at least 2."
        )
    logger.info(f"[{job_id}] calibration: loaded {len(raw_masks)} views")

    # ------------------------------------------------------------------
    # 3. Apply per-view image transforms
    # ------------------------------------------------------------------
    masks: Dict[str, np.ndarray] = {}
    images: Dict[str, np.ndarray] = {}
    transforms_applied: Dict[str, Dict[str, Any]] = {}

    for vn in CANONICAL_VIEW_ORDER:
        if vn not in raw_masks:
            continue

        overrides = camera_overrides.get(vn, {})
        rot = int(overrides.get("rotation_deg", 0))
        flip_h = bool(overrides.get("flip_h", False))
        flip_v = bool(overrides.get("flip_v", False))

        m = raw_masks[vn].copy()
        img = raw_images[vn].copy()

        m, img = _apply_image_transforms(m, img, rot, flip_h, flip_v)

        masks[vn] = m
        images[vn] = img
        transforms_applied[vn] = {
            "rotation_deg": rot,
            "flip_h": flip_h,
            "flip_v": flip_v,
        }

    # ------------------------------------------------------------------
    # 4. Build camera rig from overrides
    # ------------------------------------------------------------------
    rig = _build_rig_from_overrides(
        camera_overrides, image_size, sensor_width_mm,
    )
    logger.info(f"[{job_id}] calibration: rig built")

    # ------------------------------------------------------------------
    # 5. Compute visual hull
    # ------------------------------------------------------------------
    occupancy, grid_origin, voxel_size = compute_visual_hull(
        masks=masks,
        rig=rig,
        grid_resolution=grid_resolution,
        grid_half_extent=grid_half_extent,
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
    # 6. Render 3/4-angle preview
    # ------------------------------------------------------------------
    preview_img = render_visual_hull_preview(
        occupancy,
        grid_origin,
        voxel_size,
        level=consensus_ratio,
        image_size=(512, 512),
    )
    _annotate_preview(preview_img, n_occupied, grid_resolution, consensus_ratio)

    preview_buf = io.BytesIO()
    Image.fromarray(preview_img).save(preview_buf, format="PNG")
    preview_png = preview_buf.getvalue()

    # ------------------------------------------------------------------
    # 7. Render per-view depth maps
    # ------------------------------------------------------------------
    depth_maps = compute_depth_maps(
        occupancy, rig, grid_origin, voxel_size, image_size,
        level=consensus_ratio,
    )

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
    # 8. Render per-view mask overlays
    # ------------------------------------------------------------------
    overlay_pngs: Dict[str, bytes] = {}
    for vn in CANONICAL_VIEW_ORDER:
        if vn in masks and vn in images:
            overlay = _render_mask_projection_overlay(
                occupancy, rig, grid_origin, voxel_size,
                vn, images[vn], masks[vn],
                consensus_ratio,
                transforms_applied.get(vn, {}),
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
        "transforms_applied": transforms_applied,
        "rig": rig.to_dict(),
    }


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------


def _apply_image_transforms(
    mask: np.ndarray,
    image: np.ndarray,
    rotation_deg: int,
    flip_h: bool,
    flip_v: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply rotation and flip transforms to a mask and image pair.

    Args:
        mask: (H, W) uint8 mask.
        image: (H, W, 3) uint8 RGB image.
        rotation_deg: 0, 90, 180, or 270 degrees clockwise.
        flip_h: Flip horizontally (left ↔ right).
        flip_v: Flip vertically (top ↔ bottom).

    Returns:
        Tuple of (transformed_mask, transformed_image).
    """
    # Rotation (clockwise)
    if rotation_deg == 90:
        mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_deg == 180:
        mask = cv2.rotate(mask, cv2.ROTATE_180)
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif rotation_deg == 270:
        mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Flips
    if flip_h:
        mask = cv2.flip(mask, 1)
        image = cv2.flip(image, 1)
    if flip_v:
        mask = cv2.flip(mask, 0)
        image = cv2.flip(image, 0)

    return mask, image


# ---------------------------------------------------------------------------
# Camera rig builder
# ---------------------------------------------------------------------------


def _build_rig_from_overrides(
    camera_overrides: Dict[str, Dict[str, Any]],
    image_size: Tuple[int, int],
    sensor_width_mm: float = DEFAULT_SENSOR_WIDTH_MM,
) -> CameraRig:
    """Build a CameraRig from per-view parameter overrides."""
    image_width, image_height = image_size
    cameras: Dict[str, Dict[str, Any]] = {}

    for vn in CANONICAL_VIEW_ORDER:
        overrides = camera_overrides.get(vn, {})
        defaults = DEFAULT_VIEW_PARAMS[vn]

        yaw_deg = float(overrides.get("yaw_deg", defaults["yaw_deg"]))
        pitch_deg = float(overrides.get("pitch_deg", defaults["pitch_deg"]))
        distance = float(overrides.get("distance", defaults["distance"]))
        focal_mm = float(overrides.get("focal_length", defaults["focal_length"]))

        # Per-view up-hint
        raw_up = overrides.get("up_hint", defaults["up_hint"])
        up_hint = np.array(raw_up, dtype=np.float64)
        # Normalize (protect against zero vector)
        up_norm = np.linalg.norm(up_hint)
        if up_norm < 1e-10:
            up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            up_hint = up_hint / up_norm

        yaw_rad = math.radians(yaw_deg)
        pitch_rad = math.radians(pitch_deg)

        position = camera_position_from_angles(yaw_rad, pitch_rad, distance)
        target = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        extrinsic = look_at(position, target, up=up_hint)

        intrinsic = build_intrinsic_matrix(
            focal_mm, sensor_width_mm, image_width, image_height,
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
            sensor_width_mm,
            image_width,
        ),
        "sensor_width_mm": sensor_width_mm,
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


# ---------------------------------------------------------------------------
# Load views helper
# ---------------------------------------------------------------------------


def _load_segmented_views(
    job_id: str,
    sm: StorageManager,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load segmented view masks and RGB images (reuses coarse_recon logic)."""
    from .coarse_recon import _load_segmented_views as _load
    return _load(job_id, sm, target_size=target_size)


# ---------------------------------------------------------------------------
# Annotation / rendering helpers
# ---------------------------------------------------------------------------


def _annotate_preview(
    img: np.ndarray,
    n_occupied: int,
    grid_res: int,
    consensus: float,
) -> None:
    """Add overlay text to the preview image (modifies in-place)."""
    h, w = img.shape[:2]
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

    norm = np.zeros((h, w), dtype=np.uint8)
    if np.any(valid) and max_depth > 0:
        norm[valid] = (
            (1.0 - depth_map[valid] / max_depth) * 240
        ).clip(0, 240).astype(np.uint8)

    colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    colored[~valid] = (30, 30, 40)

    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
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
    transforms: Dict[str, Any] = None,
) -> np.ndarray:
    """
    Render an overlay showing where the visual hull projects onto a view.

    Green outline = segmentation mask boundary.
    Red fill = projected visual hull silhouette.
    The dimmed input image is shown behind.
    """
    h, w = image.shape[:2]

    # Dim the input image
    overlay = (image.astype(np.float32) * 0.4).astype(np.uint8)

    # Draw mask outline in green
    binary_mask = (mask > 127).astype(np.uint8)
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    # Project occupied voxels into this view
    binary = occupancy >= level
    surface_coords = np.argwhere(binary)
    if len(surface_coords) == 0:
        _label_overlay(overlay, view_name, transforms, h, w)
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
        projected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, proj_contours, -1, (255, 60, 60), 2)

    # Fill projected area with semi-transparent red
    red_fill = overlay.copy()
    red_fill[projected_mask > 0] = (
        red_fill[projected_mask > 0].astype(np.float32) * 0.5
        + np.array([255, 40, 40], dtype=np.float32) * 0.5
    ).astype(np.uint8)
    overlay = red_fill

    _label_overlay(overlay, view_name, transforms, h, w)
    return overlay


def _label_overlay(
    overlay: np.ndarray,
    view_name: str,
    transforms: Optional[Dict[str, Any]],
    h: int,
    w: int,
) -> None:
    """Add label bar to bottom of overlay image."""
    cv2.rectangle(overlay, (0, h - 28), (w, h), (0, 0, 0), -1)

    label = f"{view_name} — green=mask, red=hull"
    if transforms:
        parts = []
        if transforms.get("rotation_deg", 0) != 0:
            parts.append(f"rot={transforms['rotation_deg']}°")
        if transforms.get("flip_h"):
            parts.append("flipH")
        if transforms.get("flip_v"):
            parts.append("flipV")
        if parts:
            label += "  [" + ", ".join(parts) + "]"

    cv2.putText(
        overlay, label, (6, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1,
    )

