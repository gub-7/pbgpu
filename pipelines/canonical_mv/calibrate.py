"""
Camera calibration preview module.

Provides:
1. ``run_calibration_preview`` — single-shot preview with custom params.
2. ``run_calibration_sweep``   — brute-force sweep of every orientation
   combo per view, producing labelled contact sheets so the user can
   visually pick the correct one.

**Standalone operation**: These functions accept raw images directly
(as numpy arrays or PIL images) and perform segmentation inline.
They do NOT require the GPU pipeline to have run — no
``preprocess_metrics.json`` or other artifacts are needed.

Every variable that affects alignment is exposed:
    - Per-view camera placement: yaw, pitch, distance, focal_length
    - Per-view up-hint vector (controls in-plane roll)
    - Per-view image transforms: rotation (0/90/180/270), flip_h, flip_v
    - Shared sensor width (affects FOV)
    - Grid half-extent (world-space size of reconstruction volume)
    - Consensus ratio, mask dilation
"""

import io
import itertools
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

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

# Up-hint presets for sweep.
# For front/side (horizontal cameras) we only test Y-up and Y-down.
# For top (vertical camera) we test all 4 cardinal directions.
_HORIZONTAL_UP_HINTS: List[Tuple[str, List[float]]] = [
    ("Y+", [0, 1, 0]),
    ("Y-", [0, -1, 0]),
]

_TOP_UP_HINTS: List[Tuple[str, List[float]]] = [
    ("X-", [-1, 0, 0]),
    ("X+", [1, 0, 0]),
    ("Z-", [0, 0, -1]),
    ("Z+", [0, 0, 1]),
]

# Image orientation combos (dihedral group D4 — 8 elements)
_ORIENTATIONS: List[Tuple[int, bool]] = [
    (0, False),
    (90, False),
    (180, False),
    (270, False),
    (0, True),
    (90, True),
    (180, True),
    (270, True),
]


# ---------------------------------------------------------------------------
# Inline segmentation (no pipeline artifacts needed)
# ---------------------------------------------------------------------------


def _segment_image_inline(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment a single image inline, returning mask and RGB.

    Uses rembg for background removal. Falls back to simple
    thresholding if rembg is unavailable.

    Args:
        image: RGB image as (H, W, 3) uint8 numpy array.

    Returns:
        Tuple of (mask, rgb) where:
            mask: (H, W) uint8 binary mask (0 or 255)
            rgb: (H, W, 3) uint8 RGB image
    """
    try:
        from rembg import remove

        pil_img = Image.fromarray(image)
        result = remove(pil_img)
        rgba = np.array(result)

        if rgba.shape[2] == 4:
            mask = rgba[:, :, 3]
            rgb = rgba[:, :, :3]
        else:
            rgb = rgba[:, :, :3]
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            mask = (gray > 10).astype(np.uint8) * 255

        # Harden alpha to binary
        mask = ((mask > 127).astype(np.uint8)) * 255
        return mask, rgb

    except ImportError:
        logger.warning("rembg not available, using simple threshold segmentation")
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask = (gray > 10).astype(np.uint8) * 255
        return mask, image


def _prepare_views_from_images(
    raw_images: Dict[str, np.ndarray],
    target_size: Tuple[int, int] = (1024, 1024),
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Prepare view masks and images from raw uploaded images.

    Resizes all images to a common size, then segments them.

    Args:
        raw_images: Dict mapping view name → RGB image (H, W, 3) uint8.
        target_size: (width, height) to resize all images to.

    Returns:
        Tuple of (masks_dict, images_dict).
    """
    masks = {}
    images = {}

    for vn in CANONICAL_VIEW_ORDER:
        if vn not in raw_images:
            continue

        img = raw_images[vn]

        # Resize to target size
        tw, th = target_size
        if img.shape[1] != tw or img.shape[0] != th:
            pil = Image.fromarray(img)
            pil = pil.resize((tw, th), Image.Resampling.LANCZOS)
            img = np.array(pil)

        # Segment
        mask, rgb = _segment_image_inline(img)
        masks[vn] = mask
        images[vn] = rgb

    return masks, images


# ---------------------------------------------------------------------------
# Sweep: brute-force all orientation combos
# ---------------------------------------------------------------------------


def run_calibration_sweep(
    raw_images: Dict[str, np.ndarray],
    canvas_size: int = 1024,
    grid_resolution: int = 48,
    grid_half_extent: float = DEFAULT_GRID_HALF_EXTENT,
    sensor_width_mm: float = DEFAULT_SENSOR_WIDTH_MM,
    consensus_ratio: float = DEFAULT_CONSENSUS_RATIO,
    mask_dilation: int = DEFAULT_MASK_DILATION,
) -> Dict[str, bytes]:
    """
    Brute-force sweep of every orientation combo per view.

    For each view, tests all 8 image orientations (4 rotations × 2 flips)
    combined with relevant up-hint presets.  For each combo, computes a
    fresh visual hull and renders an overlay showing green = mask vs
    red = hull projection.  Results are assembled into labelled contact
    sheet PNGs — one per view.

    The user visually picks the combo where green and red overlap.

    **Standalone**: Accepts raw images directly. Does NOT require any
    pipeline artifacts to exist.

    Args:
        raw_images: Dict mapping view name → RGB image (H, W, 3) uint8.
        canvas_size: Target image size for processing.
        grid_resolution: Voxel grid resolution (lower = faster).
        grid_half_extent: World-space half-extent.
        sensor_width_mm: Camera sensor width.
        consensus_ratio: Occupancy consensus.
        mask_dilation: Mask dilation pixels.

    Returns:
        Dict mapping ``"{view}_sheet"`` → PNG bytes of the contact sheet.
    """
    logger.info(f"sweep: starting (grid={grid_resolution}, views={list(raw_images.keys())})")

    image_size = (canvas_size, canvas_size)

    # Segment all views
    logger.info("sweep: segmenting views...")
    base_masks, base_images = _prepare_views_from_images(raw_images, target_size=image_size)

    if len(base_masks) < 2:
        raise ValueError(f"Only {len(base_masks)} views found, need at least 2")

    logger.info(f"sweep: segmented {len(base_masks)} views")

    # Build combos per view
    view_combos: Dict[str, List[Dict[str, Any]]] = {}

    for vn in CANONICAL_VIEW_ORDER:
        if vn not in base_masks:
            continue

        up_hints = _TOP_UP_HINTS if vn == "top" else _HORIZONTAL_UP_HINTS

        combos = []
        for up_label, up_hint in up_hints:
            for rot, flip_h in _ORIENTATIONS:
                combos.append({
                    "up_label": up_label,
                    "up_hint": up_hint,
                    "rotation_deg": rot,
                    "flip_h": flip_h,
                    "label": f"up={up_label} rot={rot}\u00b0 {'flipH' if flip_h else ''}".strip(),
                })
        view_combos[vn] = combos

    # Run each combo
    TILE_SIZE = 256
    results: Dict[str, bytes] = {}

    for vn, combos in view_combos.items():
        logger.info(f"sweep: {vn} — {len(combos)} combos")
        tiles: List[Tuple[np.ndarray, str, int]] = []

        for ci, combo in enumerate(combos):
            try:
                tile, n_occ = _run_single_combo(
                    vn, combo,
                    base_masks, base_images,
                    image_size, sensor_width_mm,
                    grid_resolution, grid_half_extent,
                    consensus_ratio, mask_dilation,
                    TILE_SIZE,
                )
                tiles.append((tile, combo["label"], n_occ))
            except Exception as e:
                logger.warning(f"sweep {vn} combo {ci} failed: {e}")
                err_tile = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
                err_tile[:] = (40, 20, 20)
                cv2.putText(err_tile, "ERROR", (10, TILE_SIZE // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 1)
                tiles.append((err_tile, combo["label"], 0))

        # Assemble contact sheet
        sheet = _make_contact_sheet(tiles, vn, TILE_SIZE)
        buf = io.BytesIO()
        Image.fromarray(sheet).save(buf, format="PNG")
        results[f"{vn}_sheet"] = buf.getvalue()

        logger.info(f"sweep: {vn} sheet done ({len(tiles)} tiles)")

    logger.info("sweep: complete")
    return results


def _run_single_combo(
    target_view: str,
    combo: Dict[str, Any],
    raw_masks: Dict[str, np.ndarray],
    raw_images: Dict[str, np.ndarray],
    image_size: Tuple[int, int],
    sensor_width_mm: float,
    grid_resolution: int,
    grid_half_extent: float,
    consensus_ratio: float,
    mask_dilation: int,
    tile_size: int,
) -> Tuple[np.ndarray, int]:
    """
    Run a single orientation combo for one view.

    The target view gets the combo's transforms; other views use defaults.
    Returns (overlay_tile, n_occupied).
    """
    # Build per-view overrides: target view uses combo, others use defaults
    camera_overrides: Dict[str, Dict[str, Any]] = {}
    masks: Dict[str, np.ndarray] = {}
    images: Dict[str, np.ndarray] = {}

    for vn in CANONICAL_VIEW_ORDER:
        if vn not in raw_masks:
            continue

        if vn == target_view:
            # Apply combo transforms to this view's image
            rot = combo["rotation_deg"]
            flip_h = combo["flip_h"]
            m, img = _apply_image_transforms(
                raw_masks[vn].copy(), raw_images[vn].copy(),
                rot, flip_h, False,
            )
            masks[vn] = m
            images[vn] = img
            camera_overrides[vn] = {
                **DEFAULT_VIEW_PARAMS[vn],
                "up_hint": combo["up_hint"],
            }
        else:
            masks[vn] = raw_masks[vn]
            images[vn] = raw_images[vn]
            camera_overrides[vn] = dict(DEFAULT_VIEW_PARAMS[vn])

    # Build rig
    rig = _build_rig_from_overrides(camera_overrides, image_size, sensor_width_mm)

    # Compute hull
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

    # Render overlay for the target view
    overlay = _render_mask_projection_overlay(
        occupancy, rig, grid_origin, voxel_size,
        target_view, images[target_view], masks[target_view],
        consensus_ratio,
    )

    # Resize to tile size
    tile = cv2.resize(overlay, (tile_size, tile_size), interpolation=cv2.INTER_AREA)

    return tile, n_occupied


def _make_contact_sheet(
    tiles: List[Tuple[np.ndarray, str, int]],
    view_name: str,
    tile_size: int,
) -> np.ndarray:
    """
    Assemble tiles into a labelled contact sheet image.

    Layout: auto-grid with labels above each tile.
    """
    n = len(tiles)
    if n == 0:
        empty = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        return empty

    # Determine grid dimensions
    cols = min(8, n)
    rows = math.ceil(n / cols)

    LABEL_H = 32
    cell_h = tile_size + LABEL_H
    cell_w = tile_size

    # Title bar
    TITLE_H = 48
    sheet_w = cols * cell_w
    sheet_h = TITLE_H + rows * cell_h

    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)
    sheet[:] = (25, 25, 35)  # dark background

    # Title
    cv2.rectangle(sheet, (0, 0), (sheet_w, TITLE_H), (35, 35, 50), -1)
    title = f"{view_name.upper()} — {n} combinations (green=mask, red=hull)"
    cv2.putText(sheet, title, (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 230, 255), 2)

    for idx, (tile, label, n_occ) in enumerate(tiles):
        row = idx // cols
        col = idx % cols

        x0 = col * cell_w
        y0 = TITLE_H + row * cell_h

        # Label background
        cv2.rectangle(sheet, (x0, y0), (x0 + cell_w, y0 + LABEL_H), (40, 40, 55), -1)

        # Label text
        cv2.putText(sheet, label, (x0 + 4, y0 + 14),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 210, 240), 1)
        occ_text = f"{n_occ:,} vox"
        cv2.putText(sheet, occ_text, (x0 + 4, y0 + 27),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.28, (150, 170, 200), 1)

        # Tile
        ty = y0 + LABEL_H
        sheet[ty:ty + tile_size, x0:x0 + tile_size] = tile

        # Border for non-zero occupancy (highlight promising combos)
        if n_occ > 50:
            brightness = min(255, 80 + n_occ // 10)
            color = (0, brightness, 0) if n_occ > 200 else (brightness, brightness, 0)
            cv2.rectangle(sheet, (x0, ty), (x0 + tile_size - 1, ty + tile_size - 1),
                          color, 2)

    return sheet


# ---------------------------------------------------------------------------
# Single-shot preview
# ---------------------------------------------------------------------------


def run_calibration_preview(
    raw_images: Dict[str, np.ndarray],
    camera_overrides: Dict[str, Dict[str, Any]],
    canvas_size: int = 1024,
    grid_resolution: int = 64,
    grid_half_extent: float = DEFAULT_GRID_HALF_EXTENT,
    sensor_width_mm: float = DEFAULT_SENSOR_WIDTH_MM,
    consensus_ratio: float = DEFAULT_CONSENSUS_RATIO,
    mask_dilation: int = DEFAULT_MASK_DILATION,
) -> Dict[str, Any]:
    """
    Run a fast camera calibration preview with explicit overrides.

    **Standalone**: Accepts raw images directly. Does NOT require any
    pipeline artifacts to exist.

    Returns dict with preview_png, depth_pngs, overlay_pngs, n_occupied,
    occupancy_pct, and the serialized rig.
    """
    logger.info(f"calibration: starting (grid={grid_resolution})")

    image_size = (canvas_size, canvas_size)

    # Segment all views
    base_masks, base_images = _prepare_views_from_images(raw_images, target_size=image_size)

    if len(base_masks) < 2:
        raise ValueError(f"Only {len(base_masks)} views found, need at least 2")

    # Apply per-view image transforms
    masks: Dict[str, np.ndarray] = {}
    images: Dict[str, np.ndarray] = {}

    for vn in CANONICAL_VIEW_ORDER:
        if vn not in base_masks:
            continue
        overrides = camera_overrides.get(vn, {})
        rot = int(overrides.get("rotation_deg", 0))
        flip_h = bool(overrides.get("flip_h", False))
        flip_v = bool(overrides.get("flip_v", False))
        m, img = _apply_image_transforms(
            base_masks[vn].copy(), base_images[vn].copy(), rot, flip_h, flip_v,
        )
        masks[vn] = m
        images[vn] = img

    rig = _build_rig_from_overrides(camera_overrides, image_size, sensor_width_mm)

    occupancy, grid_origin, voxel_size = compute_visual_hull(
        masks=masks, rig=rig,
        grid_resolution=grid_resolution,
        grid_half_extent=grid_half_extent,
        consensus_ratio=consensus_ratio,
        mask_dilation=mask_dilation,
    )

    binary = threshold_occupancy(occupancy, consensus_ratio)
    n_occupied = int(binary.sum())
    occupancy_pct = float(n_occupied / binary.size) if binary.size > 0 else 0.0

    # 3/4-angle preview
    preview_img = render_visual_hull_preview(
        occupancy, grid_origin, voxel_size,
        level=consensus_ratio, image_size=(512, 512),
    )
    _annotate_preview(preview_img, n_occupied, grid_resolution, consensus_ratio)
    preview_buf = io.BytesIO()
    Image.fromarray(preview_img).save(preview_buf, format="PNG")

    # Depth maps
    depth_maps = compute_depth_maps(
        occupancy, rig, grid_origin, voxel_size, image_size,
        level=consensus_ratio,
    )
    max_depth = max(
        (float(dm.max()) for dm in depth_maps.values() if np.any(dm > 0)),
        default=1.0,
    )
    depth_pngs = {vn: _depth_to_colored_png(dm, max_depth) for vn, dm in depth_maps.items()}

    # Overlays
    overlay_pngs: Dict[str, bytes] = {}
    for vn in CANONICAL_VIEW_ORDER:
        if vn in masks and vn in images:
            overlay = _render_mask_projection_overlay(
                occupancy, rig, grid_origin, voxel_size,
                vn, images[vn], masks[vn], consensus_ratio,
            )
            buf = io.BytesIO()
            Image.fromarray(overlay).save(buf, format="PNG")
            overlay_pngs[vn] = buf.getvalue()

    return {
        "preview_png": preview_buf.getvalue(),
        "depth_pngs": depth_pngs,
        "overlay_pngs": overlay_pngs,
        "n_occupied": n_occupied,
        "occupancy_pct": occupancy_pct,
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
    """Apply rotation and flip transforms to a mask and image pair."""
    if rotation_deg == 90:
        mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_deg == 180:
        mask = cv2.rotate(mask, cv2.ROTATE_180)
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif rotation_deg == 270:
        mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

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

        raw_up = overrides.get("up_hint", defaults["up_hint"])
        up_hint = np.array(raw_up, dtype=np.float64)
        up_norm = np.linalg.norm(up_hint)
        if up_norm < 1e-10:
            up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            up_hint = up_hint / up_norm

        position = camera_position_from_angles(
            math.radians(yaw_deg), math.radians(pitch_deg), distance,
        )
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
            sensor_width_mm, image_width,
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
# Rendering helpers
# ---------------------------------------------------------------------------


def _annotate_preview(
    img: np.ndarray, n_occupied: int, grid_res: int, consensus: float,
) -> None:
    """Add annotation bar to the bottom of a preview image."""
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - 44), (w, h), (0, 0, 0), -1)
    text = f"Voxels: {n_occupied:,}  |  Grid: {grid_res}  |  Consensus: {consensus:.0%}"
    cv2.putText(img, text, (10, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 255), 1)


def _depth_to_colored_png(depth_map: np.ndarray, max_depth: float) -> bytes:
    """Convert a depth map to a colored PNG image (turbo colormap)."""
    h, w = depth_map.shape
    valid = depth_map > 0
    norm = np.zeros((h, w), dtype=np.uint8)
    if np.any(valid) and max_depth > 0:
        norm[valid] = ((1.0 - depth_map[valid] / max_depth) * 240).clip(0, 240).astype(np.uint8)
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
) -> np.ndarray:
    """
    Overlay showing green = mask boundary, red = projected hull.
    """
    h, w = image.shape[:2]
    overlay = (image.astype(np.float32) * 0.4).astype(np.uint8)

    # Mask outline (green)
    binary_mask = (mask > 127).astype(np.uint8)
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    # Project hull voxels
    binary = occupancy >= level
    coords = np.argwhere(binary)
    if len(coords) == 0:
        cv2.putText(overlay, "NO HULL", (w // 4, h // 2),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 80, 80), 2)
        return overlay

    world = coords.astype(np.float64) * voxel_size + grid_origin + voxel_size / 2
    ones = np.ones((len(world), 1), dtype=np.float64)
    homo = np.hstack([world, ones])

    ext = rig.get_extrinsic(view_name)
    intr = rig.get_intrinsic(view_name)
    cam = (ext @ homo.T).T
    z = cam[:, 2]
    p = (intr @ cam[:, :3].T).T

    with np.errstate(divide="ignore", invalid="ignore"):
        px = p[:, 0] / p[:, 2]
        py = p[:, 1] / p[:, 2]

    valid = (z > 0) & (px >= 0) & (px < w) & (py >= 0) & (py < h)
    vi = np.where(valid)[0]

    proj = np.zeros((h, w), dtype=np.uint8)
    if len(vi) > 0:
        proj[py[vi].astype(np.int32), px[vi].astype(np.int32)] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    proj = cv2.dilate(proj, kernel, iterations=1)

    # Red fill
    overlay[proj > 0] = (
        overlay[proj > 0].astype(np.float32) * 0.5
        + np.array([255, 40, 40], dtype=np.float32) * 0.5
    ).astype(np.uint8)

    # Red outline
    pc, _ = cv2.findContours(proj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, pc, -1, (255, 60, 60), 2)

    return overlay

