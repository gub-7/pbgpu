"""
Subject isolation: background removal and 3D point-cloud filtering.

This is stage B of the pipeline (per expert guidance):
  A. Camera/geometry solved using full images (with background)
  B. **Subject isolation** – remove background from images AND point cloud
  C. Generative completion with Trellis.2

Two-pronged isolation:
  1. **2D masking** – generate foreground masks per view using rembg/SAM
  2. **3D filtering** – back-project 2D masks into 3D occupancy votes
     and keep only points that are consistently inside the subject
     across multiple views.

The result is:
  - A set of masked (background-removed) images for Trellis.2 input
  - A filtered point cloud containing only the subject geometry
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from api.models import IsolationResult, ResolvedView

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 2D Masking
# ---------------------------------------------------------------------------


def generate_mask_rembg(image_path: Path, output_path: Path) -> Path:
    """
    Generate a foreground mask using rembg (U2-Net based).

    Returns path to the mask image (white=foreground, black=background).
    """
    try:
        from rembg import remove
    except ImportError:
        raise RuntimeError(
            "rembg is not installed. Install with: pip install rembg[gpu]"
        )

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        # rembg returns RGBA with alpha as mask
        result = remove(img)

    # Extract alpha channel as mask
    if result.mode == "RGBA":
        alpha = result.split()[-1]
    else:
        # Fallback: convert to grayscale
        alpha = result.convert("L")

    mask = alpha.point(lambda p: 255 if p > 128 else 0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(output_path)
    logger.info("Generated mask: %s", output_path)
    return output_path


def apply_mask_to_image(
    image_path: Path,
    mask_path: Path,
    output_path: Path,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> Path:
    """
    Apply a binary mask to an image, replacing background with bg_color.

    The output is an RGBA PNG where:
      - RGB = original pixels where mask is white, bg_color where black
      - Alpha = mask (255 foreground, 0 background)
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
    with Image.open(mask_path) as mask:
        mask = mask.convert("L")

    # Ensure same size
    if img.size != mask.size:
        mask = mask.resize(img.size, Image.NEAREST)

    # Create RGBA output
    result = Image.new("RGBA", img.size, (*bg_color, 0))
    img_rgba = img.copy()
    img_rgba.putalpha(mask)
    result = Image.alpha_composite(result, img_rgba)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, format="PNG")
    logger.info("Masked image: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 3D Point Cloud Filtering
# ---------------------------------------------------------------------------


def project_points_to_image(
    points: np.ndarray,
    R_w2c: np.ndarray,
    t_w2c: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points into a camera's image plane.

    Parameters
    ----------
    points : (N, 3) world-space points
    R_w2c : (3, 3) world-to-camera rotation
    t_w2c : (3,) world-to-camera translation
    fx, fy, cx, cy : pinhole intrinsics
    width, height : image dimensions

    Returns
    -------
    uv : (N, 2) pixel coordinates (u=col, v=row)
    valid : (N,) boolean mask for points in front of camera and within bounds
    """
    # Transform to camera space
    pts_cam = (R_w2c @ points.T).T + t_w2c  # (N, 3)

    # Points behind camera
    z = pts_cam[:, 2]
    in_front = z > 0

    # Project to pixels (avoid division by zero)
    z_safe = np.where(in_front, z, 1.0)
    u = fx * pts_cam[:, 0] / z_safe + cx
    v = fy * pts_cam[:, 1] / z_safe + cy

    # Bounds check
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    valid = in_front & in_bounds

    uv = np.stack([u, v], axis=1)
    return uv, valid


def filter_points_by_masks(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    views: list[ResolvedView],
    mask_dir: Path,
    min_votes: int = 2,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Filter a 3D point cloud using 2D foreground masks from multiple views.

    For each point, project it into each view and check if it falls inside
    the foreground mask.  Keep only points that are "voted" as foreground
    by at least `min_votes` views.

    This implements the expert-recommended approach:
    "back-project 2D masks into 3D occupancy votes and keep only the
    consistently occupied object core."

    Parameters
    ----------
    points : (N, 3) world-space points
    colors : optional (N, 3) colours
    views : resolved views with intrinsics and extrinsics
    mask_dir : directory containing {label}.png mask files
    min_votes : minimum number of views that must agree a point is foreground

    Returns
    -------
    filtered_points : (M, 3) retained points
    filtered_colors : (M, 3) or None
    """
    n_points = len(points)
    votes = np.zeros(n_points, dtype=np.int32)

    for view in views:
        # Load mask
        mask_path = mask_dir / f"{view.label.value}_mask.png"
        if not mask_path.exists():
            logger.warning("Mask not found for view %s: %s", view.label.value, mask_path)
            continue

        with Image.open(mask_path) as mask_img:
            mask_arr = np.array(mask_img.convert("L"))  # (H, W)

        # Build projection matrices
        R_w2c = np.array(view.extrinsics.R_w2c).reshape(3, 3)
        t_w2c = np.array(view.extrinsics.t_w2c)
        intr = view.intrinsics

        uv, valid = project_points_to_image(
            points, R_w2c, t_w2c,
            intr.fx, intr.fy, intr.cx, intr.cy,
            intr.width, intr.height,
        )

        # Check mask at projected locations
        for i in range(n_points):
            if valid[i]:
                col = int(round(uv[i, 0]))
                row = int(round(uv[i, 1]))
                col = min(max(col, 0), intr.width - 1)
                row = min(max(row, 0), intr.height - 1)
                if mask_arr[row, col] > 128:
                    votes[i] += 1

    # Keep points with enough votes
    keep = votes >= min_votes
    filtered_points = points[keep]
    filtered_colors = colors[keep] if colors is not None else None

    logger.info(
        "3D filtering: %d → %d points (min_votes=%d)",
        n_points,
        len(filtered_points),
        min_votes,
    )

    return filtered_points, filtered_colors


# ---------------------------------------------------------------------------
# Main isolation pipeline
# ---------------------------------------------------------------------------


def run_subject_isolation(
    image_dir: Path,
    views: list[ResolvedView],
    point_cloud_path: Optional[Path],
    output_dir: Path,
    mask_method: str = "rembg",
    min_votes: int = 2,
) -> IsolationResult:
    """
    Full subject isolation pipeline.

    1. Generate foreground masks for each view
    2. Apply masks to create background-removed images
    3. Filter the 3D point cloud using multi-view mask consensus

    Parameters
    ----------
    image_dir : directory containing preprocessed full images
    views : resolved views with camera parameters
    point_cloud_path : path to coarse reconstruction PLY (or None to skip 3D filtering)
    output_dir : directory for output artifacts
    mask_method : masking approach ("rembg" or "sam")
    min_votes : minimum view consensus for 3D point retention
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "masks"
    masked_dir = output_dir / "masked_images"
    mask_dir.mkdir(exist_ok=True)
    masked_dir.mkdir(exist_ok=True)

    masked_image_paths = []
    num_points_retained = 0
    num_points_removed = 0
    filtered_ply_path = None

    # Step 1 & 2: Generate masks and masked images
    for view in views:
        img_path = image_dir / view.image_filename
        if not img_path.exists():
            logger.error("Image not found: %s", img_path)
            continue

        mask_path = mask_dir / f"{view.label.value}_mask.png"
        masked_path = masked_dir / f"{view.label.value}_masked.png"

        if mask_method == "rembg":
            generate_mask_rembg(img_path, mask_path)
        else:
            raise ValueError(f"Unsupported mask method: {mask_method}")

        apply_mask_to_image(img_path, mask_path, masked_path)
        masked_image_paths.append(str(masked_path))

    # Step 3: Filter point cloud
    if point_cloud_path is not None and point_cloud_path.exists():
        from pipelines.coarse_recon import read_ply, _write_ply

        points, colors = read_ply(point_cloud_path)
        original_count = len(points)

        filtered_points, filtered_colors = filter_points_by_masks(
            points, colors, views, mask_dir, min_votes=min_votes
        )

        filtered_ply = output_dir / "filtered_pointcloud.ply"
        _write_ply(filtered_ply, filtered_points, filtered_colors)
        filtered_ply_path = str(filtered_ply)

        num_points_retained = len(filtered_points)
        num_points_removed = original_count - num_points_retained

        logger.info(
            "Point cloud filtered: %d retained, %d removed",
            num_points_retained,
            num_points_removed,
        )

    return IsolationResult(
        masked_image_paths=masked_image_paths,
        filtered_ply_path=filtered_ply_path,
        num_points_retained=num_points_retained,
        num_points_removed=num_points_removed,
    )

