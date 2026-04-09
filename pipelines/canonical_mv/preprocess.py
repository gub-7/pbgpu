"""
Preprocess-views stage for the canonical multi-view pipeline.

Responsibilities:
    - Per-view background removal (segmentation)
    - Per-view metric extraction (bbox, centroid, mask area, sharpness, color histogram)
    - Cross-view consistent framing (shared scale, shared canvas, shared center)
    - Save segmented and normalized preview images per view
    - Update per-view metadata in job state

Reuses the existing segmentation stack from ``preprocessing.segmentation``
and mask-metric utilities from ``preprocessing.preprocessing``.
"""

import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from api.job_manager import JobManager
from api.models import ViewStatus
from api.storage import StorageManager

from .config import CanonicalMVConfig, CANONICAL_VIEW_ORDER

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_preprocess_views(
    job_id: str,
    config: CanonicalMVConfig,
    jm: JobManager,
    sm: StorageManager,
) -> None:
    """
    Execute the preprocess_views stage.

    Steps:
        1. Segment each view (background removal → RGBA).
        2. Compute per-view metrics (bbox, centroid, area, sharpness, histogram).
        3. Determine cross-view consistent framing parameters.
        4. Apply unified framing to produce normalized views.
        5. Save segmented + normalized previews.
        6. Update per-view metadata in job state.
    """
    n_views = len(CANONICAL_VIEW_ORDER)
    logger.info(f"[{job_id}] preprocess_views: starting for {n_views} views")

    # ------------------------------------------------------------------
    # Step 1 & 2: per-view segmentation + metric extraction
    # ------------------------------------------------------------------
    view_data: Dict[str, Dict[str, Any]] = {}

    for idx, view_name in enumerate(CANONICAL_VIEW_ORDER):
        jm.update_job(job_id, stage_progress=idx / n_views * 0.6)

        view_path = sm.get_view_upload_path(job_id, view_name)
        if view_path is None:
            raise FileNotFoundError(
                f"View '{view_name}' upload not found during preprocessing"
            )

        logger.info(f"[{job_id}] preprocess_views: segmenting {view_name}")

        # Load raw image
        raw_img = Image.open(view_path)
        raw_arr = np.array(raw_img.convert("RGB"))

        # Segment
        rgba_arr = _segment_view(str(view_path), config.segmentation_model)

        # Compute per-view metrics
        metrics = _compute_view_metrics(rgba_arr, raw_arr)

        # Save segmented preview
        _save_preview(sm, job_id, "segmented", view_name, rgba_arr)

        view_data[view_name] = {
            "rgba": rgba_arr,
            "raw_rgb": raw_arr,
            "metrics": metrics,
        }

        # Update job metadata
        jm.update_job(
            job_id,
            view_updates={
                view_name: {
                    "status": ViewStatus.SEGMENTED.value,
                    "segmentation_confidence": metrics["segmentation_confidence"],
                    "sharpness_score": metrics["sharpness"],
                }
            },
        )

    # ------------------------------------------------------------------
    # Step 3: cross-view consistent framing
    # ------------------------------------------------------------------
    jm.update_job(job_id, stage_progress=0.65)
    logger.info(f"[{job_id}] preprocess_views: computing cross-view framing")

    framing = _compute_cross_view_framing(view_data, config.shared_canvas_size)

    # ------------------------------------------------------------------
    # Step 4 & 5: apply framing and save normalized previews
    # ------------------------------------------------------------------
    for idx, view_name in enumerate(CANONICAL_VIEW_ORDER):
        jm.update_job(job_id, stage_progress=0.7 + idx / n_views * 0.25)

        rgba_arr = view_data[view_name]["rgba"]
        normalized = _apply_framing(rgba_arr, framing)

        _save_preview(sm, job_id, "normalized", view_name, normalized)

        # Recompute metrics on the normalized image for downstream use
        norm_metrics = _compute_mask_metrics_simple(normalized[:, :, 3])
        view_data[view_name]["metrics"]["normalized_bbox"] = norm_metrics["bbox"]
        view_data[view_name]["metrics"]["normalized_far"] = norm_metrics["far"]

    # ------------------------------------------------------------------
    # Step 6: save framing metadata as artifact
    # ------------------------------------------------------------------
    framing_artifact = {
        "canvas_size": framing["canvas_size"],
        "crop_side": framing["crop_side"],
        "per_view": {},
    }
    for vn in CANONICAL_VIEW_ORDER:
        m = view_data[vn]["metrics"]
        framing_artifact["per_view"][vn] = {
            "bbox": m["bbox"],
            "centroid": m["centroid"],
            "foreground_area_ratio": m["far"],
            "sharpness": m["sharpness"],
            "segmentation_confidence": m["segmentation_confidence"],
            "color_histogram_mean": m["color_hist_mean"],
        }
    sm.save_artifact_json(job_id, "preprocess_metrics.json", framing_artifact)

    jm.update_job(job_id, stage_progress=1.0)
    logger.info(f"[{job_id}] preprocess_views: completed")


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def _segment_view(image_path: str, model_name: str = "u2net") -> np.ndarray:
    """
    Segment a single view image, returning RGBA with hard alpha.

    Reuses the existing ``Segmenter`` class from the preprocessing module.
    Falls back to a simple rembg call if the full segmenter is unavailable.
    """
    try:
        from preprocessing.segmentation import Segmenter

        segmenter = Segmenter(model_name=model_name)
        rgba = segmenter.segment_rembg(image_path, alpha_threshold=0.1, post_process=True)
        # Harden alpha edges (same as single-view TripoSR pipeline)
        rgba = segmenter.harden_alpha(rgba, low_cutoff=0.15, high_cutoff=0.85)
        rgba = segmenter.decontaminate_edges(rgba, edge_width=3)
        segmenter.cleanup_gpu()
        return rgba
    except ImportError:
        # Fallback: use rembg directly
        from rembg import remove
        from PIL import Image as PILImage

        img = PILImage.open(image_path).convert("RGB")
        result = remove(img)
        return np.array(result)


# ---------------------------------------------------------------------------
# Per-view metrics
# ---------------------------------------------------------------------------


def _compute_view_metrics(rgba: np.ndarray, raw_rgb: np.ndarray) -> Dict[str, Any]:
    """
    Compute per-view quality and geometry metrics.

    Returns dict with:
        bbox, centroid, far, bbo, sharpness, segmentation_confidence,
        color_hist_mean, color_hist (per-channel histograms)
    """
    alpha = rgba[:, :, 3]
    mask_metrics = _compute_mask_metrics_simple(alpha)

    sharpness = _compute_sharpness(raw_rgb)
    seg_confidence = _compute_segmentation_confidence(alpha)
    color_hist, color_hist_mean = _compute_color_histogram(rgba)

    return {
        "bbox": mask_metrics["bbox"],
        "centroid": mask_metrics["centroid"],
        "far": mask_metrics["far"],
        "bbo": mask_metrics["bbo"],
        "foreground_pixels": mask_metrics["foreground_pixels"],
        "image_area": mask_metrics["image_area"],
        "sharpness": sharpness,
        "segmentation_confidence": seg_confidence,
        "color_hist": color_hist,
        "color_hist_mean": color_hist_mean,
    }


def _compute_mask_metrics_simple(alpha: np.ndarray, fg_threshold: int = 128) -> Dict[str, Any]:
    """
    Compute basic mask metrics (bbox, centroid, FAR, BBO).

    Lighter version of ``preprocessing.preprocessing.compute_mask_metrics``
    that avoids the morphological cleanup (which can shift centroids for
    multi-view consistency).
    """
    h, w = alpha.shape
    image_area = h * w

    fg_mask = (alpha >= fg_threshold).astype(np.uint8)
    foreground_pixels = int(fg_mask.sum())

    if foreground_pixels == 0:
        return {
            "far": 0.0,
            "bbo": 0.0,
            "centroid": (w // 2, h // 2),
            "bbox": (0, 0, w, h),
            "foreground_pixels": 0,
            "image_area": image_area,
        }

    coords = cv2.findNonZero(fg_mask)
    bx, by, bw, bh = cv2.boundingRect(coords)
    bbox_area = bw * bh

    moments = cv2.moments(fg_mask)
    if moments["m00"] > 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx = bx + bw // 2
        cy = by + bh // 2

    far = foreground_pixels / image_area if image_area > 0 else 0.0
    bbo = foreground_pixels / bbox_area if bbox_area > 0 else 0.0

    return {
        "far": float(far),
        "bbo": float(bbo),
        "centroid": (cx, cy),
        "bbox": (bx, by, bw, bh),
        "foreground_pixels": foreground_pixels,
        "image_area": image_area,
    }


def _compute_sharpness(rgb: np.ndarray) -> float:
    """
    Compute image sharpness via Laplacian variance.

    Higher = sharper.  Typical range: 10-1000+ depending on content.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def _compute_segmentation_confidence(alpha: np.ndarray) -> float:
    """
    Estimate segmentation confidence from the alpha channel.

    A clean segmentation has mostly 0 or 255 alpha values.
    Semi-transparent pixels indicate uncertainty.

    Returns a score in [0, 1] where 1 = perfectly binary alpha.
    """
    normalized = alpha.astype(np.float32) / 255.0
    # Fraction of pixels that are near-binary (< 0.05 or > 0.95)
    binary_count = np.sum((normalized < 0.05) | (normalized > 0.95))
    total = normalized.size
    return float(binary_count / total) if total > 0 else 0.0


def _compute_color_histogram(
    rgba: np.ndarray, bins: int = 32
) -> Tuple[List[List[float]], List[float]]:
    """
    Compute per-channel color histogram of the foreground region.

    Returns:
        (histograms, channel_means) where histograms is a list of 3 lists
        (R, G, B) each with *bins* float values, and channel_means is [R, G, B]
        mean values.
    """
    alpha = rgba[:, :, 3]
    fg_mask = alpha > 128

    if not np.any(fg_mask):
        empty = [0.0] * bins
        return [empty, empty, empty], [0.0, 0.0, 0.0]

    histograms = []
    means = []
    for c in range(3):
        channel = rgba[:, :, c]
        fg_pixels = channel[fg_mask].astype(np.float32)
        hist, _ = np.histogram(fg_pixels, bins=bins, range=(0, 256))
        # Normalize histogram to sum to 1
        total = hist.sum()
        if total > 0:
            hist = hist.astype(np.float64) / total
        histograms.append(hist.tolist())
        means.append(float(fg_pixels.mean()) if len(fg_pixels) > 0 else 0.0)

    return histograms, means


# ---------------------------------------------------------------------------
# Cross-view consistent framing
# ---------------------------------------------------------------------------


def _compute_cross_view_framing(
    view_data: Dict[str, Dict[str, Any]],
    target_canvas_size: int = 1024,
) -> Dict[str, Any]:
    """
    Compute unified framing parameters across all views.

    Strategy:
        1. Find the maximum bounding-box extent (width or height) across all views.
        2. Add padding (~20%) to create a square crop side.
        3. All views will be cropped/padded to this size, then resized to
           target_canvas_size.

    This ensures every view has the same scale relationship between
    subject size and canvas size.
    """
    max_extent = 0
    for vn, vd in view_data.items():
        bbox = vd["metrics"]["bbox"]  # (x, y, w, h)
        extent = max(bbox[2], bbox[3])
        if extent > max_extent:
            max_extent = extent

    if max_extent == 0:
        # Degenerate case: no foreground in any view
        return {
            "crop_side": target_canvas_size,
            "canvas_size": target_canvas_size,
            "padding_ratio": 0.2,
        }

    # Add 20% padding so the subject doesn't touch the edges
    padding_ratio = 0.2
    crop_side = int(max_extent * (1.0 + padding_ratio))

    return {
        "crop_side": crop_side,
        "canvas_size": target_canvas_size,
        "padding_ratio": padding_ratio,
    }


def _apply_framing(
    rgba: np.ndarray,
    framing: Dict[str, Any],
) -> np.ndarray:
    """
    Apply cross-view consistent framing to a single RGBA image.

    Centers the crop on the foreground centroid, pads where necessary,
    and resizes to the target canvas size.
    """
    h, w = rgba.shape[:2]
    crop_side = framing["crop_side"]
    canvas_size = framing["canvas_size"]

    # Compute centroid for centering
    alpha = rgba[:, :, 3]
    metrics = _compute_mask_metrics_simple(alpha)
    cx, cy = metrics["centroid"]

    # Compute crop region centered on centroid
    x1 = cx - crop_side // 2
    y1 = cy - crop_side // 2

    # Ensure the crop contains the full foreground bbox
    bbox = metrics["bbox"]
    bx, by, bw, bh = bbox
    if bw > 0 and bh > 0:
        if x1 > bx:
            x1 = bx
        if x1 + crop_side < bx + bw:
            x1 = bx + bw - crop_side
        if y1 > by:
            y1 = by
        if y1 + crop_side < by + bh:
            y1 = by + bh - crop_side

    # Source region clamped to image bounds
    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(w, x1 + crop_side)
    src_y2 = min(h, y1 + crop_side)

    # Destination region inside the crop canvas
    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    # Build square canvas
    canvas = np.zeros((crop_side, crop_side, 4), dtype=np.uint8)
    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = rgba[src_y1:src_y2, src_x1:src_x2]

    # Resize to target canvas size
    pil_canvas = Image.fromarray(canvas, mode="RGBA")
    pil_canvas = pil_canvas.resize(
        (canvas_size, canvas_size), Image.Resampling.LANCZOS
    )

    return np.array(pil_canvas)


# ---------------------------------------------------------------------------
# Preview saving
# ---------------------------------------------------------------------------


def _save_preview(
    sm: StorageManager,
    job_id: str,
    substage: str,
    view_name: str,
    rgba: np.ndarray,
) -> None:
    """Save an RGBA array as a PNG preview."""
    img = Image.fromarray(rgba, mode="RGBA")

    # Thumbnail for preview (max 512px)
    max_dim = 512
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    sm.save_view_preview(job_id, substage, view_name, buf.getvalue(), ".png")

