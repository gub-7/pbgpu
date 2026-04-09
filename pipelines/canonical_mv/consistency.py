"""
View-consistency validation stage for the canonical multi-view pipeline.

Responsibilities:
    - Catch bad input early (wrong side, mirrored, bad segmentation, etc.)
    - Silhouette area consistency across paired views (front/back, left/right)
    - Segmentation confidence thresholds
    - Sharpness score validation
    - Cross-view color / lighting consistency
    - Left-right mirror plausibility
    - Top-view overlap plausibility
    - Produce hard-fail errors and soft warnings
    - Save validation metrics as artifact

Does NOT require GPU models (CLIP / DINO) — uses lightweight image
statistics.  A CLIP-based identity check can be layered on top in a
future version.
"""

import io
import json
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

# Paired views that should have similar silhouette areas
_PAIRED_VIEWS = [
    ("front", "back"),
    ("left", "right"),
]

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_validate_views(
    job_id: str,
    config: CanonicalMVConfig,
    jm: JobManager,
    sm: StorageManager,
) -> None:
    """
    Execute the validate_views stage.

    Loads segmented views from the preprocess stage, runs all consistency
    checks, and produces warnings / hard-fail errors.

    Raises:
        ValueError: on catastrophic validation failures (hard fail).
    """
    logger.info(f"[{job_id}] validate_views: starting consistency checks")

    # ------------------------------------------------------------------
    # Load per-view data from preprocess artifacts
    # ------------------------------------------------------------------
    preprocess_meta = sm.load_artifact_json(job_id, "preprocess_metrics.json")
    if preprocess_meta is None:
        raise ValueError(
            "preprocess_metrics.json not found — preprocess_views must run first"
        )

    # Load segmented images for pixel-level checks
    view_images: Dict[str, np.ndarray] = {}
    for view_name in CANONICAL_VIEW_ORDER:
        path = sm.get_view_preview_path(job_id, "segmented", view_name)
        if path is None:
            raise ValueError(f"Segmented preview for '{view_name}' not found")
        img = Image.open(path).convert("RGBA")
        view_images[view_name] = np.array(img)

    per_view = preprocess_meta.get("per_view", {})

    # ------------------------------------------------------------------
    # Run checks
    # ------------------------------------------------------------------
    warnings: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    jm.update_job(job_id, stage_progress=0.1)

    # 1. Segmentation confidence
    _check_segmentation_confidence(per_view, config, warnings, errors)
    jm.update_job(job_id, stage_progress=0.2)

    # 2. Sharpness
    _check_sharpness(per_view, warnings)
    jm.update_job(job_id, stage_progress=0.3)

    # 3. Silhouette area consistency (paired views)
    _check_silhouette_area(per_view, config, warnings, errors)
    jm.update_job(job_id, stage_progress=0.5)

    # 4. Left-right mirror plausibility
    _check_mirror_plausibility(view_images, warnings)
    jm.update_job(job_id, stage_progress=0.65)

    # 5. Color / lighting consistency
    _check_color_consistency(per_view, warnings)
    jm.update_job(job_id, stage_progress=0.8)

    # 6. Top-view overlap plausibility
    _check_top_view_plausibility(per_view, view_images, warnings)
    jm.update_job(job_id, stage_progress=0.9)

    # ------------------------------------------------------------------
    # Save validation results
    # ------------------------------------------------------------------
    validation_result = {
        "status": "failed" if errors else ("warnings" if warnings else "passed"),
        "errors": errors,
        "warnings": warnings,
        "checks_run": [
            "segmentation_confidence",
            "sharpness",
            "silhouette_area",
            "mirror_plausibility",
            "color_consistency",
            "top_view_plausibility",
        ],
    }
    sm.save_artifact_json(job_id, "validation_results.json", validation_result)

    # Update job warnings (flatten to string list for the job model)
    warning_strings = [w["code"] for w in warnings]
    error_strings = [e["code"] for e in errors]
    all_issues = warning_strings + error_strings
    if all_issues:
        jm.update_job(job_id, warnings=all_issues)

    # Update view statuses
    failed_views = set()
    for e in errors:
        if e.get("view"):
            failed_views.add(e["view"])
    for w in warnings:
        if w.get("view"):
            pass  # warnings don't fail views

    for vn in CANONICAL_VIEW_ORDER:
        if vn in failed_views:
            jm.update_job(
                job_id,
                view_updates={vn: {"status": ViewStatus.FAILED.value}},
            )
        else:
            jm.update_job(
                job_id,
                view_updates={vn: {"status": ViewStatus.VALIDATED.value}},
            )

    # ------------------------------------------------------------------
    # Hard fail on catastrophic errors
    # ------------------------------------------------------------------
    if errors:
        error_msg = "; ".join(e["message"] for e in errors)
        raise ValueError(f"View validation failed: {error_msg}")

    jm.update_job(job_id, stage_progress=1.0)
    logger.info(
        f"[{job_id}] validate_views: completed "
        f"({len(warnings)} warnings, {len(errors)} errors)"
    )


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_segmentation_confidence(
    per_view: Dict[str, Dict[str, Any]],
    config: CanonicalMVConfig,
    warnings: List[Dict[str, Any]],
    errors: List[Dict[str, Any]],
) -> None:
    """
    Check that each view has sufficient segmentation confidence.

    Hard fail if confidence is extremely low (likely total segmentation failure).
    Soft warning if below threshold but not catastrophic.
    """
    threshold = config.segmentation_confidence_threshold
    hard_fail_threshold = 0.3  # below this, segmentation is catastrophic

    for vn, meta in per_view.items():
        conf = meta.get("segmentation_confidence", 0.0)
        far = meta.get("foreground_area_ratio", 0.0)

        if far < 0.01:
            errors.append({
                "code": f"{vn}_no_foreground",
                "message": f"View '{vn}' has no detectable foreground (FAR={far:.4f})",
                "severity": "error",
                "view": vn,
            })
        elif conf < hard_fail_threshold:
            errors.append({
                "code": f"{vn}_segmentation_failure",
                "message": (
                    f"View '{vn}' segmentation confidence is catastrophically "
                    f"low ({conf:.2f} < {hard_fail_threshold})"
                ),
                "severity": "error",
                "view": vn,
            })
        elif conf < threshold:
            warnings.append({
                "code": f"{vn}_low_segmentation_confidence",
                "message": (
                    f"View '{vn}' segmentation confidence is low "
                    f"({conf:.2f} < {threshold})"
                ),
                "severity": "warning",
                "view": vn,
            })


def _check_sharpness(
    per_view: Dict[str, Dict[str, Any]],
    warnings: List[Dict[str, Any]],
) -> None:
    """
    Check per-view sharpness scores.

    Warns if a view is significantly blurrier than the median.
    """
    sharpness_values = {}
    for vn, meta in per_view.items():
        sharpness_values[vn] = meta.get("sharpness", 0.0)

    if not sharpness_values:
        return

    values = list(sharpness_values.values())
    median_sharpness = float(np.median(values))

    # Warn if a view is less than 30% of the median sharpness
    min_ratio = 0.3
    for vn, sharp in sharpness_values.items():
        if median_sharpness > 0 and sharp < median_sharpness * min_ratio:
            warnings.append({
                "code": f"{vn}_blurry",
                "message": (
                    f"View '{vn}' is significantly blurrier than other views "
                    f"(sharpness={sharp:.1f}, median={median_sharpness:.1f})"
                ),
                "severity": "warning",
                "view": vn,
            })


def _check_silhouette_area(
    per_view: Dict[str, Dict[str, Any]],
    config: CanonicalMVConfig,
    warnings: List[Dict[str, Any]],
    errors: List[Dict[str, Any]],
) -> None:
    """
    Check that paired views (front/back, left/right) have similar
    foreground area ratios.

    Large discrepancies suggest wrong-side uploads or severe segmentation
    failure in one view.
    """
    tolerance = config.silhouette_area_tolerance

    for view_a, view_b in _PAIRED_VIEWS:
        if view_a not in per_view or view_b not in per_view:
            continue

        far_a = per_view[view_a].get("foreground_area_ratio", 0.0)
        far_b = per_view[view_b].get("foreground_area_ratio", 0.0)

        if far_a == 0 and far_b == 0:
            continue

        max_far = max(far_a, far_b)
        if max_far == 0:
            continue

        relative_diff = abs(far_a - far_b) / max_far

        if relative_diff > tolerance:
            severity = "error" if relative_diff > tolerance * 2 else "warning"
            issue = {
                "code": f"{view_a}_{view_b}_area_mismatch",
                "message": (
                    f"Silhouette area mismatch between '{view_a}' and '{view_b}': "
                    f"FAR={far_a:.3f} vs {far_b:.3f} "
                    f"(relative diff={relative_diff:.2f}, tolerance={tolerance})"
                ),
                "severity": severity,
                "view": f"{view_a}/{view_b}",
            }
            if severity == "error":
                errors.append(issue)
            else:
                warnings.append(issue)


def _check_mirror_plausibility(
    view_images: Dict[str, np.ndarray],
    warnings: List[Dict[str, Any]],
) -> None:
    """
    Check that left and right views are approximately mirror images.

    Compares the left view to a horizontally flipped right view using
    silhouette IoU.  A very low IoU suggests the views may be swapped
    or not from the same subject.
    """
    if "left" not in view_images or "right" not in view_images:
        return

    left_alpha = view_images["left"][:, :, 3]
    right_alpha = view_images["right"][:, :, 3]

    # Resize to common size for comparison
    target_size = (256, 256)
    left_resized = cv2.resize(left_alpha, target_size, interpolation=cv2.INTER_AREA)
    right_resized = cv2.resize(right_alpha, target_size, interpolation=cv2.INTER_AREA)

    # Flip right horizontally to compare with left
    right_flipped = cv2.flip(right_resized, 1)

    # Compute IoU of binary masks
    left_mask = (left_resized > 128).astype(np.uint8)
    right_mask = (right_flipped > 128).astype(np.uint8)

    intersection = np.sum(left_mask & right_mask)
    union = np.sum(left_mask | right_mask)

    if union == 0:
        return

    iou = intersection / union

    # Very low IoU suggests views are not mirror-plausible
    if iou < 0.3:
        warnings.append({
            "code": "left_right_mirror_implausible",
            "message": (
                f"Left and right views do not appear to be mirror images "
                f"(silhouette IoU={iou:.2f}). Views may be swapped or "
                f"from different subjects."
            ),
            "severity": "warning",
            "view": "left/right",
        })


def _check_color_consistency(
    per_view: Dict[str, Dict[str, Any]],
    warnings: List[Dict[str, Any]],
) -> None:
    """
    Check for extreme color / lighting differences across views.

    Compares per-channel foreground color means.  Large deviations
    suggest different lighting conditions or white-balance issues.
    """
    color_means = {}
    for vn, meta in per_view.items():
        cm = meta.get("color_histogram_mean")
        if cm and len(cm) == 3:
            color_means[vn] = cm

    if len(color_means) < 2:
        return

    # Compute the global mean across all views
    all_means = np.array(list(color_means.values()))  # (N, 3)
    global_mean = all_means.mean(axis=0)

    # Check each view's deviation from the global mean
    # Use L2 distance in RGB space
    for vn, cm in color_means.items():
        dist = np.linalg.norm(np.array(cm) - global_mean)
        # Threshold: 40 units in RGB space is quite noticeable
        if dist > 40.0:
            warnings.append({
                "code": f"{vn}_color_mismatch",
                "message": (
                    f"View '{vn}' has significantly different color/lighting "
                    f"compared to other views (distance={dist:.1f})"
                ),
                "severity": "warning",
                "view": vn,
            })


def _check_top_view_plausibility(
    per_view: Dict[str, Dict[str, Any]],
    view_images: Dict[str, np.ndarray],
    warnings: List[Dict[str, Any]],
) -> None:
    """
    Check that the top view has a plausible silhouette.

    The top view should have a foreground area that is not dramatically
    different from the side views.  A very small or very large top-view
    silhouette relative to the front view is suspicious.
    """
    if "top" not in per_view or "front" not in per_view:
        return

    top_far = per_view["top"].get("foreground_area_ratio", 0.0)
    front_far = per_view["front"].get("foreground_area_ratio", 0.0)

    if front_far < 0.01:
        return  # front has no foreground, can't compare

    ratio = top_far / front_far if front_far > 0 else 0.0

    # Top view FAR should be in a reasonable range relative to front
    # Very small top (< 10% of front) or very large (> 300%) is suspicious
    if ratio < 0.1:
        warnings.append({
            "code": "top_view_too_small",
            "message": (
                f"Top view silhouette is very small relative to front view "
                f"(top FAR={top_far:.3f}, front FAR={front_far:.3f}, "
                f"ratio={ratio:.2f})"
            ),
            "severity": "warning",
            "view": "top",
        })
    elif ratio > 3.0:
        warnings.append({
            "code": "top_view_too_large",
            "message": (
                f"Top view silhouette is very large relative to front view "
                f"(top FAR={top_far:.3f}, front FAR={front_far:.3f}, "
                f"ratio={ratio:.2f})"
            ),
            "severity": "warning",
            "view": "top",
        })

