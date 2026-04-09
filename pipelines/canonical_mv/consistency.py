"""
Cross-view consistency validation stage for the canonical multi-view pipeline.

Responsibilities:
    - Check silhouette area consistency across views
    - Check image sharpness consistency
    - Check color consistency across views
    - Validate segmentation confidence
    - Produce warnings and per-view quality flags

3-view setup:
    - front: perpendicular, centered (camera on +Z axis)
    - side:  perpendicular from the right (camera on +X axis)
    - top:   bird's-eye looking straight down (camera on +Y axis)

With only 3 orthogonal views there are no natural paired views
(like front/back or left/right) for mirror-plausibility checks.
Validation focuses on area ratios, sharpness, and color consistency.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from api.job_manager import JobManager
from api.storage import StorageManager

from .config import CanonicalMVConfig, CANONICAL_VIEW_ORDER

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum allowed relative area difference between any two side views
# (front and side are both horizontal views, so their silhouette areas
#  should be in the same ballpark for most objects)
DEFAULT_AREA_TOLERANCE = 0.50

# Minimum segmentation confidence to pass validation
DEFAULT_SEG_CONFIDENCE_THRESHOLD = 0.5

# Maximum sharpness ratio between the sharpest and blurriest view
MAX_SHARPNESS_RATIO = 5.0

# Maximum color histogram distance (chi-squared) between views
MAX_COLOR_DISTANCE = 0.6


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _check_silhouette_area_consistency(
    per_view: Dict[str, Dict[str, Any]],
    tolerance: float = DEFAULT_AREA_TOLERANCE,
) -> List[Dict[str, Any]]:
    """
    Check that foreground-area ratios are roughly consistent across
    the horizontal views (front and side).

    The top view is excluded because bird's-eye foreshortening
    produces a very different silhouette area.

    Returns a list of warning dicts (empty if all OK).
    """
    warnings: List[Dict[str, Any]] = []

    # Only compare horizontal views (front and side)
    horizontal_views = ["front", "side"]
    areas = {}
    for vn in horizontal_views:
        vm = per_view.get(vn)
        if vm is None:
            continue
        far = vm.get("foreground_area_ratio", vm.get("far", 0))
        if far and far > 0:
            areas[vn] = far

    if len(areas) < 2:
        return warnings

    values = list(areas.values())
    max_area = max(values)
    min_area = min(values)

    if max_area > 0:
        relative_diff = (max_area - min_area) / max_area
        if relative_diff > tolerance:
            warnings.append({
                "code": "silhouette_area_mismatch",
                "message": (
                    f"Silhouette area differs by {relative_diff:.0%} between "
                    f"horizontal views (tolerance {tolerance:.0%}). "
                    f"Areas: {areas}"
                ),
                "severity": "warning",
            })

    return warnings


def _check_sharpness_consistency(
    per_view: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Check that sharpness scores are reasonably consistent.

    A very blurry view will produce poor reconstruction quality.
    """
    warnings: List[Dict[str, Any]] = []

    sharpness = {}
    for vn in CANONICAL_VIEW_ORDER:
        vm = per_view.get(vn)
        if vm is None:
            continue
        s = vm.get("sharpness", 0)
        if s and s > 0:
            sharpness[vn] = s

    if len(sharpness) < 2:
        return warnings

    values = list(sharpness.values())
    max_s = max(values)
    min_s = min(values)

    if min_s > 0 and max_s / min_s > MAX_SHARPNESS_RATIO:
        blurriest = min(sharpness, key=sharpness.get)
        warnings.append({
            "code": "sharpness_mismatch",
            "message": (
                f"View '{blurriest}' is much blurrier than other views "
                f"(sharpness ratio {max_s / min_s:.1f}x). "
                f"Consider re-uploading a sharper image."
            ),
            "severity": "warning",
            "view": blurriest,
        })

    return warnings


def _check_segmentation_confidence(
    per_view: Dict[str, Dict[str, Any]],
    threshold: float = DEFAULT_SEG_CONFIDENCE_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Check that segmentation confidence is above threshold for all views.
    """
    warnings: List[Dict[str, Any]] = []

    for vn in CANONICAL_VIEW_ORDER:
        vm = per_view.get(vn)
        if vm is None:
            continue
        conf = vm.get("segmentation_confidence", 1.0)
        if conf < threshold:
            warnings.append({
                "code": "low_segmentation_confidence",
                "message": (
                    f"View '{vn}' has low segmentation confidence "
                    f"({conf:.2f} < {threshold:.2f}). "
                    f"The background removal may be unreliable."
                ),
                "severity": "warning",
                "view": vn,
            })

    return warnings


def _check_color_consistency(
    per_view: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Check that foreground color distributions are broadly consistent.

    Uses the mean foreground color from each view. Large differences
    may indicate lighting inconsistency or different objects.
    """
    warnings: List[Dict[str, Any]] = []

    means = {}
    for vn in CANONICAL_VIEW_ORDER:
        vm = per_view.get(vn)
        if vm is None:
            continue
        cm = vm.get("color_histogram_mean", vm.get("color_hist_mean"))
        if cm and isinstance(cm, (list, tuple)) and len(cm) == 3:
            means[vn] = np.array(cm, dtype=np.float64)

    if len(means) < 2:
        return warnings

    # Check pairwise color distance
    view_names = list(means.keys())
    for i in range(len(view_names)):
        for j in range(i + 1, len(view_names)):
            vi, vj = view_names[i], view_names[j]
            dist = np.linalg.norm(means[vi] - means[vj]) / 255.0
            if dist > MAX_COLOR_DISTANCE:
                warnings.append({
                    "code": "color_inconsistency",
                    "message": (
                        f"Views '{vi}' and '{vj}' have significantly "
                        f"different foreground colors (distance {dist:.2f}). "
                        f"Ensure consistent lighting across all views."
                    ),
                    "severity": "warning",
                })

    return warnings


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------


def run_validate_views(
    job_id: str,
    config: CanonicalMVConfig,
    jm: JobManager,
    sm: StorageManager,
) -> None:
    """
    Execute the validate_views stage.

    Steps:
        1. Load preprocess_metrics.json from the preprocess stage.
        2. Run all consistency checks.
        3. Aggregate warnings.
        4. Save validation results.
        5. Update job metadata.

    This stage never fails the job — it only produces warnings.
    The orchestrator may choose to skip downstream stages based
    on warning severity, but that's a policy decision.
    """
    logger.info(f"[{job_id}] validate_views: starting")
    jm.update_job(job_id, stage_progress=0.0)

    # Step 1: Load metrics
    metrics = sm.load_artifact_json(job_id, "preprocess_metrics.json")
    if metrics is None:
        logger.warning(
            f"[{job_id}] validate_views: no preprocess_metrics.json found, "
            f"skipping validation"
        )
        jm.update_job(job_id, stage_progress=1.0)
        return

    per_view = metrics.get("per_view", {})
    jm.update_job(job_id, stage_progress=0.2)

    # Step 2: Run checks
    all_warnings: List[Dict[str, Any]] = []

    all_warnings.extend(
        _check_silhouette_area_consistency(
            per_view,
            tolerance=config.silhouette_area_tolerance,
        )
    )
    jm.update_job(job_id, stage_progress=0.4)

    all_warnings.extend(_check_sharpness_consistency(per_view))
    jm.update_job(job_id, stage_progress=0.5)

    all_warnings.extend(
        _check_segmentation_confidence(
            per_view,
            threshold=config.segmentation_confidence_threshold,
        )
    )
    jm.update_job(job_id, stage_progress=0.6)

    all_warnings.extend(_check_color_consistency(per_view))
    jm.update_job(job_id, stage_progress=0.8)

    # Step 3: Save results
    validation_result = {
        "n_warnings": len(all_warnings),
        "warnings": all_warnings,
        "views_checked": list(per_view.keys()),
    }
    sm.save_artifact_json(job_id, "validation_result.json", validation_result)

    # Step 4: Update job with warnings
    warning_messages = [w["message"] for w in all_warnings]
    if warning_messages:
        jm.update_job(job_id, warnings=warning_messages)

    jm.update_job(job_id, stage_progress=1.0)

    if all_warnings:
        logger.info(
            f"[{job_id}] validate_views: completed with "
            f"{len(all_warnings)} warnings"
        )
    else:
        logger.info(f"[{job_id}] validate_views: completed (all checks passed)")

