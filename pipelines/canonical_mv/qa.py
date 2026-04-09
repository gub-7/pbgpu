"""
Quality assessment stage for the canonical multi-view pipeline.

Responsibilities (TDD §9.9):
    - Silhouette IoU per view
    - Masked PSNR / SSIM per view
    - Mesh component count
    - Watertightness check
    - Self-intersection count
    - Texture seam score
    - Camera reprojection error
    - Symmetry deviation score
    - Completion coverage ratio
    - Overall quality_score, warnings[], recommended_retry[]

Artifacts produced:
    - ``metrics.json``  — comprehensive QA metrics (the canonical artifact)

Camera convention (matches camera_init.py):
    - World space: Y-up, right-handed
    - Camera space: X-right, Y-down, Z-forward (OpenCV)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from api.job_manager import JobManager
from api.storage import StorageManager

from .camera_init import CameraRig, project_point
from .config import CanonicalMVConfig, CANONICAL_VIEW_ORDER
from .refine import (
    MeshState,
    compute_vertex_normals,
    load_mesh_ply,
    render_silhouettes,
)
from .export import (
    check_manifoldness,
    check_self_intersections,
    find_connected_components,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Quality score weights for each metric
WEIGHT_SILHOUETTE_IOU = 0.30
WEIGHT_MESH_QUALITY = 0.25
WEIGHT_TEXTURE_QUALITY = 0.15
WEIGHT_SYMMETRY = 0.10
WEIGHT_COMPLETENESS = 0.20

# Thresholds for warnings
WARN_SILHOUETTE_IOU = 0.5
WARN_COMPONENT_COUNT = 3
WARN_SELF_INTERSECTIONS = 50
WARN_TEXTURE_COVERAGE = 0.3
WARN_SYMMETRY_DEVIATION = 0.15


# ---------------------------------------------------------------------------
# Per-view metrics
# ---------------------------------------------------------------------------


def compute_silhouette_iou(
    rendered_mask: np.ndarray,
    target_mask: np.ndarray,
) -> float:
    """
    Compute Intersection-over-Union between rendered and target silhouettes.

    Args:
        rendered_mask: (H, W) uint8 rendered silhouette (0 or 255).
        target_mask: (H, W) uint8 target segmentation mask.

    Returns:
        IoU in [0, 1].
    """
    r = (rendered_mask > 127).astype(np.uint8)
    t = (target_mask > 127).astype(np.uint8)

    # Resize if needed
    if r.shape != t.shape:
        t = cv2.resize(t, (r.shape[1], r.shape[0]), interpolation=cv2.INTER_NEAREST)

    intersection = np.sum(r & t)
    union = np.sum(r | t)

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def compute_masked_psnr(
    rendered: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float:
    """
    Compute PSNR between rendered and target images within the mask region.

    Args:
        rendered: (H, W, 3) uint8 rendered image.
        target: (H, W, 3) uint8 target image.
        mask: (H, W) uint8 binary mask.

    Returns:
        PSNR in dB. Returns 100.0 if images are identical.
    """
    m = mask > 127
    if not np.any(m):
        return 0.0

    r = rendered[m].astype(np.float64)
    t = target[m].astype(np.float64)

    mse = np.mean((r - t) ** 2)
    if mse < 1e-10:
        return 100.0

    return float(10.0 * math.log10(255.0 * 255.0 / mse))


def compute_masked_ssim(
    rendered: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float:
    """
    Compute a simplified SSIM between rendered and target images
    within the mask region.

    Uses the mean-based SSIM approximation (no windowed computation)
    for speed and minimal dependencies.

    Args:
        rendered: (H, W, 3) uint8 rendered image.
        target: (H, W, 3) uint8 target image.
        mask: (H, W) uint8 binary mask.

    Returns:
        SSIM in [0, 1].
    """
    m = mask > 127
    if not np.any(m):
        return 0.0

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    r = rendered.astype(np.float64)
    t = target.astype(np.float64)

    # Convert to grayscale for simplicity
    r_gray = 0.299 * r[:, :, 0] + 0.587 * r[:, :, 1] + 0.114 * r[:, :, 2]
    t_gray = 0.299 * t[:, :, 0] + 0.587 * t[:, :, 1] + 0.114 * t[:, :, 2]

    r_vals = r_gray[m]
    t_vals = t_gray[m]

    mu_r = np.mean(r_vals)
    mu_t = np.mean(t_vals)
    sigma_r = np.std(r_vals)
    sigma_t = np.std(t_vals)
    sigma_rt = np.mean((r_vals - mu_r) * (t_vals - mu_t))

    numerator = (2 * mu_r * mu_t + C1) * (2 * sigma_rt + C2)
    denominator = (mu_r ** 2 + mu_t ** 2 + C1) * (sigma_r ** 2 + sigma_t ** 2 + C2)

    if denominator < 1e-10:
        return 1.0

    return float(np.clip(numerator / denominator, 0.0, 1.0))


def compute_per_view_metrics(
    vertices: np.ndarray,
    faces: np.ndarray,
    rig: CameraRig,
    images: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    image_size: Tuple[int, int],
) -> Dict[str, Dict[str, Any]]:
    """
    Compute per-view quality metrics (silhouette IoU, PSNR, SSIM).

    Args:
        vertices: (V, 3) vertex positions.
        faces: (F, 3) face indices.
        rig: Calibrated camera rig.
        images: Per-view RGB images.
        masks: Per-view segmentation masks.
        image_size: (width, height) for silhouette rendering.

    Returns:
        Dict mapping view_name → metrics dict.
    """
    rendered_masks = render_silhouettes(vertices, faces, rig, image_size)
    per_view = {}

    for vn in CANONICAL_VIEW_ORDER:
        metrics: Dict[str, Any] = {}

        # Silhouette IoU
        if vn in rendered_masks and vn in masks:
            iou = compute_silhouette_iou(rendered_masks[vn], masks[vn])
            metrics["silhouette_iou"] = iou
        else:
            metrics["silhouette_iou"] = None

        # PSNR and SSIM (only if we have both rendered and target images)
        if vn in masks and vn in images:
            target_img = images[vn]
            target_mask = masks[vn]

            # Create a simple rendered image: project mesh with flat color
            # For a proper render we'd need the texture, but for QA
            # we compare silhouettes primarily
            h, w = target_img.shape[:2]
            rendered_img = np.zeros_like(target_img)
            if vn in rendered_masks:
                r_mask = rendered_masks[vn]
                # Resize rendered mask to match target image if needed
                if r_mask.shape[:2] != (h, w):
                    r_mask = cv2.resize(
                        r_mask, (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                rendered_img[r_mask > 127] = [180, 180, 180]

            metrics["masked_psnr"] = compute_masked_psnr(
                rendered_img, target_img, target_mask,
            )
            metrics["masked_ssim"] = compute_masked_ssim(
                rendered_img, target_img, target_mask,
            )
        else:
            metrics["masked_psnr"] = None
            metrics["masked_ssim"] = None

        per_view[vn] = metrics

    return per_view


# ---------------------------------------------------------------------------
# Mesh quality metrics
# ---------------------------------------------------------------------------


def compute_mesh_metrics(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute mesh topology and quality metrics.

    Args:
        vertices: (V, 3) vertex positions.
        faces: (F, 3) face indices.

    Returns:
        Dict with mesh quality metrics.
    """
    manifold = check_manifoldness(faces)
    n_components = len(find_connected_components(faces, len(vertices)))
    n_self_intersections = check_self_intersections(vertices, faces)

    # Bounding box
    if len(vertices) > 0:
        bb_min = vertices.min(axis=0).tolist()
        bb_max = vertices.max(axis=0).tolist()
        bb_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    else:
        bb_min = [0, 0, 0]
        bb_max = [0, 0, 0]
        bb_diag = 0.0

    return {
        "n_vertices": len(vertices),
        "n_faces": len(faces),
        "n_components": n_components,
        "is_manifold": manifold["is_manifold"],
        "is_watertight": manifold["is_watertight"],
        "boundary_edges": manifold["boundary_edges"],
        "non_manifold_edges": manifold["non_manifold_edges"],
        "self_intersections": n_self_intersections,
        "bounding_box_min": bb_min,
        "bounding_box_max": bb_max,
        "bounding_box_diagonal": bb_diag,
    }


# ---------------------------------------------------------------------------
# Symmetry deviation
# ---------------------------------------------------------------------------


def compute_symmetry_deviation(
    vertices: np.ndarray,
    axis: int = 0,
) -> float:
    """
    Compute bilateral symmetry deviation score.

    For each vertex, measures the distance to the closest vertex on
    the opposite side of the symmetry plane.

    Args:
        vertices: (V, 3) vertex positions.
        axis: Symmetry axis (0=X for left-right).

    Returns:
        Mean symmetry deviation (lower = more symmetric). Normalized
        by bounding box diagonal.
    """
    if len(vertices) < 2:
        return 0.0

    reflected = vertices.copy()
    reflected[:, axis] = -reflected[:, axis]

    bb_diag = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
    if bb_diag < 1e-10:
        return 0.0

    # Subsample for large meshes
    n = len(vertices)
    if n > 5000:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, size=5000, replace=False)
        sample = vertices[idx]
    else:
        sample = vertices

    # For each sampled vertex, find closest reflected vertex
    total_dev = 0.0
    for v in sample:
        dists = np.linalg.norm(reflected - v, axis=1)
        total_dev += dists.min()

    mean_dev = total_dev / len(sample)
    return float(mean_dev / bb_diag)


# ---------------------------------------------------------------------------
# Quality score aggregation
# ---------------------------------------------------------------------------


def compute_quality_score(
    per_view_metrics: Dict[str, Dict[str, Any]],
    mesh_metrics: Dict[str, Any],
    texture_metrics: Optional[Dict[str, Any]],
    symmetry_deviation: float,
    completion_coverage: Optional[float],
) -> Tuple[float, List[Dict[str, str]], List[str]]:
    """
    Compute an overall quality score from all metrics.

    Returns:
        Tuple of (quality_score, warnings, recommended_retry).
    """
    warnings: List[Dict[str, str]] = []
    recommended_retry: List[str] = []

    # --- Silhouette IoU score ---
    ious = [
        m["silhouette_iou"]
        for m in per_view_metrics.values()
        if m.get("silhouette_iou") is not None
    ]
    if ious:
        mean_iou = sum(ious) / len(ious)
        iou_score = mean_iou
    else:
        iou_score = 0.0
        mean_iou = 0.0

    # Check per-view IoU warnings
    for vn, m in per_view_metrics.items():
        iou = m.get("silhouette_iou")
        if iou is not None and iou < WARN_SILHOUETTE_IOU:
            warnings.append({
                "code": "low_silhouette_iou",
                "message": f"Silhouette IoU for {vn} is {iou:.2f} (< {WARN_SILHOUETTE_IOU})",
                "severity": "warning",
                "view": vn,
            })
            recommended_retry.append(f"re-upload sharper {vn} image")

    # --- Mesh quality score ---
    mesh_score = 1.0
    n_components = mesh_metrics.get("n_components", 1)
    if n_components > WARN_COMPONENT_COUNT:
        mesh_score *= 0.7
        warnings.append({
            "code": "high_component_count",
            "message": f"Mesh has {n_components} disconnected components",
            "severity": "warning",
        })

    n_si = mesh_metrics.get("self_intersections", 0)
    if n_si > WARN_SELF_INTERSECTIONS:
        mesh_score *= 0.8
        warnings.append({
            "code": "self_intersections",
            "message": f"{n_si} potential self-intersections detected",
            "severity": "warning",
        })

    if not mesh_metrics.get("is_manifold", True):
        mesh_score *= 0.9
        warnings.append({
            "code": "non_manifold",
            "message": "Mesh has non-manifold edges",
            "severity": "warning",
        })

    # --- Texture quality score ---
    tex_score = 0.5  # default if no texture
    if texture_metrics:
        coverage = texture_metrics.get("coverage_fraction", 0.0)
        tex_score = min(coverage / 0.5, 1.0)  # full score at 50% coverage

        if coverage < WARN_TEXTURE_COVERAGE:
            warnings.append({
                "code": "low_texture_coverage",
                "message": f"Texture coverage is {coverage:.1%}",
                "severity": "warning",
            })

    # --- Symmetry score ---
    sym_score = max(0.0, 1.0 - symmetry_deviation / 0.3)
    if symmetry_deviation > WARN_SYMMETRY_DEVIATION:
        warnings.append({
            "code": "high_symmetry_deviation",
            "message": f"Symmetry deviation is {symmetry_deviation:.3f}",
            "severity": "warning",
        })

    # --- Completion coverage score ---
    comp_score = completion_coverage if completion_coverage is not None else 0.5

    # --- Weighted aggregate ---
    quality_score = (
        WEIGHT_SILHOUETTE_IOU * iou_score
        + WEIGHT_MESH_QUALITY * mesh_score
        + WEIGHT_TEXTURE_QUALITY * tex_score
        + WEIGHT_SYMMETRY * sym_score
        + WEIGHT_COMPLETENESS * comp_score
    )
    quality_score = float(np.clip(quality_score, 0.0, 1.0))

    return quality_score, warnings, recommended_retry


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------


def run_qa(
    job_id: str,
    config: CanonicalMVConfig,
    jm: JobManager,
    sm: StorageManager,
) -> None:
    """
    Execute the QA stage.

    Steps:
        1. Load final mesh (from export output or latest available).
        2. Load camera rig and segmented views.
        3. Compute per-view silhouette IoU.
        4. Compute mesh quality metrics.
        5. Compute symmetry deviation.
        6. Load completion and texture metrics from earlier stages.
        7. Aggregate quality score.
        8. Save comprehensive metrics.json.

    Raises:
        ValueError: If required artifacts are missing.
    """
    logger.info(f"[{job_id}] qa: starting")
    jm.update_job(job_id, stage_progress=0.0)

    # ------------------------------------------------------------------
    # Step 1: Load mesh
    # ------------------------------------------------------------------
    # Try final output first, then fall back through pipeline artifacts
    output_file = sm.get_output_file(job_id)
    mesh_source = "output_glb"
    vertices = None
    faces = None
    normals = None

    # GLB is hard to parse back; prefer PLY artifacts
    for artifact_name, source_name in [
        ("completed_mesh.ply", "completed"),
        ("refined_mesh.ply", "refined"),
        ("coarse_visual_hull_mesh.ply", "coarse"),
    ]:
        path = sm.get_artifact_path(job_id, artifact_name)
        if path is not None:
            vertices, faces, normals = load_mesh_ply(str(path))
            mesh_source = source_name
            break

    if vertices is None or len(vertices) == 0:
        raise ValueError("No mesh found for QA")

    if normals is None or len(normals) != len(vertices):
        normals = compute_vertex_normals(vertices, faces)

    logger.info(
        f"[{job_id}] qa: loaded {mesh_source} mesh "
        f"({len(vertices)} verts, {len(faces)} faces)"
    )
    jm.update_job(job_id, stage_progress=0.1)

    # ------------------------------------------------------------------
    # Step 2: Load camera rig and views
    # ------------------------------------------------------------------
    rig_data = sm.load_artifact_json(job_id, "camera_init.json")
    rig = None
    image_size = (256, 256)

    if rig_data is not None:
        rig = CameraRig.from_dict(rig_data)
        image_size = tuple(rig.shared_params["image_size"])

    masks_dict, images_dict = _load_segmented_views(job_id, sm, target_size=image_size)
    jm.update_job(job_id, stage_progress=0.2)

    # ------------------------------------------------------------------
    # Step 3: Per-view metrics
    # ------------------------------------------------------------------
    per_view_metrics: Dict[str, Dict[str, Any]] = {}
    if rig is not None and len(faces) > 0:
        per_view_metrics = compute_per_view_metrics(
            vertices, faces, rig, images_dict, masks_dict, image_size,
        )
    jm.update_job(job_id, stage_progress=0.4)

    # ------------------------------------------------------------------
    # Step 4: Mesh quality metrics
    # ------------------------------------------------------------------
    mesh_metrics = compute_mesh_metrics(vertices, faces)
    jm.update_job(job_id, stage_progress=0.55)

    # ------------------------------------------------------------------
    # Step 5: Symmetry deviation
    # ------------------------------------------------------------------
    symmetry_deviation = 0.0
    if config.symmetry_prior and len(vertices) > 0:
        symmetry_deviation = compute_symmetry_deviation(vertices, axis=0)
    jm.update_job(job_id, stage_progress=0.65)

    # ------------------------------------------------------------------
    # Step 6: Load metrics from earlier stages
    # ------------------------------------------------------------------
    texture_metrics = sm.load_artifact_json(job_id, "texture_metrics.json")
    completion_metrics = sm.load_artifact_json(job_id, "completion_metrics.json")
    export_metrics = sm.load_artifact_json(job_id, "export_metrics.json")

    # Completion coverage ratio
    completion_coverage = None
    if completion_metrics:
        coverage = completion_metrics.get("coverage", {})
        weak_frac = coverage.get("weak_fraction", None)
        if weak_frac is not None:
            completion_coverage = 1.0 - weak_frac  # higher = more observed

    jm.update_job(job_id, stage_progress=0.75)

    # ------------------------------------------------------------------
    # Step 7: Quality score
    # ------------------------------------------------------------------
    quality_score, warnings, recommended_retry = compute_quality_score(
        per_view_metrics,
        mesh_metrics,
        texture_metrics,
        symmetry_deviation,
        completion_coverage,
    )

    logger.info(
        f"[{job_id}] qa: quality_score={quality_score:.2f}, "
        f"warnings={len(warnings)}"
    )
    jm.update_job(job_id, stage_progress=0.85)

    # ------------------------------------------------------------------
    # Step 8: Save comprehensive metrics
    # ------------------------------------------------------------------
    metrics = {
        "quality_score": quality_score,
        "per_view_metrics": per_view_metrics,
        "mesh_metrics": mesh_metrics,
        "texture_metrics": texture_metrics or {},
        "symmetry_deviation": symmetry_deviation,
        "completion_coverage": completion_coverage,
        "warnings": warnings,
        "recommended_retry": recommended_retry,
        "mesh_source": mesh_source,
    }

    sm.save_metrics(job_id, metrics)

    # Also update job with warnings
    warning_messages = [w["message"] for w in warnings]
    jm.update_job(
        job_id,
        warnings=warning_messages,
        stage_progress=1.0,
    )

    logger.info(f"[{job_id}] qa: completed (score={quality_score:.2f})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_segmented_views(
    job_id: str,
    sm: StorageManager,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load segmented view masks and RGB images.

    Args:
        job_id: Job identifier.
        sm: Storage manager.
        target_size: Optional (width, height) to resize loaded images to.
                     Should match the camera rig image_size.
    """
    from .coarse_recon import _load_segmented_views as _load
    return _load(job_id, sm, target_size=target_size)

