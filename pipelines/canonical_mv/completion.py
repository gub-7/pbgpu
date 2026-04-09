"""
Generative completion stage for the canonical multi-view pipeline.

Responsibilities (TDD §9.6):
    - Preserve evidence-backed geometry
    - Identify weak-confidence regions (poorly observed surfaces)
    - Run completion prior only on weak-confidence regions
    - Fuse completed geometry back into the refined mesh
    - Save completed mesh and diagnostic metrics

Completion strategy:
    1. Compute per-vertex view coverage / confidence
    2. Identify weak regions (vertices seen by < threshold views)
    3. Select and run completion provider:
       - SymmetryProvider:  mirror observed geometry across symmetry plane
       - LaplacianProvider: smooth/fill weak regions via Laplacian diffusion
       - Trellis2Provider:  TRELLIS.2 GPU completion (requires GPU)
       - Hunyuan3DProvider: Hunyuan3D GPU completion (requires GPU)
    4. Fuse completed geometry into original mesh (only modify low-confidence regions)

Fusion policy (TDD §9.6):
    - Do NOT replace the entire mesh with generative output
    - Use confidence-weighted blending
    - Patch replacement or displacement-field correction
    - Optional back-side fill only

Artifacts produced:
    - ``completed_mesh.ply``          — mesh after completion
    - ``completion_metrics.json``     — confidence stats, provider info, region counts

Camera convention (matches camera_init.py):
    - World space: Y-up, right-handed
    - Camera space: X-right, Y-down, Z-forward (OpenCV)
"""

import abc
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from scipy import sparse as sp_sparse
    from scipy.ndimage import binary_dilation
except ImportError:
    sp_sparse = None
    binary_dilation = None

from api.job_manager import JobManager
from api.storage import StorageManager

from .camera_init import CameraRig, project_point
from .coarse_recon import save_mesh_ply
from .config import CanonicalMVConfig, CANONICAL_VIEW_ORDER
from .refine import (
    MeshState,
    compute_edges,
    compute_vertex_normals,
    build_laplacian_matrix,
    load_mesh_ply,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum confidence for a vertex to be considered "well-observed"
DEFAULT_CONFIDENCE_THRESHOLD = 0.3

# Minimum number of views that must see a vertex for full confidence
MIN_VIEWS_FOR_FULL_CONFIDENCE = 3

# Maximum blend weight for completed geometry (prevents full replacement)
MAX_COMPLETION_BLEND = 0.8

# Laplacian diffusion iterations for hole filling
DEFAULT_DIFFUSION_ITERATIONS = 20

# Laplacian diffusion step size
DEFAULT_DIFFUSION_ALPHA = 0.5

# Minimum number of weak vertices to trigger completion
MIN_WEAK_VERTICES_FOR_COMPLETION = 1

# Front-facing dot-product threshold (vertex visible if normal · view_dir > this)
FRONT_FACING_THRESHOLD = -0.1


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CompletionConfig:
    """Configuration for the completion stage."""

    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    min_views_for_full_confidence: int = MIN_VIEWS_FOR_FULL_CONFIDENCE
    max_blend_weight: float = MAX_COMPLETION_BLEND
    diffusion_iterations: int = DEFAULT_DIFFUSION_ITERATIONS
    diffusion_alpha: float = DEFAULT_DIFFUSION_ALPHA
    use_symmetry: bool = True
    symmetry_axis: int = 0  # 0=X for left-right symmetry
    use_trellis: bool = False
    use_hunyuan: bool = False

    @classmethod
    def from_pipeline_config(cls, config: CanonicalMVConfig) -> "CompletionConfig":
        """Build completion config from the pipeline config."""
        return cls(
            use_symmetry=config.symmetry_prior,
            use_trellis=config.use_trellis_completion,
            use_hunyuan=config.use_hunyuan_completion,
        )


@dataclass
class CompletionResult:
    """Result from a completion provider."""

    vertices: np.ndarray          # (V, 3) completed vertex positions
    confidence_delta: np.ndarray  # (V,) how much confidence was added per vertex
    provider_name: str            # name of the provider that produced this
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Confidence analysis
# ---------------------------------------------------------------------------


def compute_vertex_visibility(
    vertices: np.ndarray,
    normals: np.ndarray,
    rig: CameraRig,
    masks: Dict[str, np.ndarray],
    image_size: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    """
    Compute per-vertex visibility for each camera view.

    A vertex is considered visible from a view if:
        1. It projects inside the image bounds.
        2. The projected position falls within the segmentation mask.
        3. The vertex normal faces toward the camera (front-facing check).

    Args:
        vertices: (V, 3) vertex positions.
        normals: (V, 3) vertex normals.
        rig: Calibrated camera rig.
        masks: Dict mapping view name → binary mask (H, W) uint8.
        image_size: (width, height) of the masks.

    Returns:
        Dict mapping view name → (V,) bool array of vertex visibility.
    """
    w, h = image_size
    n_verts = len(vertices)
    visibility = {}

    # Pre-compute homogeneous vertex coordinates
    ones = np.ones((n_verts, 1), dtype=np.float64)
    homo = np.hstack([vertices, ones])  # (V, 4)

    for vn in CANONICAL_VIEW_ORDER:
        if vn not in rig.cameras or vn not in masks:
            continue

        mask = masks[vn]
        ext = rig.get_extrinsic(vn)
        intr = rig.get_intrinsic(vn)

        # Get camera position for front-facing check
        cam_pos = rig.get_position(vn)

        # Project all vertices
        cam_coords = (ext @ homo.T).T  # (V, 4)
        z_cam = cam_coords[:, 2]

        p_img = (intr @ cam_coords[:, :3].T).T  # (V, 3)

        with np.errstate(divide="ignore", invalid="ignore"):
            px = p_img[:, 0] / np.maximum(p_img[:, 2], 1e-12)
            py = p_img[:, 1] / np.maximum(p_img[:, 2], 1e-12)

        # Check 1: in front of camera
        in_front = z_cam > 0

        # Check 2: within image bounds
        in_bounds = (px >= 0) & (px < w) & (py >= 0) & (py < h)

        # Check 3: within segmentation mask
        in_mask = np.zeros(n_verts, dtype=bool)
        valid = in_front & in_bounds
        if np.any(valid):
            px_int = np.clip(px[valid].astype(np.int32), 0, w - 1)
            py_int = np.clip(py[valid].astype(np.int32), 0, h - 1)

            # Resize mask if needed
            if mask.shape != (h, w):
                mask_resized = cv2.resize(
                    mask, (w, h), interpolation=cv2.INTER_NEAREST,
                )
            else:
                mask_resized = mask

            mask_vals = mask_resized[py_int, px_int]
            in_mask[valid] = mask_vals > 127

        # Check 4: front-facing (vertex normal faces camera)
        view_dirs = cam_pos - vertices  # (V, 3) direction from vertex to camera
        view_dirs_norm = np.linalg.norm(view_dirs, axis=1, keepdims=True)
        view_dirs_safe = view_dirs / np.maximum(view_dirs_norm, 1e-12)

        dots = np.sum(normals * view_dirs_safe, axis=1)
        front_facing = dots > FRONT_FACING_THRESHOLD

        # Combine all checks
        visible = in_front & in_bounds & in_mask & front_facing
        visibility[vn] = visible

    return visibility


def compute_vertex_confidence(
    visibility: Dict[str, np.ndarray],
    n_vertices: int,
    min_views: int = MIN_VIEWS_FOR_FULL_CONFIDENCE,
) -> np.ndarray:
    """
    Compute per-vertex confidence from visibility across views.

    Confidence is the fraction of views that observe the vertex,
    normalized so that ``min_views`` views gives confidence 1.0.

    Args:
        visibility: Dict mapping view name → (V,) bool visibility.
        n_vertices: Number of vertices.
        min_views: Number of views for full confidence (1.0).

    Returns:
        (V,) float64 confidence in [0, 1].
    """
    view_count = np.zeros(n_vertices, dtype=np.float64)

    for vis in visibility.values():
        view_count += vis.astype(np.float64)

    # Normalize: min_views views → confidence 1.0
    confidence = np.clip(view_count / max(min_views, 1), 0.0, 1.0)

    return confidence


def identify_weak_regions(
    confidence: np.ndarray,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> np.ndarray:
    """
    Identify vertices in weak-confidence regions.

    Args:
        confidence: (V,) per-vertex confidence in [0, 1].
        threshold: Confidence below this is considered "weak".

    Returns:
        (V,) bool mask — True for weak vertices.
    """
    return confidence < threshold


def compute_coverage_stats(
    confidence: np.ndarray,
    weak_mask: np.ndarray,
    visibility: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """
    Compute summary statistics for coverage analysis.

    Args:
        confidence: (V,) per-vertex confidence.
        weak_mask: (V,) bool mask of weak vertices.
        visibility: Per-view visibility dicts.

    Returns:
        Dict with coverage statistics.
    """
    n_verts = len(confidence)
    n_weak = int(np.sum(weak_mask))

    per_view = {}
    for vn, vis in visibility.items():
        per_view[vn] = {
            "visible_vertices": int(np.sum(vis)),
            "visible_fraction": float(np.sum(vis) / max(n_verts, 1)),
        }

    return {
        "n_vertices": n_verts,
        "n_weak_vertices": n_weak,
        "weak_fraction": float(n_weak / max(n_verts, 1)),
        "mean_confidence": float(np.mean(confidence)),
        "median_confidence": float(np.median(confidence)),
        "min_confidence": float(np.min(confidence)) if n_verts > 0 else 0.0,
        "max_confidence": float(np.max(confidence)) if n_verts > 0 else 0.0,
        "per_view_coverage": per_view,
    }


# ---------------------------------------------------------------------------
# Completion providers
# ---------------------------------------------------------------------------


class CompletionProvider(abc.ABC):
    """
    Abstract base class for geometry completion providers.

    A provider takes a mesh with a weak-region mask and produces
    completed vertex positions for the weak regions.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable name of this provider."""

    @abc.abstractmethod
    def complete(
        self,
        mesh: MeshState,
        weak_mask: np.ndarray,
        confidence: np.ndarray,
        rig: CameraRig,
        images: Dict[str, np.ndarray],
        masks: Dict[str, np.ndarray],
    ) -> CompletionResult:
        """
        Generate completed geometry for weak regions.

        Args:
            mesh: Current mesh state.
            weak_mask: (V,) bool mask of weak vertices.
            confidence: (V,) per-vertex confidence.
            rig: Calibrated camera rig.
            images: Per-view RGB images.
            masks: Per-view segmentation masks.

        Returns:
            CompletionResult with new vertex positions.
        """


class SymmetryCompletionProvider(CompletionProvider):
    """
    Complete weak regions by mirroring geometry from the observed side.

    For each weak vertex, find the closest vertex on the opposite side
    of the symmetry plane that has high confidence, and mirror its
    position.

    This is effective for subjects with approximate bilateral symmetry
    (e.g. human busts, animals).
    """

    def __init__(self, axis: int = 0):
        """
        Args:
            axis: Symmetry axis (0=X for left-right, 1=Y, 2=Z).
        """
        self.axis = axis

    @property
    def name(self) -> str:
        return "symmetry"

    def complete(
        self,
        mesh: MeshState,
        weak_mask: np.ndarray,
        confidence: np.ndarray,
        rig: CameraRig,
        images: Dict[str, np.ndarray],
        masks: Dict[str, np.ndarray],
    ) -> CompletionResult:
        """Mirror high-confidence geometry to fill weak regions."""
        vertices = mesh.vertices.copy()
        n_verts = len(vertices)
        confidence_delta = np.zeros(n_verts, dtype=np.float64)

        weak_indices = np.where(weak_mask)[0]
        strong_mask = ~weak_mask & (confidence > 0.5)
        strong_indices = np.where(strong_mask)[0]

        n_completed = 0

        if len(weak_indices) == 0 or len(strong_indices) == 0:
            return CompletionResult(
                vertices=vertices,
                confidence_delta=confidence_delta,
                provider_name=self.name,
                metadata={"n_completed": 0, "axis": self.axis},
            )

        # Reflect strong vertices across the symmetry plane
        strong_verts = vertices[strong_indices].copy()
        reflected = strong_verts.copy()
        reflected[:, self.axis] = -reflected[:, self.axis]

        # For each weak vertex, find the closest reflected strong vertex
        for wi in weak_indices:
            v = vertices[wi]

            # Distance to all reflected strong vertices
            diffs = reflected - v[np.newaxis, :]
            dists = np.linalg.norm(diffs, axis=1)
            closest_idx = np.argmin(dists)
            min_dist = dists[closest_idx]

            # Only mirror if the reflected vertex is reasonably close
            # (within 2x the median edge length)
            if min_dist < 0.5:  # reasonable threshold for normalized meshes
                # Blend: move weak vertex toward the mirrored position
                mirrored_pos = reflected[closest_idx]
                source_confidence = confidence[strong_indices[closest_idx]]

                # Blend weight proportional to source confidence
                blend = min(source_confidence * 0.8, MAX_COMPLETION_BLEND)
                vertices[wi] = (1.0 - blend) * vertices[wi] + blend * mirrored_pos

                confidence_delta[wi] = blend * source_confidence
                n_completed += 1

        return CompletionResult(
            vertices=vertices,
            confidence_delta=confidence_delta,
            provider_name=self.name,
            metadata={
                "n_completed": n_completed,
                "n_weak": len(weak_indices),
                "n_strong": len(strong_indices),
                "axis": self.axis,
            },
        )


class LaplacianCompletionProvider(CompletionProvider):
    """
    Complete weak regions by Laplacian diffusion from confident neighbors.

    Iteratively smooths weak-region vertices toward the average of their
    neighbors, weighted by confidence. This fills small holes and
    smooths transitions between observed and unobserved regions.
    """

    def __init__(
        self,
        n_iterations: int = DEFAULT_DIFFUSION_ITERATIONS,
        alpha: float = DEFAULT_DIFFUSION_ALPHA,
    ):
        self.n_iterations = n_iterations
        self.alpha = alpha

    @property
    def name(self) -> str:
        return "laplacian"

    def complete(
        self,
        mesh: MeshState,
        weak_mask: np.ndarray,
        confidence: np.ndarray,
        rig: CameraRig,
        images: Dict[str, np.ndarray],
        masks: Dict[str, np.ndarray],
    ) -> CompletionResult:
        """Smooth weak regions via confidence-weighted Laplacian diffusion."""
        vertices = mesh.vertices.copy()
        n_verts = len(vertices)
        confidence_delta = np.zeros(n_verts, dtype=np.float64)
        original_vertices = vertices.copy()

        if not np.any(weak_mask):
            return CompletionResult(
                vertices=vertices,
                confidence_delta=confidence_delta,
                provider_name=self.name,
                metadata={"n_iterations": 0, "n_weak": 0},
            )

        # Build adjacency from edges
        edges = compute_edges(mesh.faces)
        n_edges = len(edges)

        if n_edges == 0:
            return CompletionResult(
                vertices=vertices,
                confidence_delta=confidence_delta,
                provider_name=self.name,
                metadata={"n_iterations": 0, "n_weak": int(np.sum(weak_mask))},
            )

        # Build neighbor lists
        neighbors: Dict[int, List[int]] = {i: [] for i in range(n_verts)}
        for e in edges:
            neighbors[int(e[0])].append(int(e[1]))
            neighbors[int(e[1])].append(int(e[0]))

        # Iterative Laplacian diffusion — only move weak vertices
        for iteration in range(self.n_iterations):
            new_verts = vertices.copy()

            for vi in np.where(weak_mask)[0]:
                nbrs = neighbors.get(int(vi), [])
                if len(nbrs) == 0:
                    continue

                # Weighted average of neighbors (higher confidence neighbors
                # have more influence)
                nbr_verts = vertices[nbrs]
                nbr_conf = confidence[nbrs]

                # Weight by confidence (at least a small weight for all)
                weights = np.maximum(nbr_conf, 0.1)
                weights = weights / weights.sum()

                avg_pos = np.sum(
                    nbr_verts * weights[:, np.newaxis], axis=0,
                )

                # Move toward weighted average
                new_verts[vi] = (
                    (1.0 - self.alpha) * vertices[vi]
                    + self.alpha * avg_pos
                )

            vertices = new_verts

        # Compute confidence delta from displacement
        displacements = np.linalg.norm(vertices - original_vertices, axis=1)
        max_disp = displacements.max() if displacements.max() > 0 else 1.0
        # Vertices that moved more get more confidence boost (they were filled)
        confidence_delta[weak_mask] = np.clip(
            displacements[weak_mask] / max_disp * 0.5, 0.0, 0.5,
        )

        return CompletionResult(
            vertices=vertices,
            confidence_delta=confidence_delta,
            provider_name=self.name,
            metadata={
                "n_iterations": self.n_iterations,
                "alpha": self.alpha,
                "n_weak": int(np.sum(weak_mask)),
                "mean_displacement": float(np.mean(displacements[weak_mask]))
                if np.any(weak_mask) else 0.0,
                "max_displacement": float(np.max(displacements[weak_mask]))
                if np.any(weak_mask) else 0.0,
            },
        )


class Trellis2CompletionProvider(CompletionProvider):
    """
    TRELLIS.2 GPU-based completion provider.

    Uses TRELLIS.2 as a generative prior for completing low-confidence
    regions. Requires a GPU and the TRELLIS.2 model checkpoint.

    This provider is a stub — actual GPU inference is deferred to the
    worker environment where the model is loaded.
    """

    @property
    def name(self) -> str:
        return "trellis2"

    def complete(
        self,
        mesh: MeshState,
        weak_mask: np.ndarray,
        confidence: np.ndarray,
        rig: CameraRig,
        images: Dict[str, np.ndarray],
        masks: Dict[str, np.ndarray],
    ) -> CompletionResult:
        """
        Run TRELLIS.2 completion.

        Raises:
            RuntimeError: If TRELLIS.2 is not available in this environment.
        """
        raise RuntimeError(
            "TRELLIS.2 completion requires GPU and model checkpoint. "
            "Set use_trellis_completion=False or ensure the model is loaded."
        )


class Hunyuan3DCompletionProvider(CompletionProvider):
    """
    Hunyuan3D GPU-based completion provider.

    Uses Hunyuan3D 2.x as a generative prior for completing
    low-confidence regions. Requires a GPU and the Hunyuan3D checkpoint.

    This provider is a stub — actual GPU inference is deferred to the
    worker environment where the model is loaded.
    """

    @property
    def name(self) -> str:
        return "hunyuan3d"

    def complete(
        self,
        mesh: MeshState,
        weak_mask: np.ndarray,
        confidence: np.ndarray,
        rig: CameraRig,
        images: Dict[str, np.ndarray],
        masks: Dict[str, np.ndarray],
    ) -> CompletionResult:
        """
        Run Hunyuan3D completion.

        Raises:
            RuntimeError: If Hunyuan3D is not available in this environment.
        """
        raise RuntimeError(
            "Hunyuan3D completion requires GPU and model checkpoint. "
            "Set use_hunyuan_completion=False or ensure the model is loaded."
        )


def get_completion_providers(
    config: CompletionConfig,
) -> List[CompletionProvider]:
    """
    Build the ordered list of completion providers based on config.

    Providers are tried in order:
        1. TRELLIS.2 (if enabled)
        2. Hunyuan3D (if enabled)
        3. Symmetry (if enabled — always available as CPU fallback)
        4. Laplacian (always available as final CPU fallback)

    Args:
        config: Completion configuration.

    Returns:
        Ordered list of providers to try.
    """
    providers: List[CompletionProvider] = []

    if config.use_trellis:
        providers.append(Trellis2CompletionProvider())

    if config.use_hunyuan:
        providers.append(Hunyuan3DCompletionProvider())

    if config.use_symmetry:
        providers.append(SymmetryCompletionProvider(axis=config.symmetry_axis))

    # Laplacian is always the final fallback
    providers.append(LaplacianCompletionProvider(
        n_iterations=config.diffusion_iterations,
        alpha=config.diffusion_alpha,
    ))

    return providers


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def fuse_completion(
    original_mesh: MeshState,
    result: CompletionResult,
    confidence: np.ndarray,
    weak_mask: np.ndarray,
    max_blend: float = MAX_COMPLETION_BLEND,
) -> MeshState:
    """
    Fuse completion result into the original mesh.

    Only modifies vertices in weak regions, using confidence-weighted
    blending. High-confidence vertices are never modified.

    Args:
        original_mesh: The pre-completion mesh.
        result: Completion result with new vertex positions.
        confidence: (V,) per-vertex confidence before completion.
        weak_mask: (V,) bool mask of weak vertices.
        max_blend: Maximum blend weight for completed geometry.

    Returns:
        New MeshState with fused vertex positions.
    """
    vertices = original_mesh.vertices.copy()
    n_verts = len(vertices)

    if not np.any(weak_mask) or len(result.vertices) != n_verts:
        return original_mesh.copy()

    # Compute per-vertex blend weight:
    # - High confidence → low blend (preserve original)
    # - Low confidence → high blend (accept completion)
    # - Only blend in weak regions
    blend_weights = np.zeros(n_verts, dtype=np.float64)
    blend_weights[weak_mask] = np.clip(
        (1.0 - confidence[weak_mask]) * max_blend,
        0.0,
        max_blend,
    )

    # Weighted blend
    vertices = (
        (1.0 - blend_weights[:, np.newaxis]) * vertices
        + blend_weights[:, np.newaxis] * result.vertices
    )

    # Recompute normals for the new vertex positions
    normals = compute_vertex_normals(vertices, original_mesh.faces)

    return MeshState(
        vertices=vertices,
        faces=original_mesh.faces.copy(),
        normals=normals,
        vertex_colors=original_mesh.vertex_colors.copy(),
    )


def compute_blend_weights(
    confidence: np.ndarray,
    weak_mask: np.ndarray,
    max_blend: float = MAX_COMPLETION_BLEND,
) -> np.ndarray:
    """
    Compute per-vertex blend weights for fusion.

    Args:
        confidence: (V,) per-vertex confidence.
        weak_mask: (V,) bool mask of weak vertices.
        max_blend: Maximum blend weight.

    Returns:
        (V,) blend weights in [0, max_blend].
    """
    n_verts = len(confidence)
    weights = np.zeros(n_verts, dtype=np.float64)
    weights[weak_mask] = np.clip(
        (1.0 - confidence[weak_mask]) * max_blend,
        0.0,
        max_blend,
    )
    return weights


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------


def run_complete_geometry(
    job_id: str,
    config: CanonicalMVConfig,
    jm: JobManager,
    sm: StorageManager,
) -> None:
    """
    Execute the complete_geometry stage.

    Steps:
        1. Load camera rig from camera_init.json.
        2. Load refined mesh (or coarse mesh as fallback).
        3. Load segmented masks and images.
        4. Compute per-vertex visibility and confidence.
        5. Identify weak regions.
        6. Run completion providers (with fallback chain).
        7. Fuse completed geometry into the mesh.
        8. Save completed mesh and metrics.

    Raises:
        ValueError: If required artifacts are missing.
    """
    logger.info(f"[{job_id}] complete_geometry: starting")
    jm.update_job(job_id, stage_progress=0.0)

    # Build completion config
    comp_config = CompletionConfig.from_pipeline_config(config)

    # ------------------------------------------------------------------
    # Step 1: Load camera rig
    # ------------------------------------------------------------------
    rig_data = sm.load_artifact_json(job_id, "camera_init.json")
    if rig_data is None:
        raise ValueError(
            "camera_init.json not found — initialize_cameras must run first"
        )
    rig = CameraRig.from_dict(rig_data)
    image_size = tuple(rig.shared_params["image_size"])

    jm.update_job(job_id, stage_progress=0.05)

    # ------------------------------------------------------------------
    # Step 2: Load mesh (prefer refined, fall back to coarse)
    # ------------------------------------------------------------------
    mesh_path = sm.get_artifact_path(job_id, "refined_mesh.ply")
    mesh_source = "refined"
    if mesh_path is None:
        mesh_path = sm.get_artifact_path(job_id, "coarse_visual_hull_mesh.ply")
        mesh_source = "coarse"
    if mesh_path is None:
        raise ValueError(
            "No mesh found — refine_joint or reconstruct_coarse must run first"
        )

    vertices, faces, normals = load_mesh_ply(str(mesh_path))
    if len(normals) != len(vertices) or np.allclose(normals, 0):
        normals = compute_vertex_normals(vertices, faces)

    vertex_colors = np.full((len(vertices), 3), 128, dtype=np.uint8)

    mesh = MeshState(
        vertices=vertices,
        faces=faces,
        normals=normals,
        vertex_colors=vertex_colors,
    )

    logger.info(
        f"[{job_id}] complete_geometry: loaded {mesh_source} mesh "
        f"({mesh.n_vertices} verts, {mesh.n_faces} faces)"
    )
    jm.update_job(job_id, stage_progress=0.1)

    # ------------------------------------------------------------------
    # Step 3: Load segmented masks and images
    # ------------------------------------------------------------------
    masks, images = _load_segmented_views(job_id, sm, target_size=image_size)
    logger.info(
        f"[{job_id}] complete_geometry: loaded {len(masks)} view masks"
    )
    jm.update_job(job_id, stage_progress=0.15)

    # ------------------------------------------------------------------
    # Step 4: Compute visibility and confidence
    # ------------------------------------------------------------------
    visibility = compute_vertex_visibility(
        mesh.vertices, mesh.normals, rig, masks, image_size,
    )
    confidence = compute_vertex_confidence(
        visibility, mesh.n_vertices,
        min_views=comp_config.min_views_for_full_confidence,
    )
    weak_mask = identify_weak_regions(
        confidence, comp_config.confidence_threshold,
    )

    coverage_stats = compute_coverage_stats(confidence, weak_mask, visibility)
    logger.info(
        f"[{job_id}] complete_geometry: "
        f"{coverage_stats['n_weak_vertices']}/{coverage_stats['n_vertices']} "
        f"weak vertices ({coverage_stats['weak_fraction']:.1%}), "
        f"mean confidence={coverage_stats['mean_confidence']:.3f}"
    )
    jm.update_job(job_id, stage_progress=0.3)

    # ------------------------------------------------------------------
    # Step 5: Run completion providers
    # ------------------------------------------------------------------
    providers = get_completion_providers(comp_config)
    completion_result = None
    provider_errors: List[Dict[str, str]] = []

    n_weak = int(np.sum(weak_mask))

    if n_weak < MIN_WEAK_VERTICES_FOR_COMPLETION:
        logger.info(
            f"[{job_id}] complete_geometry: only {n_weak} weak vertices, "
            f"skipping completion"
        )
        # No completion needed — mesh is already well-observed
        completed_mesh = mesh.copy()
        completion_result = CompletionResult(
            vertices=mesh.vertices.copy(),
            confidence_delta=np.zeros(mesh.n_vertices, dtype=np.float64),
            provider_name="none",
            metadata={"reason": "no_weak_vertices"},
        )
    else:
        for provider in providers:
            try:
                logger.info(
                    f"[{job_id}] complete_geometry: trying provider "
                    f"'{provider.name}'"
                )
                completion_result = provider.complete(
                    mesh=mesh,
                    weak_mask=weak_mask,
                    confidence=confidence,
                    rig=rig,
                    images=images,
                    masks=masks,
                )
                logger.info(
                    f"[{job_id}] complete_geometry: provider "
                    f"'{provider.name}' succeeded"
                )
                break
            except (RuntimeError, ImportError, Exception) as e:
                logger.warning(
                    f"[{job_id}] complete_geometry: provider "
                    f"'{provider.name}' failed: {e}"
                )
                provider_errors.append({
                    "provider": provider.name,
                    "error": str(e),
                })

        if completion_result is None:
            # All providers failed — use identity (no change)
            logger.warning(
                f"[{job_id}] complete_geometry: all providers failed, "
                f"using original mesh"
            )
            completion_result = CompletionResult(
                vertices=mesh.vertices.copy(),
                confidence_delta=np.zeros(mesh.n_vertices, dtype=np.float64),
                provider_name="none",
                metadata={"reason": "all_providers_failed"},
            )
            completed_mesh = mesh.copy()
        else:
            # ----------------------------------------------------------
            # Step 6: Fuse completed geometry
            # ----------------------------------------------------------
            completed_mesh = fuse_completion(
                mesh, completion_result, confidence, weak_mask,
                max_blend=comp_config.max_blend_weight,
            )

    jm.update_job(job_id, stage_progress=0.8)

    # ------------------------------------------------------------------
    # Step 7: Save artifacts
    # ------------------------------------------------------------------
    # Completed mesh
    completed_mesh_path = sm.get_artifact_dir(job_id) / "completed_mesh.ply"
    save_mesh_ply(
        str(completed_mesh_path),
        completed_mesh.vertices,
        completed_mesh.faces,
        completed_mesh.normals,
    )
    logger.info(
        f"[{job_id}] complete_geometry: saved completed mesh "
        f"({completed_mesh.n_vertices} verts, {completed_mesh.n_faces} faces)"
    )

    # Metrics
    metrics = {
        "mesh_source": mesh_source,
        "mesh_vertices": mesh.n_vertices,
        "mesh_faces": mesh.n_faces,
        "coverage": coverage_stats,
        "provider_used": completion_result.provider_name,
        "provider_metadata": completion_result.metadata,
        "provider_errors": provider_errors,
        "n_weak_vertices": n_weak,
        "confidence_threshold": comp_config.confidence_threshold,
        "max_blend_weight": comp_config.max_blend_weight,
        "confidence_delta_mean": float(
            np.mean(completion_result.confidence_delta)
        ),
        "confidence_delta_max": float(
            np.max(completion_result.confidence_delta)
        ) if mesh.n_vertices > 0 else 0.0,
        "config": {
            "use_symmetry": comp_config.use_symmetry,
            "use_trellis": comp_config.use_trellis,
            "use_hunyuan": comp_config.use_hunyuan,
            "symmetry_axis": comp_config.symmetry_axis,
            "diffusion_iterations": comp_config.diffusion_iterations,
            "diffusion_alpha": comp_config.diffusion_alpha,
        },
    }

    sm.save_artifact_json(job_id, "completion_metrics.json", metrics)

    jm.update_job(job_id, stage_progress=1.0)
    logger.info(f"[{job_id}] complete_geometry: completed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_segmented_views(
    job_id: str,
    sm: StorageManager,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load segmented view masks and RGB images from the preprocess stage.

    Args:
        job_id: Job identifier.
        sm: Storage manager.
        target_size: Optional (width, height) to resize loaded images to.
                     Should match the camera rig image_size.

    Returns:
        Tuple of (masks_dict, images_dict).
    """
    from .coarse_recon import _load_segmented_views as _load
    return _load(job_id, sm, target_size=target_size)

