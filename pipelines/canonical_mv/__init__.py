"""
Canonical 3-view multi-view reconstruction pipeline.

Stages (in order):
    1. ingest           - validate uploaded views and create raw previews
    2. preprocess_views - per-view segmentation, cross-view framing
    3. validate_views   - cross-view consistency checks
    4. initialize_cameras - canonical camera rig setup
    5. reconstruct_coarse - visual hull / sparse Gaussian initialization
    6. refine_joint     - joint mesh + Gaussian refinement
    7. complete_geometry - generative completion on weak regions
    8. bake_texture     - multi-view texture projection
    9. export           - mesh cleanup, decimation, GLB export
   10. qa               - quality scoring and diagnostics
"""

from .config import CanonicalMVConfig
from .orchestrator import CanonicalMVOrchestrator
from .camera_init import CameraRig, build_canonical_rig
from .refine import JointRefiner, RefinementConfig, MeshState
from .completion import (
    CompletionConfig,
    CompletionProvider,
    CompletionResult,
    SymmetryCompletionProvider,
    LaplacianCompletionProvider,
)
from .texturing import UVLayout, TextureResult, unwrap_uvs, bake_texture
from .export import (
    export_glb,
    decimate_mesh,
    remove_small_components,
    find_connected_components,
    check_manifoldness,
    check_self_intersections,
    normalize_scale,
)
from .qa import (
    compute_silhouette_iou,
    compute_quality_score,
    compute_mesh_metrics,
    compute_symmetry_deviation,
)

__all__ = [
    "CanonicalMVConfig",
    "CanonicalMVOrchestrator",
    "CameraRig",
    "build_canonical_rig",
    "JointRefiner",
    "RefinementConfig",
    "MeshState",
    "CompletionConfig",
    "CompletionProvider",
    "CompletionResult",
    "SymmetryCompletionProvider",
    "LaplacianCompletionProvider",
    "UVLayout",
    "TextureResult",
    "unwrap_uvs",
    "bake_texture",
    "export_glb",
    "decimate_mesh",
    "remove_small_components",
    "find_connected_components",
    "check_manifoldness",
    "check_self_intersections",
    "normalize_scale",
    "compute_silhouette_iou",
    "compute_quality_score",
    "compute_mesh_metrics",
    "compute_symmetry_deviation",
]
