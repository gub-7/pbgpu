"""
GPU Cluster Multi-View Reconstruction Pipeline.

Architecture (per expert guidance for 3-image object reconstruction):

  Stage A – Camera/Geometry (coarse_recon)
    Use FULL images (with background) for pose recovery and dense geometry.
    Background provides stable correspondences, stronger parallax cues,
    and better global alignment with only 3 views.
    Backends: DUSt3R, MASt3R, VGGT

  Stage B – Subject Isolation (subject_isolation)
    Remove background from images using rembg/SAM.
    Filter 3D point cloud using multi-view mask consensus.
    Keep only geometry consistent with the object masks.

  Stage C – Generative Completion (trellis_completion)
    Feed masked images (+ optional cameras / coarse geometry) into
    TRELLIS.2 for plausible, high-quality 3D asset generation.

Canonical 3-view rig:
  - FRONT: azimuth=0°,  elevation=0°  → looks at object's front face
  - SIDE:  azimuth=90°, elevation=0°  → looks at object's right side
  - TOP:   azimuth=0°,  elevation=90° → looks down at object's top

Stitching geometry:
  - Front's LEFT edge  ↔ Side's RIGHT edge
  - Front's TOP edge   ↔ Top's BOTTOM edge
  - Top's LEFT edge    ↔ Side's TOP edge

World convention:
  - Origin at object centre
  - +Z = up, +Y = forward (object faces +Y), +X = right
"""

from pipelines.camera_init import (
    get_canonical_views,
    resolve_views,
    export_colmap_workspace,
    pose_to_extrinsics,
    spherical_to_camera_center,
)
from pipelines.config import (
    get_default_intrinsics,
    get_default_pipeline_config,
    ensure_directories,
)
from pipelines.orchestrator import (
    PipelineOrchestrator,
    run_pipeline,
)

__all__ = [
    "get_canonical_views",
    "resolve_views",
    "export_colmap_workspace",
    "pose_to_extrinsics",
    "spherical_to_camera_center",
    "get_default_intrinsics",
    "get_default_pipeline_config",
    "ensure_directories",
    "PipelineOrchestrator",
    "run_pipeline",
]

