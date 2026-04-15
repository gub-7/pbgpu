"""
GPU Cluster Multi-View Reconstruction Pipeline.

Architecture (per expert guidance for 3-image object reconstruction):

  Stage 1 - Preprocessing
    Remove background (rembg), resize to square, normalise.
    Every downstream stage works on clean, gray-background images.

  Stage 2 - View Normalization
    Cross-view bounding-box consistency constraints ensure the
    subject occupies consistent proportions across views.

  Stage 3 - Fiducial Markers
    Render synthetic reference geometry (2 orange squares + 1 blue
    circle) at known 3D positions onto copies of the images.  These
    give DUSt3R / MASt3R strong multi-view correspondences.

  Stage 4 - Camera Init
    Resolve canonical spherical poses to COLMAP-format extrinsics.

  Stage 5 - Coarse Reconstruction (DUSt3R / MASt3R / VGGT)
    Dense geometry recovery using images WITH fiducial markers.
    The markers provide anchor correspondences that improve
    reconstruction quality with only 3 views.

  Stage 6 - Subject Isolation
    Strip fiducial marker geometry from the 3D point cloud.

  Stage 7 - Generative Completion (TRELLIS.2)
    Feed clean images (without markers) into TRELLIS.2 for
    plausible, high-quality 3D asset generation.

  Stage 8 - Export
    Package results as GLB for downstream consumption.

Canonical 3-view rig:
  - FRONT: azimuth=0,  elevation=0   -> looks at object's front face
  - SIDE:  azimuth=90, elevation=0   -> looks at object's right side
  - TOP:   azimuth=0,  elevation=90  -> looks down at object's top

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
from pipelines.fiducial_markers import (
    add_fiducial_markers,
    render_markers_on_image,
    strip_markers_from_pointcloud,
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
    "add_fiducial_markers",
    "render_markers_on_image",
    "strip_markers_from_pointcloud",
    "PipelineOrchestrator",
    "run_pipeline",
]

