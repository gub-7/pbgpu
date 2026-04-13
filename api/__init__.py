"""
GPU-cluster 3D reconstruction API package.

Re-exports key models, enums, and service classes for convenience.
"""

from .models import (
    # Enums
    ViewLabel,
    JobStatus,
    ReconBackend,
    # Camera / geometry
    WorldConvention,
    CameraIntrinsics,
    SphericalPose,
    ViewSpec,
    CameraExtrinsics,
    ResolvedView,
    # Pipeline config
    PipelineConfig,
    # Reconstruction artifacts
    PointCloud,
    CoarseReconResult,
    IsolationResult,
    TrellisResult,
    # Job
    ReconJob,
    # API request / response
    CreateJobRequest,
    JobStatusResponse,
    JobDetailResponse,
)

from .job_manager import JobManager

__all__ = [
    # Enums
    "ViewLabel",
    "JobStatus",
    "ReconBackend",
    # Camera / geometry
    "WorldConvention",
    "CameraIntrinsics",
    "SphericalPose",
    "ViewSpec",
    "CameraExtrinsics",
    "ResolvedView",
    # Pipeline config
    "PipelineConfig",
    # Reconstruction artifacts
    "PointCloud",
    "CoarseReconResult",
    "IsolationResult",
    "TrellisResult",
    # Job
    "ReconJob",
    # API request / response
    "CreateJobRequest",
    "JobStatusResponse",
    "JobDetailResponse",
    # Services
    "JobManager",
]

