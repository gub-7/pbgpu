"""
GPU-cluster 3D reconstruction API package.

Re-exports key models, enums, and service classes for convenience.

NOTE: JobManager is intentionally NOT imported here to avoid a circular
import.  The chain would be:
    pipelines.__init__  →  pipelines.camera_init  →  api.models
    →  (triggers api.__init__)  →  api.job_manager
    →  pipelines.camera_init  (still initialising → ImportError)
Import JobManager directly where needed:
    from api.job_manager import JobManager
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
]

