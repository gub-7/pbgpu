"""
GPU-cluster 3D reconstruction API package.

Re-exports key models, enums, and service classes for convenience.
"""

from .models import (
    # Enums
    CategoryEnum,
    ModelEnum,
    PipelineEnum,
    ViewName,
    JobStatus,
    JobStage,
    StageStatus,
    ViewStatus,
    # Constants
    REQUIRED_VIEWS,
    MV_STAGE_ORDER,
    SV_STAGE_ORDER,
    # Parameter models
    TripoSRParams,
    Trellis2Params,
    CanonicalMVParams,
    # View metadata
    ViewMetadata,
    # Request models
    UploadRequest,
    # Response models
    UploadResponse,
    MultiViewUploadResponse,
    PreviewStage,
    PreviewResponse,
    MultiViewPreviewResponse,
    ArtifactInfo,
    ArtifactListResponse,
    QAWarning,
    MetricsResponse,
    JobStatusResponse,
    ErrorResponse,
    CategoryInfo,
    CategoriesResponse,
)

from .job_manager import JobManager
from .storage import StorageManager

__all__ = [
    # Enums
    "CategoryEnum",
    "ModelEnum",
    "PipelineEnum",
    "ViewName",
    "JobStatus",
    "JobStage",
    "StageStatus",
    "ViewStatus",
    # Constants
    "REQUIRED_VIEWS",
    "MV_STAGE_ORDER",
    "SV_STAGE_ORDER",
    # Parameter models
    "TripoSRParams",
    "Trellis2Params",
    "CanonicalMVParams",
    # View metadata
    "ViewMetadata",
    # Request models
    "UploadRequest",
    # Response models
    "UploadResponse",
    "MultiViewUploadResponse",
    "PreviewStage",
    "PreviewResponse",
    "MultiViewPreviewResponse",
    "ArtifactInfo",
    "ArtifactListResponse",
    "QAWarning",
    "MetricsResponse",
    "JobStatusResponse",
    "ErrorResponse",
    "CategoryInfo",
    "CategoriesResponse",
    # Services
    "JobManager",
    "StorageManager",
]

