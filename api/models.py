"""
Data models for the GPU cluster multi-view reconstruction pipeline.

Defines camera specifications, job lifecycle, view metadata, and
pipeline configuration using Pydantic for validation and serialization.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ViewLabel(str, enum.Enum):
    """Canonical view labels for the 3-image capture rig."""

    FRONT = "front"
    SIDE = "side"
    TOP = "top"


class JobStatus(str, enum.Enum):
    """Lifecycle states for a reconstruction job."""

    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    VIEW_NORMALIZATION = "view_normalization"
    FIDUCIAL_MARKERS = "fiducial_markers"
    CAMERA_INIT = "camera_init"
    COARSE_RECON = "coarse_recon"
    SUBJECT_ISOLATION = "subject_isolation"
    TRELLIS_COMPLETION = "trellis_completion"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"


class ReconBackend(str, enum.Enum):
    """Supported coarse-reconstruction backends."""

    DUST3R = "dust3r"
    MAST3R = "mast3r"
    VGGT = "vggt"


# ---------------------------------------------------------------------------
# Camera / geometry primitives
# ---------------------------------------------------------------------------


class WorldConvention(BaseModel):
    """
    Defines the world coordinate frame used throughout the pipeline.

    Convention (object-centric):
      - Origin at the object centre
      - +Z  = up
      - +Y  = object forward (the face the "front" camera looks at)
      - +X  = object right
    """

    origin: str = Field(
        default="object_center",
        description="Semantic label for the world origin",
    )
    up_axis: str = Field(default="Z", description="World up axis")
    forward_axis: str = Field(
        default="Y", description="World forward axis (object faces +Y)"
    )


class CameraIntrinsics(BaseModel):
    """Pinhole camera intrinsics (no distortion – images are pre-undistorted)."""

    width: int = Field(default=2048, gt=0)
    height: int = Field(default=2048, gt=0)
    fx: float = Field(default=1700.0, gt=0)
    fy: float = Field(default=1700.0, gt=0)
    cx: float = Field(default=1024.0, gt=0)
    cy: float = Field(default=1024.0, gt=0)


class SphericalPose(BaseModel):
    """
    Camera placement on a sphere centred on the target point.

    All angles are stored in **degrees** and converted to radians at
    computation time.  The convention follows the expert guidance:
      - azimuth: rotation around the world up axis (+Z)
      - elevation: angle above the XY plane
      - roll: rotation around the camera forward axis
    """

    radius: float = Field(default=1.2, gt=0)
    azimuth_deg: float = Field(default=0.0)
    elevation_deg: float = Field(default=0.0)
    roll_deg: float = Field(default=0.0)
    target_world: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])


class ViewSpec(BaseModel):
    """Full specification for a single capture view."""

    label: ViewLabel
    image_filename: str = Field(
        ..., description="Filename (not path) within the job storage"
    )
    pose: SphericalPose


class CameraExtrinsics(BaseModel):
    """
    World-to-camera extrinsics in COLMAP convention.

    R_w2c: 3x3 rotation matrix (row-major, flattened to 9 floats)
    t_w2c: 3-element translation vector
    Camera centre in world coords: C = -R_w2c^T @ t_w2c

    COLMAP camera axes:
      +X = right in image
      +Y = down in image
      +Z = forward (into scene)
    """

    R_w2c: list[float] = Field(..., min_length=9, max_length=9)
    t_w2c: list[float] = Field(..., min_length=3, max_length=3)
    quaternion_wxyz: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="COLMAP quaternion (w, x, y, z) derived from R_w2c",
    )


class ResolvedView(BaseModel):
    """A view with fully resolved intrinsics and extrinsics."""

    label: ViewLabel
    image_filename: str
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    pose: SphericalPose


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    """Top-level configuration for a reconstruction run."""

    world: WorldConvention = Field(default_factory=WorldConvention)
    intrinsics: CameraIntrinsics = Field(default_factory=CameraIntrinsics)
    recon_backend: ReconBackend = Field(default=ReconBackend.DUST3R)
    use_background_for_pose: bool = Field(
        default=True,
        description="Solve geometry using full images (with background) first",
    )
    trellis_enabled: bool = Field(
        default=True,
        description="Run Trellis.2 generative completion after coarse recon",
    )
    image_size: int = Field(
        default=2048,
        description="Expected square image dimension (images are resized to this)",
    )
    mask_method: str = Field(
        default="rembg",
        description="Background removal method: rembg | sam | external",
    )


# ---------------------------------------------------------------------------
# Reconstruction artifacts
# ---------------------------------------------------------------------------


class PointCloud(BaseModel):
    """Sparse or dense point cloud produced by coarse reconstruction."""

    num_points: int = Field(default=0, ge=0)
    ply_path: Optional[str] = Field(
        default=None, description="Path to .ply file on disk"
    )
    confidence_mean: Optional[float] = Field(default=None)


class CoarseReconResult(BaseModel):
    """Output of the coarse reconstruction stage."""

    point_cloud: PointCloud = Field(default_factory=PointCloud)
    views: list[ResolvedView] = Field(default_factory=list)
    backend_used: ReconBackend = Field(default=ReconBackend.DUST3R)
    alignment_error: Optional[float] = Field(
        default=None,
        description="Global alignment reprojection error (pixels)",
    )


class IsolationResult(BaseModel):
    """Output of the subject-isolation stage."""

    masked_image_paths: list[str] = Field(default_factory=list)
    filtered_ply_path: Optional[str] = Field(default=None)
    num_points_retained: int = Field(default=0, ge=0)
    num_points_removed: int = Field(default=0, ge=0)


class TrellisResult(BaseModel):
    """Output of the Trellis.2 completion stage."""

    mesh_path: Optional[str] = Field(default=None)
    texture_path: Optional[str] = Field(default=None)
    voxel_path: Optional[str] = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------


class ReconJob(BaseModel):
    """Top-level reconstruction job tracking model."""

    job_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    status: JobStatus = Field(default=JobStatus.PENDING)
    config: PipelineConfig = Field(default_factory=PipelineConfig)
    views: list[ViewSpec] = Field(default_factory=list)

    # Stage outputs (populated as pipeline progresses)
    resolved_views: list[ResolvedView] = Field(default_factory=list)
    coarse_result: Optional[CoarseReconResult] = Field(default=None)
    isolation_result: Optional[IsolationResult] = Field(default=None)
    trellis_result: Optional[TrellisResult] = Field(default=None)

    # Timing / metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    error_message: Optional[str] = Field(default=None)
    storage_dir: Optional[str] = Field(default=None)


# ---------------------------------------------------------------------------
# API request / response helpers
# ---------------------------------------------------------------------------


class CreateJobRequest(BaseModel):
    """POST body for creating a new reconstruction job."""

    config: Optional[PipelineConfig] = Field(default=None)


class JobStatusResponse(BaseModel):
    """Lightweight status response returned by the API."""

    job_id: str
    status: JobStatus
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class JobDetailResponse(BaseModel):
    """Full detail response including stage results."""

    job_id: str
    status: JobStatus
    config: PipelineConfig
    views: list[ViewSpec]
    resolved_views: list[ResolvedView]
    coarse_result: Optional[CoarseReconResult] = None
    isolation_result: Optional[IsolationResult] = None
    trellis_result: Optional[TrellisResult] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

