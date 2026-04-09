"""
Pydantic models for API request/response validation

Supports both single-view (TripoSR / Trellis2) and multi-view
canonical 3-view reconstruction pipelines.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CategoryEnum(str, Enum):
    """Supported 3D reconstruction categories"""
    HUMAN_FULL_BODY = "human_full_body"
    ANIMAL_FULL_BODY = "animal_full_body"
    HUMAN_BUST = "human_bust"
    ANIMAL_BUST = "animal_bust"
    GENERIC_OBJECT = "generic_object"


class ModelEnum(str, Enum):
    """Supported 3D reconstruction models (single-view lanes)"""
    TRIPOSR = "triposr"
    TRELLIS2 = "trellis2"


class PipelineEnum(str, Enum):
    """
    Available reconstruction pipelines.

    Multi-view pipelines consume 3 canonical views (front/side/top).
    Single-view pipelines consume a single image.
    """
    # Multi-view pipelines
    CANONICAL_MV_HYBRID = "canonical_mv_hybrid"
    CANONICAL_MV_FAST = "canonical_mv_fast"
    CANONICAL_MV_GENERATIVE = "canonical_mv_generative"
    # Single-view pipelines (backward-compatible)
    SINGLEVIEW_TRIPOSR = "singleview_triposr"
    SINGLEVIEW_TRELLIS2 = "singleview_trellis2"


class ViewName(str, Enum):
    """Canonical view identifiers for multi-view input"""
    FRONT = "front"
    SIDE = "side"
    TOP = "top"


# The 3 required canonical views
REQUIRED_VIEWS: List[str] = [v.value for v in ViewName]


class JobStatus(str, Enum):
    """Job lifecycle statuses"""
    QUEUED = "queued"
    PREPROCESSING = "preprocessing"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStage(str, Enum):
    """
    Fine-grained pipeline stages for multi-view jobs.

    Single-view jobs use a subset: ingest -> preprocess_views -> export -> qa.
    Multi-view jobs traverse all stages in order.
    """
    INGEST = "ingest"
    PREPROCESS_VIEWS = "preprocess_views"
    VALIDATE_VIEWS = "validate_views"
    INITIALIZE_CAMERAS = "initialize_cameras"
    RECONSTRUCT_COARSE = "reconstruct_coarse"
    REFINE_JOINT = "refine_joint"
    COMPLETE_GEOMETRY = "complete_geometry"
    BAKE_TEXTURE = "bake_texture"
    EXPORT = "export"
    QA = "qa"


class StageStatus(str, Enum):
    """Status of an individual pipeline stage"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


# Ordered list of stages for multi-view pipelines
MV_STAGE_ORDER: List[str] = [s.value for s in JobStage]

# Ordered list of stages for single-view pipelines (legacy)
SV_STAGE_ORDER: List[str] = [
    JobStage.INGEST.value,
    JobStage.PREPROCESS_VIEWS.value,
    JobStage.EXPORT.value,
    JobStage.QA.value,
]


# ---------------------------------------------------------------------------
# Parameter models
# ---------------------------------------------------------------------------

class TripoSRParams(BaseModel):
    """
    TripoSR generation parameters (knobs you can tweak for better output)

    Key parameters that affect output quality:
    - mc_resolution: Higher = more detail but slower (256-512 recommended for humans)
    - foreground_ratio: Legacy -- ignored when auto FFR is active (default).
      Auto FFR automatically computes optimal framing so bbox fills ~60% of frame area.
    - mc_threshold_sweep: Range of thresholds to try for mesh extraction
    - model_input_size: Input image size (512 or 768)

    Preprocessing notes:
    - Output is always RGB with flat white background (TripoSR is trained on RGB)
    - 1-2 px composite feather on alpha edges prevents floating artifact planes
    - Auto FFR replaces static per-category foreground_ratio maps
    """
    foreground_ratio: float = Field(
        0.85,
        ge=0.5,
        le=1.0,
        description=(
            "Legacy: how much of the frame the subject should fill (0.5-1.0). "
            "Ignored when auto FFR is active (default). To use this, set "
            "preprocess_overrides={'auto_ffr': false}."
        )
    )
    model_input_size: int = Field(
        512,
        ge=256,
        le=1024,
        description="Input image size for model (512 or 768 recommended). Higher = more detail but slower."
    )

    # Mesh extraction parameters (MOST IMPORTANT FOR QUALITY)
    mc_resolution: int = Field(
        256,
        ge=128,
        le=512,
        description="Marching cubes resolution. Higher = more mesh detail. 256=fast, 384=balanced, 512=detailed (for humans)"
    )
    mc_threshold: float = Field(
        0.0,
        ge=-0.1,
        le=0.5,
        description="Primary marching cubes threshold. Usually 0.0 works best."
    )
    mc_threshold_sweep: Optional[List[float]] = Field(
        None,
        description="List of thresholds to try (e.g., [-0.02, 0.0, 0.02]). If None, uses optimized default for humans."
    )

    # Texture parameters
    bake_texture: bool = Field(
        True,
        description="Bake texture to UV map (recommended for best results)"
    )
    texture_resolution: int = Field(
        2048,
        ge=512,
        le=4096,
        description="Texture resolution (1024, 2048, or 4096). Higher = better texture quality."
    )

    # Advanced parameters (usually don't need to change)
    remove_background: bool = Field(
        False,
        description="Remove background (already done in preprocessing, leave as False)"
    )
    output_format: str = Field(
        "glb",
        description="Output format (glb or obj)"
    )
    chunk_size: int = Field(
        8192,
        description="Processing chunk size (advanced)"
    )

    # Preprocessing control parameters
    use_triposr_optimization: bool = Field(
        True,
        description="Enable region-aware preprocessing tuned for TripoSR."
    )
    preprocess_overrides: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Overrides for preprocessing pipeline. Supported keys: "
            "auto_ffr (bool, default True) -- use automatic FFR framing; "
            "ffr_target (float, ~0.60) -- target Frame Fill Ratio; "
            "ffr_min (float, 0.45) -- minimum FFR before zoom-in; "
            "ffr_max (float, 0.75) -- maximum FFR before zoom-out; "
            "output_size (int, 512) -- square output size in pixels; "
            "composite_feather_radius (int, 1) -- alpha edge feather before compositing; "
            "foreground_ratio (float) -- legacy ratio, only when auto_ffr=false; "
            "enable_denoising, enable_clahe, enable_boundary_sharpening, "
            "enable_highlight_compression (all bool)."
        )
    )


class Trellis2Params(BaseModel):
    """Trellis.2 generation parameters"""
    preprocess_image: bool = Field(False, description="Preprocess image (already done)")
    resolution: int = Field(1024, ge=512, le=1536, description="Generation resolution")
    texture_size: int = Field(2048, ge=1024, le=4096, description="Texture resolution")
    ss_guidance_strength: float = Field(7.5, ge=1.0, le=15.0, description="Sparse structure guidance")
    ss_sampling_steps: int = Field(12, ge=8, le=30, description="Sparse structure sampling steps")
    shape_guidance_strength: float = Field(7.5, ge=1.0, le=15.0, description="Shape guidance strength")
    shape_sampling_steps: int = Field(12, ge=8, le=30, description="Shape sampling steps")
    tex_guidance_strength: float = Field(1.0, ge=0.5, le=2.0, description="Texture guidance strength")
    tex_sampling_steps: int = Field(12, ge=8, le=30, description="Texture sampling steps")
    decimation_target: int = Field(500000, ge=100000, le=2000000, description="Target triangle count")


class CanonicalMVParams(BaseModel):
    """
    Parameters for canonical multi-view reconstruction pipelines.

    Controls resolution, mesh quality, texture quality, and which
    optional stages (joint refinement, generative completion) are enabled.
    """
    # Resolution / quality
    output_resolution: int = Field(
        1024,
        ge=256,
        le=4096,
        description="Resolution for intermediate processing images."
    )
    shared_canvas_size: int = Field(
        1024,
        ge=128,
        le=4096,
        description="Canvas size for shared multi-view processing."
    )
    mesh_resolution: int = Field(
        256,
        ge=128,
        le=512,
        description="Target mesh extraction resolution (marching cubes grid)."
    )
    texture_resolution: int = Field(
        2048,
        ge=512,
        le=4096,
        description="Texture map resolution for final bake."
    )

    # Optional pipeline stages
    use_joint_refinement: bool = Field(
        True,
        description="Run joint mesh + Gaussian refinement stage."
    )
    use_trellis_completion: bool = Field(
        True,
        description="Use TRELLIS.2 for generative completion on low-confidence regions."
    )
    use_hunyuan_completion: bool = Field(
        False,
        description="Use Hunyuan3D as alternative/fallback completion prior."
    )

    # Priors
    symmetry_prior: bool = Field(
        True,
        description="Enable bilateral symmetry regularization."
    )
    category_prior: Optional[str] = Field(
        None,
        description="Category hint for shape priors (e.g. 'human_bust')."
    )

    # Debug / diagnostic outputs
    generate_debug_renders: bool = Field(
        False,
        description="Generate per-stage turntable debug renders."
    )
    generate_gaussian_debug: bool = Field(
        False,
        description="Export intermediate Gaussian .ply for debugging."
    )
    debug_incremental_recon: bool = Field(
        False,
        description="Run incremental visual hull passes (1, 2, 3 views) and save debug previews."
    )

    # Mesh post-processing
    decimation_target: int = Field(
        500_000,
        ge=10_000,
        le=2_000_000,
        description="Target triangle count for final mesh decimation."
    )

    # Reproducibility
    seed: Optional[int] = Field(
        None,
        description="Random seed for deterministic output. None = random."
    )


# ---------------------------------------------------------------------------
# View metadata
# ---------------------------------------------------------------------------

class ViewStatus(str, Enum):
    """Status of a single canonical view through the pipeline"""
    PENDING = "pending"
    UPLOADED = "uploaded"
    SEGMENTED = "segmented"
    VALIDATED = "validated"
    FAILED = "failed"


class ViewMetadata(BaseModel):
    """Metadata for a single canonical view image"""
    view_name: ViewName
    filename: str
    status: ViewStatus = ViewStatus.PENDING
    width: Optional[int] = None
    height: Optional[int] = None
    file_size: Optional[int] = None
    segmentation_confidence: Optional[float] = None
    sharpness_score: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class UploadRequest(BaseModel):
    """Upload and job creation request (single-view, legacy)"""
    category: CategoryEnum
    model: ModelEnum
    triposr_params: Optional[TripoSRParams] = None
    trellis2_params: Optional[Trellis2Params] = None


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    """Upload response with job ID (single-view)"""
    job_id: str
    status: JobStatus
    created_at: datetime


class MultiViewUploadResponse(BaseModel):
    """Upload response for multi-view jobs"""
    job_id: str
    status: JobStatus
    pipeline: PipelineEnum
    views_received: List[str]
    created_at: datetime


class PreviewStage(BaseModel):
    """Preview image stage information"""
    stage: str
    filename: str
    url: str
    timestamp: datetime
    size: Dict[str, int]


class PreviewResponse(BaseModel):
    """Preview images response"""
    job_id: str
    previews: List[PreviewStage]


class MultiViewPreviewResponse(BaseModel):
    """Preview images response for multi-view jobs, organized per view"""
    job_id: str
    views: Dict[str, List[PreviewStage]]


class ArtifactInfo(BaseModel):
    """Metadata for a single pipeline artifact"""
    name: str
    stage: str
    filename: str
    url: str
    file_size: Optional[int] = None
    content_type: Optional[str] = None
    created_at: Optional[datetime] = None


class ArtifactListResponse(BaseModel):
    """List of available artifacts for a job"""
    job_id: str
    artifacts: List[ArtifactInfo]


class QAWarning(BaseModel):
    """A single QA warning"""
    code: str
    message: str
    severity: str = Field("warning", description="warning or error")
    view: Optional[str] = None


class MetricsResponse(BaseModel):
    """QA metrics for a completed (or in-progress) job"""
    job_id: str
    quality_score: Optional[float] = None
    per_view_metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    mesh_metrics: Dict[str, Any] = Field(default_factory=dict)
    texture_metrics: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[QAWarning] = Field(default_factory=list)
    recommended_retry: List[str] = Field(default_factory=list)


class JobStatusResponse(BaseModel):
    """
    Job status response.

    For multi-view jobs, includes pipeline, current_stage, stage_progress,
    per-view status, warnings, and artifact availability.
    """
    job_id: str
    status: JobStatus
    model: Optional[ModelEnum] = None
    category: CategoryEnum
    pipeline: Optional[PipelineEnum] = None
    progress: int = Field(0, ge=0, le=100, description="Progress percentage")
    current_stage: Optional[str] = None
    stage_progress: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Progress within current stage (0.0-1.0)"
    )
    stages: Dict[str, str] = Field(default_factory=dict, description="Stage statuses")
    views: Optional[Dict[str, Dict[str, Any]]] = Field(
        None,
        description="Per-view metadata (multi-view jobs only)"
    )
    warnings: List[str] = Field(default_factory=list)
    artifacts_available: List[str] = Field(default_factory=list)
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    output_url: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    job_id: Optional[str] = None


class CategoryInfo(BaseModel):
    """Category information and guidance"""
    category: str
    display_name: str
    description: str
    best_model: str
    foreground_ratio: float
    texture_resolution: int
    expected_time_triposr: str
    expected_time_trellis2: str
    tips: List[str]


class CategoriesResponse(BaseModel):
    """Available categories with guidance"""
    categories: List[CategoryInfo]

