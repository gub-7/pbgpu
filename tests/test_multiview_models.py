"""
Tests for multi-view models, enums, and parameter validation.

Covers Phase 0 additions: PipelineEnum, ViewName, JobStage,
CanonicalMVParams, stage ordering, and response models.
"""

import pytest
from datetime import datetime

from api.models import (
    # Enums
    PipelineEnum,
    ViewName,
    JobStage,
    StageStatus,
    ViewStatus,
    JobStatus,
    CategoryEnum,
    ModelEnum,
    # Constants
    REQUIRED_VIEWS,
    MV_STAGE_ORDER,
    SV_STAGE_ORDER,
    # Params
    CanonicalMVParams,
    TripoSRParams,
    Trellis2Params,
    # Response models
    MultiViewUploadResponse,
    JobStatusResponse,
    ArtifactInfo,
    ArtifactListResponse,
    QAWarning,
    MetricsResponse,
    ViewMetadata,
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestPipelineEnum:
    def test_multiview_pipelines_exist(self):
        assert PipelineEnum.CANONICAL_MV_HYBRID.value == "canonical_mv_hybrid"
        assert PipelineEnum.CANONICAL_MV_FAST.value == "canonical_mv_fast"
        assert PipelineEnum.CANONICAL_MV_GENERATIVE.value == "canonical_mv_generative"

    def test_singleview_pipelines_exist(self):
        assert PipelineEnum.SINGLEVIEW_TRIPOSR.value == "singleview_triposr"
        assert PipelineEnum.SINGLEVIEW_TRELLIS2.value == "singleview_trellis2"

    def test_pipeline_from_string(self):
        p = PipelineEnum("canonical_mv_hybrid")
        assert p == PipelineEnum.CANONICAL_MV_HYBRID


class TestViewName:
    def test_all_canonical_views(self):
        assert ViewName.FRONT.value == "front"
        assert ViewName.BACK.value == "back"
        assert ViewName.LEFT.value == "left"
        assert ViewName.RIGHT.value == "right"
        assert ViewName.TOP.value == "top"

    def test_required_views_matches_enum(self):
        assert set(REQUIRED_VIEWS) == {"front", "back", "left", "right", "top"}
        assert len(REQUIRED_VIEWS) == 5


class TestJobStage:
    def test_all_stages_defined(self):
        expected = [
            "ingest", "preprocess_views", "validate_views",
            "initialize_cameras", "reconstruct_coarse", "refine_joint",
            "complete_geometry", "bake_texture", "export", "qa",
        ]
        for stage_name in expected:
            assert JobStage(stage_name) is not None

    def test_mv_stage_order_complete(self):
        assert len(MV_STAGE_ORDER) == 10
        assert MV_STAGE_ORDER[0] == "ingest"
        assert MV_STAGE_ORDER[-1] == "qa"

    def test_sv_stage_order_subset(self):
        assert len(SV_STAGE_ORDER) == 4
        for s in SV_STAGE_ORDER:
            assert s in MV_STAGE_ORDER


class TestStageStatus:
    def test_all_statuses(self):
        assert StageStatus.PENDING.value == "pending"
        assert StageStatus.IN_PROGRESS.value == "in_progress"
        assert StageStatus.COMPLETED.value == "completed"
        assert StageStatus.SKIPPED.value == "skipped"
        assert StageStatus.FAILED.value == "failed"


class TestViewStatus:
    def test_all_statuses(self):
        assert ViewStatus.PENDING.value == "pending"
        assert ViewStatus.UPLOADED.value == "uploaded"
        assert ViewStatus.SEGMENTED.value == "segmented"
        assert ViewStatus.VALIDATED.value == "validated"
        assert ViewStatus.FAILED.value == "failed"


# ---------------------------------------------------------------------------
# Parameter model tests
# ---------------------------------------------------------------------------

class TestCanonicalMVParams:
    def test_defaults(self):
        p = CanonicalMVParams()
        assert p.output_resolution == 1024
        assert p.mesh_resolution == 256
        assert p.texture_resolution == 2048
        assert p.use_joint_refinement is True
        assert p.use_trellis_completion is True
        assert p.use_hunyuan_completion is False
        assert p.symmetry_prior is True
        assert p.category_prior is None
        assert p.generate_debug_renders is False
        assert p.generate_gaussian_debug is False
        assert p.decimation_target == 500_000
        assert p.seed is None

    def test_custom_values(self):
        p = CanonicalMVParams(
            output_resolution=512,
            mesh_resolution=128,
            seed=42,
            use_joint_refinement=False,
        )
        assert p.output_resolution == 512
        assert p.mesh_resolution == 128
        assert p.seed == 42
        assert p.use_joint_refinement is False

    def test_model_dump_roundtrip(self):
        p = CanonicalMVParams(seed=42)
        d = p.model_dump()
        p2 = CanonicalMVParams(**d)
        assert p2.seed == 42
        assert p2.output_resolution == 1024

    def test_validation_bounds(self):
        with pytest.raises(Exception):
            CanonicalMVParams(output_resolution=0)  # below min
        with pytest.raises(Exception):
            CanonicalMVParams(mesh_resolution=9999)  # above max
        with pytest.raises(Exception):
            CanonicalMVParams(decimation_target=5)  # below min


# ---------------------------------------------------------------------------
# Response model tests
# ---------------------------------------------------------------------------

class TestMultiViewUploadResponse:
    def test_construction(self):
        resp = MultiViewUploadResponse(
            job_id="test-123",
            status=JobStatus.QUEUED,
            pipeline=PipelineEnum.CANONICAL_MV_HYBRID,
            views_received=["front", "back", "left", "right", "top"],
            created_at=datetime.utcnow(),
        )
        assert resp.job_id == "test-123"
        assert resp.pipeline == PipelineEnum.CANONICAL_MV_HYBRID
        assert len(resp.views_received) == 5


class TestJobStatusResponse:
    def test_singleview_compatible(self):
        """Ensure the new response model still works for single-view jobs."""
        resp = JobStatusResponse(
            job_id="sv-123",
            status=JobStatus.QUEUED,
            model=ModelEnum.TRIPOSR,
            category=CategoryEnum.HUMAN_FULL_BODY,
            progress=50,
            stages={"preprocessing": "completed", "generation": "in_progress"},
            created_at=datetime.utcnow(),
        )
        assert resp.pipeline is None
        assert resp.current_stage is None
        assert resp.views is None

    def test_multiview_fields(self):
        resp = JobStatusResponse(
            job_id="mv-123",
            status=JobStatus.GENERATING,
            category=CategoryEnum.HUMAN_BUST,
            pipeline=PipelineEnum.CANONICAL_MV_HYBRID,
            progress=30,
            current_stage="reconstruct_coarse",
            stage_progress=0.5,
            stages={s: "pending" for s in MV_STAGE_ORDER},
            views={"front": {"status": "uploaded"}},
            warnings=["low_resolution_top"],
            artifacts_available=["ingest", "preprocess_views"],
            created_at=datetime.utcnow(),
        )
        assert resp.pipeline == PipelineEnum.CANONICAL_MV_HYBRID
        assert resp.current_stage == "reconstruct_coarse"
        assert resp.stage_progress == 0.5
        assert "front" in resp.views


class TestMetricsResponse:
    def test_empty(self):
        m = MetricsResponse(job_id="test")
        assert m.quality_score is None
        assert m.per_view_metrics == {}
        assert m.warnings == []

    def test_populated(self):
        m = MetricsResponse(
            job_id="test",
            quality_score=0.84,
            per_view_metrics={"front": {"iou": 0.95}},
            mesh_metrics={"face_count": 100000},
            warnings=[QAWarning(code="low_iou", message="Low IoU on top view")],
            recommended_retry=["re-upload sharper top image"],
        )
        assert m.quality_score == 0.84
        assert len(m.warnings) == 1


class TestViewMetadata:
    def test_construction(self):
        vm = ViewMetadata(
            view_name=ViewName.FRONT,
            filename="front.png",
            status=ViewStatus.UPLOADED,
            width=1024,
            height=768,
        )
        assert vm.view_name == ViewName.FRONT
        assert vm.width == 1024

    def test_defaults(self):
        vm = ViewMetadata(view_name=ViewName.TOP, filename="top.png")
        assert vm.status == ViewStatus.PENDING
        assert vm.warnings == []
        assert vm.sharpness_score is None

