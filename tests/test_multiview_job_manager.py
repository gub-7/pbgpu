"""
Tests for multi-view job manager functionality.

Covers Phase 0 additions: multi-view job creation, stage management,
pipeline-based queuing, per-view metadata, and derived field computation.

Uses fakeredis to avoid requiring a real Redis server.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

import fakeredis

from api.models import (
    CategoryEnum,
    ModelEnum,
    PipelineEnum,
    JobStatus,
    JobStage,
    StageStatus,
    ViewStatus,
    MV_STAGE_ORDER,
)
from api.job_manager import (
    JobManager,
    _is_multiview_pipeline,
    _build_stage_dict,
    _build_views_dict,
    _current_stage,
    _overall_progress,
    _available_artifacts,
)


@pytest.fixture
def tmp_storage(tmp_path):
    """Provide a temp directory for job metadata storage."""
    return str(tmp_path / "storage")


@pytest.fixture
def jm(tmp_storage):
    """
    Create a JobManager with a fakeredis backend.
    """
    manager = JobManager(storage_root=tmp_storage)
    # Replace the real redis client with fakeredis
    manager.redis_client = fakeredis.FakeRedis(decode_responses=True)
    return manager


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    def test_is_multiview_pipeline(self):
        assert _is_multiview_pipeline("canonical_mv_hybrid") is True
        assert _is_multiview_pipeline("canonical_mv_fast") is True
        assert _is_multiview_pipeline("canonical_mv_generative") is True
        assert _is_multiview_pipeline("singleview_triposr") is False
        assert _is_multiview_pipeline(None) is False
        assert _is_multiview_pipeline("") is False

    def test_build_stage_dict(self):
        stages = _build_stage_dict(MV_STAGE_ORDER)
        assert len(stages) == 10
        for s in MV_STAGE_ORDER:
            assert stages[s] == "pending"

    def test_build_views_dict(self):
        views = _build_views_dict(["front", "back", "top"])
        assert len(views) == 3
        assert views["front"]["status"] == "pending"
        assert views["front"]["filename"] == "front.png"
        assert views["front"]["width"] is None
        assert views["front"]["warnings"] == []

    def test_current_stage_in_progress(self):
        stages = {"a": "completed", "b": "in_progress", "c": "pending"}
        assert _current_stage(stages) == "b"

    def test_current_stage_next_pending(self):
        stages = {"a": "completed", "b": "completed", "c": "pending"}
        assert _current_stage(stages) == "c"

    def test_current_stage_all_done(self):
        stages = {"a": "completed", "b": "completed", "c": "skipped"}
        assert _current_stage(stages) is None

    def test_overall_progress(self):
        stages = {"a": "completed", "b": "completed", "c": "pending", "d": "pending"}
        assert _overall_progress(stages) == 50

    def test_overall_progress_with_skipped(self):
        stages = {"a": "completed", "b": "skipped", "c": "pending"}
        assert _overall_progress(stages) == 67  # 2/3

    def test_overall_progress_empty(self):
        assert _overall_progress({}) == 0

    def test_available_artifacts(self):
        stages = {"a": "completed", "b": "in_progress", "c": "completed"}
        assert _available_artifacts(stages) == ["a", "c"]


# ---------------------------------------------------------------------------
# Single-view job tests (backward compatibility)
# ---------------------------------------------------------------------------

class TestSingleViewJob:
    def test_create_job(self, jm):
        job_id = jm.create_job(
            CategoryEnum.HUMAN_FULL_BODY,
            ModelEnum.TRIPOSR,
            {"mc_resolution": 256},
        )
        assert job_id is not None

        job = jm.get_job(job_id)
        assert job is not None
        assert job["category"] == "human_full_body"
        assert job["model"] == "triposr"
        assert job["pipeline"] is None
        assert job["status"] == "queued"
        assert job["views"] is None
        assert job["stages"]["preprocessing"] == "pending"
        assert job["stages"]["generation"] == "pending"
        assert job["warnings"] == []
        assert job["artifacts_available"] == []

    def test_update_job_status(self, jm):
        job_id = jm.create_job(CategoryEnum.GENERIC_OBJECT, ModelEnum.TRIPOSR)

        jm.update_job(job_id, status=JobStatus.PREPROCESSING, progress=10)
        job = jm.get_job(job_id)
        assert job["status"] == "preprocessing"
        assert job["started_at"] is not None
        assert job["stages"]["preprocessing"] == "in_progress"

    def test_queue_singleview_job(self, jm):
        job_id = jm.create_job(CategoryEnum.GENERIC_OBJECT, ModelEnum.TRIPOSR)
        jm.queue_job_for_generation(job_id)

        next_id = jm.get_next_job(ModelEnum.TRIPOSR)
        assert next_id == job_id

    def test_queue_empty(self, jm):
        assert jm.get_next_job(ModelEnum.TRIPOSR) is None


# ---------------------------------------------------------------------------
# Multi-view job tests
# ---------------------------------------------------------------------------

class TestMultiViewJob:
    def test_create_multiview_job(self, jm):
        job_id = jm.create_multiview_job(
            category=CategoryEnum.HUMAN_BUST,
            pipeline=PipelineEnum.CANONICAL_MV_HYBRID,
            views_received=["front", "back", "left", "right", "top"],
            params={"seed": 42, "mesh_resolution": 384},
        )

        job = jm.get_job(job_id)
        assert job is not None
        assert job["pipeline"] == "canonical_mv_hybrid"
        assert job["model"] is None
        assert job["status"] == "queued"
        assert job["current_stage"] == "ingest"
        assert len(job["stages"]) == 10

        # All stages should be pending
        for stage_name in MV_STAGE_ORDER:
            assert job["stages"][stage_name] == "pending"

        # Views should be present and marked as uploaded
        assert len(job["views"]) == 5
        for vn in ("front", "back", "left", "right", "top"):
            assert job["views"][vn]["status"] == "uploaded"

        # Params should be stored
        assert job["params"]["seed"] == 42

    def test_create_partial_views(self, jm):
        """Creating a job with fewer views should still work."""
        job_id = jm.create_multiview_job(
            category=CategoryEnum.GENERIC_OBJECT,
            pipeline=PipelineEnum.CANONICAL_MV_FAST,
            views_received=["front", "back"],
        )
        job = jm.get_job(job_id)
        assert len(job["views"]) == 2
        assert "front" in job["views"]
        assert "back" in job["views"]

    def test_queue_multiview_job(self, jm):
        job_id = jm.create_multiview_job(
            category=CategoryEnum.HUMAN_BUST,
            pipeline=PipelineEnum.CANONICAL_MV_HYBRID,
            views_received=["front", "back", "left", "right", "top"],
        )
        jm.queue_job_for_generation(job_id)

        # Should be on the pipeline queue, not a model queue
        next_id = jm.get_next_multiview_job(PipelineEnum.CANONICAL_MV_HYBRID)
        assert next_id == job_id

        # Model queues should be empty
        assert jm.get_next_job(ModelEnum.TRIPOSR) is None

    def test_multiview_queue_length(self, jm):
        for _ in range(3):
            jid = jm.create_multiview_job(
                category=CategoryEnum.GENERIC_OBJECT,
                pipeline=PipelineEnum.CANONICAL_MV_GENERATIVE,
                views_received=["front"],
            )
            jm.queue_job_for_generation(jid)

        assert jm.get_multiview_queue_length(PipelineEnum.CANONICAL_MV_GENERATIVE) == 3


# ---------------------------------------------------------------------------
# Stage management tests
# ---------------------------------------------------------------------------

class TestStageManagement:
    def _create_mv_job(self, jm):
        return jm.create_multiview_job(
            category=CategoryEnum.HUMAN_BUST,
            pipeline=PipelineEnum.CANONICAL_MV_HYBRID,
            views_received=["front", "back", "left", "right", "top"],
        )

    def test_advance_stage(self, jm):
        job_id = self._create_mv_job(jm)

        jm.advance_stage(job_id, "ingest")

        job = jm.get_job(job_id)
        assert job["stages"]["ingest"] == "completed"
        assert job["stages"]["preprocess_views"] == "in_progress"
        assert job["current_stage"] == "preprocess_views"
        assert job["progress"] == 10  # 1/10 stages done

    def test_advance_stage_explicit_next(self, jm):
        job_id = self._create_mv_job(jm)

        jm.advance_stage(job_id, "ingest", next_stage="validate_views")

        job = jm.get_job(job_id)
        assert job["stages"]["ingest"] == "completed"
        assert job["stages"]["validate_views"] == "in_progress"
        assert job["current_stage"] == "validate_views"

    def test_fail_stage(self, jm):
        job_id = self._create_mv_job(jm)

        jm.fail_stage(job_id, "preprocess_views", "Segmentation crashed")

        job = jm.get_job(job_id)
        assert job["status"] == "failed"
        assert job["stages"]["preprocess_views"] == "failed"
        assert job["error"] == "Segmentation crashed"
        assert job["current_stage"] == "preprocess_views"

    def test_skip_stage(self, jm):
        job_id = self._create_mv_job(jm)

        jm.skip_stage(job_id, "refine_joint")

        job = jm.get_job(job_id)
        assert job["stages"]["refine_joint"] == "skipped"
        # Progress should include skipped as done
        # (1 skipped out of 10 = 10%)
        assert job["progress"] == 10

    def test_full_stage_progression(self, jm):
        """Walk through all stages to completion."""
        job_id = self._create_mv_job(jm)

        for i, stage in enumerate(MV_STAGE_ORDER):
            jm.advance_stage(job_id, stage)

        job = jm.get_job(job_id)
        assert job["progress"] == 100
        for stage in MV_STAGE_ORDER:
            assert job["stages"][stage] == "completed"


# ---------------------------------------------------------------------------
# View metadata update tests
# ---------------------------------------------------------------------------

class TestViewMetadataUpdates:
    def test_update_view_metadata(self, jm):
        job_id = jm.create_multiview_job(
            category=CategoryEnum.HUMAN_BUST,
            pipeline=PipelineEnum.CANONICAL_MV_HYBRID,
            views_received=["front", "back", "left", "right", "top"],
        )

        jm.update_job(
            job_id,
            view_updates={
                "front": {
                    "status": "segmented",
                    "width": 1024,
                    "height": 768,
                    "segmentation_confidence": 0.95,
                }
            },
        )

        job = jm.get_job(job_id)
        assert job["views"]["front"]["status"] == "segmented"
        assert job["views"]["front"]["width"] == 1024
        assert job["views"]["front"]["segmentation_confidence"] == 0.95

    def test_update_nonexistent_view_ignored(self, jm):
        job_id = jm.create_multiview_job(
            category=CategoryEnum.GENERIC_OBJECT,
            pipeline=PipelineEnum.CANONICAL_MV_FAST,
            views_received=["front"],
        )

        # Updating a view that doesn't exist should not crash
        jm.update_job(
            job_id,
            view_updates={"nonexistent": {"status": "failed"}},
        )

        job = jm.get_job(job_id)
        assert "nonexistent" not in job["views"]


# ---------------------------------------------------------------------------
# Derived field computation tests
# ---------------------------------------------------------------------------

class TestDerivedFields:
    def test_artifacts_available_computed(self, jm):
        job_id = jm.create_multiview_job(
            category=CategoryEnum.HUMAN_BUST,
            pipeline=PipelineEnum.CANONICAL_MV_HYBRID,
            views_received=["front", "back", "left", "right", "top"],
        )

        jm.advance_stage(job_id, "ingest")
        jm.advance_stage(job_id, "preprocess_views")

        job = jm.get_job(job_id)
        assert "ingest" in job["artifacts_available"]
        assert "preprocess_views" in job["artifacts_available"]

    def test_progress_auto_computed(self, jm):
        job_id = jm.create_multiview_job(
            category=CategoryEnum.GENERIC_OBJECT,
            pipeline=PipelineEnum.CANONICAL_MV_HYBRID,
            views_received=["front", "back", "left", "right", "top"],
        )

        # Complete 5 of 10 stages
        for stage in MV_STAGE_ORDER[:5]:
            jm.advance_stage(job_id, stage)

        job = jm.get_job(job_id)
        assert job["progress"] == 50

    def test_warnings_update(self, jm):
        job_id = jm.create_multiview_job(
            category=CategoryEnum.HUMAN_BUST,
            pipeline=PipelineEnum.CANONICAL_MV_HYBRID,
            views_received=["front", "back", "left", "right", "top"],
        )

        jm.update_job(job_id, warnings=["low_res_top", "blurry_back"])

        job = jm.get_job(job_id)
        assert job["warnings"] == ["low_res_top", "blurry_back"]


# ---------------------------------------------------------------------------
# Job deletion tests
# ---------------------------------------------------------------------------

class TestJobDeletion:
    def test_delete_multiview_job(self, jm):
        job_id = jm.create_multiview_job(
            category=CategoryEnum.GENERIC_OBJECT,
            pipeline=PipelineEnum.CANONICAL_MV_HYBRID,
            views_received=["front"],
        )

        assert jm.get_job(job_id) is not None

        jm.delete_job(job_id)

        assert jm.get_job(job_id) is None


# ---------------------------------------------------------------------------
# Active jobs query tests
# ---------------------------------------------------------------------------

class TestActiveJobsQuery:
    def test_get_active_multiview_jobs(self, jm):
        jid1 = jm.create_multiview_job(
            category=CategoryEnum.HUMAN_BUST,
            pipeline=PipelineEnum.CANONICAL_MV_HYBRID,
            views_received=["front"],
        )
        jid2 = jm.create_multiview_job(
            category=CategoryEnum.GENERIC_OBJECT,
            pipeline=PipelineEnum.CANONICAL_MV_FAST,
            views_received=["front"],
        )
        # Create a single-view job (should not appear)
        jid3 = jm.create_job(CategoryEnum.GENERIC_OBJECT, ModelEnum.TRIPOSR)

        active = jm.get_active_multiview_jobs()
        active_ids = {j["job_id"] for j in active}
        assert jid1 in active_ids
        assert jid2 in active_ids
        assert jid3 not in active_ids

    def test_get_active_multiview_jobs_by_pipeline(self, jm):
        jid1 = jm.create_multiview_job(
            category=CategoryEnum.HUMAN_BUST,
            pipeline=PipelineEnum.CANONICAL_MV_HYBRID,
            views_received=["front"],
        )
        jid2 = jm.create_multiview_job(
            category=CategoryEnum.GENERIC_OBJECT,
            pipeline=PipelineEnum.CANONICAL_MV_FAST,
            views_received=["front"],
        )

        hybrid_jobs = jm.get_active_multiview_jobs(PipelineEnum.CANONICAL_MV_HYBRID)
        assert len(hybrid_jobs) == 1
        assert hybrid_jobs[0]["job_id"] == jid1

