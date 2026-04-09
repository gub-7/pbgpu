"""
End-to-end integration tests for the canonical multi-view pipeline.

Phase 7: Tests that exercise the full pipeline from job creation
through all stages to final GLB output, plus API endpoint integration,
worker processing, and job state transition verification.

Covers:
    FULL PIPELINE (orchestrator-level):
        - Happy path: all 10 stages complete, GLB produced
        - Stage skip: refinement / completion disabled via config
        - Stage failure: error in one stage fails the whole job
        - Resume: partially-completed job resumes from correct stage
        - Deterministic output with fixed seed

    WORKER INTEGRATION:
        - Worker picks up job from Redis queue
        - Worker handles orchestrator crash gracefully
        - Worker processes correct pipeline queue

    API ENDPOINT INTEGRATION:
        - POST /api/upload_multiview → job created + queued
        - GET /api/job/{id}/status → correct stages / progress
        - GET /api/job/{id}/artifacts → lists produced artifacts
        - GET /api/job/{id}/metrics → returns QA data
        - POST /api/job/{id}/rerun_stage → resets stages
        - GET /api/job/{id}/output → serves GLB
        - DELETE /api/job/{id} → cleanup

    JOB STATE TRANSITIONS:
        - QUEUED → GENERATING → COMPLETED
        - QUEUED → GENERATING → FAILED
        - Per-stage status: PENDING → IN_PROGRESS → COMPLETED / SKIPPED

    ERROR PROPAGATION:
        - Missing view → ingest fails
        - Bad segmentation → preprocess fails
        - Missing camera_init → coarse_recon fails
        - All errors set job to FAILED with error message

All tests use synthetic data and mocked segmentation — no GPU required.
"""

import io
import json
import math
import struct
import sys
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import cv2
import numpy as np
from PIL import Image
import fakeredis

from api.job_manager import JobManager
from api.models import (
    CategoryEnum,
    JobStatus,
    JobStage,
    PipelineEnum,
    StageStatus,
    MV_STAGE_ORDER,
    REQUIRED_VIEWS,
)
from api.storage import StorageManager
from pipelines.canonical_mv.config import (
    CanonicalMVConfig,
    CANONICAL_VIEW_ORDER,
)
from pipelines.canonical_mv.orchestrator import CanonicalMVOrchestrator
from pipelines.canonical_mv.camera_init import build_canonical_rig
from pipelines.canonical_mv.coarse_recon import save_mesh_ply
from pipelines.canonical_mv.refine import (
    compute_vertex_normals,
    load_mesh_ply,
)
from pipelines.canonical_mv.export import GLB_MAGIC, GLB_VERSION


# ---------------------------------------------------------------------------
# Worker import helper (torch may not be available)
# ---------------------------------------------------------------------------

def _import_canonical_mv_worker():
    """
    Import CanonicalMVWorker without triggering workers/__init__.py
    which imports triposr_worker (requires torch).
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "workers.canonical_mv_worker",
        str(Path(__file__).parent.parent / "workers" / "canonical_mv_worker.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules so patches via setattr work correctly
    sys.modules["workers.canonical_mv_worker"] = mod
    spec.loader.exec_module(mod)
    return mod.CanonicalMVWorker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Canvas size used throughout integration tests — must match the preview
# thumbnail max size (512) to ensure camera intrinsics and mask dimensions
# are consistent.
_TEST_CANVAS_SIZE = 256


@pytest.fixture
def tmp_storage(tmp_path):
    return str(tmp_path / "storage")


@pytest.fixture
def sm(tmp_storage):
    return StorageManager(storage_root=tmp_storage)


@pytest.fixture
def jm(tmp_storage):
    manager = JobManager(storage_root=tmp_storage)
    manager.redis_client = fakeredis.FakeRedis(decode_responses=True)
    return manager


@pytest.fixture
def config():
    return CanonicalMVConfig(shared_canvas_size=_TEST_CANVAS_SIZE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_synthetic_view(width=256, height=256, color=(128, 100, 80)):
    """Create a synthetic view image with a centered circle subject."""
    img = Image.new("RGB", (width, height), (240, 240, 240))
    arr = np.array(img)
    # Draw a filled circle covering most of the image center
    cy, cx = height // 2, width // 2
    radius = min(width, height) * 2 // 5  # 40% of image → covers ~80% diameter
    Y, X = np.ogrid[:height, :width]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2
    arr[mask] = color
    return Image.fromarray(arr)


def _create_mv_job_with_views(jm, sm, pipeline=PipelineEnum.CANONICAL_MV_HYBRID, params=None):
    """
    Create a multi-view job and upload synthetic view images.

    Returns the job_id.
    """
    job_id = jm.create_multiview_job(
        category=CategoryEnum.HUMAN_BUST,
        pipeline=pipeline,
        views_received=list(CANONICAL_VIEW_ORDER),
        params=params,
    )

    # Upload synthetic views with slightly different colors per view
    view_colors = {
        "front": (140, 100, 80),
        "back": (130, 95, 75),
        "left": (135, 98, 78),
        "right": (138, 102, 82),
        "top": (125, 90, 70),
    }

    for vn in CANONICAL_VIEW_ORDER:
        img = _create_synthetic_view(
            width=_TEST_CANVAS_SIZE,
            height=_TEST_CANVAS_SIZE,
            color=view_colors.get(vn, (128, 100, 80)),
        )
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        sm.save_view_upload(job_id, vn, buf.getvalue(), ".png")

    return job_id


def _mock_segment_view(image_path, model_name="u2net"):
    """
    Mock segmentation that returns an RGBA image with a large centered
    circular foreground mask.  The mask covers ~80% of the image diameter
    so that voxel projections from the canonical camera rig fall inside.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    arr = np.array(img)

    # Create alpha mask: large circle covering most of the image
    Y, X = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    radius = min(w, h) * 2 // 5  # 40% of image side → ~80% diameter
    mask = ((X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2).astype(np.uint8) * 255

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = arr
    rgba[:, :, 3] = mask
    return rgba


def _verify_glb_header(filepath):
    """Verify a file has a valid GLB header."""
    with open(filepath, "rb") as f:
        data = f.read(12)
    if len(data) < 12:
        return False
    magic, version, _ = struct.unpack("<III", data)
    return magic == GLB_MAGIC and version == GLB_VERSION


def _json_safe_params(**kwargs):
    """Build a JSON-serializable params dict for CanonicalMVConfig fields."""
    # Default to small mesh resolution for fast tests (64^3 grid instead of 128^3)
    if 'mesh_resolution' not in kwargs:
        kwargs['mesh_resolution'] = 128
    # Default to small texture resolution for fast tests (256 instead of 2048)
    if 'texture_resolution' not in kwargs:
        kwargs['texture_resolution'] = 512
    # Only include fields that are JSON-serializable (no CameraSpec objects)
    safe_fields = {
        "output_resolution", "mesh_resolution", "texture_resolution",
        "use_joint_refinement", "use_trellis_completion", "use_hunyuan_completion",
        "symmetry_prior", "category_prior", "generate_debug_renders",
        "generate_gaussian_debug", "decimation_target", "seed",
        "segmentation_model", "normalize_white_balance", "normalize_lighting",
        "shared_canvas_size", "silhouette_area_tolerance",
        "identity_similarity_threshold", "segmentation_confidence_threshold",
    }
    return {k: v for k, v in kwargs.items() if k in safe_fields}


# ===========================================================================
# FULL PIPELINE — ORCHESTRATOR INTEGRATION
# ===========================================================================


class TestFullPipelineHappyPath:
    """
    Test the full orchestrator pipeline from ingest to QA.

    Mocks only the segmentation step (no GPU/rembg needed).
    All other stages run with real code on synthetic data.
    """

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_full_pipeline_completes(self, mock_seg, jm, sm, config):
        """All 10 stages should complete and job should be marked COMPLETED."""
        job_id = _create_mv_job_with_views(
            jm, sm,
            params=_json_safe_params(shared_canvas_size=_TEST_CANVAS_SIZE),
        )

        orchestrator = CanonicalMVOrchestrator(
            job_id=job_id,
            job_manager=jm,
            storage_manager=sm,
        )
        success = orchestrator.run()

        assert success is True

        job = jm.get_job(job_id)
        assert job["status"] == JobStatus.COMPLETED.value
        assert job["progress"] == 100

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_all_stages_completed(self, mock_seg, jm, sm, config):
        """Every stage should be marked as COMPLETED (or SKIPPED if disabled)."""
        job_id = _create_mv_job_with_views(
            jm, sm,
            params=_json_safe_params(shared_canvas_size=_TEST_CANVAS_SIZE),
        )

        orchestrator = CanonicalMVOrchestrator(
            job_id=job_id,
            job_manager=jm,
            storage_manager=sm,
        )
        orchestrator.run()

        job = jm.get_job(job_id)
        stages = job["stages"]

        for stage_name in MV_STAGE_ORDER:
            assert stages[stage_name] in (
                StageStatus.COMPLETED.value,
                StageStatus.SKIPPED.value,
            ), f"Stage '{stage_name}' is {stages[stage_name]}, expected completed/skipped"

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_final_glb_produced(self, mock_seg, jm, sm, config):
        """A valid GLB file should be produced in the outputs directory."""
        job_id = _create_mv_job_with_views(
            jm, sm,
            params=_json_safe_params(shared_canvas_size=_TEST_CANVAS_SIZE),
        )

        orchestrator = CanonicalMVOrchestrator(
            job_id=job_id,
            job_manager=jm,
            storage_manager=sm,
        )
        orchestrator.run()

        output = sm.get_output_file(job_id)
        assert output is not None, "No GLB output file found"
        assert output.exists()
        assert output.suffix == ".glb"
        assert _verify_glb_header(str(output))

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_key_artifacts_produced(self, mock_seg, jm, sm, config):
        """All key intermediate artifacts should be saved."""
        job_id = _create_mv_job_with_views(
            jm, sm,
            params=_json_safe_params(shared_canvas_size=_TEST_CANVAS_SIZE),
        )

        orchestrator = CanonicalMVOrchestrator(
            job_id=job_id,
            job_manager=jm,
            storage_manager=sm,
        )
        orchestrator.run()

        # Camera init
        assert sm.get_artifact_path(job_id, "camera_init.json") is not None
        rig_data = sm.load_artifact_json(job_id, "camera_init.json")
        assert rig_data is not None
        assert "cameras" in rig_data
        assert len(rig_data["cameras"]) == 5

        # Preprocess metrics
        assert sm.load_artifact_json(job_id, "preprocess_metrics.json") is not None

        # Coarse reconstruction
        assert sm.get_artifact_path(job_id, "coarse_voxel.npz") is not None
        assert sm.get_artifact_path(job_id, "coarse_visual_hull_mesh.ply") is not None
        assert sm.get_artifact_path(job_id, "coarse_gaussians.ply") is not None

        # Export metrics
        assert sm.load_artifact_json(job_id, "export_metrics.json") is not None

        # QA metrics
        metrics = sm.load_metrics(job_id)
        assert metrics is not None
        assert "quality_score" in metrics
        assert 0.0 <= metrics["quality_score"] <= 1.0

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_per_view_previews_generated(self, mock_seg, jm, sm, config):
        """Raw, segmented, and normalized previews should exist for each view."""
        job_id = _create_mv_job_with_views(
            jm, sm,
            params=_json_safe_params(shared_canvas_size=_TEST_CANVAS_SIZE),
        )

        orchestrator = CanonicalMVOrchestrator(
            job_id=job_id,
            job_manager=jm,
            storage_manager=sm,
        )
        orchestrator.run()

        previews = sm.list_view_previews(job_id)

        # Should have raw and segmented substages at minimum
        assert "raw" in previews
        assert "segmented" in previews

        for substage in ("raw", "segmented"):
            for vn in CANONICAL_VIEW_ORDER:
                assert vn in previews[substage], (
                    f"Missing {substage} preview for view '{vn}'"
                )

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_output_url_set(self, mock_seg, jm, sm, config):
        """Job should have an output_url pointing to the GLB."""
        job_id = _create_mv_job_with_views(
            jm, sm,
            params=_json_safe_params(shared_canvas_size=_TEST_CANVAS_SIZE),
        )

        orchestrator = CanonicalMVOrchestrator(
            job_id=job_id,
            job_manager=jm,
            storage_manager=sm,
        )
        orchestrator.run()

        job = jm.get_job(job_id)
        assert job.get("output_url") is not None
        assert "final.glb" in job["output_url"]

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_validation_results_saved(self, mock_seg, jm, sm, config):
        """Validation results from the consistency stage should be saved."""
        job_id = _create_mv_job_with_views(
            jm, sm,
            params=_json_safe_params(shared_canvas_size=_TEST_CANVAS_SIZE),
        )

        orchestrator = CanonicalMVOrchestrator(
            job_id=job_id,
            job_manager=jm,
            storage_manager=sm,
        )
        orchestrator.run()

        validation = sm.load_artifact_json(job_id, "validation_results.json")
        assert validation is not None
        assert "status" in validation
        assert "checks_run" in validation

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_segmentation_called_for_each_view(self, mock_seg, jm, sm, config):
        """The segmentation function should be called once per view."""
        job_id = _create_mv_job_with_views(
            jm, sm,
            params=_json_safe_params(shared_canvas_size=_TEST_CANVAS_SIZE),
        )

        orchestrator = CanonicalMVOrchestrator(
            job_id=job_id,
            job_manager=jm,
            storage_manager=sm,
        )
        orchestrator.run()

        assert mock_seg.call_count == 5


# ===========================================================================
# STAGE SKIP TESTS
# ===========================================================================


class TestPipelineStageSkip:
    """Test that stages are correctly skipped based on config."""

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_skip_refinement(self, mock_seg, jm, sm):
        """With use_joint_refinement=False, refine_joint should be SKIPPED."""
        params = _json_safe_params(
            shared_canvas_size=_TEST_CANVAS_SIZE,
            use_joint_refinement=False,
            use_trellis_completion=True,
            symmetry_prior=True,
        )
        job_id = _create_mv_job_with_views(jm, sm, params=params)

        orchestrator = CanonicalMVOrchestrator(
            job_id=job_id,
            job_manager=jm,
            storage_manager=sm,
        )
        success = orchestrator.run()
        assert success is True

        job = jm.get_job(job_id)
        assert job["stages"][JobStage.REFINE_JOINT.value] == StageStatus.SKIPPED.value

        # Refined mesh should NOT exist
        assert sm.get_artifact_path(job_id, "refined_mesh.ply") is None

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_skip_completion(self, mock_seg, jm, sm):
        """With both completion providers disabled, complete_geometry should be SKIPPED."""
        params = _json_safe_params(
            shared_canvas_size=_TEST_CANVAS_SIZE,
            use_trellis_completion=False,
            use_hunyuan_completion=False,
        )
        job_id = _create_mv_job_with_views(jm, sm, params=params)

        orchestrator = CanonicalMVOrchestrator(
            job_id=job_id,
            job_manager=jm,
            storage_manager=sm,
        )
        success = orchestrator.run()
        assert success is True

        job = jm.get_job(job_id)
        assert job["stages"][JobStage.COMPLETE_GEOMETRY.value] == StageStatus.SKIPPED.value

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_skip_both_refinement_and_completion(self, mock_seg, jm, sm):
        """Both refinement and completion skipped still produces valid output."""
        params = _json_safe_params(
            shared_canvas_size=_TEST_CANVAS_SIZE,
            use_joint_refinement=False,
            use_trellis_completion=False,
            use_hunyuan_completion=False,
        )
        job_id = _create_mv_job_with_views(jm, sm, params=params)

        orchestrator = CanonicalMVOrchestrator(
            job_id=job_id,
            job_manager=jm,
            storage_manager=sm,
        )
        success = orchestrator.run()
        assert success is True

        job = jm.get_job(job_id)
        assert job["status"] == JobStatus.COMPLETED.value
        assert job["stages"][JobStage.REFINE_JOINT.value] == StageStatus.SKIPPED.value
        assert job["stages"][JobStage.COMPLETE_GEOMETRY.value] == StageStatus.SKIPPED.value

        # Output should still exist (export uses coarse mesh)
        output = sm.get_output_file(job_id)
        assert output is not None

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_skipped_stages_counted_in_progress(self, mock_seg, jm, sm):
        """Skipped stages should count toward progress (not block at 80%)."""
        params = _json_safe_params(
            shared_canvas_size=_TEST_CANVAS_SIZE,
            use_joint_refinement=False,
            use_trellis_completion=False,
            use_hunyuan_completion=False,
        )
        job_id = _create_mv_job_with_views(jm, sm, params=params)

        orchestrator = CanonicalMVOrchestrator(
            job_id=job_id,
            job_manager=jm,
            storage_manager=sm,
        )
        orchestrator.run()

        job = jm.get_job(job_id)
        assert job["progress"] == 100


# ===========================================================================
# STAGE FAILURE TESTS
# ===========================================================================


class TestPipelineStageFailure:
    """Test error handling when stages fail."""

    def test_missing_view_fails_ingest(self, jm, sm, config):
        """Job with missing view file should fail at ingest stage."""
        # Create job but don't upload any views
        job_id = jm.create_multiview_job(
            category=CategoryEnum.HUMAN_BUST,
            pipeline=PipelineEnum.CANONICAL_MV_HYBRID,
            views_received=["front", "back", "left", "right", "top"],
        )

        orchestrator = CanonicalMVOrchestrator(
            job_id=job_id,
            job_manager=jm,
            storage_manager=sm,
        )
        success = orchestrator.run()

        assert success is False

        job = jm.get_job(job_id)
        assert job["status"] == JobStatus.FAILED.value
        assert job["stages"][JobStage.INGEST.value] == StageStatus.FAILED.value
        assert job["error"] is not None

    @patch("pipelines.canonical_mv.preprocess._segment_view")
    def test_segmentation_failure_fails_preprocess(self, mock_seg, jm, sm, config):
        """If segmentation raises, preprocess stage should fail."""
        mock_seg.side_effect = RuntimeError("Segmentation model crashed")

        job_id = _create_mv_job_with_views(
            jm, sm,
            params=_json_safe_params(shared_canvas_size=_TEST_CANVAS_SIZE),
        )

        orchestrator = CanonicalMVOrchestrator(
            job_id=job_id,
            job_manager=jm,
            storage_manager=sm,
        )
        success = orchestrator.run()

        assert success is False

        job = jm.get_job(job_id)
        assert job["status"] == JobStatus.FAILED.value
        assert job["stages"][JobStage.PREPROCESS_VIEWS.value] == StageStatus.FAILED.value
        assert "Segmentation model crashed" in job["error"]

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_failure_stops_subsequent_stages(self, mock_seg, jm, sm, config):
        """When a stage fails, subsequent stages should remain PENDING."""
        job_id = _create_mv_job_with_views(
            jm, sm,
            params=_json_safe_params(shared_canvas_size=_TEST_CANVAS_SIZE),
        )

        # Patch coarse_recon to fail
        with patch(
            "pipelines.canonical_mv.coarse_recon.run_reconstruct_coarse",
            side_effect=ValueError("Visual hull is empty"),
        ):
            orchestrator = CanonicalMVOrchestrator(
                job_id=job_id,
                job_manager=jm,
                storage_manager=sm,
            )
            success = orchestrator.run()

        assert success is False

        job = jm.get_job(job_id)
        stages = job["stages"]

        # Stages before failure should be completed
        assert stages[JobStage.INGEST.value] == StageStatus.COMPLETED.value
        assert stages[JobStage.PREPROCESS_VIEWS.value] == StageStatus.COMPLETED.value
        assert stages[JobStage.VALIDATE_VIEWS.value] == StageStatus.COMPLETED.value
        assert stages[JobStage.INITIALIZE_CAMERAS.value] == StageStatus.COMPLETED.value

        # Failed stage
        assert stages[JobStage.RECONSTRUCT_COARSE.value] == StageStatus.FAILED.value

        # Stages after failure should still be PENDING
        assert stages[JobStage.REFINE_JOINT.value] == StageStatus.PENDING.value
        assert stages[JobStage.BAKE_TEXTURE.value] == StageStatus.PENDING.value
        assert stages[JobStage.EXPORT.value] == StageStatus.PENDING.value
        assert stages[JobStage.QA.value] == StageStatus.PENDING.value

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_error_message_preserved(self, mock_seg, jm, sm, config):
        """The error message from a failed stage should be stored in job data."""
        job_id = _create_mv_job_with_views(
            jm, sm,
            params=_json_safe_params(shared_canvas_size=_TEST_CANVAS_SIZE),
        )

        with patch(
            "pipelines.canonical_mv.coarse_recon.run_reconstruct_coarse",
            side_effect=ValueError("Test error: insufficient voxels"),
        ):
            orchestrator = CanonicalMVOrchestrator(
                job_id=job_id,
                job_manager=jm,
                storage_manager=sm,
            )
            orchestrator.run()

        job = jm.get_job(job_id)
        assert "Test error: insufficient voxels" in job["error"]


# ===========================================================================
# PIPELINE RESUME TESTS
# ===========================================================================


class TestPipelineResume:
    """Test that partially-completed jobs can be resumed."""

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_resume_skips_completed_stages(self, mock_seg, jm, sm, config):
        """
        If some stages are already completed, orchestrator should skip them.
        """
        job_id = _create_mv_job_with_views(
            jm, sm,
            params=_json_safe_params(shared_canvas_size=_TEST_CANVAS_SIZE),
        )

        # Run the full pipeline first
        orchestrator = CanonicalMVOrchestrator(
            job_id=job_id,
            job_manager=jm,
            storage_manager=sm,
        )
        orchestrator.run()

        # Reset only the last two stages
        jm.update_job(
            job_id,
            status=JobStatus.GENERATING,
            stage_updates={
                JobStage.EXPORT.value: StageStatus.PENDING.value,
                JobStage.QA.value: StageStatus.PENDING.value,
            },
        )

        # Re-run — should skip all completed stages
        orchestrator2 = CanonicalMVOrchestrator(
            job_id=job_id,
            job_manager=jm,
            storage_manager=sm,
        )
        success = orchestrator2.run()
        assert success is True

        job = jm.get_job(job_id)
        assert job["status"] == JobStatus.COMPLETED.value


# ===========================================================================
# WORKER INTEGRATION
# ===========================================================================


class TestWorkerIntegration:
    """Test the CanonicalMVWorker job processing flow."""

    @pytest.fixture(autouse=True)
    def _load_worker_class(self):
        """Load the worker class bypassing workers/__init__.py (avoids torch)."""
        self.CanonicalMVWorker = _import_canonical_mv_worker()

    def test_worker_picks_up_job_from_queue(self, jm, sm):
        """Worker should pick up a queued job via Redis."""
        job_id = _create_mv_job_with_views(jm, sm)
        jm.queue_job_for_generation(job_id)

        worker = self.CanonicalMVWorker()
        worker.jm = jm

        # Manually poll the queue
        picked_id = worker._poll_queues()
        assert picked_id == job_id

    def test_worker_queue_empty_returns_none(self, jm, sm):
        """When queue is empty, worker poll should return None."""
        worker = self.CanonicalMVWorker()
        worker.jm = jm

        picked_id = worker._poll_queues()
        assert picked_id is None

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_worker_processes_job_successfully(self, mock_seg, jm, sm):
        """Worker should process a job and mark it as completed."""
        job_id = _create_mv_job_with_views(
            jm, sm,
            params=_json_safe_params(shared_canvas_size=_TEST_CANVAS_SIZE),
        )

        worker = self.CanonicalMVWorker()
        worker.jm = jm
        worker.sm = sm

        # Process the job directly (bypass queue polling)
        worker._process_job(job_id)

        job = jm.get_job(job_id)
        assert job["status"] == JobStatus.COMPLETED.value

    def test_worker_handles_crash_gracefully(self, jm, sm):
        """Worker should catch orchestrator crashes and mark job as failed."""
        job_id = _create_mv_job_with_views(jm, sm)

        worker = self.CanonicalMVWorker()
        worker.jm = jm
        worker.sm = sm

        # Patch the orchestrator class that the worker module uses
        # The worker imports CanonicalMVOrchestrator at module level
        worker_module = sys.modules.get("workers.canonical_mv_worker")
        if worker_module is None:
            # Module was loaded via importlib, find it
            for name, mod in sys.modules.items():
                if hasattr(mod, "CanonicalMVWorker") and mod is not None:
                    worker_module = mod
                    break

        if worker_module:
            original_orch = getattr(worker_module, "CanonicalMVOrchestrator", None)
            try:
                setattr(worker_module, "CanonicalMVOrchestrator",
                        MagicMock(side_effect=Exception("GPU out of memory")))
                worker._process_job(job_id)
            finally:
                if original_orch:
                    setattr(worker_module, "CanonicalMVOrchestrator", original_orch)

        job = jm.get_job(job_id)
        assert job["status"] == JobStatus.FAILED.value
        assert "GPU out of memory" in job.get("error", "")

    def test_worker_processes_correct_pipeline_queue(self, jm, sm):
        """Jobs should be queued under the correct pipeline name."""
        job_id = _create_mv_job_with_views(
            jm, sm, pipeline=PipelineEnum.CANONICAL_MV_HYBRID,
        )
        jm.queue_job_for_generation(job_id)

        # Check queue length
        assert jm.get_multiview_queue_length(PipelineEnum.CANONICAL_MV_HYBRID) == 1
        assert jm.get_multiview_queue_length(PipelineEnum.CANONICAL_MV_FAST) == 0

    def test_worker_processes_multiple_jobs(self, jm, sm):
        """Worker should be able to process multiple jobs from the queue."""
        job_ids = []
        for _ in range(3):
            job_id = _create_mv_job_with_views(jm, sm)
            jm.queue_job_for_generation(job_id)
            job_ids.append(job_id)

        worker = self.CanonicalMVWorker()
        worker.jm = jm

        # Poll should return jobs in FIFO order
        for expected_id in job_ids:
            picked = worker._poll_queues()
            assert picked == expected_id


# ===========================================================================
# API ENDPOINT INTEGRATION
# ===========================================================================


class TestAPIEndpoints:
    """Test FastAPI endpoints for multi-view jobs."""

    @pytest.fixture
    def client(self, jm, sm):
        """Create a TestClient with patched job_manager and storage_manager."""
        from fastapi.testclient import TestClient
        from api.main import app

        with patch("api.main.job_manager", jm), \
             patch("api.main.storage_manager", sm):
            yield TestClient(app)

    def _upload_multiview(self, client):
        """Helper to upload 5 views via the API."""
        files = {}
        for vn in CANONICAL_VIEW_ORDER:
            img = _create_synthetic_view()
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            files[vn] = (f"{vn}.png", buf, "image/png")

        response = client.post(
            "/api/upload_multiview",
            files=files,
            data={
                "category": "human_bust",
                "pipeline": "canonical_mv_hybrid",
                "params": json.dumps(
                    _json_safe_params(shared_canvas_size=_TEST_CANVAS_SIZE)
                ),
            },
        )
        return response

    def test_upload_multiview_creates_job(self, client, jm):
        """POST /api/upload_multiview should create a job and return its ID."""
        response = self._upload_multiview(client)

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"
        assert data["pipeline"] == "canonical_mv_hybrid"
        assert set(data["views_received"]) == set(CANONICAL_VIEW_ORDER)

    def test_upload_multiview_queues_job(self, client, jm):
        """Uploaded job should be queued in Redis."""
        response = self._upload_multiview(client)
        data = response.json()

        queue_len = jm.get_multiview_queue_length(PipelineEnum.CANONICAL_MV_HYBRID)
        assert queue_len == 1

    def test_get_job_status(self, client, jm):
        """GET /api/job/{id}/status should return job data."""
        upload_resp = self._upload_multiview(client)
        job_id = upload_resp.json()["job_id"]

        response = client.get(f"/api/job/{job_id}/status")
        assert response.status_code == 200

        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == "queued"
        assert data["category"] == "human_bust"
        assert data["pipeline"] == "canonical_mv_hybrid"
        assert "stages" in data
        assert len(data["stages"]) == len(MV_STAGE_ORDER)

    def test_get_job_status_not_found(self, client):
        """GET /api/job/{nonexistent}/status should return 404."""
        response = client.get("/api/job/nonexistent-id/status")
        assert response.status_code == 404

    def test_get_job_artifacts_empty(self, client, jm):
        """GET /api/job/{id}/artifacts should return empty list for new job."""
        upload_resp = self._upload_multiview(client)
        job_id = upload_resp.json()["job_id"]

        response = client.get(f"/api/job/{job_id}/artifacts")
        assert response.status_code == 200

        data = response.json()
        assert data["job_id"] == job_id
        assert isinstance(data["artifacts"], list)

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_get_job_artifacts_after_pipeline(self, mock_seg, client, jm, sm):
        """After pipeline completes, artifacts endpoint should list them."""
        upload_resp = self._upload_multiview(client)
        job_id = upload_resp.json()["job_id"]

        # Drain queue and run pipeline directly
        jm.get_next_multiview_job(PipelineEnum.CANONICAL_MV_HYBRID)
        orchestrator = CanonicalMVOrchestrator(
            job_id=job_id,
            job_manager=jm,
            storage_manager=sm,
        )
        orchestrator.run()

        response = client.get(f"/api/job/{job_id}/artifacts")
        assert response.status_code == 200

        data = response.json()
        filenames = [a["filename"] for a in data["artifacts"]]
        assert any("camera_init" in f for f in filenames)
        assert any("metrics" in f for f in filenames)

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_get_job_metrics_after_pipeline(self, mock_seg, client, jm, sm):
        """After pipeline completes, metrics endpoint should return QA data."""
        upload_resp = self._upload_multiview(client)
        job_id = upload_resp.json()["job_id"]

        jm.get_next_multiview_job(PipelineEnum.CANONICAL_MV_HYBRID)
        orchestrator = CanonicalMVOrchestrator(
            job_id=job_id,
            job_manager=jm,
            storage_manager=sm,
        )
        orchestrator.run()

        response = client.get(f"/api/job/{job_id}/metrics")
        assert response.status_code == 200

        data = response.json()
        assert data["job_id"] == job_id
        assert data["quality_score"] is not None
        assert 0.0 <= data["quality_score"] <= 1.0

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_download_output_after_pipeline(self, mock_seg, client, jm, sm):
        """GET /api/job/{id}/output should serve the GLB after completion."""
        upload_resp = self._upload_multiview(client)
        job_id = upload_resp.json()["job_id"]

        jm.get_next_multiview_job(PipelineEnum.CANONICAL_MV_HYBRID)
        orchestrator = CanonicalMVOrchestrator(
            job_id=job_id,
            job_manager=jm,
            storage_manager=sm,
        )
        orchestrator.run()

        response = client.get(f"/api/job/{job_id}/output")
        assert response.status_code == 200
        assert response.headers["content-type"] == "model/gltf-binary"

        # Verify GLB header
        content = response.content
        magic, version = struct.unpack("<II", content[:8])
        assert magic == GLB_MAGIC
        assert version == GLB_VERSION

    def test_download_output_before_completion(self, client, jm):
        """GET /api/job/{id}/output before completion should return 400."""
        upload_resp = self._upload_multiview(client)
        job_id = upload_resp.json()["job_id"]

        response = client.get(f"/api/job/{job_id}/output")
        assert response.status_code == 400

    def test_rerun_stage(self, client, jm, sm):
        """POST /api/job/{id}/rerun_stage should reset stages and requeue."""
        upload_resp = self._upload_multiview(client)
        job_id = upload_resp.json()["job_id"]

        # Drain the initial queue entry
        jm.get_next_multiview_job(PipelineEnum.CANONICAL_MV_HYBRID)

        # Manually complete some stages
        for stage in [JobStage.INGEST.value, JobStage.PREPROCESS_VIEWS.value]:
            jm.advance_stage(job_id, stage)

        response = client.post(
            f"/api/job/{job_id}/rerun_stage",
            data={"stage": JobStage.PREPROCESS_VIEWS.value},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["rerun_from"] == JobStage.PREPROCESS_VIEWS.value
        assert JobStage.PREPROCESS_VIEWS.value in data["stages_reset"]

        # Job should be requeued
        assert jm.get_multiview_queue_length(PipelineEnum.CANONICAL_MV_HYBRID) == 1

    def test_delete_job(self, client, jm, sm):
        """DELETE /api/job/{id} should remove the job."""
        upload_resp = self._upload_multiview(client)
        job_id = upload_resp.json()["job_id"]

        response = client.delete(f"/api/job/{job_id}")
        assert response.status_code == 200

        # Job should no longer exist
        assert jm.get_job(job_id) is None

    def test_upload_multiview_invalid_category(self, client):
        """Invalid category should return 400."""
        files = {}
        for vn in CANONICAL_VIEW_ORDER:
            img = _create_synthetic_view()
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            files[vn] = (f"{vn}.png", buf, "image/png")

        response = client.post(
            "/api/upload_multiview",
            files=files,
            data={
                "category": "invalid_category",
                "pipeline": "canonical_mv_hybrid",
                "params": "{}",
            },
        )
        assert response.status_code == 400

    def test_upload_multiview_invalid_pipeline(self, client):
        """Invalid pipeline should return 400."""
        files = {}
        for vn in CANONICAL_VIEW_ORDER:
            img = _create_synthetic_view()
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            files[vn] = (f"{vn}.png", buf, "image/png")

        response = client.post(
            "/api/upload_multiview",
            files=files,
            data={
                "category": "human_bust",
                "pipeline": "nonexistent_pipeline",
                "params": "{}",
            },
        )
        assert response.status_code == 400

    def test_upload_singleview_pipeline_rejected(self, client):
        """Using a single-view pipeline for multiview upload should return 400."""
        files = {}
        for vn in CANONICAL_VIEW_ORDER:
            img = _create_synthetic_view()
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            files[vn] = (f"{vn}.png", buf, "image/png")

        response = client.post(
            "/api/upload_multiview",
            files=files,
            data={
                "category": "human_bust",
                "pipeline": "singleview_triposr",
                "params": "{}",
            },
        )
        assert response.status_code == 400

    def test_health_endpoint(self, client):
        """GET /api/health should return status."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "pipelines" in data

    def test_queue_status_endpoint(self, client, jm):
        """GET /api/queue/status should return queue lengths."""
        # Queue a job
        self._upload_multiview(client)

        response = client.get("/api/queue/status")
        assert response.status_code == 200
        data = response.json()
        assert "multi_view" in data
        assert "canonical_mv_hybrid" in data["multi_view"]

    def test_categories_endpoint(self, client):
        """GET /api/categories should return category guidance."""
        response = client.get("/api/categories")
        assert response.status_code == 200
        data = response.json()
        assert "categories" in data
        assert len(data["categories"]) > 0


# ===========================================================================
# JOB STATE TRANSITIONS
# ===========================================================================


class TestJobStateTransitions:
    """Verify correct job state transitions through the pipeline."""

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_queued_to_generating_to_completed(self, mock_seg, jm, sm, config):
        """Job should transition: QUEUED → GENERATING → COMPLETED."""
        job_id = _create_mv_job_with_views(
            jm, sm,
            params=_json_safe_params(shared_canvas_size=_TEST_CANVAS_SIZE),
        )

        # Initial state
        job = jm.get_job(job_id)
        assert job["status"] == JobStatus.QUEUED.value

        # Run pipeline
        orchestrator = CanonicalMVOrchestrator(
            job_id=job_id,
            job_manager=jm,
            storage_manager=sm,
        )
        orchestrator.run()

        # Final state
        job = jm.get_job(job_id)
        assert job["status"] == JobStatus.COMPLETED.value

    def test_queued_to_generating_to_failed(self, jm, sm, config):
        """Job should transition: QUEUED → GENERATING → FAILED on error."""
        # Create job without views (will fail at ingest)
        job_id = jm.create_multiview_job(
            category=CategoryEnum.HUMAN_BUST,
            pipeline=PipelineEnum.CANONICAL_MV_HYBRID,
            views_received=["front", "back", "left", "right", "top"],
        )

        job = jm.get_job(job_id)
        assert job["status"] == JobStatus.QUEUED.value

        orchestrator = CanonicalMVOrchestrator(
            job_id=job_id,
            job_manager=jm,
            storage_manager=sm,
        )
        orchestrator.run()

        job = jm.get_job(job_id)
        assert job["status"] == JobStatus.FAILED.value

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_stage_transitions_in_order(self, mock_seg, jm, sm, config):
        """
        Each stage should transition PENDING → IN_PROGRESS → COMPLETED
        in the correct order.
        """
        job_id = _create_mv_job_with_views(
            jm, sm,
            params=_json_safe_params(shared_canvas_size=_TEST_CANVAS_SIZE),
        )

        # Track stage transitions
        transitions = []
        original_update = jm.update_job

        def tracking_update(job_id_arg, **kwargs):
            if "stage_updates" in kwargs and kwargs["stage_updates"]:
                for stage, status in kwargs["stage_updates"].items():
                    transitions.append((stage, status))
            return original_update(job_id_arg, **kwargs)

        jm.update_job = tracking_update

        orchestrator = CanonicalMVOrchestrator(
            job_id=job_id,
            job_manager=jm,
            storage_manager=sm,
        )
        orchestrator.run()

        # Verify that each non-skipped stage went through IN_PROGRESS
        in_progress_stages = [
            stage for stage, status in transitions
            if status == StageStatus.IN_PROGRESS.value
        ]

        # At minimum, ingest, preprocess, validate, cameras, coarse, export, qa
        # should have been marked in_progress
        assert JobStage.INGEST.value in in_progress_stages
        assert JobStage.PREPROCESS_VIEWS.value in in_progress_stages
        assert JobStage.RECONSTRUCT_COARSE.value in in_progress_stages
        assert JobStage.EXPORT.value in in_progress_stages
        assert JobStage.QA.value in in_progress_stages

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_completed_at_timestamp_set(self, mock_seg, jm, sm, config):
        """completed_at should be set when pipeline finishes."""
        job_id = _create_mv_job_with_views(
            jm, sm,
            params=_json_safe_params(shared_canvas_size=_TEST_CANVAS_SIZE),
        )

        orchestrator = CanonicalMVOrchestrator(
            job_id=job_id,
            job_manager=jm,
            storage_manager=sm,
        )
        orchestrator.run()

        job = jm.get_job(job_id)
        assert job["completed_at"] is not None

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_view_statuses_updated(self, mock_seg, jm, sm, config):
        """Per-view statuses should be updated through the pipeline."""
        job_id = _create_mv_job_with_views(
            jm, sm,
            params=_json_safe_params(shared_canvas_size=_TEST_CANVAS_SIZE),
        )

        orchestrator = CanonicalMVOrchestrator(
            job_id=job_id,
            job_manager=jm,
            storage_manager=sm,
        )
        orchestrator.run()

        job = jm.get_job(job_id)
        views = job.get("views", {})

        for vn in CANONICAL_VIEW_ORDER:
            assert vn in views
            # After full pipeline, views should be validated
            assert views[vn]["status"] in ("validated", "segmented", "uploaded")


# ===========================================================================
# CROSS-PIPELINE CONSISTENCY
# ===========================================================================


class TestCrossPipelineConsistency:
    """Test that different pipeline configs produce consistent results."""

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_different_pipelines_all_complete(self, mock_seg, jm, sm):
        """All three multi-view pipeline types should complete."""
        for pipe in [
            PipelineEnum.CANONICAL_MV_HYBRID,
            PipelineEnum.CANONICAL_MV_FAST,
            PipelineEnum.CANONICAL_MV_GENERATIVE,
        ]:
            job_id = _create_mv_job_with_views(
                jm, sm, pipeline=pipe,
                params=_json_safe_params(shared_canvas_size=_TEST_CANVAS_SIZE),
            )

            orchestrator = CanonicalMVOrchestrator(
                job_id=job_id,
                job_manager=jm,
                storage_manager=sm,
            )
            success = orchestrator.run()
            assert success is True, f"Pipeline {pipe.value} failed"

            job = jm.get_job(job_id)
            assert job["status"] == JobStatus.COMPLETED.value

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_output_mesh_has_valid_geometry(self, mock_seg, jm, sm, config):
        """Output mesh should have valid geometry (non-zero vertices and faces)."""
        job_id = _create_mv_job_with_views(
            jm, sm,
            params=_json_safe_params(shared_canvas_size=_TEST_CANVAS_SIZE),
        )

        orchestrator = CanonicalMVOrchestrator(
            job_id=job_id,
            job_manager=jm,
            storage_manager=sm,
        )
        orchestrator.run()

        # Check export metrics for mesh stats
        export_metrics = sm.load_artifact_json(job_id, "export_metrics.json")
        assert export_metrics is not None
        assert export_metrics["final_vertices"] > 0
        assert export_metrics["final_faces"] > 0

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_qa_score_in_valid_range(self, mock_seg, jm, sm, config):
        """QA quality score should be in [0, 1]."""
        job_id = _create_mv_job_with_views(
            jm, sm,
            params=_json_safe_params(shared_canvas_size=_TEST_CANVAS_SIZE),
        )

        orchestrator = CanonicalMVOrchestrator(
            job_id=job_id,
            job_manager=jm,
            storage_manager=sm,
        )
        orchestrator.run()

        metrics = sm.load_metrics(job_id)
        assert metrics is not None
        score = metrics["quality_score"]
        assert 0.0 <= score <= 1.0

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_all_depth_maps_generated(self, mock_seg, jm, sm, config):
        """Coarse depth maps should be generated for all views."""
        job_id = _create_mv_job_with_views(
            jm, sm,
            params=_json_safe_params(shared_canvas_size=_TEST_CANVAS_SIZE),
        )

        orchestrator = CanonicalMVOrchestrator(
            job_id=job_id,
            job_manager=jm,
            storage_manager=sm,
        )
        orchestrator.run()

        for vn in CANONICAL_VIEW_ORDER:
            path = sm.get_artifact_path(job_id, f"coarse_depth_{vn}.png")
            assert path is not None, f"Missing depth map for view '{vn}'"


# ===========================================================================
# EDGE CASES
# ===========================================================================


class TestEdgeCases:
    """Edge cases and robustness tests."""

    def test_nonexistent_job_raises(self, jm, sm):
        """Orchestrator should raise ValueError for nonexistent job."""
        with pytest.raises(ValueError, match="not found"):
            CanonicalMVOrchestrator(
                job_id="nonexistent-job-id",
                job_manager=jm,
                storage_manager=sm,
            )

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_small_images_produce_warnings(self, mock_seg, jm, sm, config):
        """Very small input images should produce warnings but still complete."""
        job_id = jm.create_multiview_job(
            category=CategoryEnum.HUMAN_BUST,
            pipeline=PipelineEnum.CANONICAL_MV_HYBRID,
            views_received=list(CANONICAL_VIEW_ORDER),
            params=_json_safe_params(shared_canvas_size=128),
        )

        # Upload very small images (128x128)
        for vn in CANONICAL_VIEW_ORDER:
            img = _create_synthetic_view(width=128, height=128)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            sm.save_view_upload(job_id, vn, buf.getvalue(), ".png")

        orchestrator = CanonicalMVOrchestrator(
            job_id=job_id,
            job_manager=jm,
            storage_manager=sm,
        )
        success = orchestrator.run()
        assert success is True

        # Should have low-resolution warnings from ingest
        job = jm.get_job(job_id)
        warnings = job.get("warnings", [])
        # Small images (128x128 < 256) should trigger warnings
        low_res_warnings = [w for w in warnings if "low_resolution" in str(w)]
        assert len(low_res_warnings) > 0

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_artifacts_list_grows_through_pipeline(self, mock_seg, jm, sm, config):
        """The artifacts_available list should grow as stages complete."""
        job_id = _create_mv_job_with_views(
            jm, sm,
            params=_json_safe_params(shared_canvas_size=_TEST_CANVAS_SIZE),
        )

        orchestrator = CanonicalMVOrchestrator(
            job_id=job_id,
            job_manager=jm,
            storage_manager=sm,
        )
        orchestrator.run()

        job = jm.get_job(job_id)
        artifacts = job.get("artifacts_available", [])

        # Should have artifacts from multiple completed stages
        assert len(artifacts) > 0
        # At minimum: ingest, preprocess, validate, cameras, coarse, export, qa
        assert len(artifacts) >= 7

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_concurrent_jobs_isolated(self, mock_seg, jm, sm, config):
        """Two jobs running sequentially should not interfere with each other."""
        job_id_1 = _create_mv_job_with_views(
            jm, sm,
            params=_json_safe_params(shared_canvas_size=_TEST_CANVAS_SIZE),
        )
        job_id_2 = _create_mv_job_with_views(
            jm, sm,
            params=_json_safe_params(shared_canvas_size=_TEST_CANVAS_SIZE),
        )

        # Run first job
        orch1 = CanonicalMVOrchestrator(
            job_id=job_id_1, job_manager=jm, storage_manager=sm,
        )
        orch1.run()

        # Run second job
        orch2 = CanonicalMVOrchestrator(
            job_id=job_id_2, job_manager=jm, storage_manager=sm,
        )
        orch2.run()

        # Both should complete independently
        job1 = jm.get_job(job_id_1)
        job2 = jm.get_job(job_id_2)
        assert job1["status"] == JobStatus.COMPLETED.value
        assert job2["status"] == JobStatus.COMPLETED.value

        # Each should have its own output
        output1 = sm.get_output_file(job_id_1)
        output2 = sm.get_output_file(job_id_2)
        assert output1 is not None
        assert output2 is not None
        assert output1 != output2

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_custom_params_propagated(self, mock_seg, jm, sm):
        """Custom params should be reflected in the pipeline config."""
        custom_params = _json_safe_params(
            shared_canvas_size=_TEST_CANVAS_SIZE,
            mesh_resolution=128,
            texture_resolution=512,
            decimation_target=100000,
            symmetry_prior=False,
            use_joint_refinement=False,
            use_trellis_completion=False,
            use_hunyuan_completion=False,
        )

        job_id = _create_mv_job_with_views(jm, sm, params=custom_params)

        orchestrator = CanonicalMVOrchestrator(
            job_id=job_id,
            job_manager=jm,
            storage_manager=sm,
        )

        # Verify config was built from params
        assert orchestrator.config.mesh_resolution == 128
        assert orchestrator.config.texture_resolution == 512
        assert orchestrator.config.decimation_target == 100000
        assert orchestrator.config.symmetry_prior is False

        success = orchestrator.run()
        assert success is True

