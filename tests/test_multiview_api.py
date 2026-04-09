"""
Tests for multi-view API endpoints.

Covers Phase 0 additions:
    - POST /api/upload_multiview
    - GET /api/job/{id}/status (with multi-view fields)
    - GET /api/job/{id}/artifacts
    - GET /api/job/{id}/metrics
    - POST /api/job/{id}/rerun_stage
    - GET /api/job/{id}/download/{artifact}
    - GET /api/job/{id}/previews (multi-view)
    - GET /api/health (pipeline list)
    - GET /api/queue/status (multi-view queues)

Uses fakeredis and TestClient (synchronous) for fast, isolated tests.
"""

import io
import json
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
from pathlib import Path

import fakeredis
from fastapi.testclient import TestClient

from api.main import app, job_manager, storage_manager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_redis():
    """Replace the real Redis client with fakeredis for all tests."""
    fake_redis = fakeredis.FakeRedis(decode_responses=True)
    original = job_manager.redis_client
    job_manager.redis_client = fake_redis
    yield
    job_manager.redis_client = original
    fake_redis.flushall()


@pytest.fixture(autouse=True)
def _patch_storage(tmp_path):
    """Use a temp directory for storage in all tests."""
    original_root = storage_manager.storage_root
    original_uploads = storage_manager.uploads_dir
    original_previews = storage_manager.previews_dir
    original_outputs = storage_manager.outputs_dir
    original_artifacts = storage_manager.artifacts_dir

    storage_manager.storage_root = tmp_path / "storage"
    storage_manager.uploads_dir = storage_manager.storage_root / "uploads"
    storage_manager.previews_dir = storage_manager.storage_root / "previews"
    storage_manager.outputs_dir = storage_manager.storage_root / "outputs"
    storage_manager.artifacts_dir = storage_manager.storage_root / "artifacts"

    for d in (
        storage_manager.uploads_dir,
        storage_manager.previews_dir,
        storage_manager.outputs_dir,
        storage_manager.artifacts_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)

    # Also patch the jobs_dir for file-based metadata
    original_jobs_dir = job_manager.jobs_dir
    job_manager.jobs_dir = storage_manager.storage_root / "jobs"
    job_manager.jobs_dir.mkdir(parents=True, exist_ok=True)

    yield

    storage_manager.storage_root = original_root
    storage_manager.uploads_dir = original_uploads
    storage_manager.previews_dir = original_previews
    storage_manager.outputs_dir = original_outputs
    storage_manager.artifacts_dir = original_artifacts
    job_manager.jobs_dir = original_jobs_dir


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


def _make_fake_image(name: str = "test.png") -> tuple:
    """Create a minimal fake PNG file for upload."""
    # 1x1 red pixel PNG
    png_data = (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
        b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00'
        b'\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00'
        b'\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    return (name, png_data, "image/png")


# ---------------------------------------------------------------------------
# POST /api/upload_multiview
# ---------------------------------------------------------------------------

class TestUploadMultiview:
    def test_successful_upload(self, client):
        files = {
            "front": _make_fake_image("front.png"),
            "back": _make_fake_image("back.png"),
            "left": _make_fake_image("left.png"),
            "right": _make_fake_image("right.png"),
            "top": _make_fake_image("top.png"),
        }
        data = {
            "category": "human_bust",
            "pipeline": "canonical_mv_hybrid",
            "params": "{}",
        }

        resp = client.post("/api/upload_multiview", files=files, data=data)
        assert resp.status_code == 200

        body = resp.json()
        assert "job_id" in body
        assert body["status"] == "queued"
        assert body["pipeline"] == "canonical_mv_hybrid"
        assert set(body["views_received"]) == {"front", "back", "left", "right", "top"}

    def test_invalid_category(self, client):
        files = {
            "front": _make_fake_image(),
            "back": _make_fake_image(),
            "left": _make_fake_image(),
            "right": _make_fake_image(),
            "top": _make_fake_image(),
        }
        resp = client.post(
            "/api/upload_multiview",
            files=files,
            data={"category": "invalid_cat", "pipeline": "canonical_mv_hybrid"},
        )
        assert resp.status_code == 400

    def test_invalid_pipeline(self, client):
        files = {
            "front": _make_fake_image(),
            "back": _make_fake_image(),
            "left": _make_fake_image(),
            "right": _make_fake_image(),
            "top": _make_fake_image(),
        }
        resp = client.post(
            "/api/upload_multiview",
            files=files,
            data={"category": "human_bust", "pipeline": "nonexistent"},
        )
        assert resp.status_code == 400

    def test_singleview_pipeline_rejected(self, client):
        files = {
            "front": _make_fake_image(),
            "back": _make_fake_image(),
            "left": _make_fake_image(),
            "right": _make_fake_image(),
            "top": _make_fake_image(),
        }
        resp = client.post(
            "/api/upload_multiview",
            files=files,
            data={"category": "human_bust", "pipeline": "singleview_triposr"},
        )
        assert resp.status_code == 400
        assert "not a multi-view pipeline" in resp.json()["detail"]

    def test_invalid_params_json(self, client):
        files = {
            "front": _make_fake_image(),
            "back": _make_fake_image(),
            "left": _make_fake_image(),
            "right": _make_fake_image(),
            "top": _make_fake_image(),
        }
        resp = client.post(
            "/api/upload_multiview",
            files=files,
            data={
                "category": "human_bust",
                "pipeline": "canonical_mv_hybrid",
                "params": "not-json",
            },
        )
        assert resp.status_code == 400

    def test_custom_params(self, client):
        files = {
            "front": _make_fake_image(),
            "back": _make_fake_image(),
            "left": _make_fake_image(),
            "right": _make_fake_image(),
            "top": _make_fake_image(),
        }
        params = json.dumps({"seed": 42, "mesh_resolution": 384})
        resp = client.post(
            "/api/upload_multiview",
            files=files,
            data={
                "category": "generic_object",
                "pipeline": "canonical_mv_hybrid",
                "params": params,
            },
        )
        assert resp.status_code == 200

        # Verify params were stored
        job_id = resp.json()["job_id"]
        job = job_manager.get_job(job_id)
        assert job["params"]["seed"] == 42
        assert job["params"]["mesh_resolution"] == 384


# ---------------------------------------------------------------------------
# GET /api/job/{id}/status (multi-view)
# ---------------------------------------------------------------------------

class TestJobStatusMultiview:
    def _create_mv_job(self, client):
        files = {
            "front": _make_fake_image(),
            "back": _make_fake_image(),
            "left": _make_fake_image(),
            "right": _make_fake_image(),
            "top": _make_fake_image(),
        }
        resp = client.post(
            "/api/upload_multiview",
            files=files,
            data={"category": "human_bust", "pipeline": "canonical_mv_hybrid"},
        )
        return resp.json()["job_id"]

    def test_status_has_pipeline(self, client):
        job_id = self._create_mv_job(client)
        resp = client.get(f"/api/job/{job_id}/status")
        assert resp.status_code == 200

        body = resp.json()
        assert body["pipeline"] == "canonical_mv_hybrid"
        assert body["model"] is None

    def test_status_has_stages(self, client):
        job_id = self._create_mv_job(client)
        resp = client.get(f"/api/job/{job_id}/status")
        body = resp.json()

        assert "stages" in body
        assert len(body["stages"]) == 10
        assert body["stages"]["ingest"] == "pending"

    def test_status_has_views(self, client):
        job_id = self._create_mv_job(client)
        resp = client.get(f"/api/job/{job_id}/status")
        body = resp.json()

        assert "views" in body
        assert body["views"] is not None
        assert "front" in body["views"]

    def test_status_not_found(self, client):
        resp = client.get("/api/job/nonexistent/status")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/job/{id}/artifacts
# ---------------------------------------------------------------------------

class TestArtifactsEndpoint:
    def test_empty_artifacts(self, client):
        job_id = self._create_sv_job(client)
        resp = client.get(f"/api/job/{job_id}/artifacts")
        assert resp.status_code == 200
        body = resp.json()
        assert body["artifacts"] == []

    def test_artifacts_with_files(self, client):
        job_id = self._create_sv_job(client)

        # Manually create an artifact
        storage_manager.save_artifact(job_id, "coarse_mesh.glb", b"mesh-data")

        resp = client.get(f"/api/job/{job_id}/artifacts")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["artifacts"]) == 1
        assert body["artifacts"][0]["filename"] == "coarse_mesh.glb"
        assert body["artifacts"][0]["stage"] == "reconstruct_coarse"

    def test_artifacts_not_found(self, client):
        resp = client.get("/api/job/nonexistent/artifacts")
        assert resp.status_code == 404

    def _create_sv_job(self, client):
        """Helper to create a single-view job for testing."""
        job_id = job_manager.create_job(
            category=__import__("api.models", fromlist=["CategoryEnum"]).CategoryEnum.GENERIC_OBJECT,
            model=__import__("api.models", fromlist=["ModelEnum"]).ModelEnum.TRIPOSR,
        )
        return job_id


# ---------------------------------------------------------------------------
# GET /api/job/{id}/download/{artifact}
# ---------------------------------------------------------------------------

class TestDownloadArtifact:
    def test_download_artifact(self, client):
        from api.models import CategoryEnum, ModelEnum
        job_id = job_manager.create_job(CategoryEnum.GENERIC_OBJECT, ModelEnum.TRIPOSR)

        storage_manager.save_artifact(job_id, "camera_init.json", b'{"test": true}')

        resp = client.get(f"/api/job/{job_id}/download/camera_init.json")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/json"

    def test_download_not_found(self, client):
        from api.models import CategoryEnum, ModelEnum
        job_id = job_manager.create_job(CategoryEnum.GENERIC_OBJECT, ModelEnum.TRIPOSR)

        resp = client.get(f"/api/job/{job_id}/download/nonexistent.glb")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/job/{id}/metrics
# ---------------------------------------------------------------------------

class TestMetricsEndpoint:
    def test_empty_metrics(self, client):
        from api.models import CategoryEnum, ModelEnum
        job_id = job_manager.create_job(CategoryEnum.GENERIC_OBJECT, ModelEnum.TRIPOSR)

        resp = client.get(f"/api/job/{job_id}/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert body["quality_score"] is None
        assert body["warnings"] == []

    def test_populated_metrics(self, client):
        from api.models import CategoryEnum, ModelEnum
        job_id = job_manager.create_job(CategoryEnum.GENERIC_OBJECT, ModelEnum.TRIPOSR)

        storage_manager.save_metrics(job_id, {
            "quality_score": 0.84,
            "per_view_metrics": {"front": {"iou": 0.95}},
            "mesh_metrics": {"face_count": 100000},
            "warnings": [
                {"code": "low_iou", "message": "Low IoU on top", "severity": "warning"},
                "simple_warning_string",
            ],
            "recommended_retry": ["re-upload sharper top image"],
        })

        resp = client.get(f"/api/job/{job_id}/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert body["quality_score"] == 0.84
        assert len(body["warnings"]) == 2
        assert body["recommended_retry"] == ["re-upload sharper top image"]

    def test_metrics_not_found(self, client):
        resp = client.get("/api/job/nonexistent/metrics")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/job/{id}/rerun_stage
# ---------------------------------------------------------------------------

class TestRerunStage:
    def _create_mv_job_with_progress(self, client):
        """Create a multi-view job and advance through some stages."""
        files = {
            "front": _make_fake_image(),
            "back": _make_fake_image(),
            "left": _make_fake_image(),
            "right": _make_fake_image(),
            "top": _make_fake_image(),
        }
        resp = client.post(
            "/api/upload_multiview",
            files=files,
            data={"category": "human_bust", "pipeline": "canonical_mv_hybrid"},
        )
        job_id = resp.json()["job_id"]

        # Advance a few stages
        job_manager.advance_stage(job_id, "ingest")
        job_manager.advance_stage(job_id, "preprocess_views")
        job_manager.advance_stage(job_id, "validate_views")

        return job_id

    def test_rerun_stage(self, client):
        job_id = self._create_mv_job_with_progress(client)

        resp = client.post(
            f"/api/job/{job_id}/rerun_stage",
            data={"stage": "preprocess_views"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["rerun_from"] == "preprocess_views"
        assert "preprocess_views" in body["stages_reset"]
        assert "validate_views" in body["stages_reset"]

        # Check that stages were actually reset
        job = job_manager.get_job(job_id)
        assert job["stages"]["preprocess_views"] == "pending"
        assert job["stages"]["validate_views"] == "pending"
        assert job["stages"]["ingest"] == "completed"  # should NOT be reset

    def test_rerun_stage_singleview_rejected(self, client):
        from api.models import CategoryEnum, ModelEnum
        job_id = job_manager.create_job(CategoryEnum.GENERIC_OBJECT, ModelEnum.TRIPOSR)

        resp = client.post(
            f"/api/job/{job_id}/rerun_stage",
            data={"stage": "preprocessing"},
        )
        assert resp.status_code == 400

    def test_rerun_unknown_stage(self, client):
        job_id = self._create_mv_job_with_progress(client)

        resp = client.post(
            f"/api/job/{job_id}/rerun_stage",
            data={"stage": "nonexistent_stage"},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /api/health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_includes_pipelines(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "pipelines" in body
        assert "canonical_mv_hybrid" in body["pipelines"]
        assert "singleview_triposr" in body["pipelines"]
        assert body["version"] == "2.0.0"


# ---------------------------------------------------------------------------
# GET /api/queue/status
# ---------------------------------------------------------------------------

class TestQueueStatus:
    def test_queue_status_includes_multiview(self, client):
        resp = client.get("/api/queue/status")
        assert resp.status_code == 200
        body = resp.json()
        assert "multi_view" in body
        assert "canonical_mv_hybrid" in body["multi_view"]
        assert "single_view" in body
        assert "triposr" in body["single_view"]


# ---------------------------------------------------------------------------
# Legacy single-view compatibility
# ---------------------------------------------------------------------------

class TestLegacyUploadStillWorks:
    @patch("api.main.trigger_preprocessing")
    def test_single_view_upload(self, mock_preprocess, client):
        """Ensure the old /api/upload endpoint still works."""
        mock_preprocess.return_value = None

        files = {"file": _make_fake_image("input.png")}
        data = {"category": "generic_object", "model": "triposr"}

        resp = client.post("/api/upload", files=files, data=data)
        assert resp.status_code == 200
        body = resp.json()
        assert "job_id" in body
        assert body["status"] == "queued"


# ---------------------------------------------------------------------------
# DELETE /api/job/{id}
# ---------------------------------------------------------------------------

class TestDeleteJob:
    def test_delete_multiview_job(self, client):
        files = {
            "front": _make_fake_image(),
            "back": _make_fake_image(),
            "left": _make_fake_image(),
            "right": _make_fake_image(),
            "top": _make_fake_image(),
        }
        resp = client.post(
            "/api/upload_multiview",
            files=files,
            data={"category": "human_bust", "pipeline": "canonical_mv_hybrid"},
        )
        job_id = resp.json()["job_id"]

        resp = client.delete(f"/api/job/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        resp = client.get(f"/api/job/{job_id}/status")
        assert resp.status_code == 404

