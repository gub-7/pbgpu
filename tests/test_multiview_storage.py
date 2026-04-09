"""
Tests for multi-view storage layout.

Covers Phase 0 additions: view uploads, per-view previews,
artifact management, and metrics persistence.
"""

import json
import shutil
import tempfile
import pytest
from pathlib import Path

from api.storage import StorageManager


@pytest.fixture
def storage(tmp_path):
    """Create a StorageManager rooted in a temp directory."""
    sm = StorageManager(storage_root=str(tmp_path / "storage"))
    return sm


@pytest.fixture
def job_id():
    return "test-job-001"


# Minimal valid PNG (1x1 red pixel)
_TINY_PNG = (
    b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
    b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00'
    b'\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00'
    b'\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
)


# ---------------------------------------------------------------------------
# View upload tests
# ---------------------------------------------------------------------------

class TestViewUploads:
    def test_save_and_get_view(self, storage, job_id):
        path = storage.save_view_upload(job_id, "front", b"fake-png", ".png")
        assert path.exists()
        assert path.name == "front.png"
        assert path.parent.name == "views"

    def test_get_view_upload_path(self, storage, job_id):
        storage.save_view_upload(job_id, "back", b"data", ".jpg")
        p = storage.get_view_upload_path(job_id, "back")
        assert p is not None
        assert p.name == "back.jpg"

    def test_get_view_upload_path_not_found(self, storage, job_id):
        p = storage.get_view_upload_path(job_id, "front")
        assert p is None

    def test_list_uploaded_views(self, storage, job_id):
        for vn in ("front", "back", "left", "right", "top"):
            storage.save_view_upload(job_id, vn, b"data", ".png")

        views = storage.list_uploaded_views(job_id)
        assert views == ["front", "back", "left", "right", "top"]

    def test_list_uploaded_views_partial(self, storage, job_id):
        storage.save_view_upload(job_id, "front", b"data", ".png")
        storage.save_view_upload(job_id, "top", b"data", ".png")
        views = storage.list_uploaded_views(job_id)
        assert views == ["front", "top"]

    def test_list_uploaded_views_empty(self, storage, job_id):
        views = storage.list_uploaded_views(job_id)
        assert views == []

    def test_views_dir_created(self, storage, job_id):
        d = storage.get_views_dir(job_id)
        assert d.exists()
        assert d.name == "views"

    def test_multiple_extensions(self, storage, job_id):
        storage.save_view_upload(job_id, "front", b"data", ".jpg")
        p = storage.get_view_upload_path(job_id, "front")
        assert p is not None
        assert p.suffix == ".jpg"


# ---------------------------------------------------------------------------
# View preview tests
# ---------------------------------------------------------------------------

class TestViewPreviews:
    def test_save_and_get_preview(self, storage, job_id):
        path = storage.save_view_preview(job_id, "raw", "front", b"preview-data", ".png")
        assert path.exists()
        assert path.name == "front.png"

        found = storage.get_view_preview_path(job_id, "raw", "front")
        assert found is not None
        assert found == path

    def test_preview_not_found(self, storage, job_id):
        p = storage.get_view_preview_path(job_id, "segmented", "front")
        assert p is None

    def test_list_view_previews(self, storage, job_id):
        for substage in ("raw", "segmented"):
            for vn in ("front", "back"):
                storage.save_view_preview(job_id, substage, vn, b"data", ".png")

        result = storage.list_view_previews(job_id)
        assert "raw" in result
        assert "segmented" in result
        assert "front" in result["raw"]
        assert "back" in result["raw"]

    def test_list_view_previews_empty(self, storage, job_id):
        result = storage.list_view_previews(job_id)
        assert result == {}

    def test_preview_dir_created(self, storage, job_id):
        d = storage.get_view_preview_dir(job_id, "normalized")
        assert d.exists()
        assert d.name == "normalized"


# ---------------------------------------------------------------------------
# Artifact tests
# ---------------------------------------------------------------------------

class TestArtifacts:
    def test_save_and_get_artifact(self, storage, job_id):
        path = storage.save_artifact(job_id, "coarse_mesh.glb", b"mesh-data")
        assert path.exists()
        assert path.name == "coarse_mesh.glb"

        found = storage.get_artifact_path(job_id, "coarse_mesh.glb")
        assert found is not None

    def test_artifact_not_found(self, storage, job_id):
        assert storage.get_artifact_path(job_id, "nonexistent.glb") is None

    def test_save_artifact_json(self, storage, job_id):
        data = {"cameras": {"front": {"yaw": 0}}}
        path = storage.save_artifact_json(job_id, "camera_init.json", data)
        assert path.exists()

        loaded = storage.load_artifact_json(job_id, "camera_init.json")
        assert loaded == data

    def test_load_artifact_json_not_found(self, storage, job_id):
        assert storage.load_artifact_json(job_id, "missing.json") is None

    def test_save_artifact_subpath(self, storage, job_id):
        path = storage.save_artifact(job_id, "textures/diffuse.png", b"tex-data")
        assert path.exists()
        assert path.parent.name == "textures"

    def test_list_artifacts(self, storage, job_id):
        storage.save_artifact(job_id, "coarse_mesh.glb", b"mesh")
        storage.save_artifact(job_id, "metrics.json", b'{"score":0.8}')
        storage.save_artifact(job_id, "textures/diffuse.png", b"tex")

        artifacts = storage.list_artifacts(job_id)
        assert len(artifacts) == 3

        filenames = {a["filename"] for a in artifacts}
        assert "coarse_mesh.glb" in filenames
        assert "metrics.json" in filenames

        # Check that content_type is set
        for a in artifacts:
            assert "content_type" in a
            assert a["file_size"] > 0

    def test_list_artifacts_empty(self, storage, job_id):
        assert storage.list_artifacts(job_id) == []

    def test_artifact_textures_dir(self, storage, job_id):
        d = storage.get_artifact_textures_dir(job_id)
        assert d.exists()
        assert d.name == "textures"


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_save_and_load_metrics(self, storage, job_id):
        metrics = {
            "quality_score": 0.84,
            "per_view_metrics": {
                "front": {"iou": 0.95},
            },
            "warnings": ["low_confidence_top"],
        }
        storage.save_metrics(job_id, metrics)

        loaded = storage.load_metrics(job_id)
        assert loaded["quality_score"] == 0.84
        assert loaded["per_view_metrics"]["front"]["iou"] == 0.95

    def test_load_metrics_not_found(self, storage, job_id):
        assert storage.load_metrics(job_id) is None


# ---------------------------------------------------------------------------
# Cleanup tests
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_cleanup_removes_all_dirs(self, storage, job_id):
        # Create files in all areas
        storage.save_view_upload(job_id, "front", b"data", ".png")
        storage.save_view_preview(job_id, "raw", "front", b"data", ".png")
        storage.save_artifact(job_id, "mesh.glb", b"data")

        output_dir = storage.get_job_output_dir(job_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "final.glb").write_bytes(b"glb")

        storage.cleanup_job(job_id)

        assert not (storage.uploads_dir / job_id).exists()
        assert not (storage.previews_dir / job_id).exists()
        assert not (storage.artifacts_dir / job_id).exists()
        assert not (storage.outputs_dir / job_id).exists()


# ---------------------------------------------------------------------------
# Content type utility
# ---------------------------------------------------------------------------

class TestContentType:
    def test_known_types(self):
        assert StorageManager.content_type_for("test.json") == "application/json"
        assert StorageManager.content_type_for("mesh.glb") == "model/gltf-binary"
        assert StorageManager.content_type_for("image.png") == "image/png"
        assert StorageManager.content_type_for("image.jpg") == "image/jpeg"

    def test_unknown_type(self):
        assert StorageManager.content_type_for("file.xyz") == "application/octet-stream"

