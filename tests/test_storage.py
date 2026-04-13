"""Tests for api/storage.py – file storage layer."""

import tempfile
from pathlib import Path

import pytest

from api import storage as storage_mod
from api.storage import (
    StorageError,
    create_job_storage,
    delete_job_storage,
    get_artifact_path,
    get_job_dir,
    get_storage_usage,
    list_artifacts,
    save_upload,
)
from pipelines import config as cfg


@pytest.fixture(autouse=True)
def use_tmp_storage(tmp_path, monkeypatch):
    """Redirect storage to a temp directory for all tests."""
    tmp_storage = tmp_path / "storage"
    tmp_storage.mkdir()
    # Patch both the canonical source and the already-imported binding
    monkeypatch.setattr(cfg, "STORAGE_ROOT", tmp_storage)
    monkeypatch.setattr(storage_mod, "STORAGE_ROOT", tmp_storage)


class TestCreateJobStorage:
    def test_creates_directory_structure(self):
        job_dir = create_job_storage("test-job-1")
        assert job_dir.exists()
        assert (job_dir / "input").exists()
        assert (job_dir / "preprocessed").exists()
        assert (job_dir / "coarse_recon").exists()
        assert (job_dir / "isolation" / "masks").exists()
        assert (job_dir / "isolation" / "masked_images").exists()
        assert (job_dir / "trellis").exists()
        assert (job_dir / "colmap" / "sparse" / "0").exists()
        assert (job_dir / "export").exists()

    def test_duplicate_raises(self):
        create_job_storage("dup-job")
        with pytest.raises(StorageError, match="already exists"):
            create_job_storage("dup-job")


class TestSaveUpload:
    def test_saves_file(self):
        create_job_storage("upload-test")
        path = save_upload("upload-test", "test.png", b"fake image data")
        assert path.exists()
        assert path.read_bytes() == b"fake image data"

    def test_sanitises_filename(self):
        create_job_storage("sanitise-test")
        path = save_upload("sanitise-test", "../../../etc/passwd", b"nope")
        assert "etc" not in str(path)


class TestGetArtifactPath:
    def test_existing_file(self):
        job_dir = create_job_storage("artifact-test")
        (job_dir / "input" / "front.png").write_bytes(b"data")
        result = get_artifact_path("artifact-test", "input/front.png")
        assert result is not None
        assert result.exists()

    def test_nonexistent_file(self):
        create_job_storage("no-file-test")
        result = get_artifact_path("no-file-test", "input/missing.png")
        assert result is None

    def test_path_traversal_blocked(self):
        create_job_storage("traversal-test")
        result = get_artifact_path("traversal-test", "../../etc/passwd")
        assert result is None


class TestListArtifacts:
    def test_empty_job(self):
        create_job_storage("empty-list")
        result = list_artifacts("empty-list", "input")
        assert result == []

    def test_with_files(self):
        job_dir = create_job_storage("list-test")
        (job_dir / "input" / "front.png").write_bytes(b"a")
        (job_dir / "input" / "side.png").write_bytes(b"b")
        result = list_artifacts("list-test", "input")
        assert len(result) == 2


class TestDeleteJobStorage:
    def test_deletes(self):
        create_job_storage("delete-test")
        assert delete_job_storage("delete-test") is True
        assert not get_job_dir("delete-test").exists()

    def test_nonexistent(self):
        assert delete_job_storage("nonexistent") is False


class TestGetStorageUsage:
    def test_empty(self):
        create_job_storage("usage-test")
        assert get_storage_usage("usage-test") == 0

    def test_with_data(self):
        job_dir = create_job_storage("usage-data")
        (job_dir / "input" / "test.bin").write_bytes(b"x" * 1000)
        assert get_storage_usage("usage-data") == 1000

