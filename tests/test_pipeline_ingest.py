"""
Tests for the ingest stage of the canonical multi-view pipeline.

Covers:
    - Successful ingest of all 5 views
    - Missing view detection
    - Corrupt image detection
    - Raw preview thumbnail generation
    - Per-view metadata updates
    - Warning generation for low-res / extreme aspect ratio
"""

import io
import pytest
from pathlib import Path
from unittest.mock import patch

import fakeredis
from PIL import Image

from api.job_manager import JobManager
from api.models import (
    CategoryEnum,
    PipelineEnum,
    ViewStatus,
)
from api.storage import StorageManager
from pipelines.canonical_mv.config import CanonicalMVConfig
from pipelines.canonical_mv.ingest import run_ingest


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
    return CanonicalMVConfig()


def _create_test_image(width=512, height=512, mode="RGB") -> bytes:
    """Create a valid PNG image of given dimensions."""
    img = Image.new(mode, (width, height), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _create_mv_job(jm, views=None):
    """Create a multi-view job with uploaded views."""
    if views is None:
        views = ["front", "back", "left", "right", "top"]
    return jm.create_multiview_job(
        category=CategoryEnum.HUMAN_BUST,
        pipeline=PipelineEnum.CANONICAL_MV_HYBRID,
        views_received=views,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestIngestHappyPath:
    def test_all_views_ingested(self, jm, sm, config):
        job_id = _create_mv_job(jm)

        # Upload all 5 views
        for vn in ("front", "back", "left", "right", "top"):
            img_bytes = _create_test_image(width=1024, height=768)
            sm.save_view_upload(job_id, vn, img_bytes, ".png")

        run_ingest(job_id, config, jm, sm)

        job = jm.get_job(job_id)

        # All views should have metadata
        for vn in ("front", "back", "left", "right", "top"):
            view = job["views"][vn]
            assert view["width"] == 1024
            assert view["height"] == 768
            assert view["file_size"] > 0
            assert view["status"] == ViewStatus.UPLOADED.value

    def test_raw_previews_generated(self, jm, sm, config):
        job_id = _create_mv_job(jm)

        for vn in ("front", "back", "left", "right", "top"):
            img_bytes = _create_test_image(width=1024, height=768)
            sm.save_view_upload(job_id, vn, img_bytes, ".png")

        run_ingest(job_id, config, jm, sm)

        # Check that raw preview thumbnails were created
        for vn in ("front", "back", "left", "right", "top"):
            preview_path = sm.get_view_preview_path(job_id, "raw", vn)
            assert preview_path is not None, f"Missing raw preview for {vn}"
            assert preview_path.exists()

            # Verify it's a valid image
            img = Image.open(preview_path)
            assert max(img.size) <= 512  # thumbnail max size

    def test_stage_progress_updated(self, jm, sm, config):
        job_id = _create_mv_job(jm)

        for vn in ("front", "back", "left", "right", "top"):
            sm.save_view_upload(job_id, vn, _create_test_image(), ".png")

        run_ingest(job_id, config, jm, sm)

        job = jm.get_job(job_id)
        assert job["stage_progress"] == 1.0


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestIngestErrors:
    def test_missing_view_raises(self, jm, sm, config):
        job_id = _create_mv_job(jm)

        # Only upload 4 of 5 views
        for vn in ("front", "back", "left", "right"):
            sm.save_view_upload(job_id, vn, _create_test_image(), ".png")

        with pytest.raises(FileNotFoundError, match="top"):
            run_ingest(job_id, config, jm, sm)

    def test_corrupt_image_raises(self, jm, sm, config):
        job_id = _create_mv_job(jm)

        # Upload valid images for 4 views
        for vn in ("front", "back", "left", "right"):
            sm.save_view_upload(job_id, vn, _create_test_image(), ".png")

        # Upload corrupt data for top
        sm.save_view_upload(job_id, "top", b"not-a-valid-image", ".png")

        with pytest.raises(ValueError, match="top"):
            run_ingest(job_id, config, jm, sm)


# ---------------------------------------------------------------------------
# Warning cases
# ---------------------------------------------------------------------------

class TestIngestWarnings:
    def test_low_resolution_warning(self, jm, sm, config):
        job_id = _create_mv_job(jm)

        # Upload 4 normal views and 1 tiny view
        for vn in ("front", "back", "left", "right"):
            sm.save_view_upload(job_id, vn, _create_test_image(512, 512), ".png")

        # Top view is very small
        sm.save_view_upload(job_id, "top", _create_test_image(128, 128), ".png")

        run_ingest(job_id, config, jm, sm)

        job = jm.get_job(job_id)
        assert any("low_resolution" in w for w in job["warnings"])

    def test_extreme_aspect_ratio_warning(self, jm, sm, config):
        job_id = _create_mv_job(jm)

        for vn in ("front", "back", "left", "right"):
            sm.save_view_upload(job_id, vn, _create_test_image(512, 512), ".png")

        # Top view has extreme aspect ratio
        sm.save_view_upload(job_id, "top", _create_test_image(100, 800), ".png")

        run_ingest(job_id, config, jm, sm)

        job = jm.get_job(job_id)
        assert any("aspect_ratio" in w for w in job["warnings"])

    def test_no_warnings_for_good_images(self, jm, sm, config):
        job_id = _create_mv_job(jm)

        for vn in ("front", "back", "left", "right", "top"):
            sm.save_view_upload(job_id, vn, _create_test_image(1024, 1024), ".png")

        run_ingest(job_id, config, jm, sm)

        job = jm.get_job(job_id)
        assert job["warnings"] == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestIngestEdgeCases:
    def test_rgba_image(self, jm, sm, config):
        """RGBA images should be handled correctly."""
        job_id = _create_mv_job(jm)

        for vn in ("front", "back", "left", "right", "top"):
            sm.save_view_upload(job_id, vn, _create_test_image(512, 512, "RGBA"), ".png")

        run_ingest(job_id, config, jm, sm)

        # Should complete without error
        job = jm.get_job(job_id)
        assert job["views"]["front"]["width"] == 512

    def test_jpeg_upload(self, jm, sm, config):
        """JPEG uploads should work."""
        job_id = _create_mv_job(jm)

        for vn in ("front", "back", "left", "right", "top"):
            img = Image.new("RGB", (512, 512), color=(128, 64, 32))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            sm.save_view_upload(job_id, vn, buf.getvalue(), ".jpg")

        run_ingest(job_id, config, jm, sm)

        job = jm.get_job(job_id)
        assert job["views"]["front"]["width"] == 512

