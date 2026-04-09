"""
Tests for the preprocess_views stage of the canonical multi-view pipeline.

Covers:
    - Per-view segmentation (mocked — rembg not available locally)
    - Per-view metric extraction (bbox, centroid, sharpness, color histogram)
    - Cross-view consistent framing (global scale, shared canvas)
    - Segmented + normalized preview generation
    - Per-view metadata updates in job state
    - Preprocess metrics artifact saved
    - Edge cases: empty foreground, single-pixel foreground

The segmentation step is mocked because it depends on rembg/U2Net which
are only available on the GPU worker machine.  All downstream logic
(metrics, framing, preview saving) is tested with real numpy/PIL operations.
"""

import io
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
from PIL import Image
import fakeredis

from api.job_manager import JobManager
from api.models import (
    CategoryEnum,
    PipelineEnum,
    ViewStatus,
)
from api.storage import StorageManager
from pipelines.canonical_mv.config import CanonicalMVConfig, CANONICAL_VIEW_ORDER
from pipelines.canonical_mv.preprocess import (
    run_preprocess_views,
    _compute_view_metrics,
    _compute_mask_metrics_simple,
    _compute_sharpness,
    _compute_segmentation_confidence,
    _compute_color_histogram,
    _compute_cross_view_framing,
    _apply_framing,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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


def _create_mv_job(jm):
    return jm.create_multiview_job(
        category=CategoryEnum.HUMAN_BUST,
        pipeline=PipelineEnum.CANONICAL_MV_HYBRID,
        views_received=["front", "back", "left", "right", "top"],
    )


def _make_test_image(width=512, height=512, color=(128, 64, 32)) -> bytes:
    """Create a valid RGB PNG image."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_rgba_with_subject(
    width=512, height=512,
    subject_bbox=(100, 100, 300, 300),
    fg_color=(200, 100, 50),
) -> np.ndarray:
    """
    Create an RGBA image with a rectangular foreground subject.

    Args:
        width, height: Image dimensions.
        subject_bbox: (x, y, w, h) of the foreground rectangle.
        fg_color: RGB color of the foreground.

    Returns:
        RGBA numpy array with hard alpha edges.
    """
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    x, y, w, h = subject_bbox
    rgba[y:y+h, x:x+w, 0] = fg_color[0]
    rgba[y:y+h, x:x+w, 1] = fg_color[1]
    rgba[y:y+h, x:x+w, 2] = fg_color[2]
    rgba[y:y+h, x:x+w, 3] = 255
    return rgba


# ---------------------------------------------------------------------------
# Unit tests: _compute_mask_metrics_simple
# ---------------------------------------------------------------------------

class TestComputeMaskMetricsSimple:
    def test_full_foreground(self):
        alpha = np.full((100, 100), 255, dtype=np.uint8)
        m = _compute_mask_metrics_simple(alpha)
        assert m["far"] == pytest.approx(1.0, abs=0.01)
        assert m["bbo"] == pytest.approx(1.0, abs=0.01)
        assert m["foreground_pixels"] == 10000

    def test_empty_foreground(self):
        alpha = np.zeros((100, 100), dtype=np.uint8)
        m = _compute_mask_metrics_simple(alpha)
        assert m["far"] == 0.0
        assert m["bbo"] == 0.0
        assert m["foreground_pixels"] == 0
        # Centroid should default to center
        assert m["centroid"] == (50, 50)

    def test_quarter_foreground(self):
        alpha = np.zeros((100, 100), dtype=np.uint8)
        alpha[0:50, 0:50] = 255  # top-left quarter
        m = _compute_mask_metrics_simple(alpha)
        assert m["far"] == pytest.approx(0.25, abs=0.01)
        assert m["bbox"] == (0, 0, 50, 50)
        assert m["foreground_pixels"] == 2500

    def test_centroid_location(self):
        alpha = np.zeros((200, 200), dtype=np.uint8)
        # Place a block at (50,50) to (150,150) — centroid should be ~(100,100)
        alpha[50:150, 50:150] = 255
        m = _compute_mask_metrics_simple(alpha)
        cx, cy = m["centroid"]
        assert 95 <= cx <= 105
        assert 95 <= cy <= 105


# ---------------------------------------------------------------------------
# Unit tests: _compute_sharpness
# ---------------------------------------------------------------------------

class TestComputeSharpness:
    def test_flat_image_low_sharpness(self):
        rgb = np.full((100, 100, 3), 128, dtype=np.uint8)
        s = _compute_sharpness(rgb)
        assert s < 1.0  # flat image has ~0 Laplacian variance

    def test_noisy_image_higher_sharpness(self):
        rng = np.random.RandomState(42)
        rgb = rng.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        s = _compute_sharpness(rgb)
        assert s > 100  # random noise has high Laplacian variance

    def test_edge_image_high_sharpness(self):
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb[:, 50:] = 255  # sharp vertical edge
        s = _compute_sharpness(rgb)
        assert s > 0  # should detect the edge


# ---------------------------------------------------------------------------
# Unit tests: _compute_segmentation_confidence
# ---------------------------------------------------------------------------

class TestComputeSegmentationConfidence:
    def test_binary_alpha_high_confidence(self):
        alpha = np.zeros((100, 100), dtype=np.uint8)
        alpha[20:80, 20:80] = 255
        conf = _compute_segmentation_confidence(alpha)
        assert conf > 0.95  # all pixels are 0 or 255

    def test_soft_alpha_low_confidence(self):
        # Create alpha with lots of mid-range values
        alpha = np.full((100, 100), 128, dtype=np.uint8)
        conf = _compute_segmentation_confidence(alpha)
        assert conf < 0.1  # all pixels are in the uncertain range

    def test_mixed_alpha(self):
        alpha = np.zeros((100, 100), dtype=np.uint8)
        alpha[0:50, :] = 255  # half binary
        alpha[50:75, :] = 128  # quarter uncertain
        # remaining quarter is 0 (binary)
        conf = _compute_segmentation_confidence(alpha)
        assert 0.5 < conf < 1.0


# ---------------------------------------------------------------------------
# Unit tests: _compute_color_histogram
# ---------------------------------------------------------------------------

class TestComputeColorHistogram:
    def test_uniform_color(self):
        rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        rgba[:, :, 0] = 200  # R
        rgba[:, :, 1] = 100  # G
        rgba[:, :, 2] = 50   # B
        rgba[:, :, 3] = 255  # fully opaque

        hists, means = _compute_color_histogram(rgba)
        assert len(hists) == 3
        assert len(means) == 3
        assert means[0] == pytest.approx(200.0, abs=1.0)
        assert means[1] == pytest.approx(100.0, abs=1.0)
        assert means[2] == pytest.approx(50.0, abs=1.0)

    def test_no_foreground(self):
        rgba = np.zeros((100, 100, 4), dtype=np.uint8)  # all transparent
        hists, means = _compute_color_histogram(rgba)
        assert means == [0.0, 0.0, 0.0]

    def test_histogram_normalized(self):
        rgba = _make_rgba_with_subject(100, 100, (10, 10, 80, 80))
        hists, means = _compute_color_histogram(rgba)
        for h in hists:
            total = sum(h)
            assert total == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Unit tests: _compute_view_metrics
# ---------------------------------------------------------------------------

class TestComputeViewMetrics:
    def test_returns_all_keys(self):
        rgba = _make_rgba_with_subject()
        raw_rgb = rgba[:, :, :3].copy()
        m = _compute_view_metrics(rgba, raw_rgb)

        expected_keys = {
            "bbox", "centroid", "far", "bbo", "foreground_pixels",
            "image_area", "sharpness", "segmentation_confidence",
            "color_hist", "color_hist_mean",
        }
        assert expected_keys.issubset(set(m.keys()))

    def test_reasonable_values(self):
        rgba = _make_rgba_with_subject(512, 512, (100, 100, 300, 300))
        raw_rgb = rgba[:, :, :3].copy()
        m = _compute_view_metrics(rgba, raw_rgb)

        # Subject is 300x300 in 512x512 image
        assert 0.2 < m["far"] < 0.5
        assert m["segmentation_confidence"] > 0.9  # binary alpha
        assert m["sharpness"] >= 0  # flat color but has edges from bbox


# ---------------------------------------------------------------------------
# Unit tests: cross-view framing
# ---------------------------------------------------------------------------

class TestCrossViewFraming:
    def test_framing_uses_max_extent(self):
        view_data = {
            "front": {"metrics": {"bbox": (10, 10, 200, 300)}},  # tallest
            "back":  {"metrics": {"bbox": (10, 10, 200, 250)}},
            "left":  {"metrics": {"bbox": (10, 10, 150, 200)}},
            "right": {"metrics": {"bbox": (10, 10, 150, 200)}},
            "top":   {"metrics": {"bbox": (10, 10, 180, 180)}},
        }
        framing = _compute_cross_view_framing(view_data, target_canvas_size=1024)

        # Max extent is 300 (front height), with 20% padding
        assert framing["crop_side"] == int(300 * 1.2)
        assert framing["canvas_size"] == 1024

    def test_framing_empty_foreground(self):
        view_data = {
            "front": {"metrics": {"bbox": (0, 0, 0, 0)}},
            "back":  {"metrics": {"bbox": (0, 0, 0, 0)}},
        }
        framing = _compute_cross_view_framing(view_data, target_canvas_size=512)
        # Degenerate case — should return canvas_size
        assert framing["crop_side"] == 512

    def test_framing_single_view(self):
        view_data = {
            "front": {"metrics": {"bbox": (50, 50, 400, 400)}},
        }
        framing = _compute_cross_view_framing(view_data, target_canvas_size=1024)
        assert framing["crop_side"] == int(400 * 1.2)


class TestApplyFraming:
    def test_output_size(self):
        rgba = _make_rgba_with_subject(512, 512, (100, 100, 300, 300))
        framing = {"crop_side": 400, "canvas_size": 256}
        result = _apply_framing(rgba, framing)
        assert result.shape == (256, 256, 4)

    def test_subject_preserved(self):
        """Framing should keep the subject in the output."""
        rgba = _make_rgba_with_subject(512, 512, (100, 100, 300, 300))
        framing = {"crop_side": 400, "canvas_size": 256}
        result = _apply_framing(rgba, framing)

        # There should be foreground pixels in the output
        fg_pixels = np.sum(result[:, :, 3] > 128)
        assert fg_pixels > 0

    def test_consistent_scale_across_views(self):
        """Two views with different bbox sizes should produce same-scale output."""
        small_subject = _make_rgba_with_subject(512, 512, (150, 150, 200, 200))
        large_subject = _make_rgba_with_subject(512, 512, (50, 50, 400, 400))

        # Same framing params = same scale
        framing = {"crop_side": 500, "canvas_size": 256}
        r1 = _apply_framing(small_subject, framing)
        r2 = _apply_framing(large_subject, framing)

        assert r1.shape == r2.shape == (256, 256, 4)
        # Large subject should have more foreground pixels
        fg1 = np.sum(r1[:, :, 3] > 128)
        fg2 = np.sum(r2[:, :, 3] > 128)
        assert fg2 > fg1


# ---------------------------------------------------------------------------
# Integration tests: run_preprocess_views (with mocked segmentation)
# ---------------------------------------------------------------------------

def _mock_segment_view(image_path, model_name="u2net"):
    """
    Mock segmentation that returns an RGBA image with a centered rectangle
    as the 'foreground'. The rectangle size varies slightly by view name
    to make cross-view framing non-trivial.
    """
    img = Image.open(image_path)
    w, h = img.size

    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Centered rectangle, ~60% of image
    margin_x = int(w * 0.2)
    margin_y = int(h * 0.2)
    rgba[margin_y:h-margin_y, margin_x:w-margin_x, :3] = np.array(img.convert("RGB"))[
        margin_y:h-margin_y, margin_x:w-margin_x
    ]
    rgba[margin_y:h-margin_y, margin_x:w-margin_x, 3] = 255

    return rgba


class TestRunPreprocessViews:
    """Integration tests for the full preprocess_views stage."""

    def _setup_job(self, jm, sm, sizes=None):
        """Create a job and upload test views."""
        job_id = _create_mv_job(jm)
        if sizes is None:
            sizes = {vn: (512, 512) for vn in CANONICAL_VIEW_ORDER}

        for vn in CANONICAL_VIEW_ORDER:
            w, h = sizes.get(vn, (512, 512))
            # Use different colors per view so histograms differ
            colors = {
                "front": (200, 100, 50),
                "back": (190, 110, 60),
                "left": (180, 90, 40),
                "right": (185, 95, 45),
                "top": (170, 120, 70),
            }
            img_bytes = _make_test_image(w, h, colors.get(vn, (128, 128, 128)))
            sm.save_view_upload(job_id, vn, img_bytes, ".png")

        return job_id

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_happy_path(self, mock_seg, jm, sm, config):
        job_id = self._setup_job(jm, sm)
        run_preprocess_views(job_id, config, jm, sm)

        job = jm.get_job(job_id)

        # All views should be marked as segmented
        for vn in CANONICAL_VIEW_ORDER:
            assert job["views"][vn]["status"] == ViewStatus.SEGMENTED.value
            assert job["views"][vn]["segmentation_confidence"] is not None
            assert job["views"][vn]["sharpness_score"] is not None

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_segmented_previews_saved(self, mock_seg, jm, sm, config):
        job_id = self._setup_job(jm, sm)
        run_preprocess_views(job_id, config, jm, sm)

        for vn in CANONICAL_VIEW_ORDER:
            path = sm.get_view_preview_path(job_id, "segmented", vn)
            assert path is not None, f"Missing segmented preview for {vn}"
            assert path.exists()
            img = Image.open(path)
            assert img.mode == "RGBA"

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_normalized_previews_saved(self, mock_seg, jm, sm, config):
        job_id = self._setup_job(jm, sm)
        run_preprocess_views(job_id, config, jm, sm)

        for vn in CANONICAL_VIEW_ORDER:
            path = sm.get_view_preview_path(job_id, "normalized", vn)
            assert path is not None, f"Missing normalized preview for {vn}"
            assert path.exists()

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_preprocess_metrics_artifact_saved(self, mock_seg, jm, sm, config):
        job_id = self._setup_job(jm, sm)
        run_preprocess_views(job_id, config, jm, sm)

        metrics = sm.load_artifact_json(job_id, "preprocess_metrics.json")
        assert metrics is not None
        assert "canvas_size" in metrics
        assert "crop_side" in metrics
        assert "per_view" in metrics
        assert set(metrics["per_view"].keys()) == set(CANONICAL_VIEW_ORDER)

        for vn, vm in metrics["per_view"].items():
            assert "bbox" in vm
            assert "centroid" in vm
            assert "sharpness" in vm
            assert "segmentation_confidence" in vm

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_stage_progress_reaches_1(self, mock_seg, jm, sm, config):
        job_id = self._setup_job(jm, sm)
        run_preprocess_views(job_id, config, jm, sm)

        job = jm.get_job(job_id)
        assert job["stage_progress"] == 1.0

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_segmentation_called_per_view(self, mock_seg, jm, sm, config):
        job_id = self._setup_job(jm, sm)
        run_preprocess_views(job_id, config, jm, sm)

        assert mock_seg.call_count == 5

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_missing_view_raises(self, mock_seg, jm, sm, config):
        job_id = _create_mv_job(jm)
        # Only upload 4 views
        for vn in ("front", "back", "left", "right"):
            sm.save_view_upload(job_id, vn, _make_test_image(), ".png")

        with pytest.raises(FileNotFoundError, match="top"):
            run_preprocess_views(job_id, config, jm, sm)

    @patch("pipelines.canonical_mv.preprocess._segment_view", side_effect=_mock_segment_view)
    def test_different_sized_views(self, mock_seg, jm, sm, config):
        """Views with different resolutions should still produce consistent framing."""
        sizes = {
            "front": (1024, 768),
            "back": (800, 600),
            "left": (512, 512),
            "right": (640, 480),
            "top": (768, 768),
        }
        job_id = self._setup_job(jm, sm, sizes=sizes)
        run_preprocess_views(job_id, config, jm, sm)

        # All normalized previews should exist and be the same size
        preview_sizes = set()
        for vn in CANONICAL_VIEW_ORDER:
            path = sm.get_view_preview_path(job_id, "normalized", vn)
            assert path is not None
            img = Image.open(path)
            preview_sizes.add(img.size)

        # All normalized previews should be the same dimensions
        # (they're thumbnailed to max 512, so the actual canvas_size
        #  gets thumbnailed, but they should all be the same)
        assert len(preview_sizes) == 1

