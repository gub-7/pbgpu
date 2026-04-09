"""
Tests for the validate_views (consistency) stage of the canonical multi-view pipeline.

Covers:
    - Segmentation confidence checks (hard fail + soft warning)
    - Sharpness validation
    - Silhouette area consistency (front/back, left/right pairs)
    - Left-right mirror plausibility
    - Color / lighting consistency
    - Top-view overlap plausibility
    - Validation result artifact saved
    - View status updates (VALIDATED vs FAILED)
    - Hard fail raises ValueError

All tests use synthetic RGBA images — no GPU dependencies needed.
"""

import io
import json
import pytest
from pathlib import Path
from unittest.mock import patch

import cv2
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
from pipelines.canonical_mv.consistency import (
    run_validate_views,
    _check_segmentation_confidence,
    _check_sharpness,
    _check_silhouette_area,
    _check_mirror_plausibility,
    _check_color_consistency,
    _check_top_view_plausibility,
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


def _make_rgba_image(
    width=256, height=256,
    subject_bbox=None,
    fg_color=(200, 100, 50),
    alpha_value=255,
) -> np.ndarray:
    """Create an RGBA image with a rectangular subject."""
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    if subject_bbox is None:
        subject_bbox = (50, 50, width - 100, height - 100)
    x, y, w, h = subject_bbox
    rgba[y:y+h, x:x+w, 0] = fg_color[0]
    rgba[y:y+h, x:x+w, 1] = fg_color[1]
    rgba[y:y+h, x:x+w, 2] = fg_color[2]
    rgba[y:y+h, x:x+w, 3] = alpha_value
    return rgba


def _save_segmented_preview(sm, job_id, view_name, rgba):
    """Save an RGBA array as a segmented preview."""
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    sm.save_view_preview(job_id, "segmented", view_name, buf.getvalue(), ".png")


def _setup_consistent_job(jm, sm, config):
    """
    Create a job with preprocess_metrics.json and segmented previews
    that represent a consistent, well-segmented set of views.
    """
    job_id = _create_mv_job(jm)

    # Create consistent views: similar-sized subjects, similar colors
    view_configs = {
        "front": {"bbox": (50, 30, 160, 200), "color": (180, 90, 45)},
        "back":  {"bbox": (55, 35, 155, 195), "color": (175, 85, 40)},
        "left":  {"bbox": (60, 40, 140, 180), "color": (185, 95, 50)},
        "right": {"bbox": (58, 38, 142, 182), "color": (183, 93, 48)},
        "top":   {"bbox": (70, 70, 120, 120), "color": (170, 100, 55)},
    }

    per_view_meta = {}
    for vn, vc in view_configs.items():
        rgba = _make_rgba_image(256, 256, vc["bbox"], vc["color"])
        _save_segmented_preview(sm, job_id, vn, rgba)

        # Compute FAR for preprocess_metrics
        x, y, w, h = vc["bbox"]
        far = (w * h) / (256 * 256)

        per_view_meta[vn] = {
            "bbox": list(vc["bbox"]),
            "centroid": [x + w // 2, y + h // 2],
            "foreground_area_ratio": far,
            "sharpness": 500.0,  # reasonable sharpness
            "segmentation_confidence": 0.95,
            "color_histogram_mean": list(vc["color"]),
        }

    sm.save_artifact_json(job_id, "preprocess_metrics.json", {
        "canvas_size": 256,
        "crop_side": 300,
        "per_view": per_view_meta,
    })

    return job_id, per_view_meta


# ---------------------------------------------------------------------------
# Unit tests: _check_segmentation_confidence
# ---------------------------------------------------------------------------

class TestCheckSegmentationConfidence:
    def test_good_confidence_no_issues(self):
        per_view = {
            "front": {"segmentation_confidence": 0.95, "foreground_area_ratio": 0.3},
            "back": {"segmentation_confidence": 0.90, "foreground_area_ratio": 0.3},
        }
        warnings, errors = [], []
        _check_segmentation_confidence(per_view, CanonicalMVConfig(), warnings, errors)
        assert len(warnings) == 0
        assert len(errors) == 0

    def test_low_confidence_warning(self):
        per_view = {
            "front": {"segmentation_confidence": 0.4, "foreground_area_ratio": 0.3},
        }
        warnings, errors = [], []
        _check_segmentation_confidence(per_view, CanonicalMVConfig(), warnings, errors)
        assert len(warnings) == 1
        assert "low_segmentation_confidence" in warnings[0]["code"]

    def test_catastrophic_confidence_error(self):
        per_view = {
            "front": {"segmentation_confidence": 0.1, "foreground_area_ratio": 0.3},
        }
        warnings, errors = [], []
        _check_segmentation_confidence(per_view, CanonicalMVConfig(), warnings, errors)
        assert len(errors) == 1
        assert "segmentation_failure" in errors[0]["code"]

    def test_no_foreground_error(self):
        per_view = {
            "top": {"segmentation_confidence": 0.95, "foreground_area_ratio": 0.005},
        }
        warnings, errors = [], []
        _check_segmentation_confidence(per_view, CanonicalMVConfig(), warnings, errors)
        assert len(errors) == 1
        assert "no_foreground" in errors[0]["code"]


# ---------------------------------------------------------------------------
# Unit tests: _check_sharpness
# ---------------------------------------------------------------------------

class TestCheckSharpness:
    def test_uniform_sharpness_no_warnings(self):
        per_view = {vn: {"sharpness": 500.0} for vn in CANONICAL_VIEW_ORDER}
        warnings = []
        _check_sharpness(per_view, warnings)
        assert len(warnings) == 0

    def test_one_blurry_view_warning(self):
        per_view = {
            "front": {"sharpness": 500.0},
            "back": {"sharpness": 480.0},
            "left": {"sharpness": 510.0},
            "right": {"sharpness": 490.0},
            "top": {"sharpness": 10.0},  # way below median
        }
        warnings = []
        _check_sharpness(per_view, warnings)
        assert len(warnings) == 1
        assert "top" in warnings[0]["code"]
        assert "blurry" in warnings[0]["code"]

    def test_all_blurry_no_warnings(self):
        """If all views are equally blurry, no warnings (no outlier)."""
        per_view = {vn: {"sharpness": 5.0} for vn in CANONICAL_VIEW_ORDER}
        warnings = []
        _check_sharpness(per_view, warnings)
        assert len(warnings) == 0


# ---------------------------------------------------------------------------
# Unit tests: _check_silhouette_area
# ---------------------------------------------------------------------------

class TestCheckSilhouetteArea:
    def test_consistent_areas_no_issues(self):
        per_view = {
            "front": {"foreground_area_ratio": 0.30},
            "back": {"foreground_area_ratio": 0.28},
            "left": {"foreground_area_ratio": 0.25},
            "right": {"foreground_area_ratio": 0.26},
        }
        warnings, errors = [], []
        config = CanonicalMVConfig(silhouette_area_tolerance=0.35)
        _check_silhouette_area(per_view, config, warnings, errors)
        assert len(warnings) == 0
        assert len(errors) == 0

    def test_large_mismatch_warning(self):
        per_view = {
            "front": {"foreground_area_ratio": 0.30},
            "back": {"foreground_area_ratio": 0.15},  # 50% smaller
        }
        warnings, errors = [], []
        config = CanonicalMVConfig(silhouette_area_tolerance=0.35)
        _check_silhouette_area(per_view, config, warnings, errors)
        assert len(warnings) == 1
        assert "area_mismatch" in warnings[0]["code"]

    def test_extreme_mismatch_error(self):
        per_view = {
            "front": {"foreground_area_ratio": 0.30},
            "back": {"foreground_area_ratio": 0.02},  # 93% smaller
        }
        warnings, errors = [], []
        config = CanonicalMVConfig(silhouette_area_tolerance=0.35)
        _check_silhouette_area(per_view, config, warnings, errors)
        assert len(errors) == 1
        assert "area_mismatch" in errors[0]["code"]

    def test_left_right_pair_checked(self):
        per_view = {
            "front": {"foreground_area_ratio": 0.30},
            "back": {"foreground_area_ratio": 0.30},
            "left": {"foreground_area_ratio": 0.30},
            "right": {"foreground_area_ratio": 0.05},  # mismatch
        }
        warnings, errors = [], []
        config = CanonicalMVConfig(silhouette_area_tolerance=0.35)
        _check_silhouette_area(per_view, config, warnings, errors)
        total_issues = len(warnings) + len(errors)
        assert total_issues >= 1
        # Should be about left/right, not front/back
        all_codes = [w["code"] for w in warnings] + [e["code"] for e in errors]
        assert any("left_right" in c for c in all_codes)


# ---------------------------------------------------------------------------
# Unit tests: _check_mirror_plausibility
# ---------------------------------------------------------------------------

class TestCheckMirrorPlausibility:
    def test_mirror_images_no_warning(self):
        """Left and flipped-right should have high IoU."""
        # Create a left-facing L shape
        left_rgba = np.zeros((256, 256, 4), dtype=np.uint8)
        left_rgba[50:200, 50:100, 3] = 255  # vertical bar
        left_rgba[150:200, 50:150, 3] = 255  # horizontal bar

        # Right should be mirror of left
        right_rgba = np.zeros((256, 256, 4), dtype=np.uint8)
        right_rgba[50:200, 156:206, 3] = 255  # mirrored vertical
        right_rgba[150:200, 106:206, 3] = 255  # mirrored horizontal

        view_images = {"left": left_rgba, "right": right_rgba}
        warnings = []
        _check_mirror_plausibility(view_images, warnings)
        assert len(warnings) == 0

    def test_non_mirror_images_warning(self):
        """Completely different shapes should trigger warning."""
        left_rgba = np.zeros((256, 256, 4), dtype=np.uint8)
        left_rgba[10:50, 10:50, 3] = 255  # small top-left

        right_rgba = np.zeros((256, 256, 4), dtype=np.uint8)
        right_rgba[200:250, 200:250, 3] = 255  # small bottom-right

        view_images = {"left": left_rgba, "right": right_rgba}
        warnings = []
        _check_mirror_plausibility(view_images, warnings)
        assert len(warnings) == 1
        assert "mirror_implausible" in warnings[0]["code"]

    def test_missing_views_no_crash(self):
        """If left or right is missing, should not crash."""
        view_images = {"front": np.zeros((256, 256, 4), dtype=np.uint8)}
        warnings = []
        _check_mirror_plausibility(view_images, warnings)
        assert len(warnings) == 0


# ---------------------------------------------------------------------------
# Unit tests: _check_color_consistency
# ---------------------------------------------------------------------------

class TestCheckColorConsistency:
    def test_consistent_colors_no_warnings(self):
        per_view = {
            "front": {"color_histogram_mean": [180, 90, 45]},
            "back": {"color_histogram_mean": [175, 85, 40]},
            "left": {"color_histogram_mean": [185, 95, 50]},
            "right": {"color_histogram_mean": [183, 93, 48]},
            "top": {"color_histogram_mean": [178, 88, 43]},
        }
        warnings = []
        _check_color_consistency(per_view, warnings)
        assert len(warnings) == 0

    def test_one_view_different_color_warning(self):
        per_view = {
            "front": {"color_histogram_mean": [180, 90, 45]},
            "back": {"color_histogram_mean": [175, 85, 40]},
            "left": {"color_histogram_mean": [50, 200, 200]},  # very different
            "right": {"color_histogram_mean": [183, 93, 48]},
            "top": {"color_histogram_mean": [178, 88, 43]},
        }
        warnings = []
        _check_color_consistency(per_view, warnings)
        assert len(warnings) >= 1
        assert any("left" in w["code"] for w in warnings)

    def test_missing_histogram_data(self):
        per_view = {
            "front": {"color_histogram_mean": [180, 90, 45]},
            "back": {},  # no histogram
        }
        warnings = []
        _check_color_consistency(per_view, warnings)
        # Should not crash, just skip views without data


# ---------------------------------------------------------------------------
# Unit tests: _check_top_view_plausibility
# ---------------------------------------------------------------------------

class TestCheckTopViewPlausibility:
    def test_reasonable_top_no_warning(self):
        per_view = {
            "front": {"foreground_area_ratio": 0.30},
            "top": {"foreground_area_ratio": 0.20},
        }
        view_images = {}  # not used in current implementation
        warnings = []
        _check_top_view_plausibility(per_view, view_images, warnings)
        assert len(warnings) == 0

    def test_tiny_top_warning(self):
        per_view = {
            "front": {"foreground_area_ratio": 0.30},
            "top": {"foreground_area_ratio": 0.01},  # < 10% of front
        }
        warnings = []
        _check_top_view_plausibility(per_view, {}, warnings)
        assert len(warnings) == 1
        assert "too_small" in warnings[0]["code"]

    def test_huge_top_warning(self):
        per_view = {
            "front": {"foreground_area_ratio": 0.10},
            "top": {"foreground_area_ratio": 0.50},  # 5x front
        }
        warnings = []
        _check_top_view_plausibility(per_view, {}, warnings)
        assert len(warnings) == 1
        assert "too_large" in warnings[0]["code"]

    def test_missing_front_no_crash(self):
        per_view = {
            "top": {"foreground_area_ratio": 0.20},
        }
        warnings = []
        _check_top_view_plausibility(per_view, {}, warnings)
        assert len(warnings) == 0


# ---------------------------------------------------------------------------
# Integration tests: run_validate_views
# ---------------------------------------------------------------------------

class TestRunValidateViews:
    def test_consistent_views_pass(self, jm, sm, config):
        job_id, _ = _setup_consistent_job(jm, sm, config)
        run_validate_views(job_id, config, jm, sm)

        # Should complete without raising
        job = jm.get_job(job_id)
        for vn in CANONICAL_VIEW_ORDER:
            assert job["views"][vn]["status"] == ViewStatus.VALIDATED.value

    def test_validation_results_artifact_saved(self, jm, sm, config):
        job_id, _ = _setup_consistent_job(jm, sm, config)
        run_validate_views(job_id, config, jm, sm)

        results = sm.load_artifact_json(job_id, "validation_results.json")
        assert results is not None
        assert results["status"] in ("passed", "warnings")
        assert "checks_run" in results
        assert len(results["checks_run"]) >= 5

    def test_stage_progress_reaches_1(self, jm, sm, config):
        job_id, _ = _setup_consistent_job(jm, sm, config)
        run_validate_views(job_id, config, jm, sm)

        job = jm.get_job(job_id)
        assert job["stage_progress"] == 1.0

    def test_catastrophic_failure_raises(self, jm, sm, config):
        """A view with no foreground should cause hard fail."""
        job_id = _create_mv_job(jm)

        # Save normal segmented previews for 4 views
        for vn in ("front", "back", "left", "right"):
            rgba = _make_rgba_image(256, 256, (50, 50, 160, 160))
            _save_segmented_preview(sm, job_id, vn, rgba)

        # Top view: completely empty (no foreground)
        empty_rgba = np.zeros((256, 256, 4), dtype=np.uint8)
        _save_segmented_preview(sm, job_id, "top", empty_rgba)

        # preprocess_metrics with no foreground for top
        per_view = {}
        for vn in ("front", "back", "left", "right"):
            per_view[vn] = {
                "foreground_area_ratio": 0.25,
                "segmentation_confidence": 0.95,
                "sharpness": 500.0,
                "color_histogram_mean": [180, 90, 45],
            }
        per_view["top"] = {
            "foreground_area_ratio": 0.0,
            "segmentation_confidence": 0.95,
            "sharpness": 500.0,
            "color_histogram_mean": [0, 0, 0],
        }
        sm.save_artifact_json(job_id, "preprocess_metrics.json", {
            "canvas_size": 256, "crop_side": 300, "per_view": per_view,
        })

        with pytest.raises(ValueError, match="validation failed"):
            run_validate_views(job_id, config, jm, sm)

        # Top view should be marked as failed
        job = jm.get_job(job_id)
        assert job["views"]["top"]["status"] == ViewStatus.FAILED.value

    def test_missing_preprocess_metrics_raises(self, jm, sm, config):
        job_id = _create_mv_job(jm)
        # No preprocess_metrics.json saved
        with pytest.raises(ValueError, match="preprocess_metrics"):
            run_validate_views(job_id, config, jm, sm)

    def test_warnings_stored_in_job(self, jm, sm, config):
        """A blurry view should produce a warning but not a hard fail."""
        job_id = _create_mv_job(jm)

        per_view = {}
        for vn in CANONICAL_VIEW_ORDER:
            rgba = _make_rgba_image(256, 256, (50, 50, 160, 160))
            _save_segmented_preview(sm, job_id, vn, rgba)
            per_view[vn] = {
                "foreground_area_ratio": 0.25,
                "segmentation_confidence": 0.95,
                "sharpness": 500.0 if vn != "top" else 5.0,  # top is blurry
                "color_histogram_mean": [180, 90, 45],
            }

        sm.save_artifact_json(job_id, "preprocess_metrics.json", {
            "canvas_size": 256, "crop_side": 300, "per_view": per_view,
        })

        run_validate_views(job_id, config, jm, sm)

        job = jm.get_job(job_id)
        assert any("blurry" in w for w in job["warnings"])
        # Should still pass (blurry is a warning, not an error)
        for vn in CANONICAL_VIEW_ORDER:
            assert job["views"][vn]["status"] == ViewStatus.VALIDATED.value

    def test_color_mismatch_warning(self, jm, sm, config):
        """A view with very different color should produce a warning."""
        job_id = _create_mv_job(jm)

        per_view = {}
        for vn in CANONICAL_VIEW_ORDER:
            if vn == "left":
                color = (50, 200, 200)  # very different
            else:
                color = (180, 90, 45)
            rgba = _make_rgba_image(256, 256, (50, 50, 160, 160), fg_color=color)
            _save_segmented_preview(sm, job_id, vn, rgba)
            per_view[vn] = {
                "foreground_area_ratio": 0.25,
                "segmentation_confidence": 0.95,
                "sharpness": 500.0,
                "color_histogram_mean": list(color),
            }

        sm.save_artifact_json(job_id, "preprocess_metrics.json", {
            "canvas_size": 256, "crop_side": 300, "per_view": per_view,
        })

        run_validate_views(job_id, config, jm, sm)

        job = jm.get_job(job_id)
        assert any("color_mismatch" in w for w in job["warnings"])

