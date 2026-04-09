"""
Tests for the QA scoring stage (Phase 6 — qa).

Covers:
    PER-VIEW METRICS (unit tests):
        - compute_silhouette_iou: IoU between rendered and target masks
        - compute_masked_psnr: PSNR within mask region
        - compute_masked_ssim: SSIM within mask region
        - compute_per_view_metrics: full per-view metric computation

    MESH METRICS (unit tests):
        - compute_mesh_metrics: topology and quality metrics

    SYMMETRY DEVIATION (unit tests):
        - compute_symmetry_deviation: bilateral symmetry score

    QUALITY SCORE (unit tests):
        - compute_quality_score: weighted aggregation, warnings, retry suggestions

    STAGE RUNNER (integration tests):
        - run_qa: happy path with completed mesh
        - run_qa: fallback to coarse mesh when refined/completed absent
        - Missing mesh raises ValueError
        - Artifacts saved correctly (metrics.json)
        - Stage progress reaches 1.0
        - Warnings populated for low-quality results
        - Quality score in valid range

    EDGE CASES:
        - Empty masks
        - Identical images
        - Zero-area masks
        - Large mesh handling
        - Missing camera rig (graceful degradation)

All tests use synthetic meshes and masks — no GPU dependencies.
"""

import io
import json
import math
import pytest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import fakeredis

from api.job_manager import JobManager
from api.models import (
    CategoryEnum,
    PipelineEnum,
)
from api.storage import StorageManager
from pipelines.canonical_mv.config import (
    CanonicalMVConfig,
    CANONICAL_VIEW_ORDER,
)
from pipelines.canonical_mv.camera_init import (
    CameraRig,
    build_canonical_rig,
)
from pipelines.canonical_mv.coarse_recon import save_mesh_ply
from pipelines.canonical_mv.refine import (
    MeshState,
    compute_vertex_normals,
    load_mesh_ply,
    render_silhouettes,
)
from pipelines.canonical_mv.qa import (
    # Per-view metrics
    compute_silhouette_iou,
    compute_masked_psnr,
    compute_masked_ssim,
    compute_per_view_metrics,
    # Mesh metrics
    compute_mesh_metrics,
    # Symmetry
    compute_symmetry_deviation,
    # Quality score
    compute_quality_score,
    # Stage runner
    run_qa,
    # Constants
    WEIGHT_SILHOUETTE_IOU,
    WEIGHT_MESH_QUALITY,
    WEIGHT_TEXTURE_QUALITY,
    WEIGHT_SYMMETRY,
    WEIGHT_COMPLETENESS,
    WARN_SILHOUETTE_IOU,
    WARN_COMPONENT_COUNT,
    WARN_SELF_INTERSECTIONS,
    WARN_TEXTURE_COVERAGE,
    WARN_SYMMETRY_DEVIATION,
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


@pytest.fixture
def default_rig(config):
    return build_canonical_rig(config, (256, 256))


# ---------------------------------------------------------------------------
# Test mesh generators
# ---------------------------------------------------------------------------

def _make_cube_mesh(size=0.5, center=(0, 0, 0)):
    """Create a simple cube mesh with corrected CCW winding."""
    cx, cy, cz = center
    s = size / 2
    vertices = np.array([
        [cx - s, cy - s, cz - s],
        [cx + s, cy - s, cz - s],
        [cx + s, cy + s, cz - s],
        [cx - s, cy + s, cz - s],
        [cx - s, cy - s, cz + s],
        [cx + s, cy - s, cz + s],
        [cx + s, cy + s, cz + s],
        [cx - s, cy + s, cz + s],
    ], dtype=np.float64)

    faces = np.array([
        [0, 3, 2], [0, 2, 1],
        [4, 5, 6], [4, 6, 7],
        [3, 7, 6], [3, 6, 2],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [0, 4, 7], [0, 7, 3],
    ], dtype=np.int32)

    return vertices, faces


def _create_mv_job(jm):
    return jm.create_multiview_job(
        category=CategoryEnum.HUMAN_BUST,
        pipeline=PipelineEnum.CANONICAL_MV_HYBRID,
        views_received=["front", "back", "left", "right", "top"],
    )


# ===========================================================================
# SILHOUETTE IoU
# ===========================================================================


class TestComputeSilhouetteIou:
    def test_identical_masks_iou_one(self):
        """Identical masks should give IoU = 1.0."""
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:50, 10:50] = 255
        iou = compute_silhouette_iou(mask, mask)
        assert iou == pytest.approx(1.0)

    def test_no_overlap_iou_zero(self):
        """Non-overlapping masks should give IoU = 0.0."""
        r = np.zeros((64, 64), dtype=np.uint8)
        r[0:20, 0:20] = 255
        t = np.zeros((64, 64), dtype=np.uint8)
        t[40:60, 40:60] = 255
        iou = compute_silhouette_iou(r, t)
        assert iou == pytest.approx(0.0)

    def test_partial_overlap(self):
        """Partially overlapping masks should give 0 < IoU < 1."""
        r = np.zeros((64, 64), dtype=np.uint8)
        r[10:40, 10:40] = 255
        t = np.zeros((64, 64), dtype=np.uint8)
        t[20:50, 20:50] = 255
        iou = compute_silhouette_iou(r, t)
        assert 0.0 < iou < 1.0

    def test_empty_masks_iou_one(self):
        """Both empty masks should give IoU = 1.0 (no union, no intersection)."""
        r = np.zeros((64, 64), dtype=np.uint8)
        t = np.zeros((64, 64), dtype=np.uint8)
        iou = compute_silhouette_iou(r, t)
        assert iou == pytest.approx(1.0)

    def test_one_empty_one_full_iou_zero(self):
        """One empty, one full mask should give IoU = 0.0."""
        r = np.zeros((64, 64), dtype=np.uint8)
        t = np.full((64, 64), 255, dtype=np.uint8)
        iou = compute_silhouette_iou(r, t)
        assert iou == pytest.approx(0.0)

    def test_different_sizes_resized(self):
        """Masks of different sizes should be handled (target resized)."""
        r = np.zeros((64, 64), dtype=np.uint8)
        r[10:50, 10:50] = 255
        t = np.zeros((128, 128), dtype=np.uint8)
        t[20:100, 20:100] = 255
        iou = compute_silhouette_iou(r, t)
        # Should not crash, and IoU should be a valid number
        assert 0.0 <= iou <= 1.0

    def test_iou_symmetric(self):
        """IoU should be symmetric: IoU(A, B) == IoU(B, A)."""
        r = np.zeros((64, 64), dtype=np.uint8)
        r[10:40, 10:40] = 255
        t = np.zeros((64, 64), dtype=np.uint8)
        t[20:50, 20:50] = 255
        iou_rt = compute_silhouette_iou(r, t)
        iou_tr = compute_silhouette_iou(t, r)
        assert iou_rt == pytest.approx(iou_tr, abs=1e-10)


# ===========================================================================
# MASKED PSNR
# ===========================================================================


class TestComputeMaskedPsnr:
    def test_identical_images_high_psnr(self):
        """Identical images should give very high PSNR."""
        img = np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8)
        mask = np.full((64, 64), 255, dtype=np.uint8)
        psnr = compute_masked_psnr(img, img, mask)
        assert psnr == pytest.approx(100.0)

    def test_different_images_lower_psnr(self):
        """Different images should have lower PSNR."""
        rng = np.random.RandomState(42)
        img1 = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        img2 = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        mask = np.full((64, 64), 255, dtype=np.uint8)
        psnr = compute_masked_psnr(img1, img2, mask)
        assert 0.0 < psnr < 100.0

    def test_empty_mask_returns_zero(self):
        """Empty mask should return 0.0."""
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        psnr = compute_masked_psnr(img, img, mask)
        assert psnr == pytest.approx(0.0)

    def test_psnr_positive(self):
        """PSNR should always be positive for non-empty masks."""
        img1 = np.full((64, 64, 3), 100, dtype=np.uint8)
        img2 = np.full((64, 64, 3), 200, dtype=np.uint8)
        mask = np.full((64, 64), 255, dtype=np.uint8)
        psnr = compute_masked_psnr(img1, img2, mask)
        assert psnr > 0.0

    def test_partial_mask(self):
        """PSNR computed only within masked region."""
        img1 = np.zeros((64, 64, 3), dtype=np.uint8)
        img2 = np.zeros((64, 64, 3), dtype=np.uint8)
        # Make them different only outside the mask
        img2[50:, 50:] = 255
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[0:40, 0:40] = 255
        # Within the mask, both are zero → identical → high PSNR
        psnr = compute_masked_psnr(img1, img2, mask)
        assert psnr == pytest.approx(100.0)


# ===========================================================================
# MASKED SSIM
# ===========================================================================


class TestComputeMaskedSsim:
    def test_identical_images_ssim_one(self):
        """Identical images should give SSIM close to 1.0."""
        rng = np.random.RandomState(42)
        img = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        mask = np.full((64, 64), 255, dtype=np.uint8)
        ssim = compute_masked_ssim(img, img, mask)
        assert ssim == pytest.approx(1.0, abs=0.01)

    def test_different_images_lower_ssim(self):
        """Very different images should have lower SSIM."""
        img1 = np.zeros((64, 64, 3), dtype=np.uint8)
        img2 = np.full((64, 64, 3), 255, dtype=np.uint8)
        mask = np.full((64, 64), 255, dtype=np.uint8)
        ssim = compute_masked_ssim(img1, img2, mask)
        assert ssim < 0.5

    def test_ssim_in_range(self):
        """SSIM should be in [0, 1]."""
        rng = np.random.RandomState(42)
        img1 = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        img2 = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        mask = np.full((64, 64), 255, dtype=np.uint8)
        ssim = compute_masked_ssim(img1, img2, mask)
        assert 0.0 <= ssim <= 1.0

    def test_empty_mask_returns_zero(self):
        """Empty mask should return 0.0."""
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        ssim = compute_masked_ssim(img, img, mask)
        assert ssim == pytest.approx(0.0)


# ===========================================================================
# PER-VIEW METRICS
# ===========================================================================


class TestComputePerViewMetrics:
    def test_structure(self, default_rig):
        """Per-view metrics should have all expected keys for each view."""
        v, f = _make_cube_mesh(size=0.3)
        image_size = (256, 256)
        masks = render_silhouettes(v, f, default_rig, image_size)
        # Create dummy images
        images = {
            vn: np.full((256, 256, 3), 128, dtype=np.uint8)
            for vn in CANONICAL_VIEW_ORDER
        }

        per_view = compute_per_view_metrics(
            v, f, default_rig, images, masks, image_size,
        )

        for vn in CANONICAL_VIEW_ORDER:
            assert vn in per_view
            m = per_view[vn]
            assert "silhouette_iou" in m
            assert "masked_psnr" in m
            assert "masked_ssim" in m

    def test_self_comparison_high_iou(self, default_rig):
        """Rendering the mesh and comparing to its own silhouette should give high IoU."""
        v, f = _make_cube_mesh(size=0.3)
        image_size = (256, 256)
        masks = render_silhouettes(v, f, default_rig, image_size)
        images = {
            vn: np.full((256, 256, 3), 128, dtype=np.uint8)
            for vn in CANONICAL_VIEW_ORDER
        }

        per_view = compute_per_view_metrics(
            v, f, default_rig, images, masks, image_size,
        )

        for vn in CANONICAL_VIEW_ORDER:
            iou = per_view[vn]["silhouette_iou"]
            if iou is not None:
                assert iou == pytest.approx(1.0, abs=0.01), \
                    f"Self-comparison IoU for {vn} should be ~1.0, got {iou}"

    def test_empty_mesh_zero_iou(self, default_rig):
        """Empty mesh should produce zero IoU against non-empty masks."""
        v = np.zeros((0, 3), dtype=np.float64)
        f = np.zeros((0, 3), dtype=np.int32)
        image_size = (256, 256)
        # Non-empty target masks
        masks = {
            vn: np.full((256, 256), 255, dtype=np.uint8)
            for vn in CANONICAL_VIEW_ORDER
        }
        images = {
            vn: np.full((256, 256, 3), 128, dtype=np.uint8)
            for vn in CANONICAL_VIEW_ORDER
        }

        per_view = compute_per_view_metrics(
            v, f, default_rig, images, masks, image_size,
        )

        for vn in CANONICAL_VIEW_ORDER:
            iou = per_view[vn].get("silhouette_iou")
            if iou is not None:
                assert iou == pytest.approx(0.0)

    def test_no_masks_none_iou(self, default_rig):
        """Missing masks should result in None IoU."""
        v, f = _make_cube_mesh(size=0.3)
        image_size = (256, 256)

        per_view = compute_per_view_metrics(
            v, f, default_rig, {}, {}, image_size,
        )

        for vn in CANONICAL_VIEW_ORDER:
            assert per_view[vn]["silhouette_iou"] is None


# ===========================================================================
# MESH METRICS
# ===========================================================================


class TestComputeMeshMetrics:
    def test_cube_metrics(self):
        """Cube mesh should have correct topology metrics."""
        v, f = _make_cube_mesh()
        metrics = compute_mesh_metrics(v, f)

        assert metrics["n_vertices"] == 8
        assert metrics["n_faces"] == 12
        assert metrics["n_components"] == 1
        assert metrics["is_manifold"] is True
        assert metrics["is_watertight"] is True
        assert metrics["self_intersections"] >= 0
        assert metrics["bounding_box_diagonal"] > 0

    def test_empty_mesh(self):
        """Empty mesh should return valid metrics."""
        v = np.zeros((0, 3), dtype=np.float64)
        f = np.zeros((0, 3), dtype=np.int32)
        metrics = compute_mesh_metrics(v, f)

        assert metrics["n_vertices"] == 0
        assert metrics["n_faces"] == 0
        assert metrics["is_manifold"] is True
        assert metrics["bounding_box_diagonal"] == 0.0

    def test_single_triangle_not_watertight(self):
        """Single triangle should not be watertight."""
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        f = np.array([[0, 1, 2]], dtype=np.int32)
        metrics = compute_mesh_metrics(v, f)

        assert metrics["n_components"] == 1
        assert metrics["is_watertight"] is False
        assert metrics["boundary_edges"] == 3

    def test_bounding_box(self):
        """Bounding box should match mesh extents."""
        v, f = _make_cube_mesh(size=1.0, center=(0, 0, 0))
        metrics = compute_mesh_metrics(v, f)

        bb_min = metrics["bounding_box_min"]
        bb_max = metrics["bounding_box_max"]
        assert bb_min[0] == pytest.approx(-0.5)
        assert bb_max[0] == pytest.approx(0.5)

    def test_metrics_has_all_keys(self):
        """Metrics should include all expected keys."""
        v, f = _make_cube_mesh()
        metrics = compute_mesh_metrics(v, f)

        expected_keys = [
            "n_vertices", "n_faces", "n_components",
            "is_manifold", "is_watertight",
            "boundary_edges", "non_manifold_edges",
            "self_intersections",
            "bounding_box_min", "bounding_box_max", "bounding_box_diagonal",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"


# ===========================================================================
# SYMMETRY DEVIATION
# ===========================================================================


class TestComputeSymmetryDeviation:
    def test_symmetric_mesh_low_deviation(self):
        """A symmetric mesh should have low deviation."""
        v, _ = _make_cube_mesh()
        dev = compute_symmetry_deviation(v, axis=0)
        assert dev < 0.05, f"Symmetric cube should have low deviation, got {dev}"

    def test_asymmetric_mesh_higher_deviation(self):
        """An asymmetric mesh should have higher deviation."""
        v, _ = _make_cube_mesh()
        # Shift the right side vertices
        v_asym = v.copy()
        v_asym[v_asym[:, 0] > 0, 0] += 0.5
        dev = compute_symmetry_deviation(v_asym, axis=0)

        dev_sym = compute_symmetry_deviation(v, axis=0)
        assert dev > dev_sym, "Asymmetric mesh should have higher deviation"

    def test_single_vertex_zero_deviation(self):
        """Single vertex mesh should return 0.0."""
        v = np.array([[0, 0, 0]], dtype=np.float64)
        dev = compute_symmetry_deviation(v, axis=0)
        assert dev == pytest.approx(0.0)

    def test_empty_mesh_zero_deviation(self):
        """Empty mesh should return 0.0."""
        v = np.zeros((0, 3), dtype=np.float64)
        dev = compute_symmetry_deviation(v, axis=0)
        assert dev == pytest.approx(0.0)

    def test_deviation_non_negative(self):
        """Deviation should always be non-negative."""
        rng = np.random.RandomState(42)
        v = rng.randn(50, 3)
        dev = compute_symmetry_deviation(v, axis=0)
        assert dev >= 0.0

    def test_different_axes(self):
        """Symmetry deviation should work for different axes."""
        v, _ = _make_cube_mesh()
        for axis in [0, 1, 2]:
            dev = compute_symmetry_deviation(v, axis=axis)
            assert dev >= 0.0
            # Cube is symmetric along all axes
            assert dev < 0.1


# ===========================================================================
# QUALITY SCORE
# ===========================================================================


class TestComputeQualityScore:
    def test_perfect_score(self):
        """Perfect metrics should give high quality score."""
        per_view = {
            vn: {"silhouette_iou": 1.0, "masked_psnr": 50.0, "masked_ssim": 1.0}
            for vn in CANONICAL_VIEW_ORDER
        }
        mesh_metrics = {
            "n_components": 1,
            "self_intersections": 0,
            "is_manifold": True,
        }
        texture_metrics = {"coverage_fraction": 0.8}
        symmetry_deviation = 0.0
        completion_coverage = 1.0

        score, warnings, retry = compute_quality_score(
            per_view, mesh_metrics, texture_metrics,
            symmetry_deviation, completion_coverage,
        )

        assert score > 0.8
        assert len(warnings) == 0
        assert len(retry) == 0

    def test_poor_score(self):
        """Poor metrics should give low quality score."""
        per_view = {
            vn: {"silhouette_iou": 0.1}
            for vn in CANONICAL_VIEW_ORDER
        }
        mesh_metrics = {
            "n_components": 10,
            "self_intersections": 100,
            "is_manifold": False,
        }
        texture_metrics = {"coverage_fraction": 0.05}
        symmetry_deviation = 0.5
        completion_coverage = 0.0

        score, warnings, retry = compute_quality_score(
            per_view, mesh_metrics, texture_metrics,
            symmetry_deviation, completion_coverage,
        )

        assert score < 0.5
        assert len(warnings) > 0

    def test_score_in_range(self):
        """Quality score should always be in [0, 1]."""
        per_view = {
            vn: {"silhouette_iou": 0.5}
            for vn in CANONICAL_VIEW_ORDER
        }
        mesh_metrics = {"n_components": 1, "self_intersections": 0, "is_manifold": True}

        score, _, _ = compute_quality_score(
            per_view, mesh_metrics, None, 0.1, 0.5,
        )

        assert 0.0 <= score <= 1.0

    def test_low_iou_generates_warning(self):
        """Low silhouette IoU should generate warnings."""
        per_view = {
            "front": {"silhouette_iou": 0.2},
            "back": {"silhouette_iou": 0.9},
            "left": {"silhouette_iou": 0.3},
            "right": {"silhouette_iou": 0.9},
            "top": {"silhouette_iou": 0.9},
        }
        mesh_metrics = {"n_components": 1, "self_intersections": 0, "is_manifold": True}

        _, warnings, retry = compute_quality_score(
            per_view, mesh_metrics, None, 0.0, 1.0,
        )

        iou_warnings = [w for w in warnings if w["code"] == "low_silhouette_iou"]
        assert len(iou_warnings) >= 2  # front and left
        assert len(retry) >= 2

    def test_high_component_count_warning(self):
        """Many components should generate warning."""
        per_view = {
            vn: {"silhouette_iou": 0.9}
            for vn in CANONICAL_VIEW_ORDER
        }
        mesh_metrics = {
            "n_components": 10,
            "self_intersections": 0,
            "is_manifold": True,
        }

        _, warnings, _ = compute_quality_score(
            per_view, mesh_metrics, None, 0.0, 1.0,
        )

        comp_warnings = [w for w in warnings if w["code"] == "high_component_count"]
        assert len(comp_warnings) == 1

    def test_self_intersection_warning(self):
        """Many self-intersections should generate warning."""
        per_view = {
            vn: {"silhouette_iou": 0.9}
            for vn in CANONICAL_VIEW_ORDER
        }
        mesh_metrics = {
            "n_components": 1,
            "self_intersections": 100,
            "is_manifold": True,
        }

        _, warnings, _ = compute_quality_score(
            per_view, mesh_metrics, None, 0.0, 1.0,
        )

        si_warnings = [w for w in warnings if w["code"] == "self_intersections"]
        assert len(si_warnings) == 1

    def test_non_manifold_warning(self):
        """Non-manifold mesh should generate warning."""
        per_view = {
            vn: {"silhouette_iou": 0.9}
            for vn in CANONICAL_VIEW_ORDER
        }
        mesh_metrics = {
            "n_components": 1,
            "self_intersections": 0,
            "is_manifold": False,
        }

        _, warnings, _ = compute_quality_score(
            per_view, mesh_metrics, None, 0.0, 1.0,
        )

        nm_warnings = [w for w in warnings if w["code"] == "non_manifold"]
        assert len(nm_warnings) == 1

    def test_low_texture_coverage_warning(self):
        """Low texture coverage should generate warning."""
        per_view = {
            vn: {"silhouette_iou": 0.9}
            for vn in CANONICAL_VIEW_ORDER
        }
        mesh_metrics = {"n_components": 1, "self_intersections": 0, "is_manifold": True}
        texture_metrics = {"coverage_fraction": 0.1}

        _, warnings, _ = compute_quality_score(
            per_view, mesh_metrics, texture_metrics, 0.0, 1.0,
        )

        tex_warnings = [w for w in warnings if w["code"] == "low_texture_coverage"]
        assert len(tex_warnings) == 1

    def test_high_symmetry_deviation_warning(self):
        """High symmetry deviation should generate warning."""
        per_view = {
            vn: {"silhouette_iou": 0.9}
            for vn in CANONICAL_VIEW_ORDER
        }
        mesh_metrics = {"n_components": 1, "self_intersections": 0, "is_manifold": True}

        _, warnings, _ = compute_quality_score(
            per_view, mesh_metrics, None, 0.3, 1.0,
        )

        sym_warnings = [w for w in warnings if w["code"] == "high_symmetry_deviation"]
        assert len(sym_warnings) == 1

    def test_no_texture_metrics_default(self):
        """None texture_metrics should use default score."""
        per_view = {
            vn: {"silhouette_iou": 0.9}
            for vn in CANONICAL_VIEW_ORDER
        }
        mesh_metrics = {"n_components": 1, "self_intersections": 0, "is_manifold": True}

        score, _, _ = compute_quality_score(
            per_view, mesh_metrics, None, 0.0, 1.0,
        )

        # Should still produce a valid score
        assert 0.0 <= score <= 1.0

    def test_no_completion_coverage_default(self):
        """None completion_coverage should use default score."""
        per_view = {
            vn: {"silhouette_iou": 0.9}
            for vn in CANONICAL_VIEW_ORDER
        }
        mesh_metrics = {"n_components": 1, "self_intersections": 0, "is_manifold": True}

        score, _, _ = compute_quality_score(
            per_view, mesh_metrics, None, 0.0, None,
        )

        assert 0.0 <= score <= 1.0

    def test_empty_per_view_metrics(self):
        """Empty per-view metrics should not crash."""
        mesh_metrics = {"n_components": 1, "self_intersections": 0, "is_manifold": True}

        score, _, _ = compute_quality_score(
            {}, mesh_metrics, None, 0.0, 1.0,
        )

        assert 0.0 <= score <= 1.0

    def test_weights_sum_to_one(self):
        """Quality score weights should sum to 1.0."""
        total = (
            WEIGHT_SILHOUETTE_IOU
            + WEIGHT_MESH_QUALITY
            + WEIGHT_TEXTURE_QUALITY
            + WEIGHT_SYMMETRY
            + WEIGHT_COMPLETENESS
        )
        assert total == pytest.approx(1.0)


# ===========================================================================
# STAGE RUNNER INTEGRATION TESTS
# ===========================================================================


class TestRunQa:
    """Integration tests for the full QA stage runner."""

    def _setup_job_with_artifacts(
        self, jm, sm, config,
        mesh_artifact="completed_mesh.ply",
        include_camera_rig=True,
        include_views=True,
    ):
        """
        Create a job and save required artifacts from previous stages.
        """
        job_id = _create_mv_job(jm)

        # Build and save camera rig
        if include_camera_rig:
            rig = build_canonical_rig(config, (256, 256))
            sm.save_artifact_json(job_id, "camera_init.json", rig.to_dict())

        # Build and save mesh
        vertices, faces = _make_cube_mesh(size=0.3)
        normals = compute_vertex_normals(vertices, faces)
        mesh_path = sm.get_artifact_dir(job_id) / mesh_artifact
        save_mesh_ply(str(mesh_path), vertices, faces, normals)

        # Save segmented view previews
        if include_views and include_camera_rig:
            rig = build_canonical_rig(config, (256, 256))
            image_size = (256, 256)
            masks = render_silhouettes(vertices, faces, rig, image_size)

            for vn in CANONICAL_VIEW_ORDER:
                mask = masks.get(vn, np.zeros((256, 256), dtype=np.uint8))
                rgba = np.zeros((256, 256, 4), dtype=np.uint8)
                rgba[:, :, 0] = 128
                rgba[:, :, 1] = 64
                rgba[:, :, 2] = 32
                rgba[:, :, 3] = mask
                img = Image.fromarray(rgba, mode="RGBA")
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                sm.save_view_preview(job_id, "segmented", vn, buf.getvalue(), ".png")

        return job_id

    def test_happy_path_completed_mesh(self, jm, sm, config):
        """QA should complete with completed mesh and save metrics.json."""
        job_id = self._setup_job_with_artifacts(jm, sm, config)

        run_qa(job_id, config, jm, sm)

        metrics = sm.load_metrics(job_id)
        assert metrics is not None
        assert "quality_score" in metrics
        assert "per_view_metrics" in metrics
        assert "mesh_metrics" in metrics
        assert "warnings" in metrics
        assert "recommended_retry" in metrics

    def test_fallback_to_refined_mesh(self, jm, sm, config):
        """QA should fall back to refined mesh when completed is absent."""
        job_id = self._setup_job_with_artifacts(
            jm, sm, config, mesh_artifact="refined_mesh.ply",
        )

        run_qa(job_id, config, jm, sm)

        metrics = sm.load_metrics(job_id)
        assert metrics is not None
        assert metrics["mesh_source"] == "refined"

    def test_fallback_to_coarse_mesh(self, jm, sm, config):
        """QA should fall back to coarse mesh when others are absent."""
        job_id = self._setup_job_with_artifacts(
            jm, sm, config, mesh_artifact="coarse_visual_hull_mesh.ply",
        )

        run_qa(job_id, config, jm, sm)

        metrics = sm.load_metrics(job_id)
        assert metrics is not None
        assert metrics["mesh_source"] == "coarse"

    def test_missing_mesh_raises(self, jm, sm, config):
        """Missing all meshes should raise ValueError."""
        job_id = _create_mv_job(jm)
        with pytest.raises(ValueError, match="mesh"):
            run_qa(job_id, config, jm, sm)

    def test_stage_progress_reaches_one(self, jm, sm, config):
        """Stage progress should reach 1.0 on completion."""
        job_id = self._setup_job_with_artifacts(jm, sm, config)

        run_qa(job_id, config, jm, sm)

        job = jm.get_job(job_id)
        assert job["stage_progress"] == 1.0

    def test_quality_score_in_range(self, jm, sm, config):
        """Quality score should be in [0, 1]."""
        job_id = self._setup_job_with_artifacts(jm, sm, config)

        run_qa(job_id, config, jm, sm)

        metrics = sm.load_metrics(job_id)
        assert 0.0 <= metrics["quality_score"] <= 1.0

    def test_per_view_metrics_present(self, jm, sm, config):
        """Per-view metrics should include all canonical views."""
        job_id = self._setup_job_with_artifacts(jm, sm, config)

        run_qa(job_id, config, jm, sm)

        metrics = sm.load_metrics(job_id)
        per_view = metrics["per_view_metrics"]
        for vn in CANONICAL_VIEW_ORDER:
            assert vn in per_view

    def test_mesh_metrics_present(self, jm, sm, config):
        """Mesh metrics should have all expected keys."""
        job_id = self._setup_job_with_artifacts(jm, sm, config)

        run_qa(job_id, config, jm, sm)

        metrics = sm.load_metrics(job_id)
        mesh_m = metrics["mesh_metrics"]
        assert "n_vertices" in mesh_m
        assert "n_faces" in mesh_m
        assert "n_components" in mesh_m
        assert "is_manifold" in mesh_m

    def test_symmetry_deviation_present(self, jm, sm, config):
        """Symmetry deviation should be computed when symmetry_prior is True."""
        job_id = self._setup_job_with_artifacts(jm, sm, config)

        run_qa(job_id, config, jm, sm)

        metrics = sm.load_metrics(job_id)
        assert "symmetry_deviation" in metrics
        assert metrics["symmetry_deviation"] >= 0.0

    def test_symmetry_disabled(self, jm, sm):
        """With symmetry_prior=False, symmetry deviation should be 0."""
        config = CanonicalMVConfig(symmetry_prior=False)
        job_id = self._setup_job_with_artifacts(jm, sm, config)

        run_qa(job_id, config, jm, sm)

        metrics = sm.load_metrics(job_id)
        assert metrics["symmetry_deviation"] == 0.0

    def test_no_camera_rig_graceful(self, jm, sm, config):
        """Missing camera rig should still produce metrics (no per-view)."""
        job_id = self._setup_job_with_artifacts(
            jm, sm, config, include_camera_rig=False, include_views=False,
        )

        run_qa(job_id, config, jm, sm)

        metrics = sm.load_metrics(job_id)
        assert metrics is not None
        # Per-view metrics should be empty (no rig)
        assert len(metrics["per_view_metrics"]) == 0

    def test_no_views_graceful(self, jm, sm, config):
        """Missing views should still produce metrics (empty per-view IoU)."""
        job_id = self._setup_job_with_artifacts(
            jm, sm, config, include_views=False,
        )

        run_qa(job_id, config, jm, sm)

        metrics = sm.load_metrics(job_id)
        assert metrics is not None
        # Per-view metrics should exist but IoU may be None for missing views
        assert "per_view_metrics" in metrics

    def test_warnings_set_on_job(self, jm, sm, config):
        """Warnings should be set on the job metadata."""
        job_id = self._setup_job_with_artifacts(jm, sm, config)

        run_qa(job_id, config, jm, sm)

        job = jm.get_job(job_id)
        # warnings field should exist (may be empty for good quality)
        assert "warnings" in job

    def test_completion_metrics_loaded(self, jm, sm, config):
        """If completion_metrics.json exists, it should be used."""
        job_id = self._setup_job_with_artifacts(jm, sm, config)

        # Save fake completion metrics
        sm.save_artifact_json(job_id, "completion_metrics.json", {
            "coverage": {"weak_fraction": 0.2},
            "provider_used": "symmetry",
        })

        run_qa(job_id, config, jm, sm)

        metrics = sm.load_metrics(job_id)
        assert metrics["completion_coverage"] == pytest.approx(0.8)

    def test_texture_metrics_loaded(self, jm, sm, config):
        """If texture_metrics.json exists, it should be used."""
        job_id = self._setup_job_with_artifacts(jm, sm, config)

        # Save fake texture metrics
        sm.save_artifact_json(job_id, "texture_metrics.json", {
            "coverage_fraction": 0.6,
            "texture_resolution": 2048,
        })

        run_qa(job_id, config, jm, sm)

        metrics = sm.load_metrics(job_id)
        assert metrics["texture_metrics"]["coverage_fraction"] == 0.6

    def test_mesh_source_recorded(self, jm, sm, config):
        """Metrics should record which mesh was used."""
        job_id = self._setup_job_with_artifacts(jm, sm, config)

        run_qa(job_id, config, jm, sm)

        metrics = sm.load_metrics(job_id)
        assert "mesh_source" in metrics
        assert metrics["mesh_source"] == "completed"

    def test_deterministic_output(self, jm, sm, config):
        """Running QA twice with same input should produce same score."""
        job_id_1 = self._setup_job_with_artifacts(jm, sm, config)
        job_id_2 = self._setup_job_with_artifacts(jm, sm, config)

        run_qa(job_id_1, config, jm, sm)
        run_qa(job_id_2, config, jm, sm)

        m1 = sm.load_metrics(job_id_1)
        m2 = sm.load_metrics(job_id_2)

        assert m1["quality_score"] == pytest.approx(m2["quality_score"], abs=1e-6)


# ===========================================================================
# EDGE CASES
# ===========================================================================


class TestEdgeCases:
    def test_all_zero_masks(self):
        """All-zero masks should give IoU=1.0 (both empty)."""
        r = np.zeros((64, 64), dtype=np.uint8)
        t = np.zeros((64, 64), dtype=np.uint8)
        assert compute_silhouette_iou(r, t) == pytest.approx(1.0)

    def test_all_full_masks(self):
        """All-255 masks should give IoU=1.0."""
        r = np.full((64, 64), 255, dtype=np.uint8)
        t = np.full((64, 64), 255, dtype=np.uint8)
        assert compute_silhouette_iou(r, t) == pytest.approx(1.0)

    def test_psnr_with_constant_images(self):
        """Identical constant images should give high PSNR."""
        img = np.full((64, 64, 3), 100, dtype=np.uint8)
        mask = np.full((64, 64), 255, dtype=np.uint8)
        psnr = compute_masked_psnr(img, img, mask)
        assert psnr == pytest.approx(100.0)

    def test_ssim_with_noise(self):
        """SSIM should be lower for noisy comparisons."""
        rng = np.random.RandomState(42)
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        noisy = img.copy()
        noise = rng.randint(-50, 50, img.shape).astype(np.int16)
        noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        mask = np.full((64, 64), 255, dtype=np.uint8)

        ssim = compute_masked_ssim(img, noisy, mask)
        assert ssim < 1.0

    def test_symmetry_deviation_point_cloud(self):
        """Symmetric point cloud should have low deviation."""
        v = np.array([
            [-1, 0, 0], [1, 0, 0],
            [-1, 1, 0], [1, 1, 0],
            [-1, -1, 0], [1, -1, 0],
        ], dtype=np.float64)
        dev = compute_symmetry_deviation(v, axis=0)
        assert dev < 0.01

    def test_quality_score_all_none_iou(self):
        """Per-view metrics with all None IoU should not crash."""
        per_view = {
            vn: {"silhouette_iou": None}
            for vn in CANONICAL_VIEW_ORDER
        }
        mesh_metrics = {"n_components": 1, "self_intersections": 0, "is_manifold": True}

        score, _, _ = compute_quality_score(
            per_view, mesh_metrics, None, 0.0, 1.0,
        )

        assert 0.0 <= score <= 1.0

    def test_mesh_metrics_two_components(self):
        """Mesh with two components should report n_components=2."""
        v1, f1 = _make_cube_mesh(size=0.2, center=(0, 0, 0))
        v2, f2 = _make_cube_mesh(size=0.1, center=(5, 5, 5))
        v = np.vstack([v1, v2])
        f = np.vstack([f1, f2 + len(v1)])
        metrics = compute_mesh_metrics(v, f)
        assert metrics["n_components"] == 2

    def test_large_mesh_symmetry(self):
        """Symmetry deviation should handle larger meshes without crashing."""
        rng = np.random.RandomState(42)
        # Symmetric point cloud
        n = 3000
        left = rng.randn(n, 3)
        left[:, 0] = -np.abs(left[:, 0])  # all negative X
        right = left.copy()
        right[:, 0] = -right[:, 0]  # mirror
        v = np.vstack([left, right])

        dev = compute_symmetry_deviation(v, axis=0)
        assert dev < 0.01
        assert dev >= 0.0

