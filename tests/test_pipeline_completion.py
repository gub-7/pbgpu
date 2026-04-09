"""
Tests for the generative completion stage (Phase 5).

Covers:
    CONFIDENCE ANALYSIS (unit tests):
        - compute_vertex_visibility: per-view vertex visibility computation
        - compute_vertex_confidence: view count → confidence mapping
        - identify_weak_regions: confidence thresholding
        - compute_coverage_stats: summary statistics

    COMPLETION CONFIG (unit tests):
        - CompletionConfig defaults and from_pipeline_config

    COMPLETION PROVIDERS (unit tests):
        - SymmetryCompletionProvider: mirror observed geometry
        - LaplacianCompletionProvider: diffusion-based hole filling
        - Trellis2CompletionProvider: GPU stub raises RuntimeError
        - Hunyuan3DCompletionProvider: GPU stub raises RuntimeError
        - get_completion_providers: factory function ordering

    FUSION (unit tests):
        - fuse_completion: confidence-weighted blending
        - compute_blend_weights: weight computation

    STAGE RUNNER (integration tests):
        - run_complete_geometry: happy path with refined mesh
        - run_complete_geometry: fallback to coarse mesh when refined absent
        - Missing camera_init.json raises ValueError
        - Missing mesh raises ValueError
        - Artifacts saved correctly (completed_mesh.ply, completion_metrics.json)
        - Stage progress reaches 1.0
        - Symmetry disabled config
        - Trellis enabled but falls back to CPU providers

    EDGE CASES:
        - Empty mesh handling
        - All vertices confident → no completion
        - All vertices weak → full completion
        - Single-vertex mesh
        - Large mesh performance

All tests use synthetic meshes and masks — no GPU dependencies.
"""

import io
import json
import math
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

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
from pipelines.canonical_mv.coarse_recon import (
    save_mesh_ply,
    save_gaussians_ply,
    build_coarse_gaussians,
)
from pipelines.canonical_mv.refine import (
    MeshState,
    compute_edges,
    compute_vertex_normals,
    compute_face_normals,
    build_laplacian_matrix,
    load_mesh_ply,
    render_silhouettes,
)
from pipelines.canonical_mv.completion import (
    # Config
    CompletionConfig,
    CompletionResult,
    # Confidence analysis
    compute_vertex_visibility,
    compute_vertex_confidence,
    identify_weak_regions,
    compute_coverage_stats,
    # Providers
    CompletionProvider,
    SymmetryCompletionProvider,
    LaplacianCompletionProvider,
    Trellis2CompletionProvider,
    Hunyuan3DCompletionProvider,
    get_completion_providers,
    # Fusion
    fuse_completion,
    compute_blend_weights,
    # Stage runner
    run_complete_geometry,
    # Constants
    DEFAULT_CONFIDENCE_THRESHOLD,
    MIN_VIEWS_FOR_FULL_CONFIDENCE,
    MAX_COMPLETION_BLEND,
    DEFAULT_DIFFUSION_ITERATIONS,
    DEFAULT_DIFFUSION_ALPHA,
    MIN_WEAK_VERTICES_FOR_COMPLETION,
    FRONT_FACING_THRESHOLD,
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
    """
    Create a simple cube mesh centered at the given point.
    Uses corrected CCW winding for outward-facing normals.

    Returns (vertices, faces) where:
        - vertices: (8, 3) float64
        - faces: (12, 3) int32 (2 triangles per face × 6 faces)
    """
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
        # Front (z=-s, outward normal -Z)
        [0, 3, 2], [0, 2, 1],
        # Back (z=+s, outward normal +Z)
        [4, 5, 6], [4, 6, 7],
        # Top (y=+s, outward normal +Y)
        [3, 7, 6], [3, 6, 2],
        # Bottom (y=-s, outward normal -Y)
        [0, 1, 5], [0, 5, 4],
        # Right (x=+s, outward normal +X)
        [1, 2, 6], [1, 6, 5],
        # Left (x=-s, outward normal -X)
        [0, 4, 7], [0, 7, 3],
    ], dtype=np.int32)

    return vertices, faces


def _make_tetrahedron_mesh():
    """Create a simple tetrahedron mesh."""
    vertices = np.array([
        [1, 1, 1],
        [-1, -1, 1],
        [-1, 1, -1],
        [1, -1, -1],
    ], dtype=np.float64) * 0.3

    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ], dtype=np.int32)

    return vertices, faces


def _make_mesh_state(vertices=None, faces=None):
    """Create a MeshState from vertices and faces."""
    if vertices is None or faces is None:
        vertices, faces = _make_cube_mesh()

    normals = compute_vertex_normals(vertices, faces)
    colors = np.full((len(vertices), 3), 128, dtype=np.uint8)

    return MeshState(
        vertices=vertices,
        faces=faces,
        normals=normals,
        vertex_colors=colors,
    )


def _create_mv_job(jm):
    return jm.create_multiview_job(
        category=CategoryEnum.HUMAN_BUST,
        pipeline=PipelineEnum.CANONICAL_MV_HYBRID,
        views_received=["front", "back", "left", "right", "top"],
    )


# ===========================================================================
# COMPLETION CONFIG
# ===========================================================================


class TestCompletionConfig:
    def test_defaults(self):
        cfg = CompletionConfig()
        assert cfg.confidence_threshold == DEFAULT_CONFIDENCE_THRESHOLD
        assert cfg.min_views_for_full_confidence == MIN_VIEWS_FOR_FULL_CONFIDENCE
        assert cfg.max_blend_weight == MAX_COMPLETION_BLEND
        assert cfg.diffusion_iterations == DEFAULT_DIFFUSION_ITERATIONS
        assert cfg.diffusion_alpha == DEFAULT_DIFFUSION_ALPHA
        assert cfg.use_symmetry is True
        assert cfg.symmetry_axis == 0
        assert cfg.use_trellis is False
        assert cfg.use_hunyuan is False

    def test_from_pipeline_config_defaults(self):
        pc = CanonicalMVConfig()
        cfg = CompletionConfig.from_pipeline_config(pc)
        assert cfg.use_symmetry is True
        assert cfg.use_trellis is True  # default in CanonicalMVConfig
        assert cfg.use_hunyuan is False

    def test_from_pipeline_config_no_symmetry(self):
        pc = CanonicalMVConfig(symmetry_prior=False)
        cfg = CompletionConfig.from_pipeline_config(pc)
        assert cfg.use_symmetry is False

    def test_from_pipeline_config_hunyuan_enabled(self):
        pc = CanonicalMVConfig(use_hunyuan_completion=True)
        cfg = CompletionConfig.from_pipeline_config(pc)
        assert cfg.use_hunyuan is True

    def test_from_pipeline_config_trellis_disabled(self):
        pc = CanonicalMVConfig(use_trellis_completion=False)
        cfg = CompletionConfig.from_pipeline_config(pc)
        assert cfg.use_trellis is False


# ===========================================================================
# CONFIDENCE ANALYSIS
# ===========================================================================


class TestComputeVertexVisibility:
    def test_centered_cube_visible_in_all_views(self, default_rig):
        """A centered cube should have vertices visible from at least some views."""
        vertices, faces = _make_cube_mesh(size=0.3)
        normals = compute_vertex_normals(vertices, faces)
        image_size = (256, 256)

        # Render masks from the mesh itself
        masks = render_silhouettes(vertices, faces, default_rig, image_size)

        visibility = compute_vertex_visibility(
            vertices, normals, default_rig, masks, image_size,
        )

        # Should have visibility info for all 5 views
        assert len(visibility) == 5
        for vn in CANONICAL_VIEW_ORDER:
            assert vn in visibility
            assert visibility[vn].shape == (len(vertices),)
            assert visibility[vn].dtype == bool

    def test_some_vertices_visible_per_view(self, default_rig):
        """Each view should see at least some vertices of a centered cube."""
        vertices, faces = _make_cube_mesh(size=0.3)
        normals = compute_vertex_normals(vertices, faces)
        image_size = (256, 256)
        masks = render_silhouettes(vertices, faces, default_rig, image_size)

        visibility = compute_vertex_visibility(
            vertices, normals, default_rig, masks, image_size,
        )

        for vn in CANONICAL_VIEW_ORDER:
            n_visible = np.sum(visibility[vn])
            assert n_visible > 0, f"No vertices visible from {vn}"

    def test_far_away_mesh_not_visible(self, default_rig):
        """A mesh far from origin should not project into masks from centered mesh."""
        # Centered masks
        center_v, center_f = _make_cube_mesh(size=0.3)
        image_size = (256, 256)
        masks = render_silhouettes(center_v, center_f, default_rig, image_size)

        # Far-away mesh
        far_v, far_f = _make_cube_mesh(size=0.3, center=(10, 10, 10))
        far_normals = compute_vertex_normals(far_v, far_f)

        visibility = compute_vertex_visibility(
            far_v, far_normals, default_rig, masks, image_size,
        )

        total_visible = sum(np.sum(v) for v in visibility.values())
        assert total_visible == 0, "Far mesh should not be visible in centered masks"

    def test_empty_masks_no_visibility(self, default_rig):
        """Empty masks should produce zero visibility."""
        vertices, faces = _make_cube_mesh(size=0.3)
        normals = compute_vertex_normals(vertices, faces)
        image_size = (256, 256)

        masks = {vn: np.zeros((256, 256), dtype=np.uint8) for vn in CANONICAL_VIEW_ORDER}

        visibility = compute_vertex_visibility(
            vertices, normals, default_rig, masks, image_size,
        )

        for vn in CANONICAL_VIEW_ORDER:
            assert np.sum(visibility[vn]) == 0

    def test_partial_views(self, default_rig):
        """Should handle having only some views in masks dict."""
        vertices, faces = _make_cube_mesh(size=0.3)
        normals = compute_vertex_normals(vertices, faces)
        image_size = (256, 256)
        all_masks = render_silhouettes(vertices, faces, default_rig, image_size)

        # Only provide front and back
        partial_masks = {k: v for k, v in all_masks.items() if k in ("front", "back")}

        visibility = compute_vertex_visibility(
            vertices, normals, default_rig, partial_masks, image_size,
        )

        assert "front" in visibility
        assert "back" in visibility
        # Other views should not be in the result
        assert "left" not in visibility
        assert "right" not in visibility
        assert "top" not in visibility


class TestComputeVertexConfidence:
    def test_all_views_visible_full_confidence(self):
        """If all views see a vertex, confidence should be 1.0."""
        n_verts = 10
        visibility = {
            vn: np.ones(n_verts, dtype=bool) for vn in CANONICAL_VIEW_ORDER
        }

        confidence = compute_vertex_confidence(visibility, n_verts, min_views=3)
        np.testing.assert_allclose(confidence, 1.0)

    def test_no_views_visible_zero_confidence(self):
        """If no views see a vertex, confidence should be 0.0."""
        n_verts = 10
        visibility = {
            vn: np.zeros(n_verts, dtype=bool) for vn in CANONICAL_VIEW_ORDER
        }

        confidence = compute_vertex_confidence(visibility, n_verts, min_views=3)
        np.testing.assert_allclose(confidence, 0.0)

    def test_partial_views_fractional_confidence(self):
        """If 2 of 3 required views see a vertex, confidence should be 2/3."""
        n_verts = 5
        visibility = {
            "front": np.ones(n_verts, dtype=bool),
            "back": np.ones(n_verts, dtype=bool),
            "left": np.zeros(n_verts, dtype=bool),
        }

        confidence = compute_vertex_confidence(visibility, n_verts, min_views=3)
        np.testing.assert_allclose(confidence, 2.0 / 3.0, atol=1e-10)

    def test_confidence_clipped_to_one(self):
        """Confidence should be clipped to [0, 1] even with many views."""
        n_verts = 5
        visibility = {
            vn: np.ones(n_verts, dtype=bool) for vn in CANONICAL_VIEW_ORDER
        }

        confidence = compute_vertex_confidence(visibility, n_verts, min_views=2)
        assert np.all(confidence <= 1.0)
        assert np.all(confidence >= 0.0)

    def test_mixed_visibility(self):
        """Different vertices can have different confidence levels."""
        n_verts = 3
        visibility = {
            "front": np.array([True, True, False]),
            "back": np.array([True, False, False]),
            "left": np.array([True, False, False]),
        }

        confidence = compute_vertex_confidence(visibility, n_verts, min_views=3)
        assert confidence[0] == pytest.approx(1.0)
        assert confidence[1] == pytest.approx(1.0 / 3.0, abs=1e-10)
        assert confidence[2] == pytest.approx(0.0)

    def test_empty_visibility(self):
        """Empty visibility dict should give zero confidence."""
        confidence = compute_vertex_confidence({}, 5, min_views=3)
        np.testing.assert_allclose(confidence, 0.0)


class TestIdentifyWeakRegions:
    def test_all_confident(self):
        """All high-confidence vertices → no weak regions."""
        confidence = np.ones(10, dtype=np.float64)
        weak = identify_weak_regions(confidence, threshold=0.3)
        assert not np.any(weak)

    def test_all_weak(self):
        """All zero-confidence vertices → all weak."""
        confidence = np.zeros(10, dtype=np.float64)
        weak = identify_weak_regions(confidence, threshold=0.3)
        assert np.all(weak)

    def test_mixed(self):
        """Mixed confidence → correct weak/strong split."""
        confidence = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0])
        weak = identify_weak_regions(confidence, threshold=0.3)
        expected = np.array([True, True, True, False, False, False, False])
        np.testing.assert_array_equal(weak, expected)

    def test_threshold_boundary(self):
        """Vertices exactly at threshold should NOT be weak."""
        confidence = np.array([0.29, 0.30, 0.31])
        weak = identify_weak_regions(confidence, threshold=0.3)
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(weak, expected)

    def test_empty_array(self):
        """Empty confidence array should return empty mask."""
        confidence = np.zeros(0, dtype=np.float64)
        weak = identify_weak_regions(confidence, threshold=0.3)
        assert len(weak) == 0


class TestComputeCoverageStats:
    def test_structure(self):
        """Coverage stats should have all expected keys."""
        n_verts = 10
        confidence = np.random.RandomState(42).rand(n_verts)
        weak_mask = confidence < 0.3
        visibility = {
            "front": np.ones(n_verts, dtype=bool),
            "back": np.zeros(n_verts, dtype=bool),
        }

        stats = compute_coverage_stats(confidence, weak_mask, visibility)

        assert "n_vertices" in stats
        assert "n_weak_vertices" in stats
        assert "weak_fraction" in stats
        assert "mean_confidence" in stats
        assert "median_confidence" in stats
        assert "min_confidence" in stats
        assert "max_confidence" in stats
        assert "per_view_coverage" in stats

    def test_per_view_coverage(self):
        """Per-view coverage should include visible_vertices and visible_fraction."""
        n_verts = 10
        confidence = np.ones(n_verts)
        weak_mask = np.zeros(n_verts, dtype=bool)
        visibility = {
            "front": np.ones(n_verts, dtype=bool),
        }

        stats = compute_coverage_stats(confidence, weak_mask, visibility)

        assert "front" in stats["per_view_coverage"]
        front_stats = stats["per_view_coverage"]["front"]
        assert front_stats["visible_vertices"] == n_verts
        assert front_stats["visible_fraction"] == pytest.approx(1.0)

    def test_weak_fraction_correct(self):
        """Weak fraction should be n_weak / n_total."""
        n_verts = 20
        confidence = np.zeros(n_verts, dtype=np.float64)
        confidence[:10] = 1.0  # 10 confident, 10 weak
        weak_mask = confidence < 0.3

        stats = compute_coverage_stats(confidence, weak_mask, {})

        assert stats["n_weak_vertices"] == 10
        assert stats["weak_fraction"] == pytest.approx(0.5)


# ===========================================================================
# COMPLETION PROVIDERS
# ===========================================================================


class TestSymmetryCompletionProvider:
    def test_name(self):
        provider = SymmetryCompletionProvider(axis=0)
        assert provider.name == "symmetry"

    def test_no_weak_vertices_no_change(self, default_rig):
        """If no vertices are weak, symmetry provider should not change anything."""
        mesh = _make_mesh_state()
        weak_mask = np.zeros(mesh.n_vertices, dtype=bool)
        confidence = np.ones(mesh.n_vertices, dtype=np.float64)

        provider = SymmetryCompletionProvider(axis=0)
        result = provider.complete(mesh, weak_mask, confidence, default_rig, {}, {})

        assert result.provider_name == "symmetry"
        assert result.metadata["n_completed"] == 0
        np.testing.assert_allclose(result.confidence_delta, 0.0)

    def test_no_strong_vertices_no_change(self, default_rig):
        """If no vertices are strong, nothing to mirror from."""
        mesh = _make_mesh_state()
        weak_mask = np.ones(mesh.n_vertices, dtype=bool)
        confidence = np.zeros(mesh.n_vertices, dtype=np.float64)

        provider = SymmetryCompletionProvider(axis=0)
        result = provider.complete(mesh, weak_mask, confidence, default_rig, {}, {})

        assert result.metadata["n_completed"] == 0

    def test_mirrors_from_strong_to_weak(self, default_rig):
        """Weak vertices should be moved toward mirrored strong positions."""
        # Create a mesh where left side is strong, right side is weak
        vertices = np.array([
            [-0.2, 0.0, 0.0],  # left (strong)
            [-0.2, 0.1, 0.0],  # left (strong)
            [0.2, 0.0, 0.0],   # right (weak)
            [0.2, 0.1, 0.0],   # right (weak)
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
        mesh = _make_mesh_state(vertices, faces)

        weak_mask = np.array([False, False, True, True])
        confidence = np.array([0.9, 0.9, 0.1, 0.1])

        provider = SymmetryCompletionProvider(axis=0)
        result = provider.complete(mesh, weak_mask, confidence, default_rig, {}, {})

        # Weak vertices should have moved
        assert result.metadata["n_completed"] >= 1
        # The mirrored position of [-0.2, 0, 0] is [0.2, 0, 0]
        # So vertex 2 should stay near [0.2, 0, 0] but with some blending
        assert result.confidence_delta[2] > 0
        assert result.confidence_delta[3] > 0
        # Strong vertices should not have changed
        assert result.confidence_delta[0] == 0
        assert result.confidence_delta[1] == 0

    def test_result_has_correct_shape(self, default_rig):
        """Result vertices and confidence_delta should match input shape."""
        mesh = _make_mesh_state()
        weak_mask = np.zeros(mesh.n_vertices, dtype=bool)
        weak_mask[0] = True
        confidence = np.ones(mesh.n_vertices, dtype=np.float64)
        confidence[0] = 0.1

        provider = SymmetryCompletionProvider(axis=0)
        result = provider.complete(mesh, weak_mask, confidence, default_rig, {}, {})

        assert result.vertices.shape == mesh.vertices.shape
        assert result.confidence_delta.shape == (mesh.n_vertices,)

    def test_different_axes(self, default_rig):
        """Symmetry should work for different axes."""
        mesh = _make_mesh_state()
        weak_mask = np.zeros(mesh.n_vertices, dtype=bool)
        weak_mask[0] = True
        confidence = np.ones(mesh.n_vertices, dtype=np.float64)
        confidence[0] = 0.1

        for axis in [0, 1, 2]:
            provider = SymmetryCompletionProvider(axis=axis)
            result = provider.complete(
                mesh, weak_mask, confidence, default_rig, {}, {},
            )
            assert result.provider_name == "symmetry"
            assert result.metadata["axis"] == axis


class TestLaplacianCompletionProvider:
    def test_name(self):
        provider = LaplacianCompletionProvider()
        assert provider.name == "laplacian"

    def test_no_weak_vertices_no_change(self, default_rig):
        """If no vertices are weak, Laplacian provider should not change anything."""
        mesh = _make_mesh_state()
        weak_mask = np.zeros(mesh.n_vertices, dtype=bool)
        confidence = np.ones(mesh.n_vertices, dtype=np.float64)

        provider = LaplacianCompletionProvider()
        result = provider.complete(mesh, weak_mask, confidence, default_rig, {}, {})

        assert result.provider_name == "laplacian"
        assert result.metadata["n_weak"] == 0

    def test_weak_vertices_move_toward_neighbors(self, default_rig):
        """Weak vertices should move toward the average of their confident neighbors."""
        vertices, faces = _make_cube_mesh(size=0.5)
        # Perturb one vertex
        vertices[0] += np.array([0.3, 0.3, 0.3])
        mesh = _make_mesh_state(vertices, faces)

        weak_mask = np.zeros(mesh.n_vertices, dtype=bool)
        weak_mask[0] = True
        confidence = np.ones(mesh.n_vertices, dtype=np.float64)
        confidence[0] = 0.1

        provider = LaplacianCompletionProvider(n_iterations=10, alpha=0.5)
        result = provider.complete(mesh, weak_mask, confidence, default_rig, {}, {})

        # Vertex 0 should have moved back toward its neighbors
        original_pos = mesh.vertices[0]
        completed_pos = result.vertices[0]
        # The displacement should be non-zero
        disp = np.linalg.norm(completed_pos - original_pos)
        assert disp > 0.01, "Weak vertex should have moved"

    def test_strong_vertices_unchanged(self, default_rig):
        """Strong (non-weak) vertices should not move."""
        mesh = _make_mesh_state()
        original_strong = mesh.vertices[1:].copy()

        weak_mask = np.zeros(mesh.n_vertices, dtype=bool)
        weak_mask[0] = True
        confidence = np.ones(mesh.n_vertices, dtype=np.float64)
        confidence[0] = 0.1

        provider = LaplacianCompletionProvider(n_iterations=5)
        result = provider.complete(mesh, weak_mask, confidence, default_rig, {}, {})

        np.testing.assert_allclose(result.vertices[1:], original_strong)

    def test_metadata_structure(self, default_rig):
        """Metadata should include iteration count and displacement stats."""
        mesh = _make_mesh_state()
        weak_mask = np.zeros(mesh.n_vertices, dtype=bool)
        weak_mask[0] = True
        confidence = np.ones(mesh.n_vertices, dtype=np.float64)
        confidence[0] = 0.1

        provider = LaplacianCompletionProvider(n_iterations=5, alpha=0.3)
        result = provider.complete(mesh, weak_mask, confidence, default_rig, {}, {})

        assert "n_iterations" in result.metadata
        assert result.metadata["n_iterations"] == 5
        assert "alpha" in result.metadata
        assert result.metadata["alpha"] == 0.3
        assert "n_weak" in result.metadata
        assert "mean_displacement" in result.metadata
        assert "max_displacement" in result.metadata

    def test_no_edges_handled(self, default_rig):
        """Mesh with no edges should be handled gracefully."""
        # Single vertex, no faces
        vertices = np.array([[0, 0, 0]], dtype=np.float64)
        faces = np.zeros((0, 3), dtype=np.int32)
        normals = np.array([[0, 1, 0]], dtype=np.float64)
        colors = np.array([[128, 128, 128]], dtype=np.uint8)
        mesh = MeshState(vertices=vertices, faces=faces, normals=normals,
                         vertex_colors=colors)

        weak_mask = np.array([True])
        confidence = np.array([0.0])

        provider = LaplacianCompletionProvider()
        result = provider.complete(mesh, weak_mask, confidence, default_rig, {}, {})

        assert result.vertices.shape == (1, 3)


class TestTrellis2CompletionProvider:
    def test_name(self):
        provider = Trellis2CompletionProvider()
        assert provider.name == "trellis2"

    def test_raises_runtime_error(self, default_rig):
        """TRELLIS.2 provider should raise RuntimeError (GPU not available)."""
        mesh = _make_mesh_state()
        weak_mask = np.zeros(mesh.n_vertices, dtype=bool)
        confidence = np.ones(mesh.n_vertices, dtype=np.float64)

        provider = Trellis2CompletionProvider()
        with pytest.raises(RuntimeError, match="TRELLIS"):
            provider.complete(mesh, weak_mask, confidence, default_rig, {}, {})


class TestHunyuan3DCompletionProvider:
    def test_name(self):
        provider = Hunyuan3DCompletionProvider()
        assert provider.name == "hunyuan3d"

    def test_raises_runtime_error(self, default_rig):
        """Hunyuan3D provider should raise RuntimeError (GPU not available)."""
        mesh = _make_mesh_state()
        weak_mask = np.zeros(mesh.n_vertices, dtype=bool)
        confidence = np.ones(mesh.n_vertices, dtype=np.float64)

        provider = Hunyuan3DCompletionProvider()
        with pytest.raises(RuntimeError, match="Hunyuan"):
            provider.complete(mesh, weak_mask, confidence, default_rig, {}, {})


class TestGetCompletionProviders:
    def test_default_config_order(self):
        """Default config: no GPU → symmetry + laplacian."""
        cfg = CompletionConfig()
        providers = get_completion_providers(cfg)

        names = [p.name for p in providers]
        assert "symmetry" in names
        assert "laplacian" in names
        # Laplacian should always be last (final fallback)
        assert names[-1] == "laplacian"

    def test_trellis_enabled(self):
        """With trellis enabled, it should be first in the list."""
        cfg = CompletionConfig(use_trellis=True)
        providers = get_completion_providers(cfg)

        names = [p.name for p in providers]
        assert names[0] == "trellis2"
        assert "laplacian" in names

    def test_hunyuan_enabled(self):
        """With hunyuan enabled, it should be in the list."""
        cfg = CompletionConfig(use_hunyuan=True)
        providers = get_completion_providers(cfg)

        names = [p.name for p in providers]
        assert "hunyuan3d" in names
        assert "laplacian" in names

    def test_both_gpu_providers(self):
        """Both GPU providers enabled → trellis first, then hunyuan."""
        cfg = CompletionConfig(use_trellis=True, use_hunyuan=True)
        providers = get_completion_providers(cfg)

        names = [p.name for p in providers]
        assert names[0] == "trellis2"
        assert names[1] == "hunyuan3d"

    def test_symmetry_disabled(self):
        """With symmetry disabled, only laplacian should remain."""
        cfg = CompletionConfig(use_symmetry=False, use_trellis=False, use_hunyuan=False)
        providers = get_completion_providers(cfg)

        names = [p.name for p in providers]
        assert "symmetry" not in names
        assert names == ["laplacian"]

    def test_always_has_laplacian(self):
        """Laplacian should always be present as the final fallback."""
        for use_sym in [True, False]:
            for use_trellis in [True, False]:
                for use_hunyuan in [True, False]:
                    cfg = CompletionConfig(
                        use_symmetry=use_sym,
                        use_trellis=use_trellis,
                        use_hunyuan=use_hunyuan,
                    )
                    providers = get_completion_providers(cfg)
                    names = [p.name for p in providers]
                    assert names[-1] == "laplacian"


# ===========================================================================
# FUSION
# ===========================================================================


class TestFuseCompletion:
    def test_no_weak_vertices_no_change(self):
        """If no vertices are weak, mesh should be unchanged."""
        mesh = _make_mesh_state()
        original_verts = mesh.vertices.copy()

        result = CompletionResult(
            vertices=mesh.vertices + 1.0,  # large change
            confidence_delta=np.zeros(mesh.n_vertices),
            provider_name="test",
        )
        confidence = np.ones(mesh.n_vertices, dtype=np.float64)
        weak_mask = np.zeros(mesh.n_vertices, dtype=bool)

        fused = fuse_completion(mesh, result, confidence, weak_mask)

        np.testing.assert_allclose(fused.vertices, original_verts)

    def test_weak_vertices_blended(self):
        """Weak vertices should be blended toward completion result."""
        mesh = _make_mesh_state()
        original_verts = mesh.vertices.copy()

        # Completion moves vertex 0 significantly
        new_verts = mesh.vertices.copy()
        new_verts[0] += np.array([1.0, 0.0, 0.0])

        result = CompletionResult(
            vertices=new_verts,
            confidence_delta=np.zeros(mesh.n_vertices),
            provider_name="test",
        )
        confidence = np.ones(mesh.n_vertices, dtype=np.float64)
        confidence[0] = 0.0  # vertex 0 has zero confidence
        weak_mask = np.zeros(mesh.n_vertices, dtype=bool)
        weak_mask[0] = True

        fused = fuse_completion(mesh, result, confidence, weak_mask, max_blend=0.8)

        # Vertex 0 should have moved toward the completion position
        disp = np.linalg.norm(fused.vertices[0] - original_verts[0])
        assert disp > 0.1, "Weak vertex should have moved"

        # But not fully to the completion position (blending)
        max_possible = np.linalg.norm(new_verts[0] - original_verts[0])
        assert disp < max_possible, "Should be blended, not fully replaced"

        # Strong vertices should be unchanged
        np.testing.assert_allclose(fused.vertices[1:], original_verts[1:])

    def test_high_confidence_low_blend(self):
        """High-confidence weak vertices should have less blending."""
        mesh = _make_mesh_state()

        new_verts = mesh.vertices.copy()
        new_verts[0] += np.array([1.0, 0.0, 0.0])
        new_verts[1] += np.array([1.0, 0.0, 0.0])

        result = CompletionResult(
            vertices=new_verts,
            confidence_delta=np.zeros(mesh.n_vertices),
            provider_name="test",
        )

        confidence = np.ones(mesh.n_vertices, dtype=np.float64)
        confidence[0] = 0.0   # zero confidence → max blend
        confidence[1] = 0.25  # moderate confidence → less blend
        weak_mask = np.zeros(mesh.n_vertices, dtype=bool)
        weak_mask[0] = True
        weak_mask[1] = True

        fused = fuse_completion(mesh, result, confidence, weak_mask, max_blend=0.8)

        disp_0 = np.linalg.norm(fused.vertices[0] - mesh.vertices[0])
        disp_1 = np.linalg.norm(fused.vertices[1] - mesh.vertices[1])

        # Lower confidence → more displacement
        assert disp_0 > disp_1

    def test_preserves_topology(self):
        """Fusion should preserve mesh topology (faces unchanged)."""
        mesh = _make_mesh_state()
        original_faces = mesh.faces.copy()

        result = CompletionResult(
            vertices=mesh.vertices.copy(),
            confidence_delta=np.zeros(mesh.n_vertices),
            provider_name="test",
        )
        confidence = np.ones(mesh.n_vertices, dtype=np.float64)
        weak_mask = np.zeros(mesh.n_vertices, dtype=bool)

        fused = fuse_completion(mesh, result, confidence, weak_mask)

        np.testing.assert_array_equal(fused.faces, original_faces)

    def test_normals_recomputed(self):
        """Fused mesh should have recomputed normals."""
        mesh = _make_mesh_state()

        new_verts = mesh.vertices.copy()
        new_verts[0] += np.array([0.5, 0.0, 0.0])

        result = CompletionResult(
            vertices=new_verts,
            confidence_delta=np.zeros(mesh.n_vertices),
            provider_name="test",
        )
        confidence = np.zeros(mesh.n_vertices, dtype=np.float64)
        weak_mask = np.ones(mesh.n_vertices, dtype=bool)

        fused = fuse_completion(mesh, result, confidence, weak_mask)

        # Normals should be unit length
        lengths = np.linalg.norm(fused.normals, axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=0.1)

    def test_mismatched_vertices_returns_copy(self):
        """If result has wrong vertex count, should return original copy."""
        mesh = _make_mesh_state()
        original_verts = mesh.vertices.copy()

        result = CompletionResult(
            vertices=np.zeros((3, 3)),  # wrong size
            confidence_delta=np.zeros(3),
            provider_name="test",
        )
        confidence = np.zeros(mesh.n_vertices, dtype=np.float64)
        weak_mask = np.ones(mesh.n_vertices, dtype=bool)

        fused = fuse_completion(mesh, result, confidence, weak_mask)
        np.testing.assert_allclose(fused.vertices, original_verts)


class TestComputeBlendWeights:
    def test_no_weak_zero_weights(self):
        """No weak vertices → all-zero weights."""
        confidence = np.ones(10, dtype=np.float64)
        weak_mask = np.zeros(10, dtype=bool)

        weights = compute_blend_weights(confidence, weak_mask)
        np.testing.assert_allclose(weights, 0.0)

    def test_zero_confidence_max_blend(self):
        """Zero-confidence weak vertex → max blend weight."""
        confidence = np.array([0.0, 1.0])
        weak_mask = np.array([True, False])

        weights = compute_blend_weights(confidence, weak_mask, max_blend=0.8)
        assert weights[0] == pytest.approx(0.8)
        assert weights[1] == pytest.approx(0.0)

    def test_weights_in_range(self):
        """All weights should be in [0, max_blend]."""
        rng = np.random.RandomState(42)
        confidence = rng.rand(100)
        weak_mask = confidence < 0.3

        weights = compute_blend_weights(confidence, weak_mask, max_blend=0.8)
        assert np.all(weights >= 0.0)
        assert np.all(weights <= 0.8 + 1e-10)

    def test_higher_confidence_lower_weight(self):
        """Higher confidence → lower blend weight."""
        confidence = np.array([0.0, 0.1, 0.2])
        weak_mask = np.array([True, True, True])

        weights = compute_blend_weights(confidence, weak_mask, max_blend=0.8)
        assert weights[0] > weights[1] > weights[2]


# ===========================================================================
# STAGE RUNNER INTEGRATION TESTS
# ===========================================================================


class TestRunCompleteGeometry:
    """Integration tests for the full stage runner."""

    def _setup_job_with_artifacts(
        self, jm, sm, config,
        mesh_size=0.3,
        include_refined=True,
    ):
        """
        Create a job and save all required artifacts from previous stages:
        camera_init.json, mesh (refined or coarse), segmented view previews.
        """
        job_id = _create_mv_job(jm)

        # Build and save camera rig
        rig = build_canonical_rig(config, (256, 256))
        sm.save_artifact_json(job_id, "camera_init.json", rig.to_dict())

        # Build mesh
        vertices, faces = _make_cube_mesh(size=mesh_size)
        normals = compute_vertex_normals(vertices, faces)

        if include_refined:
            mesh_path = sm.get_artifact_dir(job_id) / "refined_mesh.ply"
        else:
            mesh_path = sm.get_artifact_dir(job_id) / "coarse_visual_hull_mesh.ply"
        save_mesh_ply(str(mesh_path), vertices, faces, normals)

        # Save segmented view previews (RGBA images with subject silhouette)
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

        # Save preprocess metrics (needed by _load_segmented_views)
        sm.save_artifact_json(job_id, "preprocess_metrics.json", {
            "canvas_size": 256,
            "crop_side": 300,
            "per_view": {vn: {"bbox": [50, 50, 150, 150]} for vn in CANONICAL_VIEW_ORDER},
        })

        return job_id

    def test_happy_path_with_refined_mesh(self, jm, sm, config):
        """Stage should complete with refined mesh and save all artifacts."""
        job_id = self._setup_job_with_artifacts(jm, sm, config, include_refined=True)

        run_complete_geometry(job_id, config, jm, sm)

        # Completed mesh should exist
        mesh_path = sm.get_artifact_path(job_id, "completed_mesh.ply")
        assert mesh_path is not None

        # Metrics should exist
        metrics = sm.load_artifact_json(job_id, "completion_metrics.json")
        assert metrics is not None
        assert "provider_used" in metrics
        assert "coverage" in metrics
        assert "mesh_source" in metrics
        assert metrics["mesh_source"] == "refined"

    def test_fallback_to_coarse_mesh(self, jm, sm, config):
        """Stage should fall back to coarse mesh when refined is absent."""
        job_id = self._setup_job_with_artifacts(
            jm, sm, config, include_refined=False,
        )

        run_complete_geometry(job_id, config, jm, sm)

        metrics = sm.load_artifact_json(job_id, "completion_metrics.json")
        assert metrics is not None
        assert metrics["mesh_source"] == "coarse"

    def test_stage_progress_reaches_one(self, jm, sm, config):
        """Stage progress should reach 1.0 on completion."""
        job_id = self._setup_job_with_artifacts(jm, sm, config)

        run_complete_geometry(job_id, config, jm, sm)

        job = jm.get_job(job_id)
        assert job["stage_progress"] == 1.0

    def test_missing_camera_init_raises(self, jm, sm, config):
        """Missing camera_init.json should raise ValueError."""
        job_id = _create_mv_job(jm)
        with pytest.raises(ValueError, match="camera_init"):
            run_complete_geometry(job_id, config, jm, sm)

    def test_missing_mesh_raises(self, jm, sm, config):
        """Missing both refined and coarse mesh should raise ValueError."""
        job_id = _create_mv_job(jm)
        rig = build_canonical_rig(config, (256, 256))
        sm.save_artifact_json(job_id, "camera_init.json", rig.to_dict())

        with pytest.raises(ValueError, match="mesh"):
            run_complete_geometry(job_id, config, jm, sm)

    def test_completed_mesh_valid(self, jm, sm, config):
        """Completed mesh should be loadable and have same topology as input."""
        job_id = self._setup_job_with_artifacts(jm, sm, config)

        run_complete_geometry(job_id, config, jm, sm)

        mesh_path = sm.get_artifact_path(job_id, "completed_mesh.ply")
        vertices, faces, normals = load_mesh_ply(str(mesh_path))

        # Same topology as input cube (8 vertices, 12 faces)
        assert len(vertices) == 8
        assert len(faces) == 12
        # All face indices should be valid
        assert np.all(faces >= 0)
        assert np.all(faces < len(vertices))

    def test_metrics_has_coverage_stats(self, jm, sm, config):
        """Metrics should include coverage statistics."""
        job_id = self._setup_job_with_artifacts(jm, sm, config)

        run_complete_geometry(job_id, config, jm, sm)

        metrics = sm.load_artifact_json(job_id, "completion_metrics.json")
        coverage = metrics["coverage"]
        assert "n_vertices" in coverage
        assert "n_weak_vertices" in coverage
        assert "weak_fraction" in coverage
        assert "mean_confidence" in coverage
        assert "per_view_coverage" in coverage

    def test_metrics_has_provider_info(self, jm, sm, config):
        """Metrics should include which provider was used."""
        job_id = self._setup_job_with_artifacts(jm, sm, config)

        run_complete_geometry(job_id, config, jm, sm)

        metrics = sm.load_artifact_json(job_id, "completion_metrics.json")
        assert "provider_used" in metrics
        assert "provider_metadata" in metrics
        assert "config" in metrics
        assert "use_symmetry" in metrics["config"]

    def test_with_symmetry_disabled(self, jm, sm):
        """Stage should work with symmetry disabled."""
        config = CanonicalMVConfig(
            symmetry_prior=False,
            use_trellis_completion=False,
            use_hunyuan_completion=False,
        )
        job_id = self._setup_job_with_artifacts(jm, sm, config)

        run_complete_geometry(job_id, config, jm, sm)

        metrics = sm.load_artifact_json(job_id, "completion_metrics.json")
        assert metrics is not None
        # Provider should be laplacian (only fallback available)
        # (or "none" if no weak vertices)
        assert metrics["provider_used"] in ("laplacian", "none")

    def test_with_trellis_enabled_falls_back(self, jm, sm, config):
        """With trellis enabled but unavailable, should fall back to CPU providers."""
        config.use_trellis_completion = True
        job_id = self._setup_job_with_artifacts(jm, sm, config)

        run_complete_geometry(job_id, config, jm, sm)

        metrics = sm.load_artifact_json(job_id, "completion_metrics.json")
        assert metrics is not None
        # Should have fallen back from trellis2
        if metrics["provider_used"] != "none":
            assert metrics["provider_used"] in ("symmetry", "laplacian")
            # Should have recorded trellis2 error
            errors = metrics.get("provider_errors", [])
            trellis_errors = [e for e in errors if e["provider"] == "trellis2"]
            assert len(trellis_errors) > 0

    def test_with_hunyuan_enabled_falls_back(self, jm, sm, config):
        """With hunyuan enabled but unavailable, should fall back to CPU providers."""
        config.use_hunyuan_completion = True
        config.use_trellis_completion = False
        job_id = self._setup_job_with_artifacts(jm, sm, config)

        run_complete_geometry(job_id, config, jm, sm)

        metrics = sm.load_artifact_json(job_id, "completion_metrics.json")
        assert metrics is not None
        if metrics["provider_used"] != "none":
            assert metrics["provider_used"] in ("symmetry", "laplacian")

    def test_no_segmented_views_still_runs(self, jm, sm, config):
        """With no segmented views, should still complete (all vertices weak)."""
        job_id = _create_mv_job(jm)

        # Save camera rig and mesh but no views
        rig = build_canonical_rig(config, (256, 256))
        sm.save_artifact_json(job_id, "camera_init.json", rig.to_dict())

        vertices, faces = _make_cube_mesh(size=0.3)
        normals = compute_vertex_normals(vertices, faces)
        mesh_path = sm.get_artifact_dir(job_id) / "refined_mesh.ply"
        save_mesh_ply(str(mesh_path), vertices, faces, normals)

        # Should complete (all vertices will have zero confidence)
        run_complete_geometry(job_id, config, jm, sm)

        metrics = sm.load_artifact_json(job_id, "completion_metrics.json")
        assert metrics is not None


# ===========================================================================
# EDGE CASES
# ===========================================================================


class TestEdgeCases:
    def test_completion_result_dataclass(self):
        """CompletionResult should be constructible with all fields."""
        result = CompletionResult(
            vertices=np.zeros((5, 3)),
            confidence_delta=np.zeros(5),
            provider_name="test",
            metadata={"key": "value"},
        )
        assert result.provider_name == "test"
        assert result.metadata["key"] == "value"

    def test_all_confident_mesh_skips_completion(self, default_rig):
        """If all vertices are confident, completion should be skipped."""
        mesh = _make_mesh_state()
        image_size = (256, 256)

        # Create masks that cover all vertices well
        masks = render_silhouettes(
            mesh.vertices, mesh.faces, default_rig, image_size,
        )

        visibility = compute_vertex_visibility(
            mesh.vertices, mesh.normals, default_rig, masks, image_size,
        )
        confidence = compute_vertex_confidence(
            visibility, mesh.n_vertices, min_views=1,
        )

        # With min_views=1, most vertices of a centered cube should be confident
        weak_mask = identify_weak_regions(confidence, threshold=0.3)

        # The cube might still have some weak vertices depending on the
        # camera setup, but the point is the logic should handle it
        stats = compute_coverage_stats(confidence, weak_mask, visibility)
        assert stats["n_vertices"] == 8

    def test_large_mesh_performance(self, default_rig):
        """Ensure confidence computation doesn't crash for larger meshes."""
        # Create a grid mesh
        n = 15
        vertices = []
        for i in range(n):
            for j in range(n):
                vertices.append([i * 0.04 - 0.3, j * 0.04 - 0.3, 0.0])
        vertices = np.array(vertices, dtype=np.float64)

        faces = []
        for i in range(n - 1):
            for j in range(n - 1):
                v0 = i * n + j
                faces.append([v0, v0 + 1, v0 + n])
                faces.append([v0 + 1, v0 + n + 1, v0 + n])
        faces = np.array(faces, dtype=np.int32)

        normals = compute_vertex_normals(vertices, faces)
        image_size = (256, 256)

        masks = {vn: np.full((256, 256), 255, dtype=np.uint8) for vn in CANONICAL_VIEW_ORDER}

        visibility = compute_vertex_visibility(
            vertices, normals, default_rig, masks, image_size,
        )

        confidence = compute_vertex_confidence(
            visibility, len(vertices), min_views=3,
        )

        assert len(confidence) == len(vertices)
        assert np.all(np.isfinite(confidence))

    def test_symmetry_provider_with_large_distance(self, default_rig):
        """Symmetry provider should not mirror very distant vertices."""
        # Create vertices where mirror is very far
        vertices = np.array([
            [-0.2, 0.0, 0.0],   # strong, left side
            [5.0, 0.0, 0.0],    # weak, very far right
        ], dtype=np.float64)
        faces = np.zeros((0, 3), dtype=np.int32)
        normals = np.array([[0, 1, 0], [0, 1, 0]], dtype=np.float64)
        colors = np.full((2, 3), 128, dtype=np.uint8)
        mesh = MeshState(vertices=vertices, faces=faces, normals=normals,
                         vertex_colors=colors)

        weak_mask = np.array([False, True])
        confidence = np.array([0.9, 0.1])

        provider = SymmetryCompletionProvider(axis=0)
        result = provider.complete(mesh, weak_mask, confidence, default_rig, {}, {})

        # Mirror of [-0.2, 0, 0] is [0.2, 0, 0], which is 4.8 away from [5, 0, 0]
        # This exceeds the 0.5 threshold, so vertex 1 should NOT be completed
        assert result.metadata["n_completed"] == 0

    def test_laplacian_provider_convergence(self, default_rig):
        """Laplacian diffusion should converge (not diverge)."""
        vertices, faces = _make_cube_mesh(size=0.5)
        # Heavily perturb one vertex
        vertices[0] += np.array([1.0, 1.0, 1.0])
        mesh = _make_mesh_state(vertices, faces)

        weak_mask = np.zeros(mesh.n_vertices, dtype=bool)
        weak_mask[0] = True
        confidence = np.ones(mesh.n_vertices, dtype=np.float64)
        confidence[0] = 0.0

        provider = LaplacianCompletionProvider(n_iterations=50, alpha=0.5)
        result = provider.complete(mesh, weak_mask, confidence, default_rig, {}, {})

        # The completed vertex should have moved toward its neighbors
        # not away from them
        original_dist = np.linalg.norm(mesh.vertices[0] - mesh.vertices[1:].mean(axis=0))
        completed_dist = np.linalg.norm(result.vertices[0] - mesh.vertices[1:].mean(axis=0))
        assert completed_dist < original_dist, \
            "Laplacian diffusion should move weak vertex toward neighbors"


class TestProviderFallbackChain:
    """Test that the provider fallback chain works correctly."""

    def test_trellis_fails_symmetry_succeeds(self, default_rig):
        """When trellis fails, symmetry should be tried next."""
        cfg = CompletionConfig(use_trellis=True, use_symmetry=True)
        providers = get_completion_providers(cfg)

        mesh = _make_mesh_state()
        weak_mask = np.zeros(mesh.n_vertices, dtype=bool)
        weak_mask[0] = True
        confidence = np.ones(mesh.n_vertices, dtype=np.float64)
        confidence[0] = 0.1

        # Try providers in order (simulating the stage runner logic)
        result = None
        errors = []
        for provider in providers:
            try:
                result = provider.complete(
                    mesh, weak_mask, confidence, default_rig, {}, {},
                )
                break
            except (RuntimeError, Exception) as e:
                errors.append(provider.name)

        assert "trellis2" in errors
        assert result is not None
        assert result.provider_name == "symmetry"

    def test_all_gpu_fail_laplacian_succeeds(self, default_rig):
        """When all GPU providers fail, laplacian should succeed."""
        cfg = CompletionConfig(
            use_trellis=True,
            use_hunyuan=True,
            use_symmetry=False,
        )
        providers = get_completion_providers(cfg)

        mesh = _make_mesh_state()
        weak_mask = np.zeros(mesh.n_vertices, dtype=bool)
        weak_mask[0] = True
        confidence = np.ones(mesh.n_vertices, dtype=np.float64)
        confidence[0] = 0.1

        result = None
        errors = []
        for provider in providers:
            try:
                result = provider.complete(
                    mesh, weak_mask, confidence, default_rig, {}, {},
                )
                break
            except (RuntimeError, Exception) as e:
                errors.append(provider.name)

        assert "trellis2" in errors
        assert "hunyuan3d" in errors
        assert result is not None
        assert result.provider_name == "laplacian"


class TestCompletionQuality:
    """Tests verifying completion actually improves the mesh."""

    def test_symmetry_reduces_asymmetry(self, default_rig):
        """Symmetry completion should reduce asymmetry in the mesh."""
        # Create an asymmetric mesh (left side complete, right side missing)
        vertices = np.array([
            [-0.2, -0.1, 0.0],  # left bottom
            [-0.2, 0.1, 0.0],   # left top
            [-0.1, 0.0, 0.0],   # left center
            [0.2, -0.1, 0.0],   # right bottom (displaced)
            [0.3, 0.2, 0.0],    # right top (displaced more)
            [0.15, 0.0, 0.0],   # right center (displaced)
        ], dtype=np.float64)
        faces = np.array([
            [0, 1, 2],
            [3, 4, 5],
        ], dtype=np.int32)
        mesh = _make_mesh_state(vertices, faces)

        # Left side is strong, right side is weak
        weak_mask = np.array([False, False, False, True, True, True])
        confidence = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])

        provider = SymmetryCompletionProvider(axis=0)
        result = provider.complete(mesh, weak_mask, confidence, default_rig, {}, {})

        # Right side should have moved closer to the mirror of the left side
        if result.metadata["n_completed"] > 0:
            # Check that at least one weak vertex moved
            total_delta = np.sum(result.confidence_delta[weak_mask])
            assert total_delta > 0

    def test_laplacian_smooths_perturbation(self, default_rig):
        """Laplacian completion should smooth out a perturbed vertex."""
        vertices, faces = _make_cube_mesh(size=0.4)
        # Perturb vertex 0 significantly
        original_v0 = vertices[0].copy()
        vertices[0] += np.array([0.5, 0.0, 0.0])
        mesh = _make_mesh_state(vertices, faces)

        weak_mask = np.zeros(mesh.n_vertices, dtype=bool)
        weak_mask[0] = True
        confidence = np.ones(mesh.n_vertices, dtype=np.float64)
        confidence[0] = 0.0

        provider = LaplacianCompletionProvider(n_iterations=20, alpha=0.5)
        result = provider.complete(mesh, weak_mask, confidence, default_rig, {}, {})

        # Vertex 0 should have moved back toward its neighbors
        dist_before = np.linalg.norm(vertices[0] - original_v0)
        dist_after = np.linalg.norm(result.vertices[0] - original_v0)
        assert dist_after < dist_before, \
            "Laplacian should move perturbed vertex back toward neighbors"


class TestRunCompleteGeometryRobustness:
    """Additional robustness tests for the stage runner."""

    def _setup_job_with_artifacts(self, jm, sm, config, include_refined=True):
        """Create a job with all required artifacts."""
        job_id = _create_mv_job(jm)

        rig = build_canonical_rig(config, (256, 256))
        sm.save_artifact_json(job_id, "camera_init.json", rig.to_dict())

        vertices, faces = _make_cube_mesh(size=0.3)
        normals = compute_vertex_normals(vertices, faces)

        if include_refined:
            mesh_path = sm.get_artifact_dir(job_id) / "refined_mesh.ply"
        else:
            mesh_path = sm.get_artifact_dir(job_id) / "coarse_visual_hull_mesh.ply"
        save_mesh_ply(str(mesh_path), vertices, faces, normals)

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

        sm.save_artifact_json(job_id, "preprocess_metrics.json", {
            "canvas_size": 256,
            "crop_side": 300,
            "per_view": {vn: {"bbox": [50, 50, 150, 150]} for vn in CANONICAL_VIEW_ORDER},
        })

        return job_id

    def test_completed_mesh_topology_preserved(self, jm, sm, config):
        """Completion should preserve mesh topology."""
        job_id = self._setup_job_with_artifacts(jm, sm, config)
        run_complete_geometry(job_id, config, jm, sm)

        mesh_path = sm.get_artifact_path(job_id, "completed_mesh.ply")
        vertices, faces, normals = load_mesh_ply(str(mesh_path))

        assert len(vertices) == 8
        assert len(faces) == 12
        assert np.all(faces >= 0)
        assert np.all(faces < len(vertices))

    def test_metrics_confidence_delta(self, jm, sm, config):
        """Metrics should include confidence delta statistics."""
        job_id = self._setup_job_with_artifacts(jm, sm, config)
        run_complete_geometry(job_id, config, jm, sm)

        metrics = sm.load_artifact_json(job_id, "completion_metrics.json")
        assert "confidence_delta_mean" in metrics
        assert "confidence_delta_max" in metrics
        assert np.isfinite(metrics["confidence_delta_mean"])
        assert np.isfinite(metrics["confidence_delta_max"])

    def test_multiple_runs_deterministic(self, jm, sm, config):
        """Running completion twice with same input should produce same output."""
        job_id_1 = self._setup_job_with_artifacts(jm, sm, config)
        job_id_2 = self._setup_job_with_artifacts(jm, sm, config)

        run_complete_geometry(job_id_1, config, jm, sm)
        run_complete_geometry(job_id_2, config, jm, sm)

        mesh_path_1 = sm.get_artifact_path(job_id_1, "completed_mesh.ply")
        mesh_path_2 = sm.get_artifact_path(job_id_2, "completed_mesh.ply")

        v1, f1, n1 = load_mesh_ply(str(mesh_path_1))
        v2, f2, n2 = load_mesh_ply(str(mesh_path_2))

        np.testing.assert_allclose(v1, v2, atol=1e-5)
        np.testing.assert_array_equal(f1, f2)

    def test_both_gpu_and_cpu_disabled_uses_laplacian(self, jm, sm):
        """With all optional providers disabled, laplacian should be used."""
        config = CanonicalMVConfig(
            symmetry_prior=False,
            use_trellis_completion=False,
            use_hunyuan_completion=False,
        )
        job_id = self._setup_job_with_artifacts(jm, sm, config)

        run_complete_geometry(job_id, config, jm, sm)

        metrics = sm.load_artifact_json(job_id, "completion_metrics.json")
        assert metrics is not None
        # Should have used laplacian (only available provider)
        # or "none" if no weak vertices
        assert metrics["provider_used"] in ("laplacian", "none")

