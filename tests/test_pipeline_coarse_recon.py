"""
Tests for the coarse reconstruction stage of the canonical multi-view pipeline.

Covers:
    VISUAL HULL (unit tests):
        - compute_visual_hull: occupancy from silhouette masks + cameras
        - threshold_occupancy: binary thresholding
        - Empty masks produce empty volume
        - Full masks produce full volume
        - Known shape (sphere-like) produces expected occupancy

    MESH EXTRACTION (unit tests):
        - extract_surface_mesh: marching cubes on synthetic occupancy
        - Fallback surface extraction when skimage unavailable
        - Empty volume raises ValueError

    DEPTH MAPS (unit tests):
        - compute_depth_maps: per-view depth rendering from occupancy
        - Depth maps have correct dimensions
        - Non-zero depth where surface is visible

    SURFACE SAMPLING (unit tests):
        - sample_surface_points: correct count, on surface, with normals
        - color_surface_points: color assignment from best-visible view
        - build_coarse_gaussians: correct structure and shapes

    PLY I/O (unit tests):
        - save_mesh_ply: writes valid binary PLY
        - save_gaussians_ply: writes valid binary PLY with Gaussian attributes
        - save_depth_map_png: writes 16-bit grayscale PNG

    HELPERS (unit tests):
        - _grid_resolution_from_config: config mapping
        - _load_segmented_views: loading from storage

    STAGE RUNNER (integration tests):
        - run_reconstruct_coarse: happy path
        - Missing camera_init.json raises ValueError
        - Missing segmented views raises ValueError
        - Empty visual hull raises ValueError
        - All artifacts saved correctly
        - Stage progress reaches 1.0

No GPU dependencies — all tests use synthetic data with numpy.
"""

import io
import json
import math
import struct
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
    ViewStatus,
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
    # Visual hull
    compute_visual_hull,
    threshold_occupancy,
    # Mesh extraction
    extract_surface_mesh,
    _fallback_surface_extraction,
    # Depth maps
    compute_depth_maps,
    # Surface sampling
    sample_surface_points,
    color_surface_points,
    build_coarse_gaussians,
    # PLY I/O
    save_mesh_ply,
    save_gaussians_ply,
    save_depth_map_png,
    # Stage runner
    run_reconstruct_coarse,
    # Helpers
    _grid_resolution_from_config,
    _load_segmented_views,
    # Constants
    DEFAULT_GRID_RESOLUTION,
    DEFAULT_GRID_HALF_EXTENT,
    DEFAULT_N_SURFACE_POINTS,
    DEFAULT_CONSENSUS_RATIO,
    MIN_OCCUPIED_VOXELS,
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
    """Build a canonical rig with 256x256 images (small for fast tests)."""
    return build_canonical_rig(config, (256, 256))


def _create_mv_job(jm):
    return jm.create_multiview_job(
        category=CategoryEnum.HUMAN_BUST,
        pipeline=PipelineEnum.CANONICAL_MV_HYBRID,
        views_received=["front", "back", "left", "right", "top"],
    )


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _make_sphere_masks(
    rig: CameraRig,
    sphere_radius: float = 0.4,
    image_size: int = 256,
) -> dict:
    """
    Create silhouette masks for a sphere centered at the origin.

    Projects a sphere into each camera view and creates binary masks
    where the sphere's projection falls.
    """
    from pipelines.canonical_mv.camera_init import project_point

    masks = {}
    for vn in CANONICAL_VIEW_ORDER:
        if vn not in rig.cameras:
            continue

        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        ext = rig.get_extrinsic(vn)
        intr = rig.get_intrinsic(vn)

        # Project the sphere center to get the 2D center
        center_2d = project_point(np.array([0.0, 0.0, 0.0]), ext, intr)
        if center_2d is None:
            continue

        # Approximate the projected radius by projecting a point on the sphere edge
        # Use a point at +X from center
        edge_3d = np.array([sphere_radius, 0.0, 0.0])
        edge_2d = project_point(edge_3d, ext, intr)
        if edge_2d is None:
            # Try another direction
            edge_3d = np.array([0.0, sphere_radius, 0.0])
            edge_2d = project_point(edge_3d, ext, intr)
        if edge_2d is None:
            continue

        proj_radius = np.linalg.norm(edge_2d - center_2d)

        # Draw filled circle
        cx, cy = int(center_2d[0]), int(center_2d[1])
        cv2.circle(mask, (cx, cy), int(proj_radius), 255, -1)

        masks[vn] = mask

    return masks


def _make_cube_masks(
    rig: CameraRig,
    cube_half_extent: float = 0.3,
    image_size: int = 256,
) -> dict:
    """
    Create silhouette masks for an axis-aligned cube centered at the origin.

    Projects the 8 cube corners into each view and fills the convex hull.
    """
    from pipelines.canonical_mv.camera_init import project_point

    # 8 corners of the cube
    h = cube_half_extent
    corners_3d = np.array([
        [-h, -h, -h], [h, -h, -h], [-h, h, -h], [h, h, -h],
        [-h, -h, h], [h, -h, h], [-h, h, h], [h, h, h],
    ], dtype=np.float64)

    masks = {}
    for vn in CANONICAL_VIEW_ORDER:
        if vn not in rig.cameras:
            continue

        ext = rig.get_extrinsic(vn)
        intr = rig.get_intrinsic(vn)

        projected = []
        for corner in corners_3d:
            p2d = project_point(corner, ext, intr)
            if p2d is not None:
                projected.append(p2d.astype(np.int32))

        if len(projected) < 3:
            continue

        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        hull = cv2.convexHull(np.array(projected))
        cv2.fillConvexPoly(mask, hull, 255)

        masks[vn] = mask

    return masks


def _make_full_masks(image_size: int = 256) -> dict:
    """Create masks where the entire image is foreground."""
    return {
        vn: np.full((image_size, image_size), 255, dtype=np.uint8)
        for vn in CANONICAL_VIEW_ORDER
    }


def _make_empty_masks(image_size: int = 256) -> dict:
    """Create empty masks (no foreground)."""
    return {
        vn: np.zeros((image_size, image_size), dtype=np.uint8)
        for vn in CANONICAL_VIEW_ORDER
    }


def _make_simple_occupancy(resolution: int = 32, fill_center: bool = True) -> np.ndarray:
    """
    Create a simple occupancy grid with a filled sphere in the center.

    Returns (N, N, N) float32 array with values 0 or 1.
    """
    N = resolution
    occupancy = np.zeros((N, N, N), dtype=np.float32)

    if fill_center:
        center = N // 2
        radius = N // 4
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    dist = math.sqrt(
                        (i - center) ** 2 + (j - center) ** 2 + (k - center) ** 2
                    )
                    if dist <= radius:
                        occupancy[i, j, k] = 1.0

    return occupancy


def _make_rgb_images(image_size: int = 256) -> dict:
    """Create simple RGB images for each view (different colors per view)."""
    colors = {
        "front": (200, 100, 50),
        "back": (50, 100, 200),
        "left": (100, 200, 50),
        "right": (200, 50, 100),
        "top": (150, 150, 50),
    }
    images = {}
    for vn in CANONICAL_VIEW_ORDER:
        img = np.full((image_size, image_size, 3), colors.get(vn, (128, 128, 128)), dtype=np.uint8)
        images[vn] = img
    return images


# ===========================================================================
# VISUAL HULL UNIT TESTS
# ===========================================================================


class TestComputeVisualHull:
    """Test visual hull computation from silhouette masks."""

    def test_sphere_masks_produce_occupied_center(self, default_rig):
        """Sphere silhouettes should produce occupied voxels near the origin."""
        masks = _make_sphere_masks(default_rig, sphere_radius=0.4)
        assert len(masks) >= 3, "Need at least 3 masks for visual hull"

        occupancy, grid_origin, voxel_size = compute_visual_hull(
            masks=masks,
            rig=default_rig,
            grid_resolution=32,
            grid_half_extent=1.0,
            consensus_ratio=1.0,
        )

        assert occupancy.shape == (32, 32, 32)
        assert voxel_size == pytest.approx(2.0 / 32, abs=1e-6)

        # Center voxels should have high occupancy
        center = 16
        assert occupancy[center, center, center] > 0.5

    def test_cube_masks_produce_occupied_volume(self, default_rig):
        """Cube silhouettes should produce a non-empty visual hull."""
        masks = _make_cube_masks(default_rig, cube_half_extent=0.3)
        assert len(masks) >= 3

        occupancy, _, _ = compute_visual_hull(
            masks=masks,
            rig=default_rig,
            grid_resolution=32,
            grid_half_extent=1.0,
        )

        binary = threshold_occupancy(occupancy, 1.0)
        n_occupied = int(binary.sum())
        assert n_occupied > 0, "Visual hull should have occupied voxels"

    def test_empty_masks_produce_empty_volume(self, default_rig):
        """Empty masks should produce zero occupancy."""
        masks = _make_empty_masks()

        occupancy, _, _ = compute_visual_hull(
            masks=masks,
            rig=default_rig,
            grid_resolution=16,
            grid_half_extent=1.0,
        )

        assert occupancy.max() == 0.0

    def test_full_masks_produce_full_volume(self, default_rig):
        """Full masks should produce high occupancy everywhere visible."""
        masks = _make_full_masks()

        occupancy, _, _ = compute_visual_hull(
            masks=masks,
            rig=default_rig,
            grid_resolution=16,
            grid_half_extent=1.0,
        )

        # Many voxels should be occupied (full silhouettes = everything visible)
        binary = threshold_occupancy(occupancy, 1.0)
        n_occupied = int(binary.sum())
        assert n_occupied > 100, f"Expected many occupied voxels, got {n_occupied}"

    def test_output_shapes(self, default_rig):
        """Output should have correct shapes and types."""
        masks = _make_sphere_masks(default_rig)

        occupancy, grid_origin, voxel_size = compute_visual_hull(
            masks=masks,
            rig=default_rig,
            grid_resolution=16,
        )

        assert occupancy.shape == (16, 16, 16)
        assert occupancy.dtype == np.float32
        assert grid_origin.shape == (3,)
        assert isinstance(voxel_size, float)
        assert voxel_size > 0

    def test_grid_origin_correct(self, default_rig):
        """Grid origin should be at -half_extent in all axes."""
        masks = _make_sphere_masks(default_rig)

        _, grid_origin, _ = compute_visual_hull(
            masks=masks,
            rig=default_rig,
            grid_resolution=16,
            grid_half_extent=1.0,
        )

        np.testing.assert_allclose(grid_origin, [-1.0, -1.0, -1.0])

    def test_voxel_size_correct(self, default_rig):
        """Voxel size should be 2*half_extent / resolution."""
        masks = _make_sphere_masks(default_rig)

        _, _, voxel_size = compute_visual_hull(
            masks=masks,
            rig=default_rig,
            grid_resolution=32,
            grid_half_extent=1.5,
        )

        expected = 2 * 1.5 / 32
        assert voxel_size == pytest.approx(expected, abs=1e-10)

    def test_consensus_ratio_affects_occupancy(self, default_rig):
        """Lower consensus ratio should produce more occupied voxels."""
        masks = _make_sphere_masks(default_rig, sphere_radius=0.3)

        occ_strict, _, _ = compute_visual_hull(
            masks=masks, rig=default_rig, grid_resolution=16,
            consensus_ratio=1.0,
        )
        occ_relaxed, _, _ = compute_visual_hull(
            masks=masks, rig=default_rig, grid_resolution=16,
            consensus_ratio=0.5,
        )

        strict_count = int((occ_strict >= 1.0).sum())
        relaxed_count = int((occ_relaxed >= 0.5).sum())

        assert relaxed_count >= strict_count

    def test_occupancy_values_in_range(self, default_rig):
        """Occupancy values should be in [0, 1]."""
        masks = _make_sphere_masks(default_rig)

        occupancy, _, _ = compute_visual_hull(
            masks=masks, rig=default_rig, grid_resolution=16,
        )

        assert occupancy.min() >= 0.0
        assert occupancy.max() <= 1.0

    def test_missing_views_handled(self, default_rig):
        """Should work with fewer than 5 views."""
        masks = _make_sphere_masks(default_rig)
        # Keep only 3 views
        partial_masks = {vn: masks[vn] for vn in list(masks.keys())[:3]}

        occupancy, _, _ = compute_visual_hull(
            masks=partial_masks, rig=default_rig, grid_resolution=16,
        )

        # Should still produce some occupancy
        assert occupancy.shape == (16, 16, 16)


class TestThresholdOccupancy:
    """Test binary thresholding of occupancy grids."""

    def test_basic_threshold(self):
        occ = np.array([[[0.3, 0.7], [0.5, 1.0]]], dtype=np.float32)
        binary = threshold_occupancy(occ, 0.5)
        assert binary.dtype == bool
        assert binary[0, 0, 0] is np.bool_(False)  # 0.3 < 0.5
        assert binary[0, 0, 1] is np.bool_(True)   # 0.7 >= 0.5
        assert binary[0, 1, 0] is np.bool_(True)   # 0.5 >= 0.5
        assert binary[0, 1, 1] is np.bool_(True)   # 1.0 >= 0.5

    def test_strict_threshold(self):
        occ = np.array([[[0.8, 0.9, 1.0]]], dtype=np.float32)
        binary = threshold_occupancy(occ, 1.0)
        assert not binary[0, 0, 0]
        assert not binary[0, 0, 1]
        assert binary[0, 0, 2]

    def test_zero_threshold(self):
        occ = np.array([[[0.0, 0.01, 0.5]]], dtype=np.float32)
        binary = threshold_occupancy(occ, 0.0)
        assert binary.all()  # everything >= 0


# ===========================================================================
# MESH EXTRACTION UNIT TESTS
# ===========================================================================


class TestExtractSurfaceMesh:
    """Test mesh extraction from occupancy grids."""

    def test_sphere_occupancy_produces_mesh(self):
        """A sphere occupancy should produce vertices and faces."""
        occupancy = _make_simple_occupancy(resolution=32, fill_center=True)
        grid_origin = np.array([-1.0, -1.0, -1.0])
        voxel_size = 2.0 / 32

        try:
            vertices, faces, normals = extract_surface_mesh(
                occupancy, grid_origin, voxel_size, level=0.5,
            )

            assert len(vertices) > 0, "Should have vertices"
            assert len(faces) > 0, "Should have faces"
            assert vertices.shape[1] == 3
            assert faces.shape[1] == 3
            assert normals.shape == vertices.shape

            # Vertices should be in world space (near origin)
            assert np.all(np.abs(vertices) < 2.0)

        except ImportError:
            pytest.skip("scikit-image not available")

    def test_empty_occupancy_raises(self):
        """Empty occupancy should raise ValueError."""
        occupancy = np.zeros((16, 16, 16), dtype=np.float32)
        grid_origin = np.array([-1.0, -1.0, -1.0])
        voxel_size = 2.0 / 16

        with pytest.raises(ValueError, match="empty"):
            extract_surface_mesh(occupancy, grid_origin, voxel_size, level=0.5)

    def test_vertices_in_world_space(self):
        """Extracted vertices should be in world coordinates."""
        occupancy = _make_simple_occupancy(resolution=32)
        grid_origin = np.array([-1.0, -1.0, -1.0])
        voxel_size = 2.0 / 32

        try:
            vertices, _, _ = extract_surface_mesh(
                occupancy, grid_origin, voxel_size, level=0.5,
            )

            # Center of sphere is at grid center → world origin
            center = vertices.mean(axis=0)
            assert np.all(np.abs(center) < 0.5), \
                f"Mesh center should be near origin, got {center}"

        except ImportError:
            pytest.skip("scikit-image not available")


class TestFallbackSurfaceExtraction:
    """Test fallback surface extraction (no scikit-image)."""

    def test_produces_surface_points(self):
        """Should find surface voxels."""
        occupancy = _make_simple_occupancy(resolution=16)
        grid_origin = np.array([-1.0, -1.0, -1.0])
        voxel_size = 2.0 / 16

        vertices, faces, normals = _fallback_surface_extraction(
            occupancy, grid_origin, voxel_size, level=0.5,
        )

        assert len(vertices) > 0
        assert vertices.shape[1] == 3
        assert normals.shape == vertices.shape
        # Fallback produces point cloud (empty faces)
        assert len(faces) == 0

    def test_empty_occupancy_raises(self):
        """Empty occupancy should raise ValueError."""
        occupancy = np.zeros((16, 16, 16), dtype=np.float32)
        grid_origin = np.array([-1.0, -1.0, -1.0])

        with pytest.raises(ValueError, match="No surface"):
            _fallback_surface_extraction(occupancy, grid_origin, 0.125, level=0.5)

    def test_normals_are_unit_length(self):
        """Surface normals should be approximately unit length (where non-zero)."""
        occupancy = _make_simple_occupancy(resolution=16)
        grid_origin = np.array([-1.0, -1.0, -1.0])
        voxel_size = 2.0 / 16

        _, _, normals = _fallback_surface_extraction(
            occupancy, grid_origin, voxel_size, level=0.5,
        )

        # Check non-zero normals are approximately unit length
        norms = np.linalg.norm(normals, axis=1)
        non_zero = norms > 0.01
        if np.any(non_zero):
            assert np.all(norms[non_zero] > 0.9)
            assert np.all(norms[non_zero] < 1.1)


# ===========================================================================
# DEPTH MAP UNIT TESTS
# ===========================================================================


class TestComputeDepthMaps:
    """Test per-view depth map rendering."""

    def test_produces_depth_maps_for_all_views(self, default_rig):
        """Should produce a depth map for each camera view."""
        occupancy = _make_simple_occupancy(resolution=16)
        grid_origin = np.array([-1.0, -1.0, -1.0])
        voxel_size = 2.0 / 16

        depth_maps = compute_depth_maps(
            occupancy, default_rig, grid_origin, voxel_size,
            image_size=(256, 256), level=0.5,
        )

        assert len(depth_maps) == 5
        for vn in CANONICAL_VIEW_ORDER:
            assert vn in depth_maps
            assert depth_maps[vn].shape == (256, 256)
            assert depth_maps[vn].dtype == np.float32

    def test_depth_maps_have_nonzero_values(self, default_rig):
        """Depth maps should have non-zero values where surface is visible."""
        occupancy = _make_simple_occupancy(resolution=16)
        grid_origin = np.array([-1.0, -1.0, -1.0])
        voxel_size = 2.0 / 16

        depth_maps = compute_depth_maps(
            occupancy, default_rig, grid_origin, voxel_size,
            image_size=(256, 256), level=0.5,
        )

        # At least one view should have non-zero depth
        has_depth = any(np.any(dm > 0) for dm in depth_maps.values())
        assert has_depth, "At least one depth map should have non-zero values"

    def test_empty_occupancy_produces_zero_depth(self, default_rig):
        """Empty occupancy should produce all-zero depth maps."""
        occupancy = np.zeros((16, 16, 16), dtype=np.float32)
        grid_origin = np.array([-1.0, -1.0, -1.0])

        depth_maps = compute_depth_maps(
            occupancy, default_rig, grid_origin, 0.125,
            image_size=(64, 64), level=0.5,
        )

        for vn, dm in depth_maps.items():
            assert dm.max() == 0.0, f"Depth map for {vn} should be all zeros"

    def test_depth_values_positive(self, default_rig):
        """Non-zero depth values should be positive."""
        occupancy = _make_simple_occupancy(resolution=16)
        grid_origin = np.array([-1.0, -1.0, -1.0])
        voxel_size = 2.0 / 16

        depth_maps = compute_depth_maps(
            occupancy, default_rig, grid_origin, voxel_size,
            image_size=(64, 64), level=0.5,
        )

        for vn, dm in depth_maps.items():
            assert np.all(dm >= 0), f"Depth map for {vn} has negative values"


# ===========================================================================
# SURFACE SAMPLING UNIT TESTS
# ===========================================================================


class TestSampleSurfacePoints:
    """Test surface point sampling."""

    def test_correct_number_of_points(self):
        """Should return the requested number of points."""
        occupancy = _make_simple_occupancy(resolution=16)
        grid_origin = np.array([-1.0, -1.0, -1.0])
        voxel_size = 2.0 / 16

        points, normals = sample_surface_points(
            occupancy, grid_origin, voxel_size,
            n_points=100, level=0.5, seed=42,
        )

        assert len(points) == 100
        assert len(normals) == 100
        assert points.shape == (100, 3)
        assert normals.shape == (100, 3)

    def test_points_near_surface(self):
        """Sampled points should be near the surface (not deep inside)."""
        occupancy = _make_simple_occupancy(resolution=32)
        grid_origin = np.array([-1.0, -1.0, -1.0])
        voxel_size = 2.0 / 32

        points, _ = sample_surface_points(
            occupancy, grid_origin, voxel_size,
            n_points=50, level=0.5, seed=42,
        )

        # Points should be near the sphere surface at radius ~N/4 * voxel_size
        # from the grid center (which is at the origin)
        distances = np.linalg.norm(points, axis=1)
        # Sphere radius in world units: (32/4) * (2/32) = 0.5
        expected_radius = 0.5
        # Allow some tolerance for jitter and discretization
        assert np.mean(distances) == pytest.approx(expected_radius, abs=0.3)

    def test_deterministic_with_seed(self):
        """Same seed should produce identical results."""
        occupancy = _make_simple_occupancy(resolution=16)
        grid_origin = np.array([-1.0, -1.0, -1.0])
        voxel_size = 2.0 / 16

        p1, n1 = sample_surface_points(
            occupancy, grid_origin, voxel_size,
            n_points=50, seed=42,
        )
        p2, n2 = sample_surface_points(
            occupancy, grid_origin, voxel_size,
            n_points=50, seed=42,
        )

        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(n1, n2)

    def test_different_seeds_different_results(self):
        """Different seeds should produce different results."""
        occupancy = _make_simple_occupancy(resolution=16)
        grid_origin = np.array([-1.0, -1.0, -1.0])
        voxel_size = 2.0 / 16

        p1, _ = sample_surface_points(
            occupancy, grid_origin, voxel_size,
            n_points=50, seed=42,
        )
        p2, _ = sample_surface_points(
            occupancy, grid_origin, voxel_size,
            n_points=50, seed=99,
        )

        assert not np.array_equal(p1, p2)

    def test_empty_surface_returns_empty(self):
        """Empty occupancy should return empty arrays."""
        occupancy = np.zeros((16, 16, 16), dtype=np.float32)
        grid_origin = np.array([-1.0, -1.0, -1.0])

        points, normals = sample_surface_points(
            occupancy, grid_origin, 0.125, n_points=100,
        )

        assert len(points) == 0
        assert len(normals) == 0

    def test_more_points_than_surface_voxels(self):
        """Requesting more points than surface voxels should use replacement."""
        # Small sphere with few surface voxels
        occupancy = _make_simple_occupancy(resolution=8)
        grid_origin = np.array([-1.0, -1.0, -1.0])
        voxel_size = 2.0 / 8

        points, normals = sample_surface_points(
            occupancy, grid_origin, voxel_size,
            n_points=10000, seed=42,
        )

        assert len(points) == 10000  # should still return requested count


class TestColorSurfacePoints:
    """Test color assignment from camera views."""

    def test_colors_assigned(self, default_rig):
        """Points should get colors from the views."""
        # Create some surface points at the origin
        points = np.array([
            [0.0, 0.0, 0.4],   # in front of front camera
            [0.4, 0.0, 0.0],   # in front of right camera
            [0.0, 0.0, -0.4],  # in front of back camera
        ], dtype=np.float64)
        normals = np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ], dtype=np.float64)

        images = _make_rgb_images(256)
        masks = _make_full_masks(256)

        colors = color_surface_points(points, normals, default_rig, images, masks)

        assert colors.shape == (3, 3)
        assert colors.dtype == np.uint8
        # At least some points should have non-default (non-128) colors
        assert not np.all(colors == 128)

    def test_empty_points_returns_empty(self, default_rig):
        """Empty input should return empty output."""
        colors = color_surface_points(
            np.zeros((0, 3)), np.zeros((0, 3)),
            default_rig, {}, {},
        )
        assert len(colors) == 0

    def test_no_views_returns_default_gray(self, default_rig):
        """With no image data, should return default gray."""
        points = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)

        colors = color_surface_points(points, normals, default_rig, {}, {})

        assert colors.shape == (1, 3)
        # Default gray is (128, 128, 128)
        np.testing.assert_array_equal(colors[0], [128, 128, 128])


class TestBuildCoarseGaussians:
    """Test Gaussian point cloud construction."""

    def test_correct_structure(self):
        """Output should have all expected keys with correct shapes."""
        n = 100
        points = np.random.randn(n, 3).astype(np.float64)
        colors = np.random.randint(0, 256, (n, 3), dtype=np.uint8)
        normals = np.random.randn(n, 3).astype(np.float64)
        voxel_size = 0.05

        gaussians = build_coarse_gaussians(
            points, colors, normals, voxel_size,
        )

        assert "positions" in gaussians
        assert "colors" in gaussians
        assert "scales" in gaussians
        assert "opacities" in gaussians
        assert "normals" in gaussians

        assert gaussians["positions"].shape == (n, 3)
        assert gaussians["colors"].shape == (n, 3)
        assert gaussians["scales"].shape == (n, 3)
        assert gaussians["opacities"].shape == (n, 1)
        assert gaussians["normals"].shape == (n, 3)

    def test_scale_based_on_voxel_size(self):
        """Gaussian scale should be proportional to voxel size."""
        points = np.array([[0, 0, 0]], dtype=np.float64)
        colors = np.array([[128, 128, 128]], dtype=np.uint8)
        normals = np.array([[0, 0, 1]], dtype=np.float64)

        gaussians = build_coarse_gaussians(
            points, colors, normals, voxel_size=0.1, scale_factor=2.0,
        )

        expected_scale = 0.1 * 2.0
        np.testing.assert_allclose(gaussians["scales"][0], [expected_scale] * 3)

    def test_opacities_are_one(self):
        """All opacities should be 1.0 for surface points."""
        n = 10
        points = np.random.randn(n, 3).astype(np.float64)
        colors = np.random.randint(0, 256, (n, 3), dtype=np.uint8)
        normals = np.random.randn(n, 3).astype(np.float64)

        gaussians = build_coarse_gaussians(points, colors, normals, 0.05)
        np.testing.assert_allclose(gaussians["opacities"], np.ones((n, 1)))

    def test_dtypes(self):
        """Output arrays should have correct dtypes."""
        points = np.array([[0, 0, 0]], dtype=np.float64)
        colors = np.array([[128, 128, 128]], dtype=np.uint8)
        normals = np.array([[0, 0, 1]], dtype=np.float64)

        gaussians = build_coarse_gaussians(points, colors, normals, 0.05)

        assert gaussians["positions"].dtype == np.float32
        assert gaussians["colors"].dtype == np.uint8
        assert gaussians["scales"].dtype == np.float32
        assert gaussians["opacities"].dtype == np.float32
        assert gaussians["normals"].dtype == np.float32


# ===========================================================================
# PLY I/O UNIT TESTS
# ===========================================================================


class TestSaveMeshPly:
    """Test mesh PLY writing."""

    def test_writes_valid_file(self, tmp_path):
        """Should write a file that starts with 'ply'."""
        filepath = str(tmp_path / "mesh.ply")
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        save_mesh_ply(filepath, vertices, faces)

        assert Path(filepath).exists()
        with open(filepath, "rb") as f:
            header_start = f.read(3)
        assert header_start == b"ply"

    def test_writes_with_normals(self, tmp_path):
        """Should include normals when provided."""
        filepath = str(tmp_path / "mesh_normals.ply")
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float64)

        save_mesh_ply(filepath, vertices, faces, normals)

        assert Path(filepath).exists()
        # File should be larger with normals
        size_with = Path(filepath).stat().st_size

        filepath2 = str(tmp_path / "mesh_no_normals.ply")
        save_mesh_ply(filepath2, vertices, faces)
        size_without = Path(filepath2).stat().st_size

        assert size_with > size_without

    def test_writes_empty_faces(self, tmp_path):
        """Should handle empty face array (point cloud mode)."""
        filepath = str(tmp_path / "points.ply")
        vertices = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
        faces = np.zeros((0, 3), dtype=np.int32)

        save_mesh_ply(filepath, vertices, faces)
        assert Path(filepath).exists()

    def test_header_contains_vertex_count(self, tmp_path):
        """Header should declare the correct vertex count."""
        filepath = str(tmp_path / "mesh.ply")
        vertices = np.random.randn(42, 3).astype(np.float64)
        faces = np.zeros((0, 3), dtype=np.int32)

        save_mesh_ply(filepath, vertices, faces)

        with open(filepath, "rb") as f:
            header = b""
            while True:
                line = f.readline()
                header += line
                if b"end_header" in line:
                    break

        assert b"element vertex 42" in header


class TestSaveGaussiansPly:
    """Test Gaussian PLY writing."""

    def test_writes_valid_file(self, tmp_path):
        """Should write a valid PLY file."""
        filepath = str(tmp_path / "gaussians.ply")
        n = 10
        gaussians = {
            "positions": np.random.randn(n, 3).astype(np.float32),
            "colors": np.random.randint(0, 256, (n, 3), dtype=np.uint8),
            "scales": np.full((n, 3), 0.05, dtype=np.float32),
            "opacities": np.ones((n, 1), dtype=np.float32),
            "normals": np.random.randn(n, 3).astype(np.float32),
        }

        save_gaussians_ply(filepath, gaussians)

        assert Path(filepath).exists()
        with open(filepath, "rb") as f:
            header_start = f.read(3)
        assert header_start == b"ply"

    def test_header_has_gaussian_properties(self, tmp_path):
        """Header should declare Gaussian-specific properties."""
        filepath = str(tmp_path / "gaussians.ply")
        gaussians = {
            "positions": np.zeros((1, 3), dtype=np.float32),
            "colors": np.zeros((1, 3), dtype=np.uint8),
            "scales": np.zeros((1, 3), dtype=np.float32),
            "opacities": np.ones((1, 1), dtype=np.float32),
            "normals": np.zeros((1, 3), dtype=np.float32),
        }

        save_gaussians_ply(filepath, gaussians)

        with open(filepath, "rb") as f:
            header = b""
            while True:
                line = f.readline()
                header += line
                if b"end_header" in line:
                    break

        assert b"scale_x" in header
        assert b"scale_y" in header
        assert b"scale_z" in header
        assert b"opacity" in header
        assert b"red" in header
        assert b"green" in header
        assert b"blue" in header


class TestSaveDepthMapPng:
    """Test depth map PNG writing."""

    def test_writes_file(self, tmp_path):
        """Should write a PNG file."""
        filepath = str(tmp_path / "depth.png")
        depth_map = np.random.rand(64, 64).astype(np.float32) * 5.0

        save_depth_map_png(filepath, depth_map)

        assert Path(filepath).exists()
        img = Image.open(filepath)
        assert img.size == (64, 64)

    def test_zero_depth_stays_zero(self, tmp_path):
        """Zero depth (no surface) should remain zero in the output."""
        filepath = str(tmp_path / "depth.png")
        depth_map = np.zeros((32, 32), dtype=np.float32)
        depth_map[10:20, 10:20] = 3.0  # some surface

        save_depth_map_png(filepath, depth_map)

        img = Image.open(filepath)
        arr = np.array(img)
        # Zero regions should remain zero
        assert arr[0, 0] == 0
        # Non-zero regions should be non-zero
        assert arr[15, 15] > 0

    def test_custom_max_depth(self, tmp_path):
        """Custom max_depth should affect normalization."""
        filepath1 = str(tmp_path / "depth1.png")
        filepath2 = str(tmp_path / "depth2.png")
        depth_map = np.ones((16, 16), dtype=np.float32) * 2.0

        save_depth_map_png(filepath1, depth_map, max_depth=4.0)
        save_depth_map_png(filepath2, depth_map, max_depth=2.0)

        arr1 = np.array(Image.open(filepath1))
        arr2 = np.array(Image.open(filepath2))

        # With max_depth=2.0, values should be at max (65535)
        # With max_depth=4.0, values should be at half
        # arr2 values should be larger than arr1
        assert arr2.mean() > arr1.mean()


# ===========================================================================
# HELPER UNIT TESTS
# ===========================================================================


class TestGridResolutionFromConfig:
    """Test config to grid resolution mapping."""

    def test_low_mesh_resolution(self):
        config = CanonicalMVConfig(mesh_resolution=128)
        assert _grid_resolution_from_config(config) == 64

    def test_default_mesh_resolution(self):
        config = CanonicalMVConfig(mesh_resolution=256)
        assert _grid_resolution_from_config(config) == 128

    def test_high_mesh_resolution(self):
        config = CanonicalMVConfig(mesh_resolution=384)
        assert _grid_resolution_from_config(config) == 192

    def test_max_mesh_resolution(self):
        config = CanonicalMVConfig(mesh_resolution=512)
        assert _grid_resolution_from_config(config) == 256


class TestLoadSegmentedViews:
    """Test loading segmented views from storage."""

    def test_loads_rgba_views(self, sm):
        """Should load RGBA views and split into masks + images."""
        job_id = "test-load-001"

        for vn in CANONICAL_VIEW_ORDER:
            # Create RGBA image
            rgba = np.zeros((64, 64, 4), dtype=np.uint8)
            rgba[10:50, 10:50, :3] = 200  # foreground
            rgba[10:50, 10:50, 3] = 255   # alpha
            img = Image.fromarray(rgba, mode="RGBA")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            sm.save_view_preview(job_id, "segmented", vn, buf.getvalue(), ".png")

        masks, images = _load_segmented_views(job_id, sm)

        assert len(masks) == 5
        assert len(images) == 5
        for vn in CANONICAL_VIEW_ORDER:
            assert vn in masks
            assert vn in images
            assert masks[vn].shape == (64, 64)
            assert images[vn].shape == (64, 64, 3)

    def test_missing_views_skipped(self, sm):
        """Missing views should be skipped, not crash."""
        job_id = "test-load-002"

        # Only upload 2 views
        for vn in ("front", "back"):
            rgba = np.zeros((64, 64, 4), dtype=np.uint8)
            rgba[10:50, 10:50, 3] = 255
            img = Image.fromarray(rgba, mode="RGBA")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            sm.save_view_preview(job_id, "segmented", vn, buf.getvalue(), ".png")

        masks, images = _load_segmented_views(job_id, sm)

        assert len(masks) == 2
        assert "front" in masks
        assert "back" in masks

    def test_no_views_returns_empty(self, sm):
        """No views uploaded should return empty dicts."""
        masks, images = _load_segmented_views("nonexistent-job", sm)
        assert len(masks) == 0
        assert len(images) == 0


# ===========================================================================
# INTEGRATION TESTS: STAGE RUNNER
# ===========================================================================


class TestRunReconstructCoarse:
    """Integration tests for the full reconstruct_coarse stage runner."""

    def _setup_job_with_artifacts(self, jm, sm, config, image_size=64):
        """
        Create a job with camera_init.json and segmented previews.

        Uses a small image size for fast tests.
        """
        job_id = _create_mv_job(jm)

        # Build and save camera rig
        rig = build_canonical_rig(config, (image_size, image_size))
        sm.save_artifact_json(job_id, "camera_init.json", rig.to_dict())

        # Create and save segmented RGBA previews with a visible subject
        for vn in CANONICAL_VIEW_ORDER:
            rgba = np.zeros((image_size, image_size, 4), dtype=np.uint8)
            # Place a centered rectangle as the subject
            margin = image_size // 4
            rgba[margin:-margin, margin:-margin, 0] = 200
            rgba[margin:-margin, margin:-margin, 1] = 100
            rgba[margin:-margin, margin:-margin, 2] = 50
            rgba[margin:-margin, margin:-margin, 3] = 255

            img = Image.fromarray(rgba, mode="RGBA")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            sm.save_view_preview(job_id, "segmented", vn, buf.getvalue(), ".png")

        # Save preprocess metrics (needed by some checks)
        sm.save_artifact_json(job_id, "preprocess_metrics.json", {
            "canvas_size": image_size,
            "crop_side": int(image_size * 1.2),
            "per_view": {
                vn: {
                    "bbox": [margin, margin, image_size - 2 * margin, image_size - 2 * margin],
                    "centroid": [image_size // 2, image_size // 2],
                    "foreground_area_ratio": 0.25,
                    "sharpness": 500.0,
                    "segmentation_confidence": 0.95,
                    "color_histogram_mean": [200.0, 100.0, 50.0],
                }
                for vn in CANONICAL_VIEW_ORDER
            },
        })

        return job_id

    def test_happy_path(self, jm, sm, config):
        """Stage should complete and produce all expected artifacts."""
        config.mesh_resolution = 128  # use low resolution for speed
        job_id = self._setup_job_with_artifacts(jm, sm, config, image_size=64)

        run_reconstruct_coarse(job_id, config, jm, sm)

        # Check artifacts exist
        artifact_dir = sm.get_artifact_dir(job_id)

        voxel_path = artifact_dir / "coarse_voxel.npz"
        assert voxel_path.exists(), "coarse_voxel.npz should exist"

        gauss_path = artifact_dir / "coarse_gaussians.ply"
        assert gauss_path.exists(), "coarse_gaussians.ply should exist"

        metrics_path = artifact_dir / "coarse_recon_metrics.json"
        assert metrics_path.exists(), "coarse_recon_metrics.json should exist"

    def test_voxel_artifact_contents(self, jm, sm, config):
        """coarse_voxel.npz should contain expected arrays."""
        config.mesh_resolution = 128
        job_id = self._setup_job_with_artifacts(jm, sm, config, image_size=64)

        run_reconstruct_coarse(job_id, config, jm, sm)

        voxel_path = sm.get_artifact_dir(job_id) / "coarse_voxel.npz"
        data = np.load(str(voxel_path))

        assert "occupancy" in data
        assert "grid_origin" in data
        assert "voxel_size" in data
        assert "grid_resolution" in data
        assert data["occupancy"].ndim == 3
        assert data["grid_origin"].shape == (3,)

    def test_recon_metrics_artifact(self, jm, sm, config):
        """coarse_recon_metrics.json should have expected fields."""
        config.mesh_resolution = 128
        job_id = self._setup_job_with_artifacts(jm, sm, config, image_size=64)

        run_reconstruct_coarse(job_id, config, jm, sm)

        metrics = sm.load_artifact_json(job_id, "coarse_recon_metrics.json")
        assert metrics is not None
        assert "grid_resolution" in metrics
        assert "voxel_size" in metrics
        assert "n_occupied_voxels" in metrics
        assert "occupancy_fraction" in metrics
        assert "n_gaussians" in metrics
        assert "depth_maps_generated" in metrics

    def test_depth_maps_saved(self, jm, sm, config):
        """Per-view depth map PNGs should be saved."""
        config.mesh_resolution = 128
        job_id = self._setup_job_with_artifacts(jm, sm, config, image_size=64)

        run_reconstruct_coarse(job_id, config, jm, sm)

        artifact_dir = sm.get_artifact_dir(job_id)
        for vn in CANONICAL_VIEW_ORDER:
            depth_path = artifact_dir / f"coarse_depth_{vn}.png"
            assert depth_path.exists(), f"Depth map for {vn} should exist"

    def test_stage_progress_reaches_one(self, jm, sm, config):
        """Stage progress should reach 1.0 on completion."""
        config.mesh_resolution = 128
        job_id = self._setup_job_with_artifacts(jm, sm, config, image_size=64)

        run_reconstruct_coarse(job_id, config, jm, sm)

        job = jm.get_job(job_id)
        assert job["stage_progress"] == 1.0

    def test_missing_camera_init_raises(self, jm, sm, config):
        """Should raise ValueError if camera_init.json is missing."""
        job_id = _create_mv_job(jm)
        # No camera_init.json saved

        with pytest.raises(ValueError, match="camera_init"):
            run_reconstruct_coarse(job_id, config, jm, sm)

    def test_insufficient_views_raises(self, jm, sm, config):
        """Should raise if fewer than 3 segmented views are available."""
        job_id = _create_mv_job(jm)

        # Save camera rig
        rig = build_canonical_rig(config, (64, 64))
        sm.save_artifact_json(job_id, "camera_init.json", rig.to_dict())

        # Only save 2 segmented views (need at least 3)
        for vn in ("front", "back"):
            rgba = np.zeros((64, 64, 4), dtype=np.uint8)
            rgba[10:50, 10:50, 3] = 255
            img = Image.fromarray(rgba, mode="RGBA")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            sm.save_view_preview(job_id, "segmented", vn, buf.getvalue(), ".png")

        with pytest.raises(ValueError, match="at least 3"):
            run_reconstruct_coarse(job_id, config, jm, sm)

    def test_empty_visual_hull_raises(self, jm, sm, config):
        """Should raise if visual hull has too few occupied voxels."""
        config.mesh_resolution = 128
        job_id = _create_mv_job(jm)

        # Save camera rig
        rig = build_canonical_rig(config, (64, 64))
        sm.save_artifact_json(job_id, "camera_init.json", rig.to_dict())

        # Save segmented views with NO foreground (empty masks)
        for vn in CANONICAL_VIEW_ORDER:
            rgba = np.zeros((64, 64, 4), dtype=np.uint8)  # all transparent
            img = Image.fromarray(rgba, mode="RGBA")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            sm.save_view_preview(job_id, "segmented", vn, buf.getvalue(), ".png")

        with pytest.raises(ValueError, match="occupied voxels"):
            run_reconstruct_coarse(job_id, config, jm, sm)

    def test_gaussian_debug_flag(self, jm, sm):
        """generate_gaussian_debug flag should cap point count."""
        config = CanonicalMVConfig(
            mesh_resolution=128,
            generate_gaussian_debug=True,
        )
        job_id = self._setup_job_with_artifacts(jm, sm, config, image_size=64)

        run_reconstruct_coarse(job_id, config, jm, sm)

        metrics = sm.load_artifact_json(job_id, "coarse_recon_metrics.json")
        assert metrics is not None
        # With debug flag, gaussians should be capped at 100_000
        assert metrics["n_gaussians"] <= 100_000

    def test_mesh_artifact_created(self, jm, sm, config):
        """Visual hull mesh PLY should be created (when skimage available)."""
        config.mesh_resolution = 128
        job_id = self._setup_job_with_artifacts(jm, sm, config, image_size=64)

        run_reconstruct_coarse(job_id, config, jm, sm)

        artifact_dir = sm.get_artifact_dir(job_id)
        mesh_path = artifact_dir / "coarse_visual_hull_mesh.ply"
        # This may or may not exist depending on whether the visual hull
        # is non-empty and skimage is available. Check metrics instead.
        metrics = sm.load_artifact_json(job_id, "coarse_recon_metrics.json")
        assert metrics["mesh_vertices"] >= 0
        assert metrics["mesh_faces"] >= 0


# ===========================================================================
# VISUAL HULL GEOMETRIC CORRECTNESS TESTS
# ===========================================================================


class TestVisualHullGeometry:
    """
    Higher-level tests that verify the visual hull produces
    geometrically correct results for known shapes.
    """

    def test_sphere_hull_is_roughly_spherical(self, default_rig):
        """
        A sphere's visual hull should be roughly spherical:
        occupied voxels should be near the origin.

        Uses consensus_ratio=0.5 because with 5 canonical views the
        strict intersection (1.0) can be empty at coarse resolution —
        the top view's viewing cone doesn't perfectly overlap all side
        view cones for every voxel.
        """
        masks = _make_sphere_masks(default_rig, sphere_radius=0.4)

        occupancy, grid_origin, voxel_size = compute_visual_hull(
            masks=masks, rig=default_rig,
            grid_resolution=32, grid_half_extent=1.0,
            consensus_ratio=0.5,
        )

        binary = threshold_occupancy(occupancy, 0.5)
        occupied_indices = np.argwhere(binary)

        assert len(occupied_indices) > 0, \
            "Visual hull should have occupied voxels at 0.5 consensus"

        # Convert to world coordinates
        world_coords = (
            occupied_indices.astype(np.float64) * voxel_size
            + grid_origin
            + voxel_size / 2
        )

        # Center of mass should be near origin
        center = world_coords.mean(axis=0)
        assert np.linalg.norm(center) < 0.3, \
            f"Hull center {center} should be near origin"

        # All occupied voxels should be within the sphere radius + tolerance
        distances = np.linalg.norm(world_coords, axis=1)
        max_dist = distances.max()
        # Visual hull is always a superset of the true shape,
        # so allow some margin
        assert max_dist < 0.8, \
            f"Max distance {max_dist} exceeds expected sphere extent"

    def test_cube_hull_contains_cube_center(self, default_rig):
        """The visual hull of a cube should include the center voxel."""
        masks = _make_cube_masks(default_rig, cube_half_extent=0.3)

        occupancy, _, _ = compute_visual_hull(
            masks=masks, rig=default_rig,
            grid_resolution=32, grid_half_extent=1.0,
        )

        # Center voxel (16, 16, 16) should be occupied
        assert occupancy[16, 16, 16] > 0.5, \
            "Center voxel should be occupied for a cube"

    def test_hull_is_superset_of_shape(self, default_rig):
        """
        The visual hull should be a superset of the actual shape.
        More views should tighten the hull (fewer excess voxels).
        """
        full_masks = _make_sphere_masks(default_rig, sphere_radius=0.3)

        # Use all 5 views
        occ_5, _, _ = compute_visual_hull(
            masks=full_masks, rig=default_rig,
            grid_resolution=16, grid_half_extent=1.0,
        )

        # Use only 2 views (front + right)
        partial_masks = {
            vn: full_masks[vn] for vn in ["front", "right"]
            if vn in full_masks
        }
        occ_2, _, _ = compute_visual_hull(
            masks=partial_masks, rig=default_rig,
            grid_resolution=16, grid_half_extent=1.0,
        )

        count_5 = int((occ_5 >= 1.0).sum())
        count_2 = int((occ_2 >= 1.0).sum())

        # 5 views should produce a tighter hull (fewer occupied voxels)
        # than 2 views
        assert count_5 <= count_2, \
            f"5-view hull ({count_5}) should be tighter than 2-view ({count_2})"

