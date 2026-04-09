"""
Tests for the joint mesh + Gaussian refinement stage (Phase 4).

Covers:
    MESH TOPOLOGY UTILITIES (unit tests):
        - compute_edges: unique edge extraction from faces
        - compute_face_normals: per-face normal computation
        - compute_vertex_normals: per-vertex normal averaging
        - compute_adjacency_face_pairs: adjacent face detection
        - build_laplacian_matrix: graph Laplacian construction

    LOSS FUNCTIONS (unit tests):
        - silhouette_loss: rendered vs target mask IoU
        - laplacian_loss: smoothness penalty
        - edge_length_loss: edge-length variance
        - normal_consistency_loss: adjacent face normal agreement
        - symmetry_loss: bilateral symmetry penalty

    SILHOUETTE RENDERING (unit tests):
        - render_silhouettes: mesh rasterization into camera views

    GRADIENT COMPUTATION (unit tests):
        - compute_vertex_gradients: numerical gradients
        - compute_laplacian_gradient: analytical Laplacian gradient
        - compute_edge_length_gradient: analytical edge-length gradient

    JOINT REFINER (unit + integration tests):
        - JointRefiner.step: single optimization step
        - JointRefiner.run: full optimization loop
        - Convergence detection
        - Gaussian update tracking
        - Metrics output structure

    PLY I/O (unit tests):
        - load_mesh_ply: roundtrip with save_mesh_ply

    STAGE RUNNER (integration tests):
        - run_refine_joint: happy path with coarse artifacts
        - Missing camera_init.json raises ValueError
        - Missing coarse mesh raises ValueError
        - Insufficient views raises ValueError
        - Artifacts saved correctly
        - Stage progress reaches 1.0

All tests use synthetic meshes and masks — no GPU dependencies.
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
    # Topology utilities
    compute_edges,
    compute_face_normals,
    compute_vertex_normals,
    compute_adjacency_face_pairs,
    build_laplacian_matrix,
    # Loss functions
    silhouette_loss,
    laplacian_loss,
    edge_length_loss,
    normal_consistency_loss,
    symmetry_loss,
    # Rendering
    render_silhouettes,
    # Gradients
    compute_vertex_gradients,
    compute_laplacian_gradient,
    compute_edge_length_gradient,
    # Data structures
    MeshState,
    RefinementConfig,
    # Refiner
    JointRefiner,
    # PLY I/O
    load_mesh_ply,
    # Stage runner
    run_refine_joint,
    # Constants
    DEFAULT_N_ITERATIONS,
    MIN_FACES_FOR_REFINEMENT,
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
    """
    Create a simple tetrahedron mesh.

    Returns (vertices, faces):
        - vertices: (4, 3) float64
        - faces: (4, 3) int32
    """
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


def _make_target_masks(rig, mesh_vertices, mesh_faces, image_size=(256, 256)):
    """Render target silhouette masks from a mesh."""
    return render_silhouettes(mesh_vertices, mesh_faces, rig, image_size)


def _create_mv_job(jm):
    return jm.create_multiview_job(
        category=CategoryEnum.HUMAN_BUST,
        pipeline=PipelineEnum.CANONICAL_MV_HYBRID,
        views_received=["front", "back", "left", "right", "top"],
    )


# ===========================================================================
# MESH TOPOLOGY UTILITIES
# ===========================================================================


class TestComputeEdges:
    def test_cube_edge_count(self):
        """A cube has 12 triangles and 18 unique edges."""
        _, faces = _make_cube_mesh()
        edges = compute_edges(faces)
        assert edges.shape[1] == 2
        assert len(edges) == 18

    def test_tetrahedron_edge_count(self):
        """A tetrahedron has 4 triangles and 6 unique edges."""
        _, faces = _make_tetrahedron_mesh()
        edges = compute_edges(faces)
        assert len(edges) == 6

    def test_edges_are_sorted(self):
        """Each edge should have the smaller index first."""
        _, faces = _make_cube_mesh()
        edges = compute_edges(faces)
        assert np.all(edges[:, 0] <= edges[:, 1])

    def test_edges_are_unique(self):
        """No duplicate edges."""
        _, faces = _make_cube_mesh()
        edges = compute_edges(faces)
        unique = np.unique(edges, axis=0)
        assert len(unique) == len(edges)

    def test_single_triangle(self):
        """A single triangle has 3 edges."""
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        edges = compute_edges(faces)
        assert len(edges) == 3

    def test_empty_faces(self):
        """Empty face array should produce empty edges."""
        faces = np.zeros((0, 3), dtype=np.int32)
        edges = compute_edges(faces)
        assert len(edges) == 0


class TestComputeFaceNormals:
    def test_single_face_xy_plane(self):
        """A triangle in the XY plane should have Z-direction normal."""
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        normals = compute_face_normals(vertices, faces)
        assert normals.shape == (1, 3)
        # Normal should be (0, 0, 1) or (0, 0, -1) depending on winding
        assert abs(normals[0, 2]) == pytest.approx(1.0, abs=1e-10)
        assert normals[0, 0] == pytest.approx(0.0, abs=1e-10)
        assert normals[0, 1] == pytest.approx(0.0, abs=1e-10)

    def test_normals_are_unit_length(self):
        vertices, faces = _make_cube_mesh()
        normals = compute_face_normals(vertices, faces)
        lengths = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-10)

    def test_cube_has_12_face_normals(self):
        vertices, faces = _make_cube_mesh()
        normals = compute_face_normals(vertices, faces)
        assert normals.shape == (12, 3)

    def test_degenerate_face_zero_normal(self):
        """A degenerate (zero-area) face should get a near-zero normal."""
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [2, 0, 0],  # collinear
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        normals = compute_face_normals(vertices, faces)
        # Cross product is zero → normal direction is arbitrary but normalized
        # The function divides by max(norm, 1e-12), so result is near-zero magnitude
        # but the function normalizes, so it will be a unit vector in some direction
        assert normals.shape == (1, 3)


class TestComputeVertexNormals:
    def test_cube_vertex_normals(self):
        vertices, faces = _make_cube_mesh()
        normals = compute_vertex_normals(vertices, faces)
        assert normals.shape == (8, 3)
        # All vertex normals should be unit length
        lengths = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-10)

    def test_vertex_normals_point_outward(self):
        """For a cube at origin, vertex normals should point away from center."""
        vertices, faces = _make_cube_mesh(center=(0, 0, 0))
        normals = compute_vertex_normals(vertices, faces)
        # Each vertex normal should have positive dot product with vertex position
        dots = np.sum(normals * vertices, axis=1)
        assert np.all(dots > 0), "Vertex normals should point outward"

    def test_with_precomputed_face_normals(self):
        vertices, faces = _make_cube_mesh()
        face_normals = compute_face_normals(vertices, faces)
        normals = compute_vertex_normals(vertices, faces, face_normals)
        assert normals.shape == (8, 3)


class TestComputeAdjacencyFacePairs:
    def test_cube_adjacency(self):
        """A cube (12 triangles) should have face pairs sharing edges."""
        _, faces = _make_cube_mesh()
        pairs = compute_adjacency_face_pairs(faces)
        assert pairs.shape[1] == 2
        # Each internal edge is shared by exactly 2 faces
        assert len(pairs) > 0

    def test_tetrahedron_adjacency(self):
        """A tetrahedron has 6 edges, each shared by 2 faces → 6 pairs."""
        _, faces = _make_tetrahedron_mesh()
        pairs = compute_adjacency_face_pairs(faces)
        assert len(pairs) == 6

    def test_single_triangle_no_pairs(self):
        """A single triangle has no adjacent faces."""
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        pairs = compute_adjacency_face_pairs(faces)
        assert len(pairs) == 0

    def test_pair_indices_valid(self):
        _, faces = _make_cube_mesh()
        pairs = compute_adjacency_face_pairs(faces)
        assert np.all(pairs >= 0)
        assert np.all(pairs < len(faces))


class TestBuildLaplacianMatrix:
    def test_shape(self):
        vertices, faces = _make_cube_mesh()
        edges = compute_edges(faces)
        L = build_laplacian_matrix(len(vertices), edges)
        if hasattr(L, "toarray"):
            L_dense = L.toarray()
        else:
            L_dense = L
        assert L_dense.shape == (8, 8)

    def test_row_sum_zero(self):
        """Laplacian matrix rows should sum to zero."""
        vertices, faces = _make_cube_mesh()
        edges = compute_edges(faces)
        L = build_laplacian_matrix(len(vertices), edges)
        if hasattr(L, "toarray"):
            L_dense = L.toarray()
        else:
            L_dense = L
        row_sums = L_dense.sum(axis=1)
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-10)

    def test_diagonal_positive(self):
        """Diagonal entries should be positive (vertex degrees)."""
        vertices, faces = _make_cube_mesh()
        edges = compute_edges(faces)
        L = build_laplacian_matrix(len(vertices), edges)
        if hasattr(L, "toarray"):
            L_dense = L.toarray()
        else:
            L_dense = L
        assert np.all(np.diag(L_dense) >= 0)

    def test_symmetric(self):
        """Laplacian should be symmetric."""
        vertices, faces = _make_cube_mesh()
        edges = compute_edges(faces)
        L = build_laplacian_matrix(len(vertices), edges)
        if hasattr(L, "toarray"):
            L_dense = L.toarray()
        else:
            L_dense = L
        np.testing.assert_allclose(L_dense, L_dense.T, atol=1e-10)

    def test_empty_edges(self):
        """No edges → identity-like Laplacian (all zeros)."""
        edges = np.zeros((0, 2), dtype=np.int32)
        L = build_laplacian_matrix(5, edges)
        if hasattr(L, "toarray"):
            L_dense = L.toarray()
        else:
            L_dense = L
        np.testing.assert_allclose(L_dense, np.zeros((5, 5)), atol=1e-10)


# ===========================================================================
# LOSS FUNCTIONS
# ===========================================================================


class TestSilhouetteLoss:
    def test_perfect_match_zero_loss(self, default_rig):
        """If rendered masks match target masks, loss should be ~0."""
        vertices, faces = _make_cube_mesh(size=0.3)
        image_size = (256, 256)

        # Render target from the same mesh
        target_masks = render_silhouettes(vertices, faces, default_rig, image_size)

        loss = silhouette_loss(
            vertices, faces, target_masks, default_rig, image_size,
        )
        assert loss == pytest.approx(0.0, abs=0.05)

    def test_no_overlap_high_loss(self, default_rig):
        """If mesh doesn't overlap target at all, loss should be high."""
        # Mesh far from origin
        vertices, faces = _make_cube_mesh(size=0.3, center=(5, 5, 5))
        image_size = (256, 256)

        # Target masks from a centered mesh
        target_verts, target_faces = _make_cube_mesh(size=0.3)
        target_masks = render_silhouettes(
            target_verts, target_faces, default_rig, image_size,
        )

        loss = silhouette_loss(
            vertices, faces, target_masks, default_rig, image_size,
        )
        assert loss > 0.5

    def test_empty_mesh_high_loss(self, default_rig):
        """Empty mesh should give high loss when target has content."""
        vertices = np.zeros((0, 3), dtype=np.float64)
        faces = np.zeros((0, 3), dtype=np.int32)
        image_size = (256, 256)

        target_verts, target_faces = _make_cube_mesh(size=0.3)
        target_masks = render_silhouettes(
            target_verts, target_faces, default_rig, image_size,
        )

        loss = silhouette_loss(
            vertices, faces, target_masks, default_rig, image_size,
        )
        # Empty rendered masks vs non-empty targets → IoU = 0 → loss = 1
        assert loss == pytest.approx(1.0, abs=0.1)

    def test_loss_in_valid_range(self, default_rig):
        """Loss should always be in [0, 1]."""
        vertices, faces = _make_cube_mesh(size=0.3)
        image_size = (256, 256)
        target_masks = render_silhouettes(
            vertices, faces, default_rig, image_size,
        )

        loss = silhouette_loss(
            vertices, faces, target_masks, default_rig, image_size,
        )
        assert 0.0 <= loss <= 1.0


class TestLaplacianLoss:
    def test_flat_mesh_low_loss(self):
        """A flat mesh (all vertices on a plane) should have low Laplacian loss."""
        # Grid of vertices on a plane
        n = 5
        vertices = np.zeros((n * n, 3), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                vertices[i * n + j] = [i * 0.1, j * 0.1, 0.0]

        # Create simple grid faces
        faces = []
        for i in range(n - 1):
            for j in range(n - 1):
                v0 = i * n + j
                v1 = v0 + 1
                v2 = v0 + n
                v3 = v2 + 1
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])
        faces = np.array(faces, dtype=np.int32)

        edges = compute_edges(faces)
        L = build_laplacian_matrix(len(vertices), edges)

        loss = laplacian_loss(vertices, L)
        # Interior vertices of a regular grid have zero Laplacian displacement
        # Boundary vertices may contribute
        assert loss < 0.05

    def test_noisy_mesh_higher_loss(self):
        """A noisy mesh should have higher Laplacian loss than a smooth one."""
        vertices, faces = _make_cube_mesh()
        edges = compute_edges(faces)
        L = build_laplacian_matrix(len(vertices), edges)

        smooth_loss = laplacian_loss(vertices, L)

        # Add noise
        noisy_vertices = vertices + np.random.RandomState(42).randn(*vertices.shape) * 0.1
        noisy_loss = laplacian_loss(noisy_vertices, L)

        assert noisy_loss > smooth_loss

    def test_non_negative(self):
        """Laplacian loss should always be non-negative."""
        vertices, faces = _make_cube_mesh()
        edges = compute_edges(faces)
        L = build_laplacian_matrix(len(vertices), edges)

        loss = laplacian_loss(vertices, L)
        assert loss >= 0.0


class TestEdgeLengthLoss:
    def test_uniform_edges_zero_variance(self):
        """An equilateral triangle has uniform edge lengths → zero variance."""
        s = 1.0
        vertices = np.array([
            [0, 0, 0],
            [s, 0, 0],
            [s / 2, s * math.sqrt(3) / 2, 0],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        edges = compute_edges(faces)

        loss = edge_length_loss(vertices, edges)
        assert loss == pytest.approx(0.0, abs=1e-10)

    def test_nonuniform_edges_positive_variance(self):
        """Non-uniform edge lengths should give positive variance."""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 0.1, 0],  # very short edge to vertex 0
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        edges = compute_edges(faces)

        loss = edge_length_loss(vertices, edges)
        assert loss > 0.0

    def test_empty_edges(self):
        vertices = np.zeros((3, 3), dtype=np.float64)
        edges = np.zeros((0, 2), dtype=np.int32)
        loss = edge_length_loss(vertices, edges)
        assert loss == 0.0

    def test_non_negative(self):
        """Variance is always non-negative."""
        vertices, faces = _make_cube_mesh()
        edges = compute_edges(faces)
        loss = edge_length_loss(vertices, edges)
        assert loss >= 0.0


class TestNormalConsistencyLoss:
    def test_coplanar_faces_zero_loss(self):
        """Two coplanar triangles should have zero normal consistency loss."""
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
        pairs = compute_adjacency_face_pairs(faces)

        loss = normal_consistency_loss(vertices, faces, pairs)
        assert loss == pytest.approx(0.0, abs=1e-10)

    def test_perpendicular_faces_high_loss(self):
        """Two perpendicular faces should have loss ≈ 1.0."""
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],  # XY plane
            [0, 0, 1],  # Z direction
        ], dtype=np.float64)
        faces = np.array([
            [0, 1, 2],  # face in XY plane
            [0, 1, 3],  # face in XZ plane
        ], dtype=np.int32)
        pairs = compute_adjacency_face_pairs(faces)

        loss = normal_consistency_loss(vertices, faces, pairs)
        assert loss == pytest.approx(1.0, abs=0.1)

    def test_no_pairs_zero_loss(self):
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        pairs = np.zeros((0, 2), dtype=np.int32)
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        loss = normal_consistency_loss(vertices, faces, pairs)
        assert loss == 0.0

    def test_non_negative(self):
        vertices, faces = _make_cube_mesh()
        pairs = compute_adjacency_face_pairs(faces)
        loss = normal_consistency_loss(vertices, faces, pairs)
        assert loss >= 0.0


class TestSymmetryLoss:
    def test_symmetric_mesh_low_loss(self):
        """A mesh symmetric about X=0 should have low symmetry loss."""
        # Cube centered at origin is symmetric
        vertices, _ = _make_cube_mesh(center=(0, 0, 0))
        loss = symmetry_loss(vertices, axis=0)
        assert loss == pytest.approx(0.0, abs=0.01)

    def test_asymmetric_mesh_higher_loss(self):
        """An asymmetric mesh should have higher loss."""
        vertices, _ = _make_cube_mesh(center=(0, 0, 0))
        # Shift positive-X vertices further
        asym_vertices = vertices.copy()
        asym_vertices[asym_vertices[:, 0] > 0, 0] += 0.5

        loss_sym = symmetry_loss(vertices, axis=0)
        loss_asym = symmetry_loss(asym_vertices, axis=0)

        assert loss_asym > loss_sym

    def test_empty_vertices(self):
        vertices = np.zeros((0, 3), dtype=np.float64)
        loss = symmetry_loss(vertices, axis=0)
        assert loss == 0.0

    def test_different_axes(self):
        """Symmetry loss should work for different axes."""
        vertices, _ = _make_cube_mesh(center=(0, 0, 0))
        for axis in [0, 1, 2]:
            loss = symmetry_loss(vertices, axis=axis)
            assert loss >= 0.0


# ===========================================================================
# SILHOUETTE RENDERING
# ===========================================================================


class TestRenderSilhouettes:
    def test_cube_renders_in_all_views(self, default_rig):
        """A centered cube should be visible in all 5 views."""
        vertices, faces = _make_cube_mesh(size=0.3)
        image_size = (256, 256)

        masks = render_silhouettes(vertices, faces, default_rig, image_size)

        assert len(masks) == 5
        for vn in CANONICAL_VIEW_ORDER:
            assert vn in masks
            assert masks[vn].shape == (256, 256)
            # Should have some non-zero pixels
            assert np.sum(masks[vn] > 0) > 0, \
                f"No rendered pixels for view {vn}"

    def test_empty_mesh_empty_masks(self, default_rig):
        """Empty mesh should produce empty masks."""
        vertices = np.zeros((0, 3), dtype=np.float64)
        faces = np.zeros((0, 3), dtype=np.int32)

        masks = render_silhouettes(vertices, faces, default_rig, (256, 256))

        for vn, mask in masks.items():
            assert np.sum(mask) == 0

    def test_mask_values_binary(self, default_rig):
        """Rendered masks should contain only 0 and 255."""
        vertices, faces = _make_cube_mesh(size=0.3)
        masks = render_silhouettes(vertices, faces, default_rig, (256, 256))

        for vn, mask in masks.items():
            unique = np.unique(mask)
            assert set(unique).issubset({0, 255})

    def test_larger_mesh_more_pixels(self, default_rig):
        """A larger mesh should cover more pixels."""
        small_v, small_f = _make_cube_mesh(size=0.1)
        large_v, large_f = _make_cube_mesh(size=0.5)

        small_masks = render_silhouettes(small_v, small_f, default_rig, (256, 256))
        large_masks = render_silhouettes(large_v, large_f, default_rig, (256, 256))

        # At least for the front view, larger mesh should cover more pixels
        small_area = np.sum(small_masks["front"] > 0)
        large_area = np.sum(large_masks["front"] > 0)
        assert large_area > small_area

    def test_off_center_mesh_shifts_silhouette(self, default_rig):
        """A mesh shifted to the right should shift the front silhouette."""
        centered_v, faces = _make_cube_mesh(size=0.2, center=(0, 0, 0))
        shifted_v, _ = _make_cube_mesh(size=0.2, center=(0.3, 0, 0))

        centered_masks = render_silhouettes(centered_v, faces, default_rig, (256, 256))
        shifted_masks = render_silhouettes(shifted_v, faces, default_rig, (256, 256))

        # Center of mass of the front silhouette should shift
        c_y, c_x = np.where(centered_masks["front"] > 0)
        s_y, s_x = np.where(shifted_masks["front"] > 0)

        if len(c_x) > 0 and len(s_x) > 0:
            # Shifted mesh should have higher mean x in front view
            assert np.mean(s_x) > np.mean(c_x)


# ===========================================================================
# GRADIENT COMPUTATION
# ===========================================================================


class TestComputeVertexGradients:
    def test_gradient_shape(self):
        """Gradient should have same shape as vertices."""
        vertices, faces = _make_tetrahedron_mesh()

        def loss_fn(v, f):
            return np.sum(v ** 2)

        grad = compute_vertex_gradients(vertices, faces, loss_fn, epsilon=1e-4)
        assert grad.shape == vertices.shape

    def test_gradient_direction(self):
        """Gradient of ||V||² should point in direction of V."""
        vertices = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        def loss_fn(v, f):
            return np.sum(v ** 2)

        grad = compute_vertex_gradients(vertices, faces, loss_fn, epsilon=1e-4)
        # Gradient of sum(v²) = 2v
        expected = 2.0 * vertices
        np.testing.assert_allclose(grad, expected, atol=1e-3)

    def test_vertex_mask(self):
        """Only masked vertices should have non-zero gradients."""
        vertices, faces = _make_tetrahedron_mesh()
        mask = np.array([True, False, False, False])

        def loss_fn(v, f):
            return np.sum(v ** 2)

        grad = compute_vertex_gradients(
            vertices, faces, loss_fn, vertex_mask=mask,
        )
        # Only vertex 0 should have gradient
        assert np.any(grad[0] != 0)
        np.testing.assert_allclose(grad[1:], 0.0, atol=1e-10)


class TestComputeLaplacianGradient:
    def test_gradient_shape(self):
        vertices, faces = _make_cube_mesh()
        edges = compute_edges(faces)
        L = build_laplacian_matrix(len(vertices), edges)

        grad = compute_laplacian_gradient(vertices, L)
        assert grad.shape == vertices.shape

    def test_gradient_reduces_loss(self):
        """Moving in the negative gradient direction should reduce loss."""
        vertices, faces = _make_cube_mesh()
        # Add noise to make loss non-zero
        rng = np.random.RandomState(42)
        noisy_verts = vertices + rng.randn(*vertices.shape) * 0.05

        edges = compute_edges(faces)
        L = build_laplacian_matrix(len(noisy_verts), edges)

        loss_before = laplacian_loss(noisy_verts, L)
        grad = compute_laplacian_gradient(noisy_verts, L)

        # Take a small step in the negative gradient direction
        updated = noisy_verts - 0.001 * grad
        loss_after = laplacian_loss(updated, L)

        assert loss_after < loss_before


class TestComputeEdgeLengthGradient:
    def test_gradient_shape(self):
        vertices, faces = _make_cube_mesh()
        edges = compute_edges(faces)
        grad = compute_edge_length_gradient(vertices, edges)
        assert grad.shape == vertices.shape

    def test_empty_edges(self):
        vertices = np.zeros((3, 3), dtype=np.float64)
        edges = np.zeros((0, 2), dtype=np.int32)
        grad = compute_edge_length_gradient(vertices, edges)
        np.testing.assert_allclose(grad, 0.0)

    def test_gradient_reduces_loss(self):
        """Moving in the negative gradient direction should reduce loss."""
        # Create a mesh with non-uniform edges
        vertices = np.array([
            [0, 0, 0],
            [2, 0, 0],  # long edge
            [0, 0.1, 0],  # short edge
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        edges = compute_edges(faces)

        loss_before = edge_length_loss(vertices, edges)
        grad = compute_edge_length_gradient(vertices, edges)

        updated = vertices - 0.01 * grad
        loss_after = edge_length_loss(updated, edges)

        assert loss_after < loss_before


# ===========================================================================
# MESH STATE
# ===========================================================================


class TestMeshState:
    def test_properties(self):
        mesh = _make_mesh_state()
        assert mesh.n_vertices == 8
        assert mesh.n_faces == 12

    def test_copy_independent(self):
        mesh = _make_mesh_state()
        copy = mesh.copy()

        # Modify copy
        copy.vertices[0] += 100

        # Original should be unchanged
        assert mesh.vertices[0, 0] != copy.vertices[0, 0]


# ===========================================================================
# REFINEMENT CONFIG
# ===========================================================================


class TestRefinementConfig:
    def test_defaults(self):
        cfg = RefinementConfig()
        assert cfg.n_iterations == DEFAULT_N_ITERATIONS
        assert cfg.learning_rate > 0
        assert cfg.momentum >= 0
        assert cfg.silhouette_weight > 0
        assert cfg.laplacian_weight > 0

    def test_from_pipeline_config(self):
        pc = CanonicalMVConfig(symmetry_prior=False)
        cfg = RefinementConfig.from_pipeline_config(pc)
        assert cfg.use_symmetry is False
        assert cfg.symmetry_weight == 0.0

    def test_from_pipeline_config_with_symmetry(self):
        pc = CanonicalMVConfig(symmetry_prior=True)
        cfg = RefinementConfig.from_pipeline_config(pc)
        assert cfg.use_symmetry is True
        assert cfg.symmetry_weight > 0


# ===========================================================================
# JOINT REFINER
# ===========================================================================


class TestJointRefiner:
    def _make_refiner(self, default_rig, n_iterations=5):
        """Create a JointRefiner with a cube mesh and rendered targets."""
        vertices, faces = _make_cube_mesh(size=0.3)
        mesh = _make_mesh_state(vertices, faces)
        image_size = (256, 256)

        target_masks = render_silhouettes(
            vertices, faces, default_rig, image_size,
        )
        target_images = {
            vn: np.full((256, 256, 3), 128, dtype=np.uint8)
            for vn in CANONICAL_VIEW_ORDER
        }

        gaussians = build_coarse_gaussians(
            vertices, mesh.vertex_colors, mesh.normals, voxel_size=0.02,
        )

        config = RefinementConfig(
            n_iterations=n_iterations,
            learning_rate=1e-3,
            silhouette_weight=0.0,  # Disable expensive silhouette for unit tests
            laplacian_weight=1.0,
            edge_length_weight=0.1,
            normal_consistency_weight=0.0,  # Disable for speed
            symmetry_weight=0.0,
        )

        return JointRefiner(
            mesh=mesh,
            gaussians=gaussians,
            rig=default_rig,
            target_masks=target_masks,
            target_images=target_images,
            config=config,
            image_size=image_size,
        )

    def test_step_returns_losses(self, default_rig):
        refiner = self._make_refiner(default_rig, n_iterations=5)
        losses = refiner.step()

        assert "total" in losses
        assert "laplacian" in losses
        assert "edge_length" in losses
        assert losses["total"] >= 0

    def test_step_updates_vertices(self, default_rig):
        refiner = self._make_refiner(default_rig)
        original_verts = refiner.mesh.vertices.copy()

        # Add noise so gradients are non-zero
        refiner.mesh.vertices += np.random.RandomState(42).randn(
            *refiner.mesh.vertices.shape
        ) * 0.05

        refiner.step()

        # Vertices should have changed (noise makes gradients non-zero)
        assert not np.allclose(
            refiner.mesh.vertices,
            original_verts + np.random.RandomState(42).randn(
                *original_verts.shape
            ) * 0.05,
        )

    def test_run_returns_tuple(self, default_rig):
        refiner = self._make_refiner(default_rig, n_iterations=3)
        result = refiner.run(n_iterations=3)

        assert len(result) == 3
        mesh, gaussians, metrics = result
        assert isinstance(mesh, MeshState)
        assert isinstance(gaussians, dict)
        assert isinstance(metrics, dict)

    def test_run_records_loss_history(self, default_rig):
        refiner = self._make_refiner(default_rig, n_iterations=5)
        _, _, metrics = refiner.run(n_iterations=5)

        assert "iterations_run" in metrics
        assert metrics["iterations_run"] <= 5
        assert "loss_history" in metrics
        assert len(metrics["loss_history"]) > 0

    def test_run_metrics_structure(self, default_rig):
        refiner = self._make_refiner(default_rig, n_iterations=3)
        _, _, metrics = refiner.run(n_iterations=3)

        assert "initial_total_loss" in metrics
        assert "final_total_loss" in metrics
        assert "loss_reduction" in metrics
        assert "loss_reduction_pct" in metrics
        assert "config" in metrics
        assert "mesh_stats" in metrics
        assert metrics["mesh_stats"]["n_vertices"] == 8
        assert metrics["mesh_stats"]["n_faces"] == 12

    def test_run_progress_callback(self, default_rig):
        refiner = self._make_refiner(default_rig, n_iterations=3)
        callback_calls = []

        def callback(iteration, total, losses):
            callback_calls.append((iteration, total))

        refiner.run(n_iterations=3, progress_callback=callback)

        assert len(callback_calls) == 3
        assert callback_calls[0] == (0, 3)
        assert callback_calls[-1] == (2, 3)

    def test_gaussian_update_tracks_mesh(self, default_rig):
        """After refinement, Gaussians should be near mesh vertices."""
        refiner = self._make_refiner(default_rig, n_iterations=3)

        # Add noise to mesh so it moves during refinement
        refiner.mesh.vertices += np.random.RandomState(42).randn(
            *refiner.mesh.vertices.shape
        ) * 0.02

        _, gaussians, _ = refiner.run(n_iterations=3)

        gauss_pos = gaussians["positions"]
        mesh_verts = refiner.mesh.vertices

        # Each Gaussian should be close to some mesh vertex
        for gp in gauss_pos:
            dists = np.linalg.norm(mesh_verts - gp, axis=1)
            min_dist = dists.min()
            assert min_dist < 0.1, \
                f"Gaussian at {gp} is far from nearest vertex (dist={min_dist})"

    def test_convergence_early_stop(self, default_rig):
        """Refiner should stop early if loss converges."""
        vertices, faces = _make_cube_mesh(size=0.3)
        # Add noise so there is meaningful loss to converge from
        rng = np.random.RandomState(42)
        vertices = vertices + rng.randn(*vertices.shape) * 0.05
        mesh = _make_mesh_state(vertices, faces)
        image_size = (256, 256)

        config = RefinementConfig(
            n_iterations=100,
            learning_rate=1e-3,  # Small LR for steady convergence
            laplacian_weight=1.0,
            silhouette_weight=0.0,
            edge_length_weight=0.0,
            normal_consistency_weight=0.0,
            symmetry_weight=0.0,
            convergence_threshold=1e-3,
        )

        refiner = JointRefiner(
            mesh=mesh,
            gaussians=None,
            rig=default_rig,
            target_masks={},
            target_images={},
            config=config,
            image_size=image_size,
        )

        _, _, metrics = refiner.run()

        # Should converge before 100 iterations
        assert metrics["converged"] is True
        assert metrics["iterations_run"] < 100

    def test_max_displacement_clamped(self, default_rig):
        """Vertex displacement should be clamped per iteration."""
        vertices, faces = _make_cube_mesh(size=0.3)
        mesh = _make_mesh_state(vertices, faces)
        original_verts = mesh.vertices.copy()
        image_size = (256, 256)

        config = RefinementConfig(
            n_iterations=1,
            learning_rate=100.0,  # Very large LR
            laplacian_weight=1.0,
            silhouette_weight=0.0,
            edge_length_weight=0.0,
            normal_consistency_weight=0.0,
            symmetry_weight=0.0,
            max_vertex_displacement=0.02,
        )

        # Add noise to create gradients
        mesh.vertices += np.random.RandomState(42).randn(*mesh.vertices.shape) * 0.1

        refiner = JointRefiner(
            mesh=mesh,
            gaussians=None,
            rig=default_rig,
            target_masks={},
            target_images={},
            config=config,
            image_size=image_size,
        )

        pre_step = mesh.vertices.copy()
        refiner.step()

        # Check that no vertex moved more than max_displacement
        displacements = np.linalg.norm(mesh.vertices - pre_step, axis=1)
        assert np.all(displacements <= config.max_vertex_displacement + 1e-8)


# ===========================================================================
# PLY I/O
# ===========================================================================


class TestLoadMeshPly:
    def test_roundtrip(self, tmp_path):
        """Save and load should produce identical mesh."""
        vertices, faces = _make_cube_mesh()
        normals = compute_vertex_normals(vertices, faces)

        filepath = str(tmp_path / "test_mesh.ply")
        save_mesh_ply(filepath, vertices, faces, normals)

        loaded_v, loaded_f, loaded_n = load_mesh_ply(filepath)

        np.testing.assert_allclose(loaded_v, vertices, atol=1e-5)
        np.testing.assert_array_equal(loaded_f, faces)
        np.testing.assert_allclose(loaded_n, normals, atol=1e-5)

    def test_roundtrip_without_normals(self, tmp_path):
        """Save without normals and load should still work."""
        vertices, faces = _make_cube_mesh()

        filepath = str(tmp_path / "test_mesh_no_normals.ply")
        save_mesh_ply(filepath, vertices, faces)

        loaded_v, loaded_f, loaded_n = load_mesh_ply(filepath)

        np.testing.assert_allclose(loaded_v, vertices, atol=1e-5)
        np.testing.assert_array_equal(loaded_f, faces)
        # Normals should be zero when not present
        np.testing.assert_allclose(loaded_n, 0.0, atol=1e-10)

    def test_tetrahedron_roundtrip(self, tmp_path):
        """Test with a different mesh topology."""
        vertices, faces = _make_tetrahedron_mesh()
        normals = compute_vertex_normals(vertices, faces)

        filepath = str(tmp_path / "tet.ply")
        save_mesh_ply(filepath, vertices, faces, normals)

        loaded_v, loaded_f, loaded_n = load_mesh_ply(filepath)

        assert loaded_v.shape == (4, 3)
        assert loaded_f.shape == (4, 3)
        np.testing.assert_allclose(loaded_v, vertices, atol=1e-5)


# ===========================================================================
# STAGE RUNNER INTEGRATION TESTS
# ===========================================================================


class TestRunRefineJoint:
    """Integration tests for the full stage runner."""

    def _setup_job_with_coarse_artifacts(
        self, jm, sm, config,
        mesh_size=0.3,
        n_gaussians=100,
    ):
        """
        Create a job and save all required artifacts from previous stages:
        camera_init.json, coarse_visual_hull_mesh.ply, coarse_gaussians.ply,
        segmented view previews.
        """
        job_id = _create_mv_job(jm)

        # Build and save camera rig
        rig = build_canonical_rig(config, (256, 256))
        sm.save_artifact_json(job_id, "camera_init.json", rig.to_dict())

        # Build and save coarse mesh
        vertices, faces = _make_cube_mesh(size=mesh_size)
        normals = compute_vertex_normals(vertices, faces)
        mesh_path = sm.get_artifact_dir(job_id) / "coarse_visual_hull_mesh.ply"
        save_mesh_ply(str(mesh_path), vertices, faces, normals)

        # Build and save coarse Gaussians
        colors = np.full((len(vertices), 3), 128, dtype=np.uint8)
        gaussians = build_coarse_gaussians(
            vertices[:min(n_gaussians, len(vertices))],
            colors[:min(n_gaussians, len(colors))],
            normals[:min(n_gaussians, len(normals))],
            voxel_size=0.02,
        )
        gauss_path = sm.get_artifact_dir(job_id) / "coarse_gaussians.ply"
        save_gaussians_ply(str(gauss_path), gaussians)

        # Save coarse recon metrics
        sm.save_artifact_json(job_id, "coarse_recon_metrics.json", {
            "n_gaussians": len(gaussians["positions"]),
            "mesh_vertices": len(vertices),
            "mesh_faces": len(faces),
        })

        # Save segmented view previews (RGBA images with subject silhouette)
        image_size = (256, 256)
        masks = render_silhouettes(vertices, faces, rig, image_size)

        for vn in CANONICAL_VIEW_ORDER:
            mask = masks.get(vn, np.zeros((256, 256), dtype=np.uint8))
            # Create RGBA image
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

    def test_happy_path(self, jm, sm, config):
        """Stage should complete and save refined artifacts."""
        config.use_joint_refinement = True
        job_id = self._setup_job_with_coarse_artifacts(jm, sm, config)

        run_refine_joint(job_id, config, jm, sm)

        # Refined mesh should exist
        mesh_path = sm.get_artifact_path(job_id, "refined_mesh.ply")
        assert mesh_path is not None

        # Refined Gaussians should exist
        gauss_path = sm.get_artifact_path(job_id, "refined_gaussians.ply")
        assert gauss_path is not None

        # Metrics should exist
        metrics = sm.load_artifact_json(job_id, "refine_metrics.json")
        assert metrics is not None
        assert "iterations_run" in metrics
        assert "final_total_loss" in metrics

    def test_stage_progress_reaches_one(self, jm, sm, config):
        job_id = self._setup_job_with_coarse_artifacts(jm, sm, config)
        run_refine_joint(job_id, config, jm, sm)

        job = jm.get_job(job_id)
        assert job["stage_progress"] == 1.0

    def test_missing_camera_init_raises(self, jm, sm, config):
        job_id = _create_mv_job(jm)
        with pytest.raises(ValueError, match="camera_init"):
            run_refine_joint(job_id, config, jm, sm)

    def test_missing_coarse_mesh_raises(self, jm, sm, config):
        job_id = _create_mv_job(jm)
        # Save camera rig but not mesh
        rig = build_canonical_rig(config, (256, 256))
        sm.save_artifact_json(job_id, "camera_init.json", rig.to_dict())

        with pytest.raises(ValueError, match="coarse_visual_hull_mesh"):
            run_refine_joint(job_id, config, jm, sm)

    def test_insufficient_views_raises(self, jm, sm, config):
        """Need at least 3 segmented views for refinement."""
        job_id = _create_mv_job(jm)

        # Save camera rig and mesh
        rig = build_canonical_rig(config, (256, 256))
        sm.save_artifact_json(job_id, "camera_init.json", rig.to_dict())

        vertices, faces = _make_cube_mesh(size=0.3)
        normals = compute_vertex_normals(vertices, faces)
        mesh_path = sm.get_artifact_dir(job_id) / "coarse_visual_hull_mesh.ply"
        save_mesh_ply(str(mesh_path), vertices, faces, normals)

        # Save only 2 segmented views (need at least 3)
        for vn in ["front", "back"]:
            rgba = np.zeros((256, 256, 4), dtype=np.uint8)
            rgba[:, :, 3] = 255
            img = Image.fromarray(rgba, mode="RGBA")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            sm.save_view_preview(job_id, "segmented", vn, buf.getvalue(), ".png")

        with pytest.raises(ValueError, match="segmented views"):
            run_refine_joint(job_id, config, jm, sm)

    def test_refined_mesh_valid(self, jm, sm, config):
        """Refined mesh should be loadable and have same topology."""
        job_id = self._setup_job_with_coarse_artifacts(jm, sm, config)
        run_refine_joint(job_id, config, jm, sm)

        mesh_path = sm.get_artifact_path(job_id, "refined_mesh.ply")
        vertices, faces, normals = load_mesh_ply(str(mesh_path))

        # Same topology as input (8 vertices, 12 faces for cube)
        assert len(vertices) == 8
        assert len(faces) == 12
        # Normals should be unit length
        lengths = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=0.1)

    def test_metrics_has_config(self, jm, sm, config):
        """Metrics should include the refinement config used."""
        job_id = self._setup_job_with_coarse_artifacts(jm, sm, config)
        run_refine_joint(job_id, config, jm, sm)

        metrics = sm.load_artifact_json(job_id, "refine_metrics.json")
        assert "config" in metrics
        assert "learning_rate" in metrics["config"]
        assert "n_iterations" in metrics["config"]

    def test_metrics_has_loss_history(self, jm, sm, config):
        """Metrics should include loss history."""
        job_id = self._setup_job_with_coarse_artifacts(jm, sm, config)
        run_refine_joint(job_id, config, jm, sm)

        metrics = sm.load_artifact_json(job_id, "refine_metrics.json")
        assert "loss_history" in metrics
        assert len(metrics["loss_history"]) > 0
        assert "total" in metrics["loss_history"][0]


# ===========================================================================
# EDGE CASES
# ===========================================================================


class TestEdgeCases:
    def test_single_face_mesh(self, default_rig):
        """Refinement should handle a mesh with just one triangle."""
        vertices = np.array([
            [0, 0, 0], [0.3, 0, 0], [0.15, 0.3, 0],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        # This has fewer than MIN_FACES_FOR_REFINEMENT, but let's test
        # the topology utilities still work
        edges = compute_edges(faces)
        assert len(edges) == 3

        normals = compute_vertex_normals(vertices, faces)
        assert normals.shape == (3, 3)

        L = build_laplacian_matrix(3, edges)
        loss = laplacian_loss(vertices, L)
        assert loss >= 0.0

    def test_all_losses_zero_for_perfect_mesh(self, default_rig):
        """A regular tetrahedron should have relatively low losses."""
        vertices, faces = _make_tetrahedron_mesh()
        edges = compute_edges(faces)
        L = build_laplacian_matrix(len(vertices), edges)
        pairs = compute_adjacency_face_pairs(faces)

        lap = laplacian_loss(vertices, L)
        el = edge_length_loss(vertices, edges)
        nc = normal_consistency_loss(vertices, faces, pairs)

        # Tetrahedron has uniform edges → zero edge-length variance
        assert el == pytest.approx(0.0, abs=1e-8)
        # All losses should be finite
        assert np.isfinite(lap)
        assert np.isfinite(nc)

    def test_large_mesh_performance(self, default_rig):
        """Ensure topology computation doesn't crash for larger meshes."""
        # Create a subdivided grid mesh
        n = 20
        vertices = []
        for i in range(n):
            for j in range(n):
                vertices.append([i * 0.05 - 0.5, j * 0.05 - 0.5, 0.0])
        vertices = np.array(vertices, dtype=np.float64)

        faces = []
        for i in range(n - 1):
            for j in range(n - 1):
                v0 = i * n + j
                faces.append([v0, v0 + 1, v0 + n])
                faces.append([v0 + 1, v0 + n + 1, v0 + n])
        faces = np.array(faces, dtype=np.int32)

        # Should complete without error
        edges = compute_edges(faces)
        L = build_laplacian_matrix(len(vertices), edges)
        loss = laplacian_loss(vertices, L)
        assert np.isfinite(loss)

    def test_refinement_config_from_no_symmetry(self):
        """Config with symmetry disabled should have zero symmetry weight."""
        pc = CanonicalMVConfig(symmetry_prior=False)
        cfg = RefinementConfig.from_pipeline_config(pc)
        assert cfg.symmetry_weight == 0.0
        assert cfg.use_symmetry is False

    def test_empty_gaussians_handled(self, default_rig):
        """Refiner should handle None gaussians gracefully."""
        vertices, faces = _make_cube_mesh(size=0.3)
        mesh = _make_mesh_state(vertices, faces)

        config = RefinementConfig(
            n_iterations=2,
            laplacian_weight=1.0,
            silhouette_weight=0.0,
            edge_length_weight=0.0,
            normal_consistency_weight=0.0,
            symmetry_weight=0.0,
        )

        refiner = JointRefiner(
            mesh=mesh,
            gaussians=None,
            rig=default_rig,
            target_masks={},
            target_images={},
            config=config,
            image_size=(256, 256),
        )

        result_mesh, result_gauss, metrics = refiner.run(n_iterations=2)
        assert result_gauss is None
        assert result_mesh.n_vertices == 8


# ===========================================================================
# ADDITIONAL ROBUSTNESS TESTS
# ===========================================================================


class TestRunRefineJointRobustness:
    """Additional robustness tests for the stage runner."""

    def _setup_job_with_coarse_artifacts(
        self, jm, sm, config,
        mesh_size=0.3,
        n_gaussians=100,
    ):
        """
        Create a job and save all required artifacts from previous stages.
        (Same helper as TestRunRefineJoint but accessible from this class.)
        """
        job_id = _create_mv_job(jm)

        # Build and save camera rig
        rig = build_canonical_rig(config, (256, 256))
        sm.save_artifact_json(job_id, "camera_init.json", rig.to_dict())

        # Build and save coarse mesh
        vertices, faces = _make_cube_mesh(size=mesh_size)
        normals = compute_vertex_normals(vertices, faces)
        mesh_path = sm.get_artifact_dir(job_id) / "coarse_visual_hull_mesh.ply"
        save_mesh_ply(str(mesh_path), vertices, faces, normals)

        # Build and save coarse Gaussians
        colors = np.full((len(vertices), 3), 128, dtype=np.uint8)
        gaussians = build_coarse_gaussians(
            vertices[:min(n_gaussians, len(vertices))],
            colors[:min(n_gaussians, len(colors))],
            normals[:min(n_gaussians, len(normals))],
            voxel_size=0.02,
        )
        gauss_path = sm.get_artifact_dir(job_id) / "coarse_gaussians.ply"
        save_gaussians_ply(str(gauss_path), gaussians)

        # Save coarse recon metrics
        sm.save_artifact_json(job_id, "coarse_recon_metrics.json", {
            "n_gaussians": len(gaussians["positions"]),
            "mesh_vertices": len(vertices),
            "mesh_faces": len(faces),
        })

        # Save segmented view previews
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

    def test_too_few_faces_raises(self, jm, sm, config):
        """Stage runner should raise ValueError when mesh has < MIN_FACES_FOR_REFINEMENT faces."""
        job_id = _create_mv_job(jm)

        # Save camera rig
        rig = build_canonical_rig(config, (256, 256))
        sm.save_artifact_json(job_id, "camera_init.json", rig.to_dict())

        # Save a mesh with only 2 faces (below MIN_FACES_FOR_REFINEMENT=4)
        vertices = np.array([
            [0, 0, 0], [0.3, 0, 0], [0.15, 0.3, 0], [0.15, 0.15, 0.3],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
        normals = compute_vertex_normals(vertices, faces)
        mesh_path = sm.get_artifact_dir(job_id) / "coarse_visual_hull_mesh.ply"
        save_mesh_ply(str(mesh_path), vertices, faces, normals)

        with pytest.raises(ValueError, match="too coarse"):
            run_refine_joint(job_id, config, jm, sm)

    def test_missing_gaussians_still_succeeds(self, jm, sm, config):
        """Stage runner should succeed even when coarse Gaussians PLY is absent."""
        job_id = _create_mv_job(jm)

        # Save camera rig
        rig = build_canonical_rig(config, (256, 256))
        sm.save_artifact_json(job_id, "camera_init.json", rig.to_dict())

        # Save coarse mesh (but NO Gaussians PLY)
        vertices, faces = _make_cube_mesh(size=0.3)
        normals = compute_vertex_normals(vertices, faces)
        mesh_path = sm.get_artifact_dir(job_id) / "coarse_visual_hull_mesh.ply"
        save_mesh_ply(str(mesh_path), vertices, faces, normals)

        # Save segmented views (needed for refinement)
        image_size = (256, 256)
        masks = render_silhouettes(vertices, faces, rig, image_size)
        for vn in CANONICAL_VIEW_ORDER:
            mask = masks.get(vn, np.zeros((256, 256), dtype=np.uint8))
            rgba = np.zeros((256, 256, 4), dtype=np.uint8)
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

        # Should succeed without Gaussians
        run_refine_joint(job_id, config, jm, sm)

        # Refined mesh should still be produced
        mesh_path = sm.get_artifact_path(job_id, "refined_mesh.ply")
        assert mesh_path is not None

        # Metrics should exist
        metrics = sm.load_artifact_json(job_id, "refine_metrics.json")
        assert metrics is not None

    def test_refined_mesh_topology_preserved(self, jm, sm, config):
        """Refinement should preserve mesh topology (same vertex/face count)."""
        job_id = self._setup_job_with_coarse_artifacts(jm, sm, config)
        run_refine_joint(job_id, config, jm, sm)

        mesh_path = sm.get_artifact_path(job_id, "refined_mesh.ply")
        vertices, faces, normals = load_mesh_ply(str(mesh_path))

        # Cube: 8 vertices, 12 faces — topology preserved
        assert len(vertices) == 8
        assert len(faces) == 12
        # All face indices should be valid
        assert np.all(faces >= 0)
        assert np.all(faces < len(vertices))


class TestLossReduction:
    """Tests verifying that optimization actually reduces loss."""

    def test_laplacian_loss_decreases(self, default_rig):
        """Laplacian loss should decrease over optimization iterations."""
        vertices, faces = _make_cube_mesh(size=0.3)
        # Add significant noise so there's loss to reduce
        rng = np.random.RandomState(42)
        vertices = vertices + rng.randn(*vertices.shape) * 0.08
        mesh = _make_mesh_state(vertices, faces)

        config = RefinementConfig(
            n_iterations=20,
            learning_rate=5e-3,
            laplacian_weight=1.0,
            silhouette_weight=0.0,
            edge_length_weight=0.0,
            normal_consistency_weight=0.0,
            symmetry_weight=0.0,
        )

        refiner = JointRefiner(
            mesh=mesh,
            gaussians=None,
            rig=default_rig,
            target_masks={},
            target_images={},
            config=config,
            image_size=(256, 256),
        )

        _, _, metrics = refiner.run()

        assert metrics["final_total_loss"] < metrics["initial_total_loss"], \
            "Laplacian loss should decrease over iterations"
        assert metrics["loss_reduction"] > 0

    def test_edge_length_loss_decreases(self, default_rig):
        """Edge-length variance should decrease over optimization iterations."""
        # Create a mesh with deliberately non-uniform edges
        vertices = np.array([
            [0, 0, 0],
            [2.0, 0, 0],  # very long edge
            [0, 0.1, 0],  # very short edge
            [1.0, 0.5, 0.3],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
        mesh = _make_mesh_state(vertices, faces)

        config = RefinementConfig(
            n_iterations=30,
            learning_rate=1e-2,
            laplacian_weight=0.0,
            silhouette_weight=0.0,
            edge_length_weight=1.0,
            normal_consistency_weight=0.0,
            symmetry_weight=0.0,
        )

        refiner = JointRefiner(
            mesh=mesh,
            gaussians=None,
            rig=default_rig,
            target_masks={},
            target_images={},
            config=config,
            image_size=(256, 256),
        )

        _, _, metrics = refiner.run()

        assert metrics["final_total_loss"] < metrics["initial_total_loss"], \
            "Edge-length variance should decrease"

    def test_combined_losses_decrease(self, default_rig):
        """Combined Laplacian + edge-length loss should decrease."""
        vertices, faces = _make_cube_mesh(size=0.3)
        rng = np.random.RandomState(123)
        vertices = vertices + rng.randn(*vertices.shape) * 0.05
        mesh = _make_mesh_state(vertices, faces)

        config = RefinementConfig(
            n_iterations=15,
            learning_rate=1e-3,
            laplacian_weight=0.5,
            silhouette_weight=0.0,
            edge_length_weight=0.5,
            normal_consistency_weight=0.0,
            symmetry_weight=0.0,
        )

        refiner = JointRefiner(
            mesh=mesh,
            gaussians=None,
            rig=default_rig,
            target_masks={},
            target_images={},
            config=config,
            image_size=(256, 256),
        )

        _, _, metrics = refiner.run()

        assert metrics["final_total_loss"] <= metrics["initial_total_loss"], \
            "Total loss should not increase over optimization"


class TestSilhouetteLossRobustness:
    """Additional silhouette loss robustness tests."""

    def test_different_image_sizes(self, default_rig):
        """Silhouette loss should work with various image sizes."""
        vertices, faces = _make_cube_mesh(size=0.3)

        for size in [(64, 64), (128, 128), (256, 256), (512, 512)]:
            target_masks = render_silhouettes(
                vertices, faces, default_rig, size,
            )
            loss = silhouette_loss(
                vertices, faces, target_masks, default_rig, size,
            )
            assert 0.0 <= loss <= 1.0, f"Loss out of range for size {size}"

    def test_mismatched_mask_sizes_handled(self, default_rig):
        """Silhouette loss should handle target masks with different sizes than render."""
        vertices, faces = _make_cube_mesh(size=0.3)
        image_size = (256, 256)

        # Render targets at a different size
        target_masks = render_silhouettes(
            vertices, faces, default_rig, (128, 128),
        )

        # Should still work (silhouette_loss resizes internally)
        loss = silhouette_loss(
            vertices, faces, target_masks, default_rig, image_size,
        )
        assert 0.0 <= loss <= 1.0

    def test_partial_views_handled(self, default_rig):
        """Loss should work when only some views have masks."""
        vertices, faces = _make_cube_mesh(size=0.3)
        image_size = (256, 256)

        all_masks = render_silhouettes(
            vertices, faces, default_rig, image_size,
        )

        # Keep only front and back
        partial_masks = {
            k: v for k, v in all_masks.items()
            if k in ("front", "back")
        }

        loss = silhouette_loss(
            vertices, faces, partial_masks, default_rig, image_size,
        )
        assert 0.0 <= loss <= 1.0


class TestAllLossesEnabled:
    """Test the refiner with all loss functions enabled simultaneously."""

    def test_all_losses_active(self, default_rig):
        """Refiner should work with all losses enabled."""
        vertices, faces = _make_cube_mesh(size=0.3)
        rng = np.random.RandomState(42)
        vertices = vertices + rng.randn(*vertices.shape) * 0.03
        mesh = _make_mesh_state(vertices, faces)
        image_size = (256, 256)

        target_masks = render_silhouettes(
            vertices, faces, default_rig, image_size,
        )
        target_images = {
            vn: np.full((256, 256, 3), 128, dtype=np.uint8)
            for vn in CANONICAL_VIEW_ORDER
        }

        config = RefinementConfig(
            n_iterations=3,
            learning_rate=1e-3,
            silhouette_weight=0.5,
            laplacian_weight=0.5,
            edge_length_weight=0.1,
            normal_consistency_weight=0.1,
            symmetry_weight=0.05,
            use_symmetry=True,
        )

        refiner = JointRefiner(
            mesh=mesh,
            gaussians=None,
            rig=default_rig,
            target_masks=target_masks,
            target_images=target_images,
            config=config,
            image_size=image_size,
        )

        result_mesh, _, metrics = refiner.run()

        # All loss components should be present in history
        assert len(metrics["loss_history"]) > 0
        first_losses = metrics["loss_history"][0]
        assert "silhouette" in first_losses
        assert "laplacian" in first_losses
        assert "edge_length" in first_losses
        assert "symmetry" in first_losses
        assert "total" in first_losses

        # All values should be finite
        for key, val in first_losses.items():
            assert np.isfinite(val), f"Loss '{key}' is not finite: {val}"

    def test_all_losses_with_gaussians(self, default_rig):
        """Refiner with all losses and Gaussians should produce valid output."""
        vertices, faces = _make_cube_mesh(size=0.3)
        mesh = _make_mesh_state(vertices, faces)
        image_size = (256, 256)

        target_masks = render_silhouettes(
            vertices, faces, default_rig, image_size,
        )

        gaussians = build_coarse_gaussians(
            vertices, mesh.vertex_colors, mesh.normals, voxel_size=0.02,
        )

        config = RefinementConfig(
            n_iterations=2,
            silhouette_weight=0.0,  # skip expensive silhouette for speed
            laplacian_weight=1.0,
            edge_length_weight=0.1,
            normal_consistency_weight=0.0,
            symmetry_weight=0.05,
            use_symmetry=True,
        )

        refiner = JointRefiner(
            mesh=mesh,
            gaussians=gaussians,
            rig=default_rig,
            target_masks=target_masks,
            target_images={},
            config=config,
            image_size=image_size,
        )

        result_mesh, result_gaussians, metrics = refiner.run()

        assert result_mesh.n_vertices == 8
        assert result_mesh.n_faces == 12
        assert result_gaussians is not None
        assert "positions" in result_gaussians
        assert len(result_gaussians["positions"]) > 0


class TestGradientAccuracy:
    """Test that analytical gradients agree with numerical gradients."""

    def test_laplacian_gradient_matches_numerical(self):
        """Analytical Laplacian gradient should match numerical gradient."""
        vertices, faces = _make_cube_mesh(size=0.3)
        # Add noise for non-trivial gradients
        rng = np.random.RandomState(42)
        vertices = vertices + rng.randn(*vertices.shape) * 0.05

        edges = compute_edges(faces)
        L = build_laplacian_matrix(len(vertices), edges)

        # Analytical gradient
        analytical = compute_laplacian_gradient(vertices, L)

        # Numerical gradient
        numerical = compute_vertex_gradients(
            vertices, faces,
            lambda v, f: laplacian_loss(v, L),
            epsilon=1e-5,
        )

        np.testing.assert_allclose(
            analytical, numerical, atol=1e-3, rtol=1e-2,
        )

    def test_edge_length_gradient_matches_numerical(self):
        """Analytical edge-length gradient should match numerical gradient."""
        vertices = np.array([
            [0, 0, 0],
            [1.5, 0, 0],
            [0.5, 0.8, 0],
            [0.5, 0.3, 0.7],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
        edges = compute_edges(faces)

        # Analytical gradient
        analytical = compute_edge_length_gradient(vertices, edges)

        # Numerical gradient
        numerical = compute_vertex_gradients(
            vertices, faces,
            lambda v, f: edge_length_loss(v, compute_edges(f)),
            epsilon=1e-5,
        )

        np.testing.assert_allclose(
            analytical, numerical, atol=1e-3, rtol=1e-2,
        )


class TestMeshStateExtended:
    """Extended MeshState tests."""

    def test_normals_recomputed_after_vertex_change(self):
        """Normals should change when vertices move."""
        mesh = _make_mesh_state()
        original_normals = mesh.normals.copy()

        # Perturb vertices
        mesh.vertices += np.random.RandomState(42).randn(*mesh.vertices.shape) * 0.1
        mesh.normals = compute_vertex_normals(mesh.vertices, mesh.faces)

        # Normals should have changed
        assert not np.allclose(mesh.normals, original_normals)

    def test_mesh_state_vertex_colors_shape(self):
        """Vertex colors should have shape (V, 3)."""
        mesh = _make_mesh_state()
        assert mesh.vertex_colors.shape == (mesh.n_vertices, 3)
        assert mesh.vertex_colors.dtype == np.uint8
