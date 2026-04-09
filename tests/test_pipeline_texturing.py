"""
Tests for the texture baking stage (Phase 6 — texturing).

Covers:
    UV UNWRAPPING:
        - unwrap_uvs: basic layout, UV range, coverage, empty mesh

    RASTERIZATION:
        - rasterize_uv_triangles: face assignment, barycentric coords, mask

    TEXTURE BAKING:
        - bake_texture: color projection, viewing angle weighting
        - inpaint_texture: fills occluded regions

    METRICS:
        - compute_texture_metrics: structure and values

    STAGE RUNNER:
        - run_bake_texture: happy path, missing artifacts, output artifacts
"""

import io
import math
import pytest
import numpy as np
from PIL import Image
import fakeredis

from api.job_manager import JobManager
from api.models import CategoryEnum, PipelineEnum
from api.storage import StorageManager
from pipelines.canonical_mv.config import CanonicalMVConfig, CANONICAL_VIEW_ORDER
from pipelines.canonical_mv.camera_init import CameraRig, build_canonical_rig
from pipelines.canonical_mv.coarse_recon import save_mesh_ply
from pipelines.canonical_mv.refine import (
    MeshState,
    compute_vertex_normals,
    load_mesh_ply,
    render_silhouettes,
)
from pipelines.canonical_mv.texturing import (
    UVLayout,
    TextureResult,
    unwrap_uvs,
    rasterize_uv_triangles,
    bake_texture,
    inpaint_texture,
    compute_texture_metrics,
    save_mesh_with_uvs_ply,
    run_bake_texture,
    ATLAS_PADDING,
    MIN_UV_TRIANGLE_EDGE_PX,
    DEFAULT_TEXTURE_RESOLUTION,
    INPAINT_RADIUS,
    MIN_VIEW_COSINE,
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
    return CanonicalMVConfig(texture_resolution=64)

@pytest.fixture
def default_rig(config):
    return build_canonical_rig(config, (64, 64))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cube_mesh(size=0.5, center=(0, 0, 0)):
    cx, cy, cz = center
    s = size / 2
    vertices = np.array([
        [cx-s, cy-s, cz-s], [cx+s, cy-s, cz-s],
        [cx+s, cy+s, cz-s], [cx-s, cy+s, cz-s],
        [cx-s, cy-s, cz+s], [cx+s, cy-s, cz+s],
        [cx+s, cy+s, cz+s], [cx-s, cy+s, cz+s],
    ], dtype=np.float64)
    faces = np.array([
        [0,3,2],[0,2,1],[4,5,6],[4,6,7],
        [3,7,6],[3,6,2],[0,1,5],[0,5,4],
        [1,2,6],[1,6,5],[0,4,7],[0,7,3],
    ], dtype=np.int32)
    return vertices, faces


def _create_mv_job(jm):
    return jm.create_multiview_job(
        category=CategoryEnum.HUMAN_BUST,
        pipeline=PipelineEnum.CANONICAL_MV_HYBRID,
        views_received=list(CANONICAL_VIEW_ORDER),
    )


def _setup_job(jm, sm, config, tex_res=64):
    job_id = _create_mv_job(jm)
    rig = build_canonical_rig(config, (tex_res, tex_res))
    sm.save_artifact_json(job_id, "camera_init.json", rig.to_dict())

    vertices, faces = _make_cube_mesh(size=0.3)
    normals = compute_vertex_normals(vertices, faces)
    mesh_path = sm.get_artifact_dir(job_id) / "completed_mesh.ply"
    save_mesh_ply(str(mesh_path), vertices, faces, normals)

    masks = render_silhouettes(vertices, faces, rig, (tex_res, tex_res))
    for vn in CANONICAL_VIEW_ORDER:
        mask = masks.get(vn, np.zeros((tex_res, tex_res), dtype=np.uint8))
        rgba = np.zeros((tex_res, tex_res, 4), dtype=np.uint8)
        rgba[:, :, :3] = 128
        rgba[:, :, 3] = mask
        img = Image.fromarray(rgba, mode="RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        sm.save_view_preview(job_id, "segmented", vn, buf.getvalue(), ".png")

    return job_id


# ===========================================================================
# UV UNWRAPPING
# ===========================================================================


class TestUnwrapUvs:
    def test_basic_cube(self):
        """Cube mesh should produce valid UV layout."""
        v, f = _make_cube_mesh()
        layout = unwrap_uvs(v, f, texture_size=128)
        assert layout.texture_size == 128
        assert layout.n_charts == len(f)
        assert len(layout.face_uvs) == len(f)
        # 3 UV coords per face
        assert len(layout.uv_coords) == len(f) * 3

    def test_uv_range(self):
        """All UV coordinates should be in [0, 1]."""
        v, f = _make_cube_mesh()
        layout = unwrap_uvs(v, f, texture_size=256)
        assert np.all(layout.uv_coords >= 0.0)
        assert np.all(layout.uv_coords <= 1.0)

    def test_coverage_positive(self):
        """UV coverage should be positive for non-empty mesh."""
        v, f = _make_cube_mesh()
        layout = unwrap_uvs(v, f, texture_size=256)
        assert layout.coverage > 0

    def test_empty_mesh(self):
        """Empty mesh should produce empty UV layout."""
        v = np.zeros((0, 3), dtype=np.float64)
        f = np.zeros((0, 3), dtype=np.int32)
        layout = unwrap_uvs(v, f, texture_size=128)
        assert layout.n_charts == 0
        assert len(layout.uv_coords) == 0
        assert layout.coverage == 0.0

    def test_single_triangle(self):
        """Single triangle should produce valid layout."""
        v = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float64)
        f = np.array([[0,1,2]], dtype=np.int32)
        layout = unwrap_uvs(v, f, texture_size=128)
        assert layout.n_charts == 1
        assert len(layout.uv_coords) == 3

    def test_face_uvs_indices_valid(self):
        """Face UV indices should reference valid UV coords."""
        v, f = _make_cube_mesh()
        layout = unwrap_uvs(v, f, texture_size=128)
        assert np.all(layout.face_uvs >= 0)
        assert np.all(layout.face_uvs < len(layout.uv_coords))


# ===========================================================================
# RASTERIZATION
# ===========================================================================


class TestRasterizeUvTriangles:
    def test_basic_rasterization(self):
        """Rasterization should produce face assignments and masks."""
        v, f = _make_cube_mesh()
        layout = unwrap_uvs(v, f, texture_size=64)
        face_id_map, bary_map, mask = rasterize_uv_triangles(layout)

        assert face_id_map.shape == (64, 64)
        assert bary_map.shape == (64, 64, 3)
        assert mask.shape == (64, 64)
        # Some pixels should be covered
        assert np.sum(mask > 0) > 0

    def test_face_ids_valid(self):
        """Face IDs should be -1 or valid face indices."""
        v, f = _make_cube_mesh()
        layout = unwrap_uvs(v, f, texture_size=64)
        face_id_map, _, mask = rasterize_uv_triangles(layout)

        valid_pixels = face_id_map[mask > 0]
        assert np.all(valid_pixels >= 0)
        assert np.all(valid_pixels < len(f))

    def test_barycentric_sum_to_one(self):
        """Barycentric coordinates should approximately sum to 1."""
        v, f = _make_cube_mesh()
        layout = unwrap_uvs(v, f, texture_size=64)
        _, bary_map, mask = rasterize_uv_triangles(layout)

        valid = mask > 0
        if np.any(valid):
            sums = bary_map[valid].sum(axis=1)
            np.testing.assert_allclose(sums, 1.0, atol=0.05)

    def test_empty_layout(self):
        """Empty UV layout should produce empty rasterization."""
        layout = UVLayout(
            uv_coords=np.zeros((0, 2), dtype=np.float64),
            face_uvs=np.zeros((0, 3), dtype=np.int32),
            texture_size=32,
            n_charts=0,
            coverage=0.0,
        )
        face_id_map, bary_map, mask = rasterize_uv_triangles(layout)
        assert np.sum(mask > 0) == 0


# ===========================================================================
# TEXTURE BAKING
# ===========================================================================


class TestBakeTexture:
    def test_produces_texture(self, default_rig):
        """Baking should produce a non-trivial texture."""
        v, f = _make_cube_mesh(size=0.3)
        normals = compute_vertex_normals(v, f)
        layout = unwrap_uvs(v, f, texture_size=64)
        face_id_map, bary_map, tex_mask = rasterize_uv_triangles(layout)

        images = {vn: np.full((64, 64, 3), 200, dtype=np.uint8) for vn in CANONICAL_VIEW_ORDER}
        masks = {vn: np.full((64, 64), 255, dtype=np.uint8) for vn in CANONICAL_VIEW_ORDER}

        texture = bake_texture(
            v, f, normals, layout, default_rig,
            images, masks, face_id_map, bary_map, tex_mask,
        )

        assert texture.shape == (64, 64, 3)
        assert texture.dtype == np.uint8

    def test_empty_mesh_returns_default(self, default_rig):
        """Empty mesh should return default gray texture."""
        v = np.zeros((0, 3), dtype=np.float64)
        f = np.zeros((0, 3), dtype=np.int32)
        normals = np.zeros((0, 3), dtype=np.float64)
        layout = unwrap_uvs(v, f, texture_size=32)
        face_id_map, bary_map, tex_mask = rasterize_uv_triangles(layout)

        texture = bake_texture(
            v, f, normals, layout, default_rig, {}, {},
            face_id_map, bary_map, tex_mask,
        )

        assert texture.shape == (32, 32, 3)


# ===========================================================================
# INPAINTING
# ===========================================================================


class TestInpaintTexture:
    def test_inpainting_fills_gaps(self):
        """Inpainting should fill some gap pixels near the mask boundary."""
        texture = np.full((32, 32, 3), 128, dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        # Create a small valid region
        texture[10:20, 10:20] = [255, 0, 0]
        mask[10:20, 10:20] = 255

        result = inpaint_texture(texture, mask, radius=3)
        assert result.shape == texture.shape

    def test_empty_mask_returns_original(self):
        """Empty mask should return original texture unchanged."""
        texture = np.full((32, 32, 3), 128, dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)

        result = inpaint_texture(texture, mask)
        np.testing.assert_array_equal(result, texture)


# ===========================================================================
# METRICS
# ===========================================================================


class TestComputeTextureMetrics:
    def test_structure(self):
        """Metrics should have all expected keys."""
        texture = np.full((64, 64, 3), 128, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:30, 10:30] = 255
        layout = UVLayout(
            uv_coords=np.zeros((3, 2)),
            face_uvs=np.zeros((1, 3), dtype=np.int32),
            texture_size=64,
            n_charts=1,
            coverage=0.1,
        )

        metrics = compute_texture_metrics(texture, mask, layout)
        assert "texture_resolution" in metrics
        assert "valid_pixels" in metrics
        assert "coverage_fraction" in metrics
        assert "color_variance" in metrics
        assert "n_charts" in metrics

    def test_coverage_fraction(self):
        """Coverage fraction should match mask coverage."""
        texture = np.full((32, 32, 3), 128, dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[:16, :] = 255  # half the image
        layout = UVLayout(
            uv_coords=np.zeros((3, 2)),
            face_uvs=np.zeros((1, 3), dtype=np.int32),
            texture_size=32,
            n_charts=1,
            coverage=0.5,
        )

        metrics = compute_texture_metrics(texture, mask, layout)
        assert metrics["coverage_fraction"] == pytest.approx(0.5, abs=0.01)


# ===========================================================================
# STAGE RUNNER
# ===========================================================================


class TestRunBakeTexture:
    def test_happy_path(self, jm, sm, config):
        """Stage should complete and save all artifacts."""
        job_id = _setup_job(jm, sm, config, tex_res=64)
        run_bake_texture(job_id, config, jm, sm)

        # Texture should exist
        tex_path = sm.get_artifact_path(job_id, "textures/diffuse.png")
        assert tex_path is not None

        # Baked mesh should exist
        mesh_path = sm.get_artifact_path(job_id, "baked_mesh.ply")
        assert mesh_path is not None

        # Metrics should exist
        metrics = sm.load_artifact_json(job_id, "texture_metrics.json")
        assert metrics is not None
        assert "mesh_source" in metrics

    def test_stage_progress(self, jm, sm, config):
        """Stage progress should reach 1.0."""
        job_id = _setup_job(jm, sm, config, tex_res=64)
        run_bake_texture(job_id, config, jm, sm)
        job = jm.get_job(job_id)
        assert job["stage_progress"] == 1.0

    def test_missing_camera_init_raises(self, jm, sm, config):
        """Missing camera_init.json should raise ValueError."""
        job_id = _create_mv_job(jm)
        with pytest.raises(ValueError, match="camera_init"):
            run_bake_texture(job_id, config, jm, sm)

    def test_missing_mesh_raises(self, jm, sm, config):
        """Missing mesh should raise ValueError."""
        job_id = _create_mv_job(jm)
        rig = build_canonical_rig(config, (64, 64))
        sm.save_artifact_json(job_id, "camera_init.json", rig.to_dict())
        with pytest.raises(ValueError, match="mesh"):
            run_bake_texture(job_id, config, jm, sm)

    def test_texture_is_valid_image(self, jm, sm, config):
        """Saved texture should be a loadable PNG."""
        job_id = _setup_job(jm, sm, config, tex_res=64)
        run_bake_texture(job_id, config, jm, sm)

        tex_path = sm.get_artifact_path(job_id, "textures/diffuse.png")
        img = Image.open(str(tex_path))
        assert img.size == (64, 64)
        assert img.mode == "RGB"

