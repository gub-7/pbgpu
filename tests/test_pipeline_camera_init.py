"""
Tests for the camera initialization stage of the canonical multi-view pipeline.

Covers:
    CAMERA MATH (unit tests):
        - camera_position_from_angles: spherical → Cartesian for all 5 views
        - look_at: world-to-camera matrix correctness, orthogonality, determinant
        - build_intrinsic_matrix: focal length conversion, principal point centering
        - focal_length_pixels: mm-to-pixel conversion
        - project_point: 3D→2D projection, behind-camera rejection
        - project_bbox_3d: 3D bbox → 2D bbox projection
        - compute_reprojection_error: error metric computation

    CAMERA RIG (unit tests):
        - build_canonical_rig: all 5 cameras created, correct positions/orientations
        - CameraRig.to_dict / from_dict: serialization roundtrip
        - CameraRig.project: per-view projection convenience method
        - _validate_rig: catches bad extrinsics, intrinsics, positions

    REFINEMENT (unit tests):
        - refine_rig_from_silhouettes: scale adjustment from bbox metrics
        - Refinement clamping (no extreme adjustments)
        - Refinement skipped when not enough data or adjustment too small

    STAGE RUNNER (integration tests):
        - run_initialize_cameras: happy path with preprocess metrics
        - Missing preprocess_metrics raises ValueError
        - camera_init.json artifact saved correctly
        - Stage progress reaches 1.0
        - Refinement applied when silhouettes indicate scale mismatch

No GPU dependencies — all tests use pure numpy/math.
"""

import io
import json
import math
import pytest
from pathlib import Path

import numpy as np
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
    CameraSpec,
    CANONICAL_CAMERAS,
    CANONICAL_VIEW_ORDER,
)
from pipelines.canonical_mv.camera_init import (
    # Camera math
    camera_position_from_angles,
    look_at,
    build_intrinsic_matrix,
    focal_length_pixels,
    project_point,
    project_bbox_3d,
    compute_reprojection_error,
    # Rig
    CameraRig,
    build_canonical_rig,
    # Refinement
    refine_rig_from_silhouettes,
    EXPECTED_BBOX_IMAGE_RATIO,
    MAX_SCALE_ADJUSTMENT,
    DEFAULT_SENSOR_WIDTH_MM,
    # Stage runner
    run_initialize_cameras,
    # Validation
    _validate_rig,
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
    """Build a default canonical rig with 1024x1024 images."""
    return build_canonical_rig(config, (1024, 1024))


def _create_mv_job(jm):
    return jm.create_multiview_job(
        category=CategoryEnum.HUMAN_BUST,
        pipeline=PipelineEnum.CANONICAL_MV_HYBRID,
        views_received=["front", "back", "left", "right", "top"],
    )


def _make_preprocess_metrics(
    canvas_size=1024,
    bbox_ratio=0.65,
    views=None,
):
    """
    Create a mock preprocess_metrics.json dict.

    Args:
        canvas_size: Canvas size in pixels.
        bbox_ratio: Ratio of max bbox dimension to canvas size.
        views: List of view names (default: all 5 canonical views).
    """
    if views is None:
        views = CANONICAL_VIEW_ORDER

    bbox_dim = int(canvas_size * bbox_ratio)
    offset = (canvas_size - bbox_dim) // 2

    per_view = {}
    for vn in views:
        per_view[vn] = {
            "bbox": [offset, offset, bbox_dim, bbox_dim],
            "centroid": [canvas_size // 2, canvas_size // 2],
            "foreground_area_ratio": bbox_ratio ** 2,
            "sharpness": 500.0,
            "segmentation_confidence": 0.95,
            "color_histogram_mean": [180.0, 90.0, 45.0],
        }

    return {
        "canvas_size": canvas_size,
        "crop_side": int(canvas_size * 1.2),
        "per_view": per_view,
    }


# ===========================================================================
# CAMERA MATH UNIT TESTS
# ===========================================================================


class TestCameraPositionFromAngles:
    """Test spherical-to-Cartesian camera positioning."""

    def test_front_camera_on_positive_z(self):
        """Front (yaw=0) should be on the +Z axis."""
        pos = camera_position_from_angles(0.0, 0.0, 2.5)
        assert pos[0] == pytest.approx(0.0, abs=1e-10)
        assert pos[1] == pytest.approx(0.0, abs=1e-10)
        assert pos[2] == pytest.approx(2.5, abs=1e-10)

    def test_right_camera_on_positive_x(self):
        """Right (yaw=90°) should be on the +X axis."""
        pos = camera_position_from_angles(math.radians(90), 0.0, 2.5)
        assert pos[0] == pytest.approx(2.5, abs=1e-10)
        assert pos[1] == pytest.approx(0.0, abs=1e-10)
        assert pos[2] == pytest.approx(0.0, abs=1e-10)

    def test_back_camera_on_negative_z(self):
        """Back (yaw=180°) should be on the -Z axis."""
        pos = camera_position_from_angles(math.radians(180), 0.0, 2.5)
        assert pos[0] == pytest.approx(0.0, abs=1e-10)
        assert pos[1] == pytest.approx(0.0, abs=1e-10)
        assert pos[2] == pytest.approx(-2.5, abs=1e-10)

    def test_left_camera_on_negative_x(self):
        """Left (yaw=270°) should be on the -X axis."""
        pos = camera_position_from_angles(math.radians(270), 0.0, 2.5)
        assert pos[0] == pytest.approx(-2.5, abs=1e-10)
        assert pos[1] == pytest.approx(0.0, abs=1e-10)
        assert pos[2] == pytest.approx(0.0, abs=1e-10)

    def test_top_camera_above_origin(self):
        """Top (pitch=-90°) should be directly above on +Y axis."""
        pos = camera_position_from_angles(0.0, math.radians(-90), 2.5)
        assert pos[0] == pytest.approx(0.0, abs=1e-10)
        assert pos[1] == pytest.approx(2.5, abs=1e-10)
        assert pos[2] == pytest.approx(0.0, abs=1e-10)

    def test_distance_preserved(self):
        """Camera should always be at the specified distance from origin."""
        for yaw in [0, 45, 90, 135, 180, 225, 270, 315]:
            for pitch in [0, -30, -45, -60, -90]:
                pos = camera_position_from_angles(
                    math.radians(yaw), math.radians(pitch), 3.0
                )
                dist = np.linalg.norm(pos)
                assert dist == pytest.approx(3.0, abs=1e-10), \
                    f"Distance wrong for yaw={yaw}, pitch={pitch}: {dist}"

    def test_different_distances(self):
        """Various distances should work correctly."""
        for d in [0.5, 1.0, 2.5, 5.0, 10.0]:
            pos = camera_position_from_angles(0.0, 0.0, d)
            assert np.linalg.norm(pos) == pytest.approx(d, abs=1e-10)


class TestLookAt:
    """Test the look-at matrix computation."""

    def test_front_camera_looks_at_origin(self):
        """Front camera at (0,0,d) looking at origin."""
        eye = np.array([0.0, 0.0, 2.5])
        target = np.array([0.0, 0.0, 0.0])
        M = look_at(eye, target)

        # Should be 4×4
        assert M.shape == (4, 4)

        # Origin should project to camera center (0, 0, d_cam)
        origin_cam = M @ np.array([0, 0, 0, 1])
        # In camera space, origin should be in front (z > 0)
        assert origin_cam[2] > 0

    def test_rotation_is_orthogonal(self):
        """Rotation part should be orthogonal (R^T R = I)."""
        eye = np.array([2.5, 0.0, 0.0])
        target = np.array([0.0, 0.0, 0.0])
        M = look_at(eye, target)

        R = M[:3, :3]
        RtR = R.T @ R
        np.testing.assert_allclose(RtR, np.eye(3), atol=1e-10)

    def test_rotation_determinant_is_one(self):
        """Rotation should be a proper rotation (det = 1)."""
        for yaw in [0, 90, 180, 270]:
            pos = camera_position_from_angles(math.radians(yaw), 0.0, 2.5)
            M = look_at(pos, np.zeros(3))
            det = np.linalg.det(M[:3, :3])
            assert det == pytest.approx(1.0, abs=1e-10), \
                f"det(R) = {det} for yaw={yaw}"

    def test_top_camera_uses_fallback_up(self):
        """Top camera (looking straight down) needs a fallback up vector."""
        eye = np.array([0.0, 2.5, 0.0])
        target = np.array([0.0, 0.0, 0.0])
        # This should not raise even though forward is parallel to world up
        M = look_at(eye, target)
        assert M.shape == (4, 4)

        R = M[:3, :3]
        RtR = R.T @ R
        np.testing.assert_allclose(RtR, np.eye(3), atol=1e-10)

    def test_eye_equals_target_raises(self):
        """Should raise if eye and target are the same point."""
        with pytest.raises(ValueError, match="same position"):
            look_at(np.zeros(3), np.zeros(3))

    def test_all_canonical_views_valid(self):
        """All 5 canonical camera positions should produce valid look-at matrices."""
        for vn, spec in CANONICAL_CAMERAS.items():
            pos = camera_position_from_angles(
                spec.yaw_rad, spec.pitch_rad, spec.distance
            )
            M = look_at(pos, np.zeros(3))

            # Verify orthogonality
            R = M[:3, :3]
            np.testing.assert_allclose(
                R.T @ R, np.eye(3), atol=1e-10,
                err_msg=f"Non-orthogonal rotation for {vn}"
            )

            # Verify origin is in front of camera
            origin_cam = M @ np.array([0, 0, 0, 1])
            assert origin_cam[2] > 0, f"Origin behind camera for {vn}"

    def test_translation_consistent_with_position(self):
        """t = -R @ eye should hold."""
        eye = np.array([1.0, 2.0, 3.0])
        M = look_at(eye, np.zeros(3))
        R = M[:3, :3]
        t = M[:3, 3]
        expected_t = -R @ eye
        np.testing.assert_allclose(t, expected_t, atol=1e-10)


class TestBuildIntrinsicMatrix:
    """Test intrinsic matrix construction."""

    def test_shape(self):
        K = build_intrinsic_matrix(50.0, 36.0, 1024, 1024)
        assert K.shape == (3, 3)

    def test_principal_point_at_center(self):
        K = build_intrinsic_matrix(50.0, 36.0, 1024, 768)
        assert K[0, 2] == pytest.approx(512.0)
        assert K[1, 2] == pytest.approx(384.0)

    def test_focal_length_conversion(self):
        """50mm on 36mm sensor with 1024px width → fx = 50/36 * 1024 ≈ 1422.2"""
        K = build_intrinsic_matrix(50.0, 36.0, 1024, 1024)
        expected_fx = 50.0 * 1024 / 36.0
        assert K[0, 0] == pytest.approx(expected_fx)
        assert K[1, 1] == pytest.approx(expected_fx)  # square pixels

    def test_zero_skew(self):
        K = build_intrinsic_matrix(50.0, 36.0, 1024, 1024)
        assert K[0, 1] == 0.0
        assert K[1, 0] == 0.0

    def test_bottom_row(self):
        K = build_intrinsic_matrix(50.0, 36.0, 1024, 1024)
        assert K[2, 0] == 0.0
        assert K[2, 1] == 0.0
        assert K[2, 2] == 1.0

    def test_different_image_sizes(self):
        K1 = build_intrinsic_matrix(50.0, 36.0, 512, 512)
        K2 = build_intrinsic_matrix(50.0, 36.0, 1024, 1024)
        # Focal length in pixels should double when image size doubles
        assert K2[0, 0] == pytest.approx(K1[0, 0] * 2)


class TestFocalLengthPixels:
    def test_conversion(self):
        fpx = focal_length_pixels(50.0, 36.0, 1024)
        assert fpx == pytest.approx(50.0 * 1024 / 36.0)

    def test_different_sensor_sizes(self):
        # Larger sensor → smaller focal length in pixels for same mm
        f1 = focal_length_pixels(50.0, 36.0, 1024)
        f2 = focal_length_pixels(50.0, 24.0, 1024)  # APS-C
        assert f2 > f1  # smaller sensor → larger effective focal length


class TestProjectPoint:
    """Test 3D to 2D projection."""

    def test_origin_projects_to_center(self):
        """Origin should project to image center for any camera looking at it."""
        config = CanonicalMVConfig()
        rig = build_canonical_rig(config, (1024, 1024))

        for vn in CANONICAL_VIEW_ORDER:
            p2d = rig.project(vn, np.array([0.0, 0.0, 0.0]))
            assert p2d is not None, f"Origin behind camera for {vn}"
            assert p2d[0] == pytest.approx(512.0, abs=2.0), \
                f"Origin x={p2d[0]:.1f} for {vn}, expected 512"
            assert p2d[1] == pytest.approx(512.0, abs=2.0), \
                f"Origin y={p2d[1]:.1f} for {vn}, expected 512"

    def test_point_behind_camera_returns_none(self):
        """A point behind the camera should return None."""
        # Front camera at (0, 0, 2.5) looking at origin
        eye = np.array([0.0, 0.0, 2.5])
        M = look_at(eye, np.zeros(3))
        K = build_intrinsic_matrix(50.0, 36.0, 1024, 1024)

        # Point far behind the camera
        p_behind = np.array([0.0, 0.0, 5.0])
        result = project_point(p_behind, M, K)
        assert result is None

    def test_point_in_front_of_camera(self):
        """A point in front should project to valid coordinates."""
        eye = np.array([0.0, 0.0, 2.5])
        M = look_at(eye, np.zeros(3))
        K = build_intrinsic_matrix(50.0, 36.0, 1024, 1024)

        # Point at origin — in front of the camera
        result = project_point(np.array([0.0, 0.0, 0.0]), M, K)
        assert result is not None
        assert len(result) == 2

    def test_positive_x_projects_right(self):
        """A point to the right in world space should project to the right in image."""
        config = CanonicalMVConfig()
        rig = build_canonical_rig(config, (1024, 1024))

        # Front camera: +X world → +X image (right)
        origin = rig.project("front", np.array([0.0, 0.0, 0.0]))
        right_pt = rig.project("front", np.array([0.3, 0.0, 0.0]))
        assert right_pt is not None
        assert origin is not None
        assert right_pt[0] > origin[0], "Point to the right should have larger x"

    def test_positive_y_projects_up(self):
        """A point above in world space should project upward (smaller y) in image."""
        config = CanonicalMVConfig()
        rig = build_canonical_rig(config, (1024, 1024))

        origin = rig.project("front", np.array([0.0, 0.0, 0.0]))
        up_pt = rig.project("front", np.array([0.0, 0.3, 0.0]))
        assert up_pt is not None
        assert origin is not None
        # OpenCV convention: Y down, so world +Y → camera -Y → smaller pixel y
        assert up_pt[1] < origin[1], "Point above should have smaller y"


class TestProjectBbox3D:
    """Test 3D bounding box projection."""

    def test_unit_cube_projects(self):
        """A unit cube at origin should project to a valid 2D bbox."""
        config = CanonicalMVConfig()
        rig = build_canonical_rig(config, (1024, 1024))

        bbox_3d = np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])

        for vn in CANONICAL_VIEW_ORDER:
            ext = rig.get_extrinsic(vn)
            intr = rig.get_intrinsic(vn)
            result = project_bbox_3d(bbox_3d, ext, intr)
            assert result is not None, f"Bbox behind camera for {vn}"
            x_min, y_min, x_max, y_max = result
            assert x_min < x_max
            assert y_min < y_max
            # Should be roughly centered
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            assert 400 < cx < 624, f"Bbox center x={cx} for {vn}"
            assert 400 < cy < 624, f"Bbox center y={cy} for {vn}"

    def test_bbox_behind_camera_returns_none(self):
        """If any corner is behind the camera, should return None."""
        eye = np.array([0.0, 0.0, 2.5])
        M = look_at(eye, np.zeros(3))
        K = build_intrinsic_matrix(50.0, 36.0, 1024, 1024)

        # Bbox entirely behind the camera (z > 2.5, camera looks toward z=0)
        bbox_3d = np.array([[-0.5, -0.5, 3.0], [0.5, 0.5, 4.0]])
        result = project_bbox_3d(bbox_3d, M, K)
        assert result is None


class TestComputeReprojectionError:
    """Test reprojection error computation."""

    def test_perfect_projection_zero_error(self):
        """Points that project perfectly should have zero error."""
        config = CanonicalMVConfig()
        rig = build_canonical_rig(config, (1024, 1024))

        ext = rig.get_extrinsic("front")
        intr = rig.get_intrinsic("front")

        points_3d = np.array([
            [0.0, 0.0, 0.0],
            [0.2, 0.1, -0.1],
            [-0.1, 0.3, 0.1],
        ])

        # Project to get "observed" 2D points
        points_2d = []
        for p in points_3d:
            p2d = project_point(p, ext, intr)
            assert p2d is not None
            points_2d.append(p2d)
        points_2d = np.array(points_2d)

        error = compute_reprojection_error(points_3d, points_2d, ext, intr)
        assert error == pytest.approx(0.0, abs=1e-6)

    def test_noisy_observations_nonzero_error(self):
        """Noisy 2D observations should give non-zero error."""
        config = CanonicalMVConfig()
        rig = build_canonical_rig(config, (1024, 1024))

        ext = rig.get_extrinsic("front")
        intr = rig.get_intrinsic("front")

        points_3d = np.array([[0.0, 0.0, 0.0]])
        # Add noise to observed point
        p2d = project_point(points_3d[0], ext, intr)
        noisy_2d = np.array([p2d + np.array([5.0, 3.0])])

        error = compute_reprojection_error(points_3d, noisy_2d, ext, intr)
        expected = np.sqrt(5.0**2 + 3.0**2)
        assert error == pytest.approx(expected, abs=0.01)

    def test_empty_points_zero_error(self):
        """Empty point arrays should return 0."""
        ext = np.eye(4)
        intr = np.eye(3)
        error = compute_reprojection_error(np.zeros((0, 3)), np.zeros((0, 2)), ext, intr)
        assert error == 0.0


# ===========================================================================
# CAMERA RIG UNIT TESTS
# ===========================================================================


class TestBuildCanonicalRig:
    """Test the canonical rig builder."""

    def test_all_views_present(self, default_rig):
        assert set(default_rig.cameras.keys()) == set(CANONICAL_VIEW_ORDER)

    def test_shared_params(self, default_rig):
        sp = default_rig.shared_params
        assert sp["image_size"] == [1024, 1024]
        assert sp["focal_length_mm"] == 50.0
        assert sp["sensor_width_mm"] == DEFAULT_SENSOR_WIDTH_MM
        assert sp["focal_length_px"] > 0
        assert sp["near_plane"] > 0
        assert sp["far_plane"] > sp["near_plane"]

    def test_front_camera_position(self, default_rig):
        pos = default_rig.get_position("front")
        assert pos[0] == pytest.approx(0.0, abs=1e-10)
        assert pos[1] == pytest.approx(0.0, abs=1e-10)
        assert pos[2] == pytest.approx(2.5, abs=1e-10)

    def test_top_camera_position(self, default_rig):
        pos = default_rig.get_position("top")
        assert pos[0] == pytest.approx(0.0, abs=1e-10)
        assert pos[1] == pytest.approx(2.5, abs=1e-10)
        assert pos[2] == pytest.approx(0.0, abs=1e-10)

    def test_all_cameras_see_origin(self, default_rig):
        """Every camera should be able to project the origin."""
        for vn in CANONICAL_VIEW_ORDER:
            p2d = default_rig.project(vn, np.array([0.0, 0.0, 0.0]))
            assert p2d is not None, f"Origin behind {vn} camera"

    def test_cameras_at_correct_distance(self, default_rig):
        for vn in CANONICAL_VIEW_ORDER:
            pos = default_rig.get_position(vn)
            dist = np.linalg.norm(pos)
            assert dist == pytest.approx(2.5, abs=1e-10)

    def test_extrinsics_are_valid(self, default_rig):
        """All extrinsic matrices should have orthogonal rotation."""
        for vn in CANONICAL_VIEW_ORDER:
            ext = default_rig.get_extrinsic(vn)
            R = ext[:3, :3]
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)
            assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)

    def test_intrinsics_positive_focal(self, default_rig):
        for vn in CANONICAL_VIEW_ORDER:
            K = default_rig.get_intrinsic(vn)
            assert K[0, 0] > 0
            assert K[1, 1] > 0

    def test_custom_image_size(self, config):
        rig = build_canonical_rig(config, (512, 512))
        K = rig.get_intrinsic("front")
        assert K[0, 2] == pytest.approx(256.0)
        assert K[1, 2] == pytest.approx(256.0)

    def test_custom_distance(self):
        """Custom camera distance should be respected."""
        from pipelines.canonical_mv.config import CameraSpec

        cameras = {}
        for vn, spec in CANONICAL_CAMERAS.items():
            cameras[vn] = CameraSpec(
                view_name=spec.view_name,
                yaw_deg=spec.yaw_deg,
                pitch_deg=spec.pitch_deg,
                distance=5.0,
                focal_length=spec.focal_length,
            )

        config = CanonicalMVConfig()
        config.cameras = cameras
        rig = build_canonical_rig(config, (1024, 1024))

        for vn in CANONICAL_VIEW_ORDER:
            pos = rig.get_position(vn)
            dist = np.linalg.norm(pos)
            assert dist == pytest.approx(5.0, abs=1e-10)

    def test_opposite_views_face_opposite_directions(self, default_rig):
        """Front and back cameras should face opposite directions."""
        front_pos = default_rig.get_position("front")
        back_pos = default_rig.get_position("back")

        # Positions should be on opposite sides
        dot = np.dot(front_pos, back_pos)
        assert dot < 0, "Front and back should be on opposite sides of origin"

    def test_left_right_symmetry(self, default_rig):
        """Left and right cameras should be mirror images across the YZ plane."""
        left_pos = default_rig.get_position("left")
        right_pos = default_rig.get_position("right")

        # X should be negated, Y and Z should match
        assert left_pos[0] == pytest.approx(-right_pos[0], abs=1e-10)
        assert left_pos[1] == pytest.approx(right_pos[1], abs=1e-10)
        assert left_pos[2] == pytest.approx(right_pos[2], abs=1e-10)


class TestCameraRigSerialization:
    """Test CameraRig serialization and deserialization."""

    def test_to_dict_structure(self, default_rig):
        d = default_rig.to_dict()
        assert d["rig_type"] == "canonical_5view"
        assert "cameras" in d
        assert "shared_params" in d
        assert "refinement" in d
        assert set(d["cameras"].keys()) == set(CANONICAL_VIEW_ORDER)

    def test_to_dict_serializable(self, default_rig):
        """to_dict output should be JSON-serializable."""
        d = default_rig.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0
        # Verify roundtrip
        parsed = json.loads(json_str)
        assert parsed["rig_type"] == "canonical_5view"

    def test_from_dict_roundtrip(self, default_rig):
        """Serialization → deserialization should preserve data."""
        d = default_rig.to_dict()
        restored = CameraRig.from_dict(d)

        for vn in CANONICAL_VIEW_ORDER:
            np.testing.assert_allclose(
                restored.get_extrinsic(vn),
                default_rig.get_extrinsic(vn),
                atol=1e-10,
            )
            np.testing.assert_allclose(
                restored.get_intrinsic(vn),
                default_rig.get_intrinsic(vn),
                atol=1e-10,
            )
            np.testing.assert_allclose(
                restored.get_position(vn),
                default_rig.get_position(vn),
                atol=1e-10,
            )

    def test_from_dict_after_json_roundtrip(self, default_rig):
        """Full JSON roundtrip should work (numpy → list → JSON → list → numpy)."""
        d = default_rig.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        restored = CameraRig.from_dict(parsed)

        for vn in CANONICAL_VIEW_ORDER:
            np.testing.assert_allclose(
                restored.get_extrinsic(vn),
                default_rig.get_extrinsic(vn),
                atol=1e-10,
            )

    def test_refinement_preserved(self, default_rig):
        """Refinement metadata should survive serialization."""
        default_rig.refinement = {
            "applied": True,
            "method": "silhouette_scale",
            "scale_factor": 1.1,
            "focal_adjustment": 0.0,
        }
        d = default_rig.to_dict()
        restored = CameraRig.from_dict(d)
        assert restored.refinement["applied"] is True
        assert restored.refinement["scale_factor"] == 1.1

    def test_camera_data_complete(self, default_rig):
        """Each camera dict should have all expected keys."""
        d = default_rig.to_dict()
        expected_keys = {
            "extrinsic", "intrinsic", "position",
            "yaw_deg", "pitch_deg", "distance",
            "focal_length_mm", "image_width", "image_height",
        }
        for vn, cam in d["cameras"].items():
            assert expected_keys.issubset(set(cam.keys())), \
                f"Camera '{vn}' missing keys: {expected_keys - set(cam.keys())}"


class TestValidateRig:
    """Test rig validation."""

    def test_valid_rig_passes(self, default_rig):
        """A correctly built rig should pass validation."""
        _validate_rig(default_rig)  # should not raise

    def test_missing_view_fails(self, default_rig):
        """Removing a view should fail validation."""
        del default_rig.cameras["top"]
        with pytest.raises(ValueError, match="missing views"):
            _validate_rig(default_rig)

    def test_nan_extrinsic_fails(self, default_rig):
        """NaN in extrinsic should fail."""
        ext = default_rig.cameras["front"]["extrinsic"].copy()
        ext[0, 0] = float("nan")
        default_rig.cameras["front"]["extrinsic"] = ext
        with pytest.raises(ValueError, match="NaN"):
            _validate_rig(default_rig)

    def test_non_orthogonal_rotation_fails(self, default_rig):
        """Non-orthogonal rotation should fail."""
        ext = default_rig.cameras["front"]["extrinsic"].copy()
        ext[0, 0] = 2.0  # break orthogonality
        default_rig.cameras["front"]["extrinsic"] = ext
        with pytest.raises(ValueError, match="orthogonal"):
            _validate_rig(default_rig)

    def test_negative_focal_length_fails(self, default_rig):
        """Negative focal length should fail."""
        K = default_rig.cameras["front"]["intrinsic"].copy()
        K[0, 0] = -100.0
        default_rig.cameras["front"]["intrinsic"] = K
        with pytest.raises(ValueError, match="focal length"):
            _validate_rig(default_rig)

    def test_wrong_distance_fails(self, default_rig):
        """Camera position not matching expected distance should fail."""
        default_rig.cameras["front"]["position"] = np.array([0.0, 0.0, 99.0])
        with pytest.raises(ValueError, match="distance"):
            _validate_rig(default_rig)


# ===========================================================================
# REFINEMENT UNIT TESTS
# ===========================================================================


class TestRefineRigFromSilhouettes:
    """Test scale refinement from silhouette metrics."""

    def test_no_refinement_when_well_framed(self, default_rig, config):
        """If bbox ratio matches expected, no refinement should happen."""
        metrics = _make_preprocess_metrics(
            canvas_size=1024,
            bbox_ratio=EXPECTED_BBOX_IMAGE_RATIO,
        )
        refined = refine_rig_from_silhouettes(default_rig, metrics, config)
        assert refined.refinement["applied"] is False

    def test_refinement_when_subject_too_small(self, default_rig, config):
        """If subject appears small (low bbox ratio), camera should move closer."""
        metrics = _make_preprocess_metrics(
            canvas_size=1024,
            bbox_ratio=0.30,  # much smaller than expected 0.65
        )
        refined = refine_rig_from_silhouettes(default_rig, metrics, config)
        assert refined.refinement["applied"] is True
        # Scale factor < 1 means camera moved closer (shorter distance)
        assert refined.refinement["scale_factor"] < 1.0

    def test_refinement_when_subject_too_large(self, default_rig, config):
        """If subject appears large (high bbox ratio), camera should move back."""
        metrics = _make_preprocess_metrics(
            canvas_size=1024,
            bbox_ratio=0.90,  # much larger than expected 0.65
        )
        refined = refine_rig_from_silhouettes(default_rig, metrics, config)
        assert refined.refinement["applied"] is True
        assert refined.refinement["scale_factor"] > 1.0

    def test_refinement_clamped(self, default_rig, config):
        """Scale adjustment should be clamped to prevent extreme values."""
        metrics = _make_preprocess_metrics(
            canvas_size=1024,
            bbox_ratio=0.01,  # extremely small — would cause huge adjustment
        )
        refined = refine_rig_from_silhouettes(default_rig, metrics, config)
        if refined.refinement["applied"]:
            sf = refined.refinement["scale_factor"]
            assert sf >= 1.0 - MAX_SCALE_ADJUSTMENT
            assert sf <= 1.0 + MAX_SCALE_ADJUSTMENT

    def test_refinement_skipped_with_empty_metrics(self, default_rig, config):
        """No per-view metrics → no refinement."""
        metrics = {"per_view": {}, "canvas_size": 1024}
        refined = refine_rig_from_silhouettes(default_rig, metrics, config)
        assert refined.refinement["applied"] is False

    def test_refinement_skipped_with_insufficient_views(self, default_rig, config):
        """Fewer than 2 side views → no refinement."""
        metrics = _make_preprocess_metrics(canvas_size=1024, bbox_ratio=0.30)
        # Remove all side views except front
        for vn in ["back", "left", "right"]:
            del metrics["per_view"][vn]
        refined = refine_rig_from_silhouettes(default_rig, metrics, config)
        assert refined.refinement["applied"] is False

    def test_refined_rig_passes_validation(self, default_rig, config):
        """A refined rig should still pass all validation checks."""
        metrics = _make_preprocess_metrics(canvas_size=1024, bbox_ratio=0.30)
        refined = refine_rig_from_silhouettes(default_rig, metrics, config)
        _validate_rig(refined)  # should not raise

    def test_refined_rig_has_correct_views(self, default_rig, config):
        """Refined rig should still have all 5 views."""
        metrics = _make_preprocess_metrics(canvas_size=1024, bbox_ratio=0.30)
        refined = refine_rig_from_silhouettes(default_rig, metrics, config)
        assert set(refined.cameras.keys()) == set(CANONICAL_VIEW_ORDER)

    def test_refinement_metadata_complete(self, default_rig, config):
        """Refinement metadata should include all expected fields."""
        metrics = _make_preprocess_metrics(canvas_size=1024, bbox_ratio=0.30)
        refined = refine_rig_from_silhouettes(default_rig, metrics, config)
        if refined.refinement["applied"]:
            ref = refined.refinement
            assert "method" in ref
            assert "scale_factor" in ref
            assert "original_distance" in ref
            assert "new_distance" in ref
            assert "median_observed_ratio" in ref
            assert "side_views_used" in ref

    def test_top_view_excluded_from_refinement(self, default_rig, config):
        """Top view should not affect scale refinement (foreshortening)."""
        # Make side views have expected ratio, but top view is extreme
        metrics = _make_preprocess_metrics(
            canvas_size=1024,
            bbox_ratio=EXPECTED_BBOX_IMAGE_RATIO,
        )
        # Override top view with extreme value
        metrics["per_view"]["top"]["bbox"] = [0, 0, 50, 50]

        refined = refine_rig_from_silhouettes(default_rig, metrics, config)
        # Should not be refined because side views are fine
        assert refined.refinement["applied"] is False


# ===========================================================================
# INTEGRATION TESTS: STAGE RUNNER
# ===========================================================================


class TestRunInitializeCameras:
    """Integration tests for the full stage runner."""

    def _setup_job_with_metrics(self, jm, sm, bbox_ratio=0.65):
        """Create a job and save preprocess_metrics.json."""
        job_id = _create_mv_job(jm)
        metrics = _make_preprocess_metrics(canvas_size=1024, bbox_ratio=bbox_ratio)
        sm.save_artifact_json(job_id, "preprocess_metrics.json", metrics)
        return job_id

    def test_happy_path(self, jm, sm, config):
        """Stage should complete and save camera_init.json."""
        job_id = self._setup_job_with_metrics(jm, sm)
        run_initialize_cameras(job_id, config, jm, sm)

        # camera_init.json should exist
        rig_data = sm.load_artifact_json(job_id, "camera_init.json")
        assert rig_data is not None
        assert rig_data["rig_type"] == "canonical_5view"
        assert len(rig_data["cameras"]) == 5
        assert "shared_params" in rig_data
        assert "refinement" in rig_data

    def test_stage_progress_reaches_one(self, jm, sm, config):
        job_id = self._setup_job_with_metrics(jm, sm)
        run_initialize_cameras(job_id, config, jm, sm)

        job = jm.get_job(job_id)
        assert job["stage_progress"] == 1.0

    def test_missing_preprocess_metrics_raises(self, jm, sm, config):
        job_id = _create_mv_job(jm)
        # No preprocess_metrics.json saved
        with pytest.raises(ValueError, match="preprocess_metrics"):
            run_initialize_cameras(job_id, config, jm, sm)

    def test_artifact_is_valid_json(self, jm, sm, config):
        """Saved artifact should be valid JSON that can be loaded back."""
        job_id = self._setup_job_with_metrics(jm, sm)
        run_initialize_cameras(job_id, config, jm, sm)

        rig_data = sm.load_artifact_json(job_id, "camera_init.json")
        # Should be deserializable into a CameraRig
        rig = CameraRig.from_dict(rig_data)
        assert len(rig.cameras) == 5

    def test_all_cameras_have_matrices(self, jm, sm, config):
        """Each camera in the artifact should have extrinsic and intrinsic."""
        job_id = self._setup_job_with_metrics(jm, sm)
        run_initialize_cameras(job_id, config, jm, sm)

        rig_data = sm.load_artifact_json(job_id, "camera_init.json")
        for vn in CANONICAL_VIEW_ORDER:
            cam = rig_data["cameras"][vn]
            ext = np.array(cam["extrinsic"])
            intr = np.array(cam["intrinsic"])
            assert ext.shape == (4, 4)
            assert intr.shape == (3, 3)

    def test_refinement_applied_for_small_subject(self, jm, sm, config):
        """If subject is small, refinement should be applied."""
        job_id = self._setup_job_with_metrics(jm, sm, bbox_ratio=0.30)
        run_initialize_cameras(job_id, config, jm, sm)

        rig_data = sm.load_artifact_json(job_id, "camera_init.json")
        assert rig_data["refinement"]["applied"] is True

    def test_no_refinement_for_well_framed(self, jm, sm, config):
        """If subject is well-framed, no refinement needed."""
        job_id = self._setup_job_with_metrics(jm, sm, bbox_ratio=EXPECTED_BBOX_IMAGE_RATIO)
        run_initialize_cameras(job_id, config, jm, sm)

        rig_data = sm.load_artifact_json(job_id, "camera_init.json")
        assert rig_data["refinement"]["applied"] is False

    def test_image_size_from_canvas(self, jm, sm, config):
        """Image size should be derived from preprocess canvas_size."""
        job_id = _create_mv_job(jm)
        metrics = _make_preprocess_metrics(canvas_size=512, bbox_ratio=0.65)
        sm.save_artifact_json(job_id, "preprocess_metrics.json", metrics)

        run_initialize_cameras(job_id, config, jm, sm)

        rig_data = sm.load_artifact_json(job_id, "camera_init.json")
        assert rig_data["shared_params"]["image_size"] == [512, 512]

        # Intrinsic principal point should be at (256, 256)
        K = np.array(rig_data["cameras"]["front"]["intrinsic"])
        assert K[0, 2] == pytest.approx(256.0)

    def test_custom_config_respected(self, jm, sm):
        """Custom config (e.g. different seed) should be used."""
        from pipelines.canonical_mv.config import CameraSpec

        config = CanonicalMVConfig()
        # Override camera distance
        for vn in CANONICAL_VIEW_ORDER:
            spec = config.cameras[vn]
            config.cameras[vn] = CameraSpec(
                view_name=spec.view_name,
                yaw_deg=spec.yaw_deg,
                pitch_deg=spec.pitch_deg,
                distance=4.0,
                focal_length=spec.focal_length,
            )

        job_id = _create_mv_job(jm)
        metrics = _make_preprocess_metrics(canvas_size=1024, bbox_ratio=0.65)
        sm.save_artifact_json(job_id, "preprocess_metrics.json", metrics)

        run_initialize_cameras(job_id, config, jm, sm)

        rig_data = sm.load_artifact_json(job_id, "camera_init.json")
        assert rig_data["shared_params"]["distance"] == 4.0

    def test_cameras_consistent_across_runs(self, jm, sm, config):
        """Same input should produce identical camera rigs (deterministic)."""
        job_id1 = self._setup_job_with_metrics(jm, sm)
        run_initialize_cameras(job_id1, config, jm, sm)
        rig1 = sm.load_artifact_json(job_id1, "camera_init.json")

        job_id2 = self._setup_job_with_metrics(jm, sm)
        run_initialize_cameras(job_id2, config, jm, sm)
        rig2 = sm.load_artifact_json(job_id2, "camera_init.json")

        for vn in CANONICAL_VIEW_ORDER:
            np.testing.assert_allclose(
                rig1["cameras"][vn]["extrinsic"],
                rig2["cameras"][vn]["extrinsic"],
                atol=1e-10,
            )


# ===========================================================================
# GEOMETRIC CONSISTENCY TESTS
# ===========================================================================


class TestGeometricConsistency:
    """
    High-level geometric tests that verify the camera rig produces
    correct spatial relationships.
    """

    def test_front_back_opposite_projections(self, default_rig):
        """
        A point to the right of center should project right in the front
        view and left in the back view.
        """
        p = np.array([0.3, 0.0, 0.0])  # +X = right of center

        front_proj = default_rig.project("front", p)
        back_proj = default_rig.project("back", p)

        assert front_proj is not None
        assert back_proj is not None

        # In front view: +X → right of center (x > 512)
        assert front_proj[0] > 512
        # In back view: +X → left of center (x < 512) because camera is behind
        assert back_proj[0] < 512

    def test_left_right_opposite_projections(self, default_rig):
        """
        A point in front of center (+Z) should project differently
        in left vs right views.
        """
        p = np.array([0.0, 0.0, 0.3])  # +Z = in front

        left_proj = default_rig.project("left", p)
        right_proj = default_rig.project("right", p)

        assert left_proj is not None
        assert right_proj is not None

        # In right view (camera on +X looking at -X), +Z should appear to the left
        # In left view (camera on -X looking at +X), +Z should appear to the right
        # These should be on opposite sides of center
        assert (left_proj[0] - 512) * (right_proj[0] - 512) < 0

    def test_top_view_sees_horizontal_extent(self, default_rig):
        """
        The top view should see horizontal extent (X and Z) but not
        vertical extent (Y).
        """
        # Points spread in X
        p_left = default_rig.project("top", np.array([-0.3, 0.0, 0.0]))
        p_right = default_rig.project("top", np.array([0.3, 0.0, 0.0]))

        assert p_left is not None
        assert p_right is not None
        # Should be spread horizontally in the top view
        assert abs(p_left[0] - p_right[0]) > 50

        # Points spread in Y should NOT spread much in the top view
        p_up = default_rig.project("top", np.array([0.0, 0.3, 0.0]))
        p_down = default_rig.project("top", np.array([0.0, -0.3, 0.0]))

        # Y changes should barely affect top-view projection
        # (they're along the camera's view axis)
        if p_up is not None and p_down is not None:
            # Both should be near center in x
            assert abs(p_up[0] - 512) < 50
            assert abs(p_down[0] - 512) < 50

    def test_all_views_see_unit_sphere(self, default_rig):
        """All cameras should be able to see all points on a unit sphere."""
        # Sample points on unit sphere
        for theta in np.linspace(0, 2 * math.pi, 8):
            for phi in np.linspace(-math.pi / 2, math.pi / 2, 5):
                p = np.array([
                    0.5 * math.cos(phi) * math.cos(theta),
                    0.5 * math.sin(phi),
                    0.5 * math.cos(phi) * math.sin(theta),
                ])
                # At least some views should see each point
                visible_count = 0
                for vn in CANONICAL_VIEW_ORDER:
                    proj = default_rig.project(vn, p)
                    if proj is not None:
                        visible_count += 1
                assert visible_count >= 1, \
                    f"Point {p} not visible from any camera"

    def test_projection_within_image_bounds(self, default_rig):
        """Origin and nearby points should project within image bounds."""
        for vn in CANONICAL_VIEW_ORDER:
            for offset in [
                [0, 0, 0], [0.1, 0, 0], [-0.1, 0, 0],
                [0, 0.1, 0], [0, -0.1, 0],
                [0, 0, 0.1], [0, 0, -0.1],
            ]:
                p = np.array(offset, dtype=np.float64)
                proj = default_rig.project(vn, p)
                if proj is not None:
                    assert 0 <= proj[0] <= 1024, \
                        f"x={proj[0]:.1f} out of bounds for {vn}, point={offset}"
                    assert 0 <= proj[1] <= 1024, \
                        f"y={proj[1]:.1f} out of bounds for {vn}, point={offset}"

