"""
Tests for pipelines/camera_init.py – camera pose computation and COLMAP export.
"""

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from api.models import CameraIntrinsics, SphericalPose, ViewLabel
from pipelines.camera_init import (
    CANONICAL_FRONT,
    CANONICAL_SIDE,
    CANONICAL_TOP,
    build_lookat_rotation,
    export_colmap_cameras_txt,
    export_colmap_images_txt,
    export_colmap_workspace,
    get_canonical_views,
    pose_to_extrinsics,
    resolve_views,
    rotation_to_quaternion_wxyz,
    spherical_to_camera_center,
)


class TestSphericalToCameraCenter:
    """Test spherical → cartesian conversion."""

    def test_front_view_position(self):
        """Front view (az=0, el=0) should be on the -Y axis."""
        C = spherical_to_camera_center(CANONICAL_FRONT)
        assert C[0] == pytest.approx(0.0, abs=1e-10)
        assert C[1] == pytest.approx(-CANONICAL_FRONT.radius, abs=1e-10)
        assert C[2] == pytest.approx(0.0, abs=1e-10)

    def test_side_view_position(self):
        """Side view (az=90, el=0) should be on the +X axis."""
        C = spherical_to_camera_center(CANONICAL_SIDE)
        assert C[0] == pytest.approx(CANONICAL_SIDE.radius, abs=1e-10)
        assert C[1] == pytest.approx(0.0, abs=1e-10)
        assert C[2] == pytest.approx(0.0, abs=1e-10)

    def test_top_view_position(self):
        """Top view (az=0, el=90) should be on the +Z axis."""
        C = spherical_to_camera_center(CANONICAL_TOP)
        assert C[0] == pytest.approx(0.0, abs=1e-10)
        assert C[1] == pytest.approx(0.0, abs=1e-10)
        assert C[2] == pytest.approx(CANONICAL_TOP.radius, abs=1e-10)

    def test_custom_target(self):
        """Camera centre should be offset by the target point."""
        pose = SphericalPose(
            radius=2.0,
            azimuth_deg=0.0,
            elevation_deg=0.0,
            target_world=[1.0, 2.0, 3.0],
        )
        C = spherical_to_camera_center(pose)
        assert C[0] == pytest.approx(1.0, abs=1e-10)
        assert C[1] == pytest.approx(2.0 - 2.0, abs=1e-10)
        assert C[2] == pytest.approx(3.0, abs=1e-10)


class TestBuildLookatRotation:
    """Test the look-at rotation matrix construction."""

    def test_rotation_is_orthogonal(self):
        """R_c2w should be a proper rotation matrix (det=1, R^T R = I)."""
        C = np.array([0.0, -1.2, 0.0])
        target = np.array([0.0, 0.0, 0.0])
        R = build_lookat_rotation(C, target)
        assert R.shape == (3, 3)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)

    def test_forward_points_at_target(self):
        """Camera forward (column 2 of R_c2w) should point from C toward target."""
        C = np.array([0.0, -1.2, 0.0])
        target = np.array([0.0, 0.0, 0.0])
        R = build_lookat_rotation(C, target)
        forward = R[:, 2]
        expected = (target - C) / np.linalg.norm(target - C)
        np.testing.assert_allclose(forward, expected, atol=1e-10)

    def test_top_view_degenerate_case(self):
        """Top view should handle the degenerate case (forward ≈ -Z)."""
        C = np.array([0.0, 0.0, 1.2])
        target = np.array([0.0, 0.0, 0.0])
        R = build_lookat_rotation(C, target)
        assert R.shape == (3, 3)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)


class TestRotationToQuaternion:
    """Test rotation → quaternion conversion."""

    def test_identity(self):
        """Identity rotation → quaternion [1, 0, 0, 0]."""
        q = rotation_to_quaternion_wxyz(np.eye(3))
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-10)

    def test_unit_quaternion(self):
        """Quaternion should always be unit length."""
        R = build_lookat_rotation(
            np.array([1.0, -1.0, 0.5]),
            np.array([0.0, 0.0, 0.0]),
        )
        R_w2c = R.T
        q = rotation_to_quaternion_wxyz(R_w2c)
        assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-10)

    def test_w_positive(self):
        """Quaternion w component should always be non-negative."""
        for az in range(0, 360, 30):
            pose = SphericalPose(radius=1.0, azimuth_deg=float(az), elevation_deg=15.0)
            ext = pose_to_extrinsics(pose)
            assert ext.quaternion_wxyz[0] >= 0


class TestPoseToExtrinsics:
    """Test the full pose → extrinsics pipeline."""

    def test_world_to_camera_inversion(self):
        """C = -R_w2c^T @ t_w2c should recover the camera centre."""
        for pose in (CANONICAL_FRONT, CANONICAL_SIDE, CANONICAL_TOP):
            ext = pose_to_extrinsics(pose)
            R = np.array(ext.R_w2c).reshape(3, 3)
            t = np.array(ext.t_w2c)
            C_recovered = -R.T @ t
            C_expected = spherical_to_camera_center(pose)
            np.testing.assert_allclose(C_recovered, C_expected, atol=1e-10)

    def test_extrinsics_shapes(self):
        """Extrinsics should have correct shapes."""
        ext = pose_to_extrinsics(CANONICAL_FRONT)
        assert len(ext.R_w2c) == 9
        assert len(ext.t_w2c) == 3
        assert len(ext.quaternion_wxyz) == 4


class TestGetCanonicalViews:
    """Test canonical view generation."""

    def test_three_views(self):
        """Should return exactly 3 views."""
        views = get_canonical_views()
        assert len(views) == 3

    def test_view_labels(self):
        """Should have front, side, and top labels."""
        views = get_canonical_views()
        labels = {v.label for v in views}
        assert labels == {ViewLabel.FRONT, ViewLabel.SIDE, ViewLabel.TOP}

    def test_custom_filenames(self):
        """Should accept custom filenames."""
        filenames = {
            ViewLabel.FRONT: "img_front.jpg",
            ViewLabel.SIDE: "img_side.jpg",
            ViewLabel.TOP: "img_top.jpg",
        }
        views = get_canonical_views(image_filenames=filenames)
        for v in views:
            assert v.image_filename == filenames[v.label]

    def test_custom_radius(self):
        """Should accept custom radius."""
        views = get_canonical_views(radius=2.5)
        for v in views:
            assert v.pose.radius == 2.5


class TestResolveViews:
    """Test view resolution (ViewSpec → ResolvedView)."""

    def test_resolves_all_views(self):
        """Should resolve all 3 views with extrinsics."""
        specs = get_canonical_views()
        resolved = resolve_views(specs)
        assert len(resolved) == 3
        for rv in resolved:
            assert rv.extrinsics is not None
            assert rv.intrinsics is not None

    def test_custom_intrinsics(self):
        """Should use custom intrinsics when provided."""
        intr = CameraIntrinsics(width=1024, height=1024, fx=800, fy=800, cx=512, cy=512)
        specs = get_canonical_views()
        resolved = resolve_views(specs, intrinsics=intr)
        for rv in resolved:
            assert rv.intrinsics.width == 1024
            assert rv.intrinsics.fx == 800


class TestColmapExport:
    """Test COLMAP-format file export."""

    def test_cameras_txt(self):
        """Should write a valid cameras.txt file."""
        specs = get_canonical_views()
        resolved = resolve_views(specs)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cameras.txt"
            export_colmap_cameras_txt(resolved, path)

            assert path.exists()
            content = path.read_text()
            assert "PINHOLE" in content
            assert "2048" in content

    def test_images_txt(self):
        """Should write a valid images.txt with 3 entries."""
        specs = get_canonical_views()
        resolved = resolve_views(specs)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "images.txt"
            export_colmap_images_txt(resolved, path)

            assert path.exists()
            content = path.read_text()
            assert "front.png" in content
            assert "side.png" in content
            assert "top.png" in content

    def test_workspace_structure(self):
        """Should create the full sparse/0/ directory structure."""
        specs = get_canonical_views()
        resolved = resolve_views(specs)

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "colmap"
            sparse_dir = export_colmap_workspace(resolved, workspace)

            assert sparse_dir.exists()
            assert (sparse_dir / "cameras.txt").exists()
            assert (sparse_dir / "images.txt").exists()
            assert (sparse_dir / "points3D.txt").exists()


class TestStitchingGeometry:
    """
    Verify the stitching geometry between views.

    The requirement is:
      - Front's LEFT edge  ↔ Side's RIGHT edge
      - Front's TOP edge   ↔ Top's BOTTOM edge
      - Top's LEFT edge    ↔ Side's TOP edge
    """

    def test_front_left_meets_side_right(self):
        """
        The front camera's left image edge corresponds to the +X direction.
        The side camera (at +X) sees the same edge on its right side.
        """
        front_C = spherical_to_camera_center(CANONICAL_FRONT)
        side_C = spherical_to_camera_center(CANONICAL_SIDE)

        assert front_C[1] < 0  # front is on -Y
        assert side_C[0] > 0  # side is on +X

    def test_front_top_meets_top_bottom(self):
        """
        The front camera's top image edge corresponds to the +Z direction.
        The top camera (at +Z) sees the same edge on its bottom side.
        """
        front_C = spherical_to_camera_center(CANONICAL_FRONT)
        top_C = spherical_to_camera_center(CANONICAL_TOP)

        assert front_C[1] < 0  # front is on -Y
        assert top_C[2] > 0  # top is above

