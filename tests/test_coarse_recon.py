"""Tests for pipelines/coarse_recon.py – PLY I/O and backend factory."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from api.models import ReconBackend
from pipelines.coarse_recon import (
    _write_ply,
    read_ply,
    get_backend,
)


class TestPlyIO:
    """Test PLY read/write utilities."""

    def test_write_and_read_points(self, tmp_path):
        points = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        path = tmp_path / "test.ply"
        _write_ply(path, points)

        read_pts, read_colors = read_ply(path)
        np.testing.assert_allclose(read_pts, points, atol=1e-5)
        assert read_colors is None

    def test_write_and_read_with_colors(self, tmp_path):
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        path = tmp_path / "test_color.ply"
        _write_ply(path, points, colors)

        read_pts, read_colors = read_ply(path)
        np.testing.assert_allclose(read_pts, points, atol=1e-5)
        assert read_colors is not None
        np.testing.assert_allclose(read_colors, colors, atol=0.01)

    def test_empty_point_cloud(self, tmp_path):
        points = np.zeros((0, 3))
        path = tmp_path / "empty.ply"
        _write_ply(path, points)

        read_pts, _ = read_ply(path)
        assert len(read_pts) == 0

    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "test.ply"
        points = np.array([[1.0, 2.0, 3.0]])
        _write_ply(path, points)
        assert path.exists()


class TestGetBackend:
    """Test backend factory (without actually loading models)."""

    def test_dust3r_backend(self):
        backend = get_backend(ReconBackend.DUST3R, device="cpu", checkpoint_path="/fake/path")
        assert backend is not None
        assert hasattr(backend, "reconstruct")

    def test_mast3r_backend(self):
        backend = get_backend(ReconBackend.MAST3R, device="cpu", checkpoint_path="/fake/path")
        assert backend is not None

    def test_vggt_backend(self):
        backend = get_backend(ReconBackend.VGGT, device="cpu")
        assert backend is not None

    def test_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_backend("nonexistent", device="cpu")

