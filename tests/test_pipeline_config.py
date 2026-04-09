"""
Tests for the canonical multi-view pipeline configuration.
"""

import math
import pytest

from pipelines.canonical_mv.config import (
    CanonicalMVConfig,
    CameraSpec,
    CANONICAL_CAMERAS,
    CANONICAL_VIEW_ORDER,
)


class TestCameraSpec:
    def test_front_camera(self):
        cam = CANONICAL_CAMERAS["front"]
        assert cam.yaw_deg == 0.0
        assert cam.pitch_deg == 0.0
        assert cam.yaw_rad == 0.0

    def test_right_camera(self):
        cam = CANONICAL_CAMERAS["right"]
        assert cam.yaw_deg == 90.0
        assert abs(cam.yaw_rad - math.pi / 2) < 1e-10

    def test_back_camera(self):
        cam = CANONICAL_CAMERAS["back"]
        assert cam.yaw_deg == 180.0

    def test_left_camera(self):
        cam = CANONICAL_CAMERAS["left"]
        assert cam.yaw_deg == 270.0

    def test_top_camera(self):
        cam = CANONICAL_CAMERAS["top"]
        assert cam.pitch_deg == -90.0
        assert abs(cam.pitch_rad - (-math.pi / 2)) < 1e-10

    def test_all_cameras_present(self):
        assert set(CANONICAL_CAMERAS.keys()) == {"front", "back", "left", "right", "top"}


class TestCanonicalViewOrder:
    def test_order(self):
        assert len(CANONICAL_VIEW_ORDER) == 5
        assert CANONICAL_VIEW_ORDER[0] == "front"


class TestCanonicalMVConfig:
    def test_defaults(self):
        cfg = CanonicalMVConfig()
        assert cfg.output_resolution == 1024
        assert cfg.mesh_resolution == 256
        assert cfg.texture_resolution == 2048
        assert cfg.use_joint_refinement is True
        assert cfg.use_trellis_completion is True
        assert cfg.use_hunyuan_completion is False
        assert cfg.symmetry_prior is True
        assert cfg.decimation_target == 500_000
        assert cfg.seed is None
        assert len(cfg.cameras) == 5

    def test_from_params(self):
        params = {
            "seed": 42,
            "mesh_resolution": 384,
            "use_joint_refinement": False,
            "extra_unknown_field": "ignored",
        }
        cfg = CanonicalMVConfig.from_params(params)
        assert cfg.seed == 42
        assert cfg.mesh_resolution == 384
        assert cfg.use_joint_refinement is False
        # Unknown fields should be ignored
        assert not hasattr(cfg, "extra_unknown_field")

    def test_from_empty_params(self):
        cfg = CanonicalMVConfig.from_params({})
        assert cfg.output_resolution == 1024  # defaults

    def test_camera_rig_independent_copies(self):
        """Ensure each config gets its own camera dict (no shared mutable state)."""
        cfg1 = CanonicalMVConfig()
        cfg2 = CanonicalMVConfig()
        assert cfg1.cameras is not cfg2.cameras

