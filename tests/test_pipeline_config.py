"""Tests for pipelines/config.py – pipeline configuration."""

import os
from pathlib import Path

import pytest

from api.models import CameraIntrinsics, PipelineConfig, ReconBackend
from pipelines.config import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_RADIUS,
    STORAGE_ROOT,
    get_default_intrinsics,
    get_default_pipeline_config,
    ensure_directories,
)


class TestGetDefaultIntrinsics:
    def test_default_size(self):
        intr = get_default_intrinsics()
        assert intr.width == DEFAULT_IMAGE_SIZE
        assert intr.height == DEFAULT_IMAGE_SIZE
        assert intr.cx == DEFAULT_IMAGE_SIZE / 2.0

    def test_custom_size(self):
        intr = get_default_intrinsics(image_size=1024)
        assert intr.width == 1024
        assert intr.cx == 512.0


class TestGetDefaultPipelineConfig:
    def test_returns_valid_config(self):
        cfg = get_default_pipeline_config()
        assert isinstance(cfg, PipelineConfig)
        assert cfg.recon_backend in ReconBackend
        assert cfg.image_size > 0

    def test_config_uses_env_defaults(self):
        cfg = get_default_pipeline_config()
        assert cfg.use_background_for_pose is True
        assert cfg.trellis_enabled is True


class TestEnsureDirectories:
    def test_creates_directories(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pipelines.config.STORAGE_ROOT", tmp_path / "storage")
        monkeypatch.setattr("pipelines.config.LOG_DIR", tmp_path / "logs")
        monkeypatch.setattr("pipelines.config.MODEL_CACHE_DIR", tmp_path / "cache")

        ensure_directories()

        assert (tmp_path / "storage").exists()
        assert (tmp_path / "logs").exists()
        assert (tmp_path / "cache").exists()

