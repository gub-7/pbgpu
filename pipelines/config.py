"""
Pipeline configuration and environment-level defaults.

Reads settings from environment variables with sensible defaults.
All path constants, model cache locations, and feature flags live here.
"""

from __future__ import annotations

import os
from pathlib import Path

from api.models import (
    CameraIntrinsics,
    PipelineConfig,
    ReconBackend,
    WorldConvention,
)

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORAGE_ROOT = Path(os.getenv("STORAGE_ROOT", str(PROJECT_ROOT / "storage")))
LOG_DIR = Path(os.getenv("LOG_DIR", str(PROJECT_ROOT / "logs")))
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", str(PROJECT_ROOT / "model_cache")))

# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ---------------------------------------------------------------------------
# External model paths
# ---------------------------------------------------------------------------

# DUSt3R / MASt3R checkpoints.
# Default to HuggingFace model IDs so `from_pretrained()` auto-downloads
# the weights on first use.  Override with a local file path if you
# pre-downloaded checkpoints (e.g. via setup/setup_dust3r.sh).
DUST3R_CHECKPOINT = os.getenv(
    "DUST3R_CHECKPOINT",
    "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
)
MAST3R_CHECKPOINT = os.getenv(
    "MAST3R_CHECKPOINT",
    "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
)

# TRELLIS.2 repo location (setup by setup/setup_trellis.sh)
TRELLIS_REPO_DIR = Path(os.getenv("TRELLIS_REPO_DIR", "/root/TRELLIS.2"))

# TripoSR repo location (setup by setup/setup_triposr.sh)
TRIPOSR_REPO_DIR = Path(os.getenv("TRIPOSR_REPO_DIR", "/root/TripoSR"))

# ---------------------------------------------------------------------------
# Image defaults
# ---------------------------------------------------------------------------

DEFAULT_IMAGE_SIZE = int(os.getenv("DEFAULT_IMAGE_SIZE", "2048"))
SUPPORTED_IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

# ---------------------------------------------------------------------------
# Pipeline feature flags
# ---------------------------------------------------------------------------

USE_BACKGROUND_FOR_POSE = os.getenv("USE_BACKGROUND_FOR_POSE", "true").lower() == "true"
TRELLIS_ENABLED = os.getenv("TRELLIS_ENABLED", "true").lower() == "true"
DEFAULT_RECON_BACKEND = ReconBackend(os.getenv("DEFAULT_RECON_BACKEND", "dust3r"))
DEFAULT_MASK_METHOD = os.getenv("DEFAULT_MASK_METHOD", "rembg")

# ---------------------------------------------------------------------------
# Camera defaults for the canonical 3-view rig
# ---------------------------------------------------------------------------

DEFAULT_RADIUS = float(os.getenv("DEFAULT_CAMERA_RADIUS", "1.2"))
DEFAULT_FOCAL_LENGTH = float(os.getenv("DEFAULT_FOCAL_LENGTH", "1700.0"))


def get_default_intrinsics(image_size: int | None = None) -> CameraIntrinsics:
    """Return default pinhole intrinsics for a square image."""
    size = image_size or DEFAULT_IMAGE_SIZE
    cx = cy = size / 2.0
    return CameraIntrinsics(
        width=size,
        height=size,
        fx=DEFAULT_FOCAL_LENGTH,
        fy=DEFAULT_FOCAL_LENGTH,
        cx=cx,
        cy=cy,
    )


def get_default_pipeline_config() -> PipelineConfig:
    """Build a PipelineConfig from environment defaults."""
    return PipelineConfig(
        world=WorldConvention(),
        intrinsics=get_default_intrinsics(),
        recon_backend=DEFAULT_RECON_BACKEND,
        use_background_for_pose=USE_BACKGROUND_FOR_POSE,
        trellis_enabled=TRELLIS_ENABLED,
        image_size=DEFAULT_IMAGE_SIZE,
        mask_method=DEFAULT_MASK_METHOD,
    )


def ensure_directories() -> None:
    """Create required directories if they don't exist."""
    for d in (STORAGE_ROOT, LOG_DIR, MODEL_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)

