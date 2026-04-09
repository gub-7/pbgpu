"""
Configuration for the canonical multi-view reconstruction pipeline.

Centralizes all default parameters, stage ordering, and camera rig
definitions so that the orchestrator and individual stages can share
a single source of truth.

3-view setup:
    - front: perpendicular, centered (camera on +Z axis)
    - side:  perpendicular from the right (camera on +X axis)
    - top:   bird's-eye looking straight down (camera on +Y axis)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math


# ---------------------------------------------------------------------------
# Canonical camera rig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraSpec:
    """Specification for a single canonical camera."""
    view_name: str
    yaw_deg: float      # azimuth in degrees (0 = front)
    pitch_deg: float     # elevation in degrees (0 = horizontal, -90 = top)
    distance: float = 2.5  # camera distance from origin (subject-centered)
    focal_length: float = 50.0  # focal length in mm (default 50mm equiv)

    @property
    def yaw_rad(self) -> float:
        return math.radians(self.yaw_deg)

    @property
    def pitch_rad(self) -> float:
        return math.radians(self.pitch_deg)


# Fixed canonical camera rig for 3 views
CANONICAL_CAMERAS: Dict[str, CameraSpec] = {
    "front": CameraSpec(view_name="front", yaw_deg=0.0, pitch_deg=0.0),
    "side":  CameraSpec(view_name="side",  yaw_deg=90.0, pitch_deg=0.0),
    "top":   CameraSpec(view_name="top",   yaw_deg=0.0, pitch_deg=-90.0),
}

# Canonical view names in processing order
CANONICAL_VIEW_ORDER: List[str] = ["front", "side", "top"]


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class CanonicalMVConfig:
    """
    Full configuration for a canonical multi-view reconstruction run.

    Populated from job params (CanonicalMVParams) + system defaults.
    """
    # Resolution / quality
    output_resolution: int = 1024
    mesh_resolution: int = 256
    texture_resolution: int = 2048

    # Stage toggles
    use_joint_refinement: bool = True
    use_trellis_completion: bool = True
    use_hunyuan_completion: bool = False

    # Priors
    symmetry_prior: bool = True
    category_prior: Optional[str] = None

    # Debug
    generate_debug_renders: bool = False
    generate_gaussian_debug: bool = False
    debug_incremental_recon: bool = False

    # Mesh post-processing
    decimation_target: int = 500_000

    # Reproducibility
    seed: Optional[int] = None

    # Camera rig
    cameras: Dict[str, CameraSpec] = field(
        default_factory=lambda: dict(CANONICAL_CAMERAS)
    )

    # Cross-view preprocessing
    segmentation_model: str = "u2net"
    normalize_white_balance: bool = True
    normalize_lighting: bool = True
    shared_canvas_size: int = 1024

    # View consistency thresholds
    silhouette_area_tolerance: float = 0.35  # max relative area difference
    identity_similarity_threshold: float = 0.65  # min CLIP/DINO cosine sim
    segmentation_confidence_threshold: float = 0.5

    @classmethod
    def from_params(cls, params: dict) -> "CanonicalMVConfig":
        """
        Build a CanonicalMVConfig from a CanonicalMVParams dict
        (as stored in job metadata).
        """
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in params.items() if k in known_fields}
        return cls(**filtered)

