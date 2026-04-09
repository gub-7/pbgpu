"""
Preprocessing module for 3D reconstruction pipeline
Enhanced with TripoSR-optimized preprocessing and segmentation
"""

# Legacy functions (backward compatibility)
from .segmentation import segment_image, Segmenter
from .preprocessing import (
    preprocess_for_category,
    enhance_alpha_quality,
    center_and_scale,
    adjust_for_category,
    CategoryType
)

# New TripoSR-optimized functions (recommended)
from .segmentation import segment_for_triposr
from .preprocessing import (
    preprocess_for_triposr,
    apply_region_aware_processing,
    rgba_to_rgb_with_background,
    apply_bilateral_filter,
    apply_clahe_luminance,
    compress_highlights,
    apply_boundary_sharpening,
    TRIPOSR_BACKGROUND_COLOR,
    COMPOSITE_FEATHER_RADIUS_DEFAULT,
    RegionType,
    # Auto FFR framing (replaces static foreground_ratio)
    compute_mask_metrics,
    auto_frame_for_triposr,
    FFR_TARGET_DEFAULT,
    FFR_MIN_DEFAULT,
    FFR_MAX_DEFAULT,
    BBO_MIN_DEFAULT,
)

from .preview_generator import PreviewGenerator, PreprocessingPipeline

__all__ = [
    # Legacy
    'segment_image',
    'Segmenter',
    'preprocess_for_category',
    'enhance_alpha_quality',
    'center_and_scale',
    'adjust_for_category',
    'CategoryType',

    # New TripoSR-optimized
    'segment_for_triposr',
    'preprocess_for_triposr',
    'apply_region_aware_processing',
    'rgba_to_rgb_with_background',
    'apply_bilateral_filter',
    'apply_clahe_luminance',
    'compress_highlights',
    'apply_boundary_sharpening',
    'TRIPOSR_BACKGROUND_COLOR',
    'COMPOSITE_FEATHER_RADIUS_DEFAULT',
    'RegionType',

    # Auto FFR framing
    'compute_mask_metrics',
    'auto_frame_for_triposr',
    'FFR_TARGET_DEFAULT',
    'FFR_MIN_DEFAULT',
    'FFR_MAX_DEFAULT',
    'BBO_MIN_DEFAULT',

    # Pipeline
    'PreviewGenerator',
    'PreprocessingPipeline'
]


