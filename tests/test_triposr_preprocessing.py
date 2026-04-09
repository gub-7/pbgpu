"""
Comprehensive tests for TripoSR-optimized preprocessing pipeline.
Tests all expert-recommended features for optimal TripoSR output.
"""
import pytest
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path

from preprocessing.preprocessing import (
    CategoryType,
    RegionType,
    TRIPOSR_BACKGROUND_COLOR,
    rgba_to_rgb_with_background,
    apply_bilateral_filter,
    apply_clahe_luminance,
    compress_highlights,
    get_boundary_ring_mask,
    apply_boundary_sharpening,
    detect_specular_regions,
    apply_region_aware_processing,
    preprocess_for_triposr,
)

from preprocessing.segmentation import (
    Segmenter,
    segment_for_triposr,
)


@pytest.fixture
def sample_rgba_image():
    """Create a sample RGBA image for testing."""
    # Create 512x512 RGBA image with a centered circle
    img = np.zeros((512, 512, 4), dtype=np.uint8)

    # Create a circle in the center
    center_y, center_x = 256, 256
    radius = 150

    y, x = np.ogrid[:512, :512]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2

    # Fill circle with color and alpha
    img[mask, 0] = 150  # R
    img[mask, 1] = 100  # G
    img[mask, 2] = 200  # B
    img[mask, 3] = 255  # A (opaque)

    # Add some semi-transparent edges (simulate hair/fur)
    edge_mask = ((x - center_x)**2 + (y - center_y)**2 > radius**2) & \
                ((x - center_x)**2 + (y - center_y)**2 <= (radius + 10)**2)
    img[edge_mask, 0] = 150
    img[edge_mask, 1] = 100
    img[edge_mask, 2] = 200
    img[edge_mask, 3] = 128  # Semi-transparent

    return img


@pytest.fixture
def sample_rgba_file(sample_rgba_image, tmp_path):
    """Save sample RGBA image to a temporary file."""
    file_path = tmp_path / "test_rgba.png"
    Image.fromarray(sample_rgba_image, mode='RGBA').save(file_path)
    return str(file_path)


class TestTripoSRBackgroundHandling:
    """Test TripoSR-specific background handling (50% gray)."""

    def test_triposr_background_color(self):
        """Test that TRIPOSR_BACKGROUND_COLOR is 50% gray."""
        assert TRIPOSR_BACKGROUND_COLOR == (128, 128, 128)

    def test_rgba_to_rgb_with_gray_background(self, sample_rgba_image):
        """Test RGBA to RGB conversion with gray background."""
        rgb_result = rgba_to_rgb_with_background(sample_rgba_image)

        # Check output shape
        assert rgb_result.shape == (512, 512, 3)
        assert rgb_result.dtype == np.uint8

        # Check that transparent areas are gray
        transparent_area = sample_rgba_image[:, :, 3] == 0
        if np.any(transparent_area):
            bg_pixels = rgb_result[transparent_area]
            assert np.allclose(bg_pixels[:, 0], 128, atol=2)
            assert np.allclose(bg_pixels[:, 1], 128, atol=2)
            assert np.allclose(bg_pixels[:, 2], 128, atol=2)

    def test_rgba_to_rgb_custom_background(self, sample_rgba_image):
        """Test RGBA to RGB with custom background color."""
        custom_bg = (255, 0, 0)  # Red
        rgb_result = rgba_to_rgb_with_background(sample_rgba_image, custom_bg)

        # Check that transparent areas are red
        transparent_area = sample_rgba_image[:, :, 3] == 0
        if np.any(transparent_area):
            bg_pixels = rgb_result[transparent_area]
            assert np.allclose(bg_pixels[:, 0], 255, atol=2)


class TestEdgePreservingDenoising:
    """Test bilateral filtering for edge-preserving denoising."""

    def test_bilateral_filter_preserves_shape(self, sample_rgba_image):
        """Test that bilateral filter preserves image shape."""
        rgb = sample_rgba_image[:, :, :3]
        filtered = apply_bilateral_filter(rgb, d=9, sigma_color=75, sigma_space=75)

        assert filtered.shape == rgb.shape
        assert filtered.dtype == rgb.dtype

    def test_bilateral_filter_reduces_noise(self):
        """Test that bilateral filter reduces noise."""
        # Create noisy image
        clean = np.ones((100, 100, 3), dtype=np.uint8) * 128
        noisy = clean + np.random.randint(-20, 20, clean.shape, dtype=np.int16)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        # Apply bilateral filter
        filtered = apply_bilateral_filter(noisy, d=9, sigma_color=75, sigma_space=75)

        # Filtered should be closer to clean than noisy
        noise_before = np.mean(np.abs(noisy.astype(float) - clean.astype(float)))
        noise_after = np.mean(np.abs(filtered.astype(float) - clean.astype(float)))

        assert noise_after < noise_before


class TestCLAHELuminance:
    """Test CLAHE on luminance channel only."""

    def test_clahe_preserves_shape(self, sample_rgba_image):
        """Test that CLAHE preserves image shape."""
        rgb = sample_rgba_image[:, :, :3]
        enhanced = apply_clahe_luminance(rgb, clip_limit=2.0)

        assert enhanced.shape == rgb.shape
        assert enhanced.dtype == rgb.dtype

    def test_clahe_enhances_contrast(self):
        """Test that CLAHE enhances local contrast."""
        # Create low-contrast image
        low_contrast = np.ones((200, 200, 3), dtype=np.uint8) * 100
        low_contrast[50:150, 50:150] = 120  # Slight difference

        # Apply CLAHE
        enhanced = apply_clahe_luminance(low_contrast, clip_limit=2.0)

        # Enhanced should have higher contrast
        original_std = np.std(low_contrast[:, :, 0])
        enhanced_std = np.std(enhanced[:, :, 0])

        assert enhanced_std > original_std


class TestHighlightCompression:
    """Test specular highlight compression."""

    def test_compress_highlights_preserves_shape(self, sample_rgba_image):
        """Test that highlight compression preserves shape."""
        rgb = sample_rgba_image[:, :, :3]
        compressed = compress_highlights(rgb, threshold=200, compression_factor=0.7)

        assert compressed.shape == rgb.shape
        assert compressed.dtype == rgb.dtype

    def test_compress_highlights_reduces_brightness(self):
        """Test that highlight compression reduces bright regions."""
        # Create image with highlights
        rgb = np.ones((100, 100, 3), dtype=np.uint8) * 100
        rgb[25:75, 25:75] = 250  # Bright highlight

        # Apply compression
        compressed = compress_highlights(rgb, threshold=200, compression_factor=0.7)

        # Bright region should be darker
        original_max = np.max(rgb[25:75, 25:75])
        compressed_max = np.max(compressed[25:75, 25:75])

        assert compressed_max < original_max


class TestBoundarySharpening:
    """Test selective boundary ring sharpening."""

    def test_boundary_ring_mask_creation(self, sample_rgba_image):
        """Test boundary ring mask extraction."""
        alpha = sample_rgba_image[:, :, 3]
        boundary_mask = get_boundary_ring_mask(alpha, ring_width=3)

        assert boundary_mask.shape == alpha.shape
        assert boundary_mask.dtype == np.uint8

        # Boundary mask should have some non-zero pixels
        assert np.any(boundary_mask > 0)

        # Boundary mask should be smaller than full alpha mask
        full_mask = (alpha > 127).astype(np.uint8) * 255
        assert np.sum(boundary_mask > 0) < np.sum(full_mask > 0)

    def test_boundary_sharpening_preserves_shape(self, sample_rgba_image):
        """Test that boundary sharpening preserves shape."""
        rgb = sample_rgba_image[:, :, :3]
        alpha = sample_rgba_image[:, :, 3]

        sharpened = apply_boundary_sharpening(rgb, alpha, amount=0.5, radius=1, ring_width=3)

        assert sharpened.shape == rgb.shape
        assert sharpened.dtype == rgb.dtype


class TestSpecularDetection:
    """Test specular region detection."""

    def test_detect_specular_regions(self):
        """Test specular region detection."""
        # Create image with specular highlight
        rgb = np.ones((100, 100, 3), dtype=np.uint8) * 100
        rgb[25:75, 25:75] = 250  # Bright specular region

        specular_mask = detect_specular_regions(rgb, threshold=200)

        assert specular_mask.shape == (100, 100)
        assert specular_mask.dtype == np.uint8

        # Specular region should be detected
        assert np.any(specular_mask[25:75, 25:75] > 0)


class TestRegionAwareProcessing:
    """Test region-aware processing for different categories."""

    def test_human_category_processing(self, sample_rgba_image):
        """Test region-aware processing for human category."""
        processed, metadata = apply_region_aware_processing(
            sample_rgba_image,
            CategoryType.HUMAN_BUST,
            enable_denoising=True,
            enable_clahe=True,
            enable_boundary_sharpening=True,
            enable_highlight_compression=False
        )

        assert processed.shape == sample_rgba_image.shape
        assert metadata["category"] == "human_bust"
        assert "bilateral_filter_foreground" in metadata["operations_applied"]
        assert "clahe_luminance_foreground" in metadata["operations_applied"]
        assert "boundary_sharpening" in metadata["operations_applied"]

    def test_animal_category_processing(self, sample_rgba_image):
        """Test region-aware processing for animal category."""
        processed, metadata = apply_region_aware_processing(
            sample_rgba_image,
            CategoryType.ANIMAL_BUST,
            enable_denoising=True,
            enable_clahe=True,
            enable_boundary_sharpening=True,
            enable_highlight_compression=False
        )

        assert processed.shape == sample_rgba_image.shape
        assert metadata["category"] == "animal_bust"
        assert "bilateral_filter_fur_aware" in metadata["operations_applied"]
        assert "clahe_luminance_fur" in metadata["operations_applied"]

    def test_object_category_processing(self, sample_rgba_image):
        """Test region-aware processing for object category."""
        # Add specular highlight to trigger compression
        sample_rgba_image[100:200, 100:200, :3] = 250

        processed, metadata = apply_region_aware_processing(
            sample_rgba_image,
            CategoryType.GENERIC_OBJECT,
            enable_denoising=True,
            enable_clahe=True,
            enable_boundary_sharpening=True,
            enable_highlight_compression=True
        )

        assert processed.shape == sample_rgba_image.shape
        assert metadata["category"] == "generic_object"
        assert "bilateral_filter_object" in metadata["operations_applied"]
        assert "clahe_luminance_object" in metadata["operations_applied"]


class TestTripoSRPreprocessingPipeline:
    """Test complete TripoSR preprocessing pipeline."""

    def test_preprocess_for_triposr_rgb_output(self, sample_rgba_file, tmp_path):
        """Test complete preprocessing with RGB output."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = preprocess_for_triposr(
            rgba_path=sample_rgba_file,
            category=CategoryType.HUMAN_BUST,
            foreground_ratio=0.85,
            enable_region_processing=True,
            enable_denoising=True,
            enable_clahe=True,
            enable_boundary_sharpening=True,
            output_format="rgb",
            output_dir=str(output_dir)
        )

        # Check metadata
        assert result["category"] == "human_bust"
        assert result["foreground_ratio"] == 0.85
        assert result["output_format"] == "rgb"
        assert result["background_color"] == TRIPOSR_BACKGROUND_COLOR

        # Check output file exists
        assert Path(result["output_path"]).exists()

        # Load and verify output
        output_img = Image.open(result["output_path"])
        assert output_img.mode == "RGB"

        output_array = np.array(output_img)
        assert output_array.shape[2] == 3  # RGB

    def test_preprocess_for_triposr_rgba_output(self, sample_rgba_file, tmp_path):
        """Test complete preprocessing with RGBA output."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = preprocess_for_triposr(
            rgba_path=sample_rgba_file,
            category=CategoryType.GENERIC_OBJECT,
            foreground_ratio=0.90,
            enable_region_processing=True,
            output_format="rgba",
            output_dir=str(output_dir)
        )

        # Check output format
        assert result["output_format"] == "rgba"

        # Load and verify output
        output_img = Image.open(result["output_path"])
        assert output_img.mode == "RGBA"

    def test_foreground_ratio_control(self, sample_rgba_file, tmp_path):
        """Test foreground ratio control."""
        result = preprocess_for_triposr(
            rgba_path=sample_rgba_file,
            category=CategoryType.HUMAN_BUST,
            foreground_ratio=0.75,
            enable_region_processing=False,
            output_format="rgb",
            output_dir=str(tmp_path)
        )

        # Check that ratio is applied
        actual_ratio = result["actual_ratio"]
        assert 0.70 <= actual_ratio <= 0.80  # Close to target

    def test_category_specific_parameters(self, sample_rgba_file, tmp_path):
        """Test that different categories use different parameters."""
        # Human bust
        result_human = preprocess_for_triposr(
            rgba_path=sample_rgba_file,
            category=CategoryType.HUMAN_BUST,
            foreground_ratio=0.85,
            enable_region_processing=True,
            output_dir=str(tmp_path)
        )

        # Animal bust
        result_animal = preprocess_for_triposr(
            rgba_path=sample_rgba_file,
            category=CategoryType.ANIMAL_BUST,
            foreground_ratio=0.85,
            enable_region_processing=True,
            output_dir=str(tmp_path)
        )

        # Both should have different operations applied
        human_ops = result_human["region_processing"]["operations_applied"]
        animal_ops = result_animal["region_processing"]["operations_applied"]

        assert "bilateral_filter_foreground" in human_ops
        assert "bilateral_filter_fur_aware" in animal_ops


class TestSegmenterEnhancements:
    """Test enhanced segmentation features."""

    def test_segmenter_initialization(self):
        """Test segmenter initialization with different models."""
        segmenter = Segmenter(model_name="u2net")
        assert segmenter.model_name == "u2net"

    def test_edge_decontamination(self, sample_rgba_image):
        """Test edge decontamination."""
        segmenter = Segmenter()
        decontaminated = segmenter.decontaminate_edges(sample_rgba_image, edge_width=3)

        assert decontaminated.shape == sample_rgba_image.shape
        assert decontaminated.dtype == sample_rgba_image.dtype

    def test_edge_feathering(self, sample_rgba_image):
        """Test edge feathering."""
        segmenter = Segmenter()
        feathered = segmenter.feather_edges(sample_rgba_image, feather_radius=2)

        assert feathered.shape == sample_rgba_image.shape

        # Alpha channel should be smoothed
        original_alpha = sample_rgba_image[:, :, 3]
        feathered_alpha = feathered[:, :, 3]

        # Feathered alpha should have smoother transitions
        original_edges = np.sum(np.abs(np.diff(original_alpha.astype(float), axis=0)))
        feathered_edges = np.sum(np.abs(np.diff(feathered_alpha.astype(float), axis=0)))

        assert feathered_edges < original_edges

    def test_hair_fur_matting_enhancement(self, sample_rgba_image):
        """Test hair/fur matting enhancement."""
        segmenter = Segmenter()
        enhanced = segmenter.enhance_hair_fur_matting(sample_rgba_image, preserve_detail=True)

        assert enhanced.shape == sample_rgba_image.shape
        assert enhanced.dtype == sample_rgba_image.dtype


class TestCategoryTypeEnum:
    """Test CategoryType enum."""

    def test_category_types_exist(self):
        """Test that all category types are defined."""
        assert hasattr(CategoryType, "HUMAN_FULL_BODY")
        assert hasattr(CategoryType, "ANIMAL_FULL_BODY")
        assert hasattr(CategoryType, "HUMAN_BUST")
        assert hasattr(CategoryType, "ANIMAL_BUST")
        assert hasattr(CategoryType, "GENERIC_OBJECT")


class TestRegionTypeEnum:
    """Test RegionType enum."""

    def test_region_types_exist(self):
        """Test that all region types are defined."""
        assert hasattr(RegionType, "BACKGROUND")
        assert hasattr(RegionType, "BOUNDARY_RING")
        assert hasattr(RegionType, "FACE")
        assert hasattr(RegionType, "HAIR")
        assert hasattr(RegionType, "SKIN")
        assert hasattr(RegionType, "CLOTHING")
        assert hasattr(RegionType, "HANDS")
        assert hasattr(RegionType, "FUR")
        assert hasattr(RegionType, "EARS")
        assert hasattr(RegionType, "WHISKERS")
        assert hasattr(RegionType, "THIN_STRUCTURES")
        assert hasattr(RegionType, "SPECULAR")
        assert hasattr(RegionType, "TEXT_LOGO")


class TestIntegration:
    """Integration tests for complete pipeline."""

    def test_end_to_end_preprocessing(self, sample_rgba_file, tmp_path):
        """Test complete end-to-end preprocessing pipeline."""
        # Complete pipeline: segmentation + preprocessing
        result = preprocess_for_triposr(
            rgba_path=sample_rgba_file,
            category=CategoryType.HUMAN_BUST,
            foreground_ratio=0.85,
            enable_region_processing=True,
            enable_denoising=True,
            enable_clahe=True,
            enable_boundary_sharpening=True,
            enable_highlight_compression=False,
            output_format="rgb",
            output_dir=str(tmp_path)
        )

        # Verify all expected operations were applied
        operations = result["region_processing"]["operations_applied"]
        assert len(operations) > 0

        # Verify output is valid
        output_img = Image.open(result["output_path"])
        output_array = np.array(output_img)

        # Check that image is not all black or all white
        assert np.mean(output_array) > 10
        assert np.mean(output_array) < 245

        # Check that background is gray
        alpha_original = np.array(Image.open(sample_rgba_file))[:, :, 3]
        bg_mask = alpha_original < 10
        if np.any(bg_mask):
            bg_pixels = output_array[bg_mask]
            # Should be close to gray (128, 128, 128)
            assert np.abs(np.mean(bg_pixels) - 128) < 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

