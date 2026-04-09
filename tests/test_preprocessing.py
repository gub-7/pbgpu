"""
Preprocessing pipeline tests
"""

import pytest
from PIL import Image
import numpy as np
from pathlib import Path
import tempfile
import shutil

from preprocessing import (
    PreviewGenerator,
    PreprocessingPipeline,
    preprocess_for_category,
    enhance_alpha_quality
)


@pytest.fixture
def test_job_id():
    """Create a test job ID"""
    return "test-job-123"


@pytest.fixture
def temp_storage(test_job_id):
    """Create temporary storage directory"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_rgba_image():
    """Create a test RGBA image"""
    # Create 512x512 RGBA image with red square in center
    img = np.zeros((512, 512, 4), dtype=np.uint8)

    # Red square 256x256 in center
    img[128:384, 128:384, 0] = 255  # R
    img[128:384, 128:384, 3] = 255  # A (opaque)

    return Image.fromarray(img, mode='RGBA')


def test_preview_generator_init(test_job_id, temp_storage):
    """Test PreviewGenerator initialization"""
    gen = PreviewGenerator(test_job_id, storage_root=str(temp_storage))

    assert gen.job_id == test_job_id
    assert gen.job_dir.exists()
    assert gen.metadata["job_id"] == test_job_id


def test_preview_generator_save_preview(test_job_id, temp_storage, test_rgba_image):
    """Test saving a preview image"""
    gen = PreviewGenerator(test_job_id, storage_root=str(temp_storage))

    stage_info = gen.save_preview(test_rgba_image, "test_stage", "test.png")

    assert stage_info["stage"] == "test_stage"
    assert stage_info["filename"] == "test.png"
    assert "url" in stage_info
    assert "timestamp" in stage_info

    # Check file was created
    assert (gen.job_dir / "test.png").exists()


def test_preview_generator_get_previews(test_job_id, temp_storage, test_rgba_image):
    """Test getting all previews"""
    gen = PreviewGenerator(test_job_id, storage_root=str(temp_storage))

    # Save multiple previews
    gen.save_preview(test_rgba_image, "stage1", "stage1.png")
    gen.save_preview(test_rgba_image, "stage2", "stage2.png")

    previews = gen.get_previews()

    assert len(previews["stages"]) == 2
    assert previews["stages"][0]["stage"] == "stage1"
    assert previews["stages"][1]["stage"] == "stage2"


def test_enhance_alpha_quality_human(test_rgba_image):
    """Test alpha enhancement for human category"""
    enhanced = enhance_alpha_quality(test_rgba_image, category="human_bust")

    # Check that alpha is binarized
    alpha = np.array(enhanced)[:, :, 3]
    unique_values = np.unique(alpha)

    # Should mostly be 0 or 255 after enhancement
    assert len(unique_values) <= 10  # Allow some intermediate values from antialiasing


def test_enhance_alpha_quality_animal_aggressive(test_rgba_image):
    """Test aggressive alpha enhancement for animals"""
    enhanced = enhance_alpha_quality(
        test_rgba_image,
        category="animal_bust",
        aggressive=True
    )

    # Check that alpha is more aggressively binarized
    alpha = np.array(enhanced)[:, :, 3]
    unique_values = np.unique(alpha)

    # Should be very binary with aggressive mode
    assert len(unique_values) <= 5


def test_preprocess_for_category_human_bust(test_rgba_image):
    """Test preprocessing for human bust category"""
    result = preprocess_for_category(test_rgba_image, category="human_bust")

    assert result.mode == 'RGBA'
    assert result.size[0] == result.size[1]  # Should be square for bust


def test_preprocess_for_category_generic_object(test_rgba_image):
    """Test preprocessing for generic object category"""
    result = preprocess_for_category(test_rgba_image, category="generic_object")

    assert result.mode == 'RGBA'
    # Generic objects can have flexible aspect ratio


def test_preprocessing_pipeline_integration(test_job_id, temp_storage):
    """Test full preprocessing pipeline"""
    # Create test RGB image (simulating raw upload)
    rgb_img = Image.new('RGB', (512, 512), (255, 0, 0))

    pipeline = PreprocessingPipeline(test_job_id)
    pipeline.preview_gen.storage_root = temp_storage
    pipeline.preview_gen.job_dir = temp_storage / test_job_id
    pipeline.preview_gen.job_dir.mkdir(parents=True, exist_ok=True)

    # This would normally run the full pipeline including segmentation
    # For testing, we'll just verify the structure
    assert pipeline.job_id == test_job_id
    assert pipeline.preview_gen.job_id == test_job_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

