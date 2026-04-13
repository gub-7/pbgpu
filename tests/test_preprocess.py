"""Tests for pipelines/preprocess.py – image preprocessing."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from api.models import SphericalPose, ViewLabel, ViewSpec
from pipelines.preprocess import (
    PreprocessingError,
    image_to_numpy,
    normalise_image,
    preprocess_single_image,
    preprocess_views,
    resize_to_square,
    validate_image_dimensions,
    validate_image_file,
)


def _create_test_image(path: Path, width: int = 1024, height: int = 1024, mode: str = "RGB"):
    """Create a test image at the given path."""
    img = Image.new(mode, (width, height), color=(128, 64, 32))
    img.save(path)
    return path


class TestValidateImageFile:
    def test_valid_png(self, tmp_path):
        path = _create_test_image(tmp_path / "test.png")
        validate_image_file(path)  # should not raise

    def test_missing_file(self, tmp_path):
        with pytest.raises(PreprocessingError, match="does not exist"):
            validate_image_file(tmp_path / "nonexistent.png")

    def test_unsupported_format(self, tmp_path):
        path = tmp_path / "test.xyz"
        path.write_text("not an image")
        with pytest.raises(PreprocessingError, match="Unsupported"):
            validate_image_file(path)

    def test_corrupt_file(self, tmp_path):
        path = tmp_path / "corrupt.png"
        path.write_bytes(b"not a real png")
        with pytest.raises(PreprocessingError, match="corrupt"):
            validate_image_file(path)


class TestValidateImageDimensions:
    def test_valid_dimensions(self, tmp_path):
        path = _create_test_image(tmp_path / "test.png", 1024, 1024)
        w, h = validate_image_dimensions(path)
        assert w == 1024
        assert h == 1024

    def test_too_small(self, tmp_path):
        path = _create_test_image(tmp_path / "small.png", 256, 256)
        with pytest.raises(PreprocessingError, match="too small"):
            validate_image_dimensions(path, min_size=512)


class TestResizeToSquare:
    def test_already_square(self):
        img = Image.new("RGB", (512, 512))
        result = resize_to_square(img, 256)
        assert result.size == (256, 256)

    def test_rectangular(self):
        img = Image.new("RGB", (800, 600))
        result = resize_to_square(img, 512)
        assert result.size == (512, 512)


class TestNormaliseImage:
    def test_rgb_passthrough(self):
        img = Image.new("RGB", (100, 100))
        result = normalise_image(img)
        assert result.mode == "RGB"

    def test_rgba_to_rgb(self):
        img = Image.new("RGBA", (100, 100))
        result = normalise_image(img)
        assert result.mode == "RGB"

    def test_grayscale_to_rgb(self):
        img = Image.new("L", (100, 100))
        result = normalise_image(img)
        assert result.mode == "RGB"


class TestPreprocessSingleImage:
    def test_full_pipeline(self, tmp_path):
        src = _create_test_image(tmp_path / "input.png", 800, 600)
        dst = tmp_path / "output.png"
        result = preprocess_single_image(src, dst, target_size=512)
        assert result.exists()
        with Image.open(result) as img:
            assert img.size == (512, 512)
            assert img.mode == "RGB"


class TestPreprocessViews:
    def test_processes_all_views(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # Create test images
        for name in ["front.png", "side.png", "top.png"]:
            _create_test_image(input_dir / name, 800, 800)

        specs = [
            ViewSpec(label=ViewLabel.FRONT, image_filename="front.png", pose=SphericalPose()),
            ViewSpec(label=ViewLabel.SIDE, image_filename="side.png", pose=SphericalPose()),
            ViewSpec(label=ViewLabel.TOP, image_filename="top.png", pose=SphericalPose()),
        ]

        updated = preprocess_views(specs, input_dir, output_dir, target_size=512)

        assert len(updated) == 3
        for vs in updated:
            assert (output_dir / vs.image_filename).exists()


class TestImageToNumpy:
    def test_shape_and_range(self, tmp_path):
        path = _create_test_image(tmp_path / "test.png", 64, 64)
        arr = image_to_numpy(path)
        assert arr.shape == (64, 64, 3)
        assert arr.dtype == np.float32
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

