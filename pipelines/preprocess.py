"""
Image preprocessing for the multi-view reconstruction pipeline.

Pipeline order (revised):
  1. Validate image (format, dimensions, corruption)
  2. EXIF orientation correction + convert to RGB
  3. **Background removal** (rembg) → place subject on flat gray canvas
  4. Resize to the pipeline's expected square dimension
  5. Save to the job workspace

Background removal happens *first* because every downstream stage
(view normalisation, fiducial markers, coarse reconstruction, Trellis)
operates on background-free images.  The gray canvas
(BACKGROUND_GRAY) gives DUSt3R / MASt3R a uniform, low-texture
background that won't generate spurious correspondences.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

from api.models import ViewLabel, ViewSpec
from pipelines.config import DEFAULT_IMAGE_SIZE, SUPPORTED_IMAGE_FORMATS

logger = logging.getLogger(__name__)

# Flat gray used as the background after rembg removal.
# Mid-gray is chosen so it doesn't bias colour statistics and is
# visually distinct from both the subject and the fiducial markers.
BACKGROUND_GRAY = (180, 180, 180)


class PreprocessingError(Exception):
    """Raised when an image fails validation or preprocessing."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_image_file(path: Path) -> None:
    """
    Validate that a file is a supported, non-corrupt image.

    Raises PreprocessingError on failure.
    """
    if not path.exists():
        raise PreprocessingError(f"Image file does not exist: {path}")

    if path.suffix.lower() not in SUPPORTED_IMAGE_FORMATS:
        raise PreprocessingError(
            f"Unsupported image format '{path.suffix}' for {path.name}. "
            f"Supported: {SUPPORTED_IMAGE_FORMATS}"
        )

    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as exc:
        raise PreprocessingError(
            f"Image file is corrupt or unreadable: {path.name}"
        ) from exc


def validate_image_dimensions(
    path: Path,
    min_size: int = 512,
) -> tuple[int, int]:
    """
    Check that the image meets minimum size requirements.

    Returns (width, height).
    """
    with Image.open(path) as img:
        w, h = img.size

    if w < min_size or h < min_size:
        raise PreprocessingError(
            f"Image {path.name} is too small ({w}×{h}). "
            f"Minimum dimension: {min_size}px"
        )

    return w, h


# ---------------------------------------------------------------------------
# Background removal
# ---------------------------------------------------------------------------


def remove_background(img: Image.Image) -> Image.Image:
    """
    Remove background using rembg and place subject on flat gray canvas.

    Returns an RGB image with the subject on BACKGROUND_GRAY.
    """
    try:
        from rembg import remove
    except ImportError:
        logger.warning(
            "rembg not installed – falling back to threshold-based masking. "
            "Install with: pip install rembg[gpu]"
        )
        return _remove_background_threshold(img)

    # rembg returns RGBA with alpha as the foreground mask
    img_rgb = img.convert("RGB")
    result_rgba = remove(img_rgb)

    if result_rgba.mode != "RGBA":
        result_rgba = result_rgba.convert("RGBA")

    # Composite onto gray canvas
    canvas = Image.new("RGB", img_rgb.size, BACKGROUND_GRAY)
    canvas.paste(result_rgba, mask=result_rgba.split()[-1])

    return canvas


def _remove_background_threshold(img: Image.Image) -> Image.Image:
    """
    Fallback background removal using simple white-threshold masking.

    Assumes a predominantly white background.
    """
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    # Pixels far from white are likely foreground
    dist = np.sqrt(((arr - 255.0) ** 2).sum(axis=2))
    mask = (dist > 30).astype(np.uint8) * 255

    mask_img = Image.fromarray(mask, mode="L")
    rgba = img.convert("RGB").copy()
    rgba.putalpha(mask_img)

    canvas = Image.new("RGB", img.size, BACKGROUND_GRAY)
    canvas.paste(rgba, mask=mask_img)
    return canvas


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def resize_to_square(
    img: Image.Image,
    target_size: int,
    resample: int = Image.LANCZOS,
) -> Image.Image:
    """
    Resize an image to a square canvas, padding with gray if needed.

    If the image is already square, it is simply resized.
    Otherwise the image is centred on a gray square canvas and then
    resized to the target dimension.
    """
    w, h = img.size

    if w == h:
        return img.resize((target_size, target_size), resample)

    # Pad to square first
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), BACKGROUND_GRAY)
    offset_x = (side - w) // 2
    offset_y = (side - h) // 2
    canvas.paste(img, (offset_x, offset_y))

    return canvas.resize((target_size, target_size), resample)


def normalise_image(img: Image.Image) -> Image.Image:
    """
    Apply standard normalisation:
      - EXIF orientation correction
      - Convert to RGB (drop alpha)
    """
    img = ImageOps.exif_transpose(img)

    if img.mode != "RGB":
        img = img.convert("RGB")

    return img


# ---------------------------------------------------------------------------
# Per-view preprocessing
# ---------------------------------------------------------------------------


def preprocess_single_image(
    src_path: Path,
    dst_path: Path,
    target_size: int = DEFAULT_IMAGE_SIZE,
) -> Path:
    """
    Full preprocessing pipeline for one image.

    1. Validate
    2. Load and normalise (EXIF, RGB)
    3. Remove background → flat gray canvas
    4. Resize to square
    5. Save to dst_path as PNG

    Returns the output path.
    """
    validate_image_file(src_path)
    validate_image_dimensions(src_path)

    with Image.open(src_path) as img:
        img = normalise_image(img)
        img = remove_background(img)
        img = resize_to_square(img, target_size)

    # Always save as PNG for lossless downstream processing
    dst_path = dst_path.with_suffix(".png")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst_path, format="PNG")

    logger.info(
        "Preprocessed %s → %s (%dx%d, bg removed)",
        src_path.name, dst_path, target_size, target_size,
    )
    return dst_path


# ---------------------------------------------------------------------------
# Batch preprocessing for all views
# ---------------------------------------------------------------------------


def preprocess_views(
    view_specs: list[ViewSpec],
    input_dir: Path,
    output_dir: Path,
    target_size: int = DEFAULT_IMAGE_SIZE,
) -> list[ViewSpec]:
    """
    Preprocess all view images and return updated ViewSpecs with new filenames.

    Images are read from input_dir/{view.image_filename} and written to
    output_dir/{label}.png.

    Background is removed and images are placed on a flat gray canvas
    so that every downstream stage works on clean, background-free images.

    Parameters
    ----------
    view_specs : list of ViewSpec with original filenames
    input_dir : directory containing the raw uploaded images
    output_dir : directory to write preprocessed images
    target_size : target square dimension

    Returns
    -------
    Updated list of ViewSpec with preprocessed filenames.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    updated_specs = []

    for vs in view_specs:
        src = input_dir / vs.image_filename
        dst_filename = f"{vs.label.value}.png"
        dst = output_dir / dst_filename

        preprocess_single_image(src, dst, target_size)

        updated = vs.model_copy(update={"image_filename": dst_filename})
        updated_specs.append(updated)

    logger.info(
        "Preprocessed %d views → %s (backgrounds removed)",
        len(updated_specs),
        output_dir,
    )
    return updated_specs


# ---------------------------------------------------------------------------
# Numpy helpers (used by downstream stages)
# ---------------------------------------------------------------------------


def image_to_numpy(path: Path) -> np.ndarray:
    """Load a preprocessed image as a float32 numpy array in [0, 1] range, HWC."""
    with Image.open(path) as img:
        arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def images_to_batch(paths: list[Path]) -> np.ndarray:
    """Load multiple images into a batch array (N, H, W, 3) float32 in [0, 1]."""
    arrays = [image_to_numpy(p) for p in paths]
    return np.stack(arrays, axis=0)

