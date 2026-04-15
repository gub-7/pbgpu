"""
Cross-view subject normalization for the 3-view reconstruction pipeline.

Problem
-------
The backend generates front, side, and top views where the subject fills
each frame independently.  A wide dog's side profile might be short
(to fit the width), while the front view is tall (face fills the frame).
This creates inconsistent 3D proportions across views.

Solution – bounding-box consistency constraints
------------------------------------------------
Draw a tight bounding box around the subject in each view.  The three
views share object dimensions:

    Front view shows (X, Y)   Side view shows (Z, Y)   Top view shows (X, Z)

Constraints:
    front_width  == top_width      (both measure X)
    front_height == side_height    (both measure Y)
    side_width   == top_height     (both measure Z)

We enforce these by *shrinking* the subject in whichever view is too
large, never enlarging.  The target for each shared dimension is the
minimum of the two measurements.

Simplified for gray-background images
--------------------------------------
Because background removal happens before normalization, the background
is always a flat gray canvas (BACKGROUND_GRAY from preprocess.py).
This means we do NOT need complex trapezoid warping to fill gaps.
Instead, we simply:
  1. Crop the subject from the image
  2. Resize the crop to the target dimensions
  3. Paste it centred onto a fresh gray canvas
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from api.models import ViewLabel, ViewSpec
from pipelines.preprocess import BACKGROUND_GRAY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bounding-box detection (simplified for gray-background images)
# ---------------------------------------------------------------------------


def _get_subject_mask(image: Image.Image) -> np.ndarray:
    """
    Generate a binary foreground mask by thresholding against the known
    gray background colour.

    Since preprocessing has already removed the background and placed
    the subject on BACKGROUND_GRAY, we don't need rembg here – a simple
    colour-distance threshold suffices.

    Returns an (H, W) uint8 array: 255 = foreground, 0 = background.
    """
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    bg = np.array(BACKGROUND_GRAY, dtype=np.float32)

    # Per-pixel distance from the known background colour
    dist = np.sqrt(((arr - bg) ** 2).sum(axis=2))

    # Anything more than 15 units from the background is foreground.
    # This threshold is intentionally low because the background is a
    # known exact colour, not a noisy photograph.
    mask = (dist > 15).astype(np.uint8) * 255
    return mask


def _mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """
    Find the tight bounding box of non-zero pixels in a mask.

    Returns (x1, y1, x2, y2) in pixel coordinates.
    Falls back to the full image dimensions if the mask is empty.
    """
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)

    if not rows.any() or not cols.any():
        h, w = mask.shape[:2]
        logger.warning("Empty mask – using full image as bounding box")
        return 0, 0, w, h

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    # x2, y2 are inclusive indices; make them exclusive
    return int(x1), int(y1), int(x2) + 1, int(y2) + 1


def detect_subject_bbox(
    image: Image.Image,
) -> tuple[int, int, int, int]:
    """
    Detect the subject bounding box in an image.

    Returns (x1, y1, x2, y2) – the tight axis-aligned bounding box
    around the foreground subject.
    """
    mask = _get_subject_mask(image)
    return _mask_to_bbox(mask)


# ---------------------------------------------------------------------------
# Constraint solver
# ---------------------------------------------------------------------------


def compute_target_dimensions(
    front_bbox: tuple[int, int, int, int],
    side_bbox: tuple[int, int, int, int],
    top_bbox: tuple[int, int, int, int],
) -> dict[str, tuple[int, int]]:
    """
    Compute target subject dimensions for each view to satisfy the
    cross-view consistency constraints.

    Constraints (shrink-only – take the minimum):
        X = min(front_width,  top_width)
        Y = min(front_height, side_height)
        Z = min(side_width,   top_height)

    Returns a dict mapping view label to (target_width, target_height):
        {"front": (X, Y), "side": (Z, Y), "top": (X, Z)}
    """
    front_w = front_bbox[2] - front_bbox[0]
    front_h = front_bbox[3] - front_bbox[1]
    side_w = side_bbox[2] - side_bbox[0]
    side_h = side_bbox[3] - side_bbox[1]
    top_w = top_bbox[2] - top_bbox[0]
    top_h = top_bbox[3] - top_bbox[1]

    target_x = min(front_w, top_w)
    target_y = min(front_h, side_h)
    target_z = min(side_w, top_h)

    logger.info(
        "Bbox dimensions – front: %dx%d, side: %dx%d, top: %dx%d",
        front_w, front_h, side_w, side_h, top_w, top_h,
    )
    logger.info(
        "Target dimensions – X=%d, Y=%d, Z=%d",
        target_x, target_y, target_z,
    )

    return {
        ViewLabel.FRONT: (target_x, target_y),
        ViewLabel.SIDE: (target_z, target_y),
        ViewLabel.TOP: (target_x, target_z),
    }


# ---------------------------------------------------------------------------
# Subject resize (simplified for gray background)
# ---------------------------------------------------------------------------


def resize_subject_in_image(
    image: Image.Image,
    orig_bbox: tuple[int, int, int, int],
    target_w: int,
    target_h: int,
) -> Image.Image:
    """
    Shrink the subject within *image* to (target_w, target_h) and
    re-centre on a fresh gray canvas.

    Because the background is a flat gray, we don't need any complex
    warping – just crop the subject, resize it, and paste onto a new
    gray canvas of the same image dimensions.

    Parameters
    ----------
    image : original image (RGB, square, gray background)
    orig_bbox : (x1, y1, x2, y2) tight bounding box of the subject
    target_w : desired subject width  (≤ orig bbox width)
    target_h : desired subject height (≤ orig bbox height)

    Returns
    -------
    A new PIL Image of the same size with the resized subject centred
    on a gray canvas.
    """
    W, H = image.size
    ox1, oy1, ox2, oy2 = orig_bbox
    orig_w = ox2 - ox1
    orig_h = oy2 - oy1

    # Nothing to do if already at or below target
    if orig_w <= target_w and orig_h <= target_h:
        logger.debug("Subject already within target – no resize needed")
        return image

    # Clamp target to not exceed original (shrink only)
    tw = min(target_w, orig_w)
    th = min(target_h, orig_h)

    # Crop the subject from the original image
    subject_crop = image.crop((ox1, oy1, ox2, oy2))

    # Resize the crop to target dimensions
    resized_subject = subject_crop.resize((tw, th), Image.LANCZOS)

    # Create a fresh gray canvas and paste the resized subject centred
    canvas = Image.new("RGB", (W, H), BACKGROUND_GRAY)
    paste_x = (W - tw) // 2
    paste_y = (H - th) // 2
    canvas.paste(resized_subject, (paste_x, paste_y))

    logger.info(
        "Resized subject from %dx%d to %dx%d (image stays %dx%d)",
        orig_w, orig_h, tw, th, W, H,
    )
    return canvas


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def normalize_views(
    view_specs: list[ViewSpec],
    image_dir: Path,
    output_dir: Optional[Path] = None,
) -> list[ViewSpec]:
    """
    Normalize subject sizes across the three canonical views so that
    bounding-box dimensions are consistent.

    Reads preprocessed images from *image_dir*, detects subject bounding
    boxes, computes the shrink-only target dimensions, resizes subjects
    as needed, and writes the results.

    If *output_dir* is None the images are overwritten in place.

    Parameters
    ----------
    view_specs : the three ViewSpecs (front, side, top)
    image_dir  : directory containing the preprocessed images
    output_dir : where to write normalised images (defaults to image_dir)

    Returns
    -------
    The (unchanged) list of ViewSpecs – filenames stay the same since
    the images are written to the same names.
    """
    if output_dir is None:
        output_dir = image_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build label → view mapping
    by_label: dict[ViewLabel, ViewSpec] = {}
    for vs in view_specs:
        by_label[vs.label] = vs

    required = {ViewLabel.FRONT, ViewLabel.SIDE, ViewLabel.TOP}
    if not required.issubset(by_label):
        logger.warning(
            "Cannot normalize views – missing labels: %s",
            required - set(by_label),
        )
        return view_specs

    # Load images and detect bounding boxes
    images: dict[ViewLabel, Image.Image] = {}
    bboxes: dict[ViewLabel, tuple[int, int, int, int]] = {}

    for label in (ViewLabel.FRONT, ViewLabel.SIDE, ViewLabel.TOP):
        path = image_dir / by_label[label].image_filename
        img = Image.open(path).convert("RGB")
        images[label] = img
        bbox = detect_subject_bbox(img)
        bboxes[label] = bbox
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        logger.info(
            "View %s bbox: (%d,%d,%d,%d) → %dx%d",
            label.value, *bbox, bw, bh,
        )

    # Compute targets
    targets = compute_target_dimensions(
        bboxes[ViewLabel.FRONT],
        bboxes[ViewLabel.SIDE],
        bboxes[ViewLabel.TOP],
    )

    # Resize each view as needed
    any_changed = False
    for label in (ViewLabel.FRONT, ViewLabel.SIDE, ViewLabel.TOP):
        tw, th = targets[label]
        bbox = bboxes[label]
        orig_w = bbox[2] - bbox[0]
        orig_h = bbox[3] - bbox[1]

        needs_resize = orig_w > tw or orig_h > th
        if not needs_resize:
            logger.info("View %s: already consistent – skipping", label.value)
            # Still copy to output_dir if different from image_dir
            if output_dir != image_dir:
                out_path = output_dir / by_label[label].image_filename
                images[label].save(out_path)
            continue

        logger.info(
            "View %s: resizing subject from %dx%d → %dx%d",
            label.value, orig_w, orig_h, tw, th,
        )
        result = resize_subject_in_image(images[label], bbox, tw, th)
        out_path = output_dir / by_label[label].image_filename
        result.save(out_path)
        any_changed = True

    if not any_changed:
        logger.info("All views already consistent – no normalization needed")
    else:
        logger.info("View normalization complete")

    return view_specs

