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

Subject resizing method
-----------------------
Images must remain the same pixel size (DUSt3R/MASt3R require uniform
dimensions).  To shrink a subject:

1. Segment the subject and find its bounding box.
2. Cut the subject out, resize it to the target dimensions.
3. Place it back, centred in the image.
4. Fill the gap between the old background edge and the new subject
   using a **trapezoid warp**: draw diagonals from the four image
   corners to the four corners of the new subject bbox, creating four
   trapezoids (top, bottom, left, right).  Perspective-warp each
   original background strip into its new (larger) trapezoid so the
   background stretches smoothly inward to meet the resized subject.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from api.models import ViewLabel, ViewSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bounding-box detection
# ---------------------------------------------------------------------------


def _get_subject_mask(image: Image.Image) -> np.ndarray:
    """
    Generate a binary foreground mask for the subject.

    Returns an (H, W) uint8 array: 255 = foreground, 0 = background.
    """
    try:
        from rembg import remove

        rgba = remove(image.convert("RGB"))
        if rgba.mode == "RGBA":
            alpha = np.array(rgba.split()[-1])
        else:
            alpha = np.array(rgba.convert("L"))
        mask = (alpha > 128).astype(np.uint8) * 255
    except ImportError:
        # Fallback: simple white-background threshold
        logger.warning("rembg not available, falling back to threshold masking")
        arr = np.array(image.convert("RGB"), dtype=np.float32)
        # Pixels far from white are likely foreground
        dist = np.sqrt(((arr - 255.0) ** 2).sum(axis=2))
        mask = (dist > 30).astype(np.uint8) * 255

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
# Trapezoid background warp
# ---------------------------------------------------------------------------


def _perspective_warp_region(
    src_img: np.ndarray,
    src_quad: np.ndarray,
    dst_quad: np.ndarray,
    output_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perspective-warp a quadrilateral region from *src_img*.

    Parameters
    ----------
    src_img : (H, W, C) source image
    src_quad : (4, 2) float32 – corners in the source image
    dst_quad : (4, 2) float32 – corresponding corners in the output image
    output_shape : (H, W) of the output canvas

    Returns
    -------
    warped : (H, W, C) – the warped image (zeros outside the quad)
    mask   : (H, W) uint8 – 255 inside the warped quad, 0 outside
    """
    import cv2

    M = cv2.getPerspectiveTransform(
        src_quad.astype(np.float32),
        dst_quad.astype(np.float32),
    )
    H_out, W_out = output_shape
    warped = cv2.warpPerspective(
        src_img, M, (W_out, H_out),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Build a mask for the destination quadrilateral
    mask = np.zeros((H_out, W_out), dtype=np.uint8)
    pts = dst_quad.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillConvexPoly(mask, pts, 255)

    return warped, mask


def resize_subject_in_image(
    image: Image.Image,
    orig_bbox: tuple[int, int, int, int],
    target_w: int,
    target_h: int,
) -> Image.Image:
    """
    Shrink the subject within *image* to (target_w, target_h) and fill
    the background gap using trapezoid warping.

    The image pixel dimensions are preserved.

    Parameters
    ----------
    image : original image (RGB, square)
    orig_bbox : (x1, y1, x2, y2) tight bounding box of the subject
    target_w : desired subject width  (≤ orig bbox width)
    target_h : desired subject height (≤ orig bbox height)

    Returns
    -------
    A new PIL Image of the same size with the resized subject centred
    and the background smoothly stretched to fill gaps.
    """
    import cv2

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

    # New bbox – centred in the image
    nx1 = (W - tw) // 2
    ny1 = (H - th) // 2
    nx2 = nx1 + tw
    ny2 = ny1 + th

    src_arr = np.array(image)  # (H, W, 3) or (H, W, 4)
    channels = src_arr.shape[2] if src_arr.ndim == 3 else 1

    # --- 1. Resize the subject crop ---
    subject_crop = src_arr[oy1:oy2, ox1:ox2]
    resized_subject = cv2.resize(
        subject_crop, (tw, th), interpolation=cv2.INTER_AREA,
    )

    # --- 2. Build output canvas ---
    out = np.zeros_like(src_arr)

    # Place resized subject
    out[ny1:ny2, nx1:nx2] = resized_subject

    # --- 3. Warp the four background trapezoids ---
    # Each trapezoid maps an original background strip to its new
    # (larger) region.  Corners are listed in consistent winding order.

    # Original background quad corners (in source image)
    # New (destination) quad corners (in output image)
    # The diagonals go from each image corner to the corresponding
    # new-subject-bbox corner.

    trapezoids = [
        # TOP trapezoid
        {
            "src": np.array(
                [[0, 0], [W, 0], [ox2, oy1], [ox1, oy1]], dtype=np.float32
            ),
            "dst": np.array(
                [[0, 0], [W, 0], [nx2, ny1], [nx1, ny1]], dtype=np.float32
            ),
        },
        # BOTTOM trapezoid
        {
            "src": np.array(
                [[ox1, oy2], [ox2, oy2], [W, H], [0, H]], dtype=np.float32
            ),
            "dst": np.array(
                [[nx1, ny2], [nx2, ny2], [W, H], [0, H]], dtype=np.float32
            ),
        },
        # LEFT trapezoid
        {
            "src": np.array(
                [[0, 0], [ox1, oy1], [ox1, oy2], [0, H]], dtype=np.float32
            ),
            "dst": np.array(
                [[0, 0], [nx1, ny1], [nx1, ny2], [0, H]], dtype=np.float32
            ),
        },
        # RIGHT trapezoid
        {
            "src": np.array(
                [[ox2, oy1], [W, 0], [W, H], [ox2, oy2]], dtype=np.float32
            ),
            "dst": np.array(
                [[nx2, ny1], [W, 0], [W, H], [nx2, ny2]], dtype=np.float32
            ),
        },
    ]

    for trap in trapezoids:
        warped, mask = _perspective_warp_region(
            src_arr, trap["src"], trap["dst"], (H, W),
        )
        # Composite: only write where the output is still empty
        # (the subject region is already filled)
        fill_mask = (mask > 0) & (
            np.all(out == 0, axis=2) if out.ndim == 3 else (out == 0)
        )
        if out.ndim == 3:
            out[fill_mask] = warped[fill_mask]
        else:
            out[fill_mask] = warped[fill_mask]

    # Fill any remaining zero pixels (corner gaps, etc.) with nearest
    # non-zero content to avoid black seams.
    if out.ndim == 3:
        gray = np.any(out > 0, axis=2).astype(np.uint8)
    else:
        gray = (out > 0).astype(np.uint8)

    if not gray.all():
        # Use inpaint-like nearest fill for tiny gaps
        missing = gray == 0
        # Simple approach: flood from the subject outward
        from scipy.ndimage import distance_transform_edt

        _, indices = distance_transform_edt(missing, return_indices=True)
        if out.ndim == 3:
            for c in range(out.shape[2]):
                channel = out[:, :, c]
                out[:, :, c] = channel[indices[0], indices[1]]
        else:
            out[:] = out[indices[0], indices[1]]

    result = Image.fromarray(out)
    logger.info(
        "Resized subject from %dx%d to %dx%d (image stays %dx%d)",
        orig_w, orig_h, tw, th, W, H,
    )
    return result


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
                src = image_dir / by_label[label].image_filename
                dst = output_dir / by_label[label].image_filename
                images[label].save(dst)
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

