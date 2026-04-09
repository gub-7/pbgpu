"""
Image preprocessing module for 3D reconstruction.
Handles cropping, centering, scaling, and category-specific adjustments.
Optimized for TripoSR with automatic FFR-based framing, region-aware processing,
edge-preserving denoising, local contrast enhancement, and proper background handling.

TripoSR optimization notes (based on reconstruction research):
- TripoSR is trained on RGB images — NEVER feed RGBA (causes domain shift)
- Automatic FFR (Frame Fill Ratio) targets ~0.60 (range 0.45–0.75)
- Square crop centered on mask centroid for consistent geometry inference
- Flat 50% gray background (128,128,128) matches the official TripoSR inference script
- Slight 1–2px alpha feather before compositing prevents floating artifact planes
- Minimum 512px output to preserve geometry detail
- Hard alpha edges in segmentation, slight feather only at composite step
"""
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
from PIL import Image
import cv2


class CategoryType(Enum):
    """Category types for preprocessing."""
    HUMAN_FULL_BODY = "human_full_body"
    ANIMAL_FULL_BODY = "animal_full_body"
    HUMAN_BUST = "human_bust"
    ANIMAL_BUST = "animal_bust"
    GENERIC_OBJECT = "generic_object"


class RegionType(Enum):
    """Region types for region-aware processing."""
    BACKGROUND = "background"
    BOUNDARY_RING = "boundary_ring"
    FACE = "face"
    HAIR = "hair"
    SKIN = "skin"
    CLOTHING = "clothing"
    HANDS = "hands"
    FUR = "fur"
    EARS = "ears"
    WHISKERS = "whiskers"
    THIN_STRUCTURES = "thin_structures"
    SPECULAR = "specular"
    TEXT_LOGO = "text_logo"


# TripoSR background color: white per reconstruction research.
# White / very-light-gray produces the most stable geometry because it
# matches the clean-background training distribution.  50 % gray (128)
# is acceptable but white is preferred.  Black is AVOIDED — it causes
# shadow hallucination, depth confusion, and shrink-wrapped geometry.
TRIPOSR_BACKGROUND_COLOR = (128, 128, 128)  # RGB 50% gray (official TripoSR parity)

# Minimum output resolution — below this TripoSR geometry degrades noticeably
MIN_OUTPUT_RESOLUTION = 512

# --- Auto FFR constants ---
# Target Frame Fill Ratio: bbox_area / crop_area ≈ 0.60
FFR_TARGET_DEFAULT = 0.60
FFR_MIN_DEFAULT = 0.45
FFR_MAX_DEFAULT = 0.75
# Minimum Bounding-Box Occupancy — below this the mask is likely fragmented
BBO_MIN_DEFAULT = 0.70

# --- Composite feather ---
# Slight feather (1–2 px) applied to alpha BEFORE compositing onto flat
# background.  This is NOT segmentation feathering (which at radius ≥ 2
# causes TripoSR sheet/blob meshes).  It softens the composite boundary
# just enough to prevent floating artifact planes from perfectly sharp
# mask cutouts.
COMPOSITE_FEATHER_RADIUS_DEFAULT = 1


def get_foreground_bbox(alpha_channel: np.ndarray, padding: int = 10) -> Tuple[int, int, int, int]:
    """
    Get bounding box of foreground object from alpha channel.

    Args:
        alpha_channel: Alpha channel as numpy array
        padding: Padding around bbox in pixels

    Returns:
        Tuple of (x, y, width, height)
    """
    # Find non-zero pixels
    coords = cv2.findNonZero(alpha_channel)
    if coords is None:
        # No foreground found, return full image
        h, w = alpha_channel.shape
        return 0, 0, w, h

    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(coords)

    # Add padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(alpha_channel.shape[1] - x, w + 2 * padding)
    h = min(alpha_channel.shape[0] - y, h + 2 * padding)

    return x, y, w, h


def compute_mask_metrics(alpha_channel: np.ndarray, fg_threshold: int = 128, *, morph_cleanup: bool = True) -> Dict:
    """
    Compute mask metrics for automatic FFR-based framing.

    Metrics:
    - FAR (Foreground Area Ratio): foreground_pixels / image_area
    - FFR (Frame Fill Ratio): bbox_area / image_area
    - BBO (Bounding-Box Occupancy): foreground_pixels / bbox_area

    Args:
        alpha_channel: Alpha channel as numpy array (H, W), values 0-255
        fg_threshold: Alpha value above which a pixel is considered foreground
        morph_cleanup: Apply morphological operations to stabilize bbox/centroid

    Returns:
        Dict with far, ffr, bbo, centroid, bbox, and raw counts
    """
    h, w = alpha_channel.shape
    image_area = h * w

    # Foreground mask
    fg_mask = (alpha_channel >= fg_threshold).astype(np.uint8)
    # Optional morphology cleanup to stabilize bbox/centroid (remove specks, fill pinholes)
    if morph_cleanup:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, k, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, k, iterations=1)
    foreground_pixels = int(fg_mask.sum())

    # Handle empty mask
    if foreground_pixels == 0:
        return {
            "far": 0.0,
            "ffr": 0.0,
            "bbo": 0.0,
            "centroid": (w // 2, h // 2),
            "bbox": (0, 0, w, h),
            "foreground_pixels": 0,
            "image_area": image_area,
            "bbox_area": image_area,
        }

    # Bounding box (no padding — raw tight bbox for metric computation)
    coords = cv2.findNonZero(fg_mask)
    bx, by, bw, bh = cv2.boundingRect(coords)
    bbox_area = bw * bh

    # Centroid via image moments
    moments = cv2.moments(fg_mask)
    if moments["m00"] > 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx = bx + bw // 2
        cy = by + bh // 2

    # Compute ratios
    far = foreground_pixels / image_area if image_area > 0 else 0.0
    ffr = bbox_area / image_area if image_area > 0 else 0.0
    bbo = foreground_pixels / bbox_area if bbox_area > 0 else 0.0

    return {
        "far": float(far),
        "ffr": float(ffr),
        "bbo": float(bbo),
        "centroid": (cx, cy),
        "bbox": (bx, by, bw, bh),
        "foreground_pixels": foreground_pixels,
        "image_area": image_area,
        "bbox_area": bbox_area,
    }


def calculate_target_size(
    bbox_width: int,
    bbox_height: int,
    foreground_ratio: float = 0.85,
    max_size: int = 2048,
    min_size: int = MIN_OUTPUT_RESOLUTION
) -> Tuple[int, int]:
    """
    Calculate target canvas size to achieve desired foreground ratio.
    LEGACY — prefer auto_frame_for_triposr() for new code.

    Args:
        bbox_width: Width of foreground bounding box
        bbox_height: Height of foreground bounding box
        foreground_ratio: Target ratio of foreground to canvas (0.7-0.95)
        max_size: Maximum dimension size
        min_size: Minimum dimension size (ensures enough pixels for geometry)

    Returns:
        Tuple of (canvas_width, canvas_height)
    """
    # Calculate required canvas size
    canvas_width = int(bbox_width / foreground_ratio)
    canvas_height = int(bbox_height / foreground_ratio)

    # Maintain aspect ratio and respect max size
    max_dim = max(canvas_width, canvas_height)
    if max_dim > max_size:
        scale = max_size / max_dim
        canvas_width = int(canvas_width * scale)
        canvas_height = int(canvas_height * scale)

    # Enforce minimum resolution — below this TripoSR geometry degrades
    min_dim = min(canvas_width, canvas_height)
    if min_dim < min_size:
        scale_up = min_size / min_dim
        canvas_width = int(canvas_width * scale_up)
        canvas_height = int(canvas_height * scale_up)

    # Make dimensions even for better compatibility
    canvas_width = (canvas_width // 2) * 2
    canvas_height = (canvas_height // 2) * 2

    return canvas_width, canvas_height


def rgba_to_rgb_with_background(
    rgba_image: np.ndarray,
    background_color: Tuple[int, int, int] = TRIPOSR_BACKGROUND_COLOR
) -> np.ndarray:
    """
    Convert RGBA to RGB by compositing onto a flat background color.

    This is the ONLY correct way to prepare images for TripoSR:
    TripoSR is trained on RGB — RGBA causes domain shift.
    The flat background eliminates distracting depth cues.

    Args:
        rgba_image: Input RGBA image as numpy array (H, W, 4)
        background_color: Background RGB color (default: white for TripoSR)

    Returns:
        RGB numpy array (H, W, 3) with background filled
    """
    rgb = rgba_image[:, :, :3].astype(np.float32)
    alpha = rgba_image[:, :, 3:4].astype(np.float32) / 255.0

    # Create background
    bg = np.full_like(rgb, background_color, dtype=np.float32)

    # Alpha composite: foreground * alpha + background * (1 - alpha)
    result = rgb * alpha + bg * (1.0 - alpha)

    return result.astype(np.uint8)


def apply_composite_feather(
    alpha_channel: np.ndarray,
    radius: int = COMPOSITE_FEATHER_RADIUS_DEFAULT
) -> np.ndarray:
    """
    Apply a slight Gaussian feather to the alpha channel before compositing.

    This is NOT segmentation feathering.  Segmentation should produce hard
    alpha edges (feathering at radius ≥ 2 causes TripoSR sheet/blob meshes).

    This is a subtle 1–2 px softening applied only at the composite step.
    It prevents "floating artifact planes" that TripoSR sometimes generates
    from perfectly sharp mask cutouts, by giving the silhouette boundary a
    realistic sub-pixel gradient.

    For a binary alpha (0 or 255) a 3×3 Gaussian kernel only modifies the
    1-pixel-wide boundary ring — interior and exterior stay untouched.

    Args:
        alpha_channel: Alpha channel (H, W), uint8 0-255
        radius: Feather radius in pixels (1–2 recommended, 0 = disabled)

    Returns:
        Feathered alpha channel, same shape and dtype
    """
    if radius <= 0:
        return alpha_channel

    kernel_size = radius * 2 + 1  # 1 px → 3×3, 2 px → 5×5
    feathered = cv2.GaussianBlur(
        alpha_channel,
        (kernel_size, kernel_size),
        sigmaX=0,  # auto-compute from kernel size
    )
    return feathered


def apply_bilateral_filter(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75
) -> np.ndarray:
    """
    Apply edge-preserving bilateral filter for denoising.
    Critical for TripoSR: removes noise without blurring edges.

    Args:
        image: Input image (RGB or grayscale)
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space

    Returns:
        Filtered image
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def apply_clahe_luminance(
    rgb_image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on luminance only.
    Enhances local contrast without affecting color. Critical for TripoSR surface detail.

    Args:
        rgb_image: Input RGB image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization

    Returns:
        RGB image with enhanced luminance
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE to L channel only
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel_clahe = clahe.apply(l_channel)

    # Merge back
    lab_clahe = cv2.merge([l_channel_clahe, a_channel, b_channel])
    rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    return rgb_clahe


def compress_highlights(
    rgb_image: np.ndarray,
    threshold: int = 200,
    compression_factor: float = 0.7
) -> np.ndarray:
    """
    Compress specular highlights to prevent them from being interpreted as geometry edges.
    Critical for objects with reflective/metallic surfaces.

    Args:
        rgb_image: Input RGB image
        threshold: Brightness threshold for highlights
        compression_factor: How much to compress (0.0-1.0, lower = more compression)

    Returns:
        RGB image with compressed highlights
    """
    # Convert to HSV to work with brightness
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV).astype(np.float32)
    v_channel = hsv[:, :, 2]

    # Find highlight regions
    highlight_mask = v_channel > threshold

    # Compress highlights
    v_channel[highlight_mask] = threshold + (v_channel[highlight_mask] - threshold) * compression_factor

    hsv[:, :, 2] = v_channel
    rgb_compressed = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return rgb_compressed


def get_boundary_ring_mask(
    alpha_channel: np.ndarray,
    ring_width: int = 3
) -> np.ndarray:
    """
    Extract boundary ring mask for selective edge sharpening.
    Only sharpen the silhouette edge, not the entire image.

    Args:
        alpha_channel: Alpha channel as numpy array
        ring_width: Width of boundary ring in pixels

    Returns:
        Binary mask of boundary ring
    """
    # Create binary mask
    binary_mask = (alpha_channel > 127).astype(np.uint8) * 255

    # Erode to get inner boundary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_width * 2 + 1, ring_width * 2 + 1))
    eroded = cv2.erode(binary_mask, kernel, iterations=1)

    # Boundary ring is original minus eroded
    boundary_ring = cv2.subtract(binary_mask, eroded)

    return boundary_ring


def apply_boundary_sharpening(
    rgb_image: np.ndarray,
    alpha_channel: np.ndarray,
    amount: float = 0.5,
    radius: int = 1,
    ring_width: int = 3
) -> np.ndarray:
    """
    Apply selective sharpening only to boundary ring.
    Avoids global sharpening halos that TripoSR interprets as fake geometry.

    Args:
        rgb_image: Input RGB image
        alpha_channel: Alpha channel for boundary detection
        amount: Sharpening amount (0.0-2.0)
        radius: Unsharp mask radius
        ring_width: Width of boundary ring

    Returns:
        RGB image with boundary sharpening
    """
    # Get boundary ring mask
    boundary_mask = get_boundary_ring_mask(alpha_channel, ring_width)
    boundary_mask = boundary_mask.astype(np.float32) / 255.0

    # Create unsharp mask
    if radius > 0:
        blurred = cv2.GaussianBlur(rgb_image, (0, 0), radius)
        sharpened = cv2.addWeighted(rgb_image, 1.0 + amount, blurred, -amount, 0)
    else:
        sharpened = rgb_image

    # Apply only to boundary ring
    result = rgb_image.copy().astype(np.float32)
    sharpened = sharpened.astype(np.float32)

    for c in range(3):
        result[:, :, c] = result[:, :, c] * (1.0 - boundary_mask) + sharpened[:, :, c] * boundary_mask

    return result.astype(np.uint8)


# ---------------------------------------------------------------------------
# Auto FFR framing (replaces static foreground_ratio for TripoSR)
# ---------------------------------------------------------------------------

def auto_frame_for_triposr(
    rgba_image: np.ndarray,
    ffr_target: float = FFR_TARGET_DEFAULT,
    ffr_min: float = FFR_MIN_DEFAULT,
    ffr_max: float = FFR_MAX_DEFAULT,
    bbo_min: float = BBO_MIN_DEFAULT,
    output_size: int = MIN_OUTPUT_RESOLUTION,
    max_iterations: int = 5,
) -> Tuple[np.ndarray, Dict]:
    """
    Automatically frame subject for TripoSR using FFR-based square cropping.

    Computes optimal square crop so that:
        bbox_area / crop_area  ≈  ffr_target  (~0.60)

    The crop is centered on the foreground mask centroid, clamped to image
    bounds, and padded with transparent pixels where necessary.  The result
    is resized to *output_size × output_size* via Lanczos.

    Returns RGBA so the caller can apply composite feathering and then
    convert to RGB with the correct background color.

    Adjustment loop:
        • FFR < ffr_min (0.45) → zoom in  (smaller crop side)
        • FFR > ffr_max (0.75) → zoom out (larger crop side)
        • BBO < bbo_min (0.70) → log warning (mask may be fragmented)

    Args:
        rgba_image: Input RGBA image as numpy array (H, W, 4)
        ffr_target: Target Frame Fill Ratio (~0.60)
        ffr_min: Minimum acceptable FFR before zoom-in
        ffr_max: Maximum acceptable FFR before zoom-out
        bbo_min: Minimum BBO; below this a warning is emitted
        output_size: Square output side in pixels (default 512)
        max_iterations: Maximum FFR adjustment iterations

    Returns:
        Tuple of (square RGBA array at output_size, metadata dict)
    """
    alpha = rgba_image[:, :, 3]
    h, w = alpha.shape

    # --- Step 1: Compute mask metrics on the source image ----------------
    metrics = compute_mask_metrics(alpha)

    if metrics["foreground_pixels"] == 0:
        # No foreground — return a transparent square canvas
        canvas = np.zeros((output_size, output_size, 4), dtype=np.uint8)
        return canvas, {
            "auto_ffr_status": "no_foreground",
            "input_size": (h, w),
            "output_size": (output_size, output_size),
        }

    bx, by, bw, bh = metrics["bbox"]
    cx, cy = metrics["centroid"]
    bbox_area = metrics["bbox_area"]

    # BBO warning
    bbo_warning = metrics["bbo"] < bbo_min
    if bbo_warning:
        print(
            f"[auto_frame] WARNING: BBO={metrics['bbo']:.2f} < {bbo_min} — "
            f"mask may be fragmented or have large holes"
        )

    # --- Step 2: Initial square crop side from FFR target ----------------
    # FFR = bbox_area / crop_area  →  crop_area = bbox_area / FFR
    # S = sqrt(crop_area)
    S = int(np.ceil(np.sqrt(bbox_area / ffr_target)))

    # S must be at least as large as the bbox longest side so nothing is clipped
    S = max(S, bw, bh)

    # --- Step 3–6: Iterative crop + FFR verification ---------------------
    actual_ffr = 0.0
    iterations_used = 0

    for iteration in range(max_iterations + 1):  # +1 so first pass counts as iteration 0
        # Center crop on mask centroid
        crop_x = cx - S // 2
        crop_y = cy - S // 2


        # Ensure the crop contains the full foreground bbox (centroid can be off-center
        # for asymmetric shapes, which can otherwise clip limbs/tails).
        bbox_x1, bbox_y1 = bx, by
        bbox_x2, bbox_y2 = bx + bw, by + bh

        if crop_x > bbox_x1:
            crop_x = bbox_x1
        if crop_x + S < bbox_x2:
            crop_x = bbox_x2 - S

        if crop_y > bbox_y1:
            crop_y = bbox_y1
        if crop_y + S < bbox_y2:
            crop_y = bbox_y2 - S
        # Source region clamped to image bounds
        src_x1 = max(0, crop_x)
        src_y1 = max(0, crop_y)
        src_x2 = min(w, crop_x + S)
        src_y2 = min(h, crop_y + S)

        # Destination region inside the S×S canvas
        dst_x1 = src_x1 - crop_x
        dst_y1 = src_y1 - crop_y
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        # Build canvas
        canvas = np.zeros((S, S, 4), dtype=np.uint8)
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = rgba_image[src_y1:src_y2, src_x1:src_x2]

        # Measure FFR on the canvas
        canvas_metrics = compute_mask_metrics(canvas[:, :, 3])
        actual_ffr = canvas_metrics["ffr"]
        iterations_used = iteration

        # Check convergence
        if ffr_min <= actual_ffr <= ffr_max:
            break

        # Adjust S for next iteration
        if actual_ffr < ffr_min:
            # Too much empty space → shrink crop (zoom in)
            S_new = int(S * np.sqrt(actual_ffr / ffr_target))
            S_new = max(S_new, bw, bh)  # never clip the bbox
            if S_new >= S:
                break  # can't shrink further without clipping
            S = S_new
        elif actual_ffr > ffr_max:
            # Too tight → enlarge crop (zoom out)
            S = int(S * np.sqrt(actual_ffr / ffr_target))

    # --- Step 7: Resize to output_size -----------------------------------
    canvas_pil = Image.fromarray(canvas, mode="RGBA")
    canvas_pil = canvas_pil.resize((output_size, output_size), Image.Resampling.LANCZOS)
    result = np.array(canvas_pil)

    # Final metrics on the output image
    final_metrics = compute_mask_metrics(result[:, :, 3])

    metadata = {
        "auto_ffr_status": "ok",
        "input_size": (h, w),
        "crop_side": S,
        "crop_origin": (int(crop_x), int(crop_y)),
        "centroid": (cx, cy),
        "bbox": (bx, by, bw, bh),
        "output_size": (output_size, output_size),
        "ffr_target": ffr_target,
        "ffr_actual": float(final_metrics["ffr"]),
        "far": float(final_metrics["far"]),
        "bbo": float(final_metrics["bbo"]),
        "bbo_warning": bbo_warning,
        "iterations": iterations_used,
    }

    return result, metadata


def official_frame_for_triposr(
    rgba_image: np.ndarray,
    foreground_ratio: float = 0.85,
    output_size: int = MIN_OUTPUT_RESOLUTION,
    fg_threshold: int = 128,
) -> Tuple[np.ndarray, Dict]:
    """
    Frame the subject using the same logic as the official TripoSR script (resize_foreground):
    - Compute tight bbox from alpha
    - Crop to bbox
    - Pad to square
    - Pad again so that the cropped square occupies `foreground_ratio` of the final square SIDE LENGTH
    - Resize to output_size

    NOTE: This `foreground_ratio` is a SIDE ratio (not area).
    """
    alpha = rgba_image[:, :, 3]
    h, w = alpha.shape

    mask = (alpha >= fg_threshold).astype(np.uint8)
    coords = cv2.findNonZero(mask)
    if coords is None:
        canvas = np.zeros((output_size, output_size, 4), dtype=np.uint8)
        return canvas, {
            "official_frame_status": "no_foreground",
            "input_size": (h, w),
            "output_size": (output_size, output_size),
            "foreground_ratio": float(foreground_ratio),
        }

    x, y, bw, bh = cv2.boundingRect(coords)
    crop = rgba_image[y:y+bh, x:x+bw]

    # Pad to square
    s = int(max(bw, bh))
    square = np.zeros((s, s, 4), dtype=np.uint8)
    ox = (s - bw) // 2
    oy = (s - bh) // 2
    square[oy:oy+bh, ox:ox+bw] = crop

    if foreground_ratio <= 0.0 or foreground_ratio >= 1.0:
        raise ValueError("foreground_ratio must be in (0, 1)")

    # Pad so square occupies `foreground_ratio` of the final side
    new_size = int(round(s / float(foreground_ratio)))
    new_size = max(new_size, s)
    padded = np.zeros((new_size, new_size, 4), dtype=np.uint8)
    px = (new_size - s) // 2
    py = (new_size - s) // 2
    padded[py:py+s, px:px+s] = square

    pil = Image.fromarray(padded, mode="RGBA")
    pil = pil.resize((output_size, output_size), Image.Resampling.LANCZOS)
    out = np.array(pil)

    mm = compute_mask_metrics(out[:, :, 3], fg_threshold=fg_threshold, morph_cleanup=True)
    meta = {
        "official_frame_status": "ok",
        "input_size": (h, w),
        "bbox": (int(x), int(y), int(bw), int(bh)),
        "foreground_ratio": float(foreground_ratio),
        "output_size": (output_size, output_size),
        "ffr_area": float(mm["ffr"]),
        "far": float(mm["far"]),
        "bbo": float(mm["bbo"]),
    }
    return out, meta


# ---------------------------------------------------------------------------
# Legacy framing helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------

def center_and_scale(
    rgba_image: np.ndarray,
    foreground_ratio: float = 0.85,
    max_size: int = 2048,
    min_size: int = MIN_OUTPUT_RESOLUTION,
    square_output: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    Center and scale foreground object to fill target ratio of frame.
    LEGACY — prefer auto_frame_for_triposr() for new TripoSR code.

    Args:
        rgba_image: Input RGBA image as numpy array
        foreground_ratio: Target foreground ratio (0.7-0.95)
        max_size: Maximum output dimension
        min_size: Minimum output dimension (512px default for TripoSR)
        square_output: Force square output

    Returns:
        Tuple of (processed RGBA array, metadata dict)
    """
    alpha = rgba_image[:, :, 3]

    # Get foreground bounding box
    x, y, w, h = get_foreground_bbox(alpha)

    # Crop to bounding box
    cropped = rgba_image[y:y+h, x:x+w]

    # Calculate target canvas size (now enforces min_size)
    canvas_w, canvas_h = calculate_target_size(w, h, foreground_ratio, max_size, min_size)

    if square_output:
        canvas_size = max(canvas_w, canvas_h)
        canvas_w = canvas_h = canvas_size

    # Calculate the target size for the foreground object within the canvas
    target_fg_w = int(canvas_w * foreground_ratio)
    target_fg_h = int(canvas_h * foreground_ratio)

    # Resize cropped image to fit within target foreground size while maintaining aspect ratio
    scale = min(target_fg_w / w, target_fg_h / h)
    if scale < 1.0 or scale > 1.0:  # Resize if needed (either up or down)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Convert to PIL Image for high-quality resizing
        cropped_pil = Image.fromarray(cropped, mode='RGBA')
        cropped_pil = cropped_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        cropped = np.array(cropped_pil)

        # Update dimensions
        w, h = new_w, new_h

    # Create new canvas
    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

    # Calculate centering position
    paste_x = (canvas_w - w) // 2
    paste_y = (canvas_h - h) // 2

    # Paste cropped image onto canvas
    canvas[paste_y:paste_y+h, paste_x:paste_x+w] = cropped

    metadata = {
        "original_size": rgba_image.shape[:2],
        "bbox": {"x": x, "y": y, "width": w, "height": h},
        "canvas_size": (canvas_h, canvas_w),
        "foreground_ratio": foreground_ratio,
        "actual_ratio": min(w / canvas_w, h / canvas_h),
        "centered_at": (paste_x, paste_y),
        "scale_applied": scale,
        "min_size_enforced": min_size
    }

    return canvas, metadata


def adjust_for_category(
    rgba_image: np.ndarray,
    category: CategoryType,
    foreground_ratio: float
) -> Tuple[np.ndarray, Dict]:
    """
    Apply category-specific preprocessing adjustments.
    LEGACY — prefer auto_frame_for_triposr() for new TripoSR code.

    Args:
        rgba_image: Input RGBA image
        category: Category type
        foreground_ratio: Desired foreground ratio

    Returns:
        Tuple of (adjusted RGBA array, adjustment metadata)
    """
    adjustments = {}

    # Category-specific adjustments — aggressive frame fill for TripoSR
    if category == CategoryType.HUMAN_FULL_BODY:
        adjustments["aspect_preference"] = "portrait"
        adjustments["min_foreground_ratio"] = 0.87
        adjustments["max_foreground_ratio"] = 0.93

    elif category == CategoryType.ANIMAL_FULL_BODY:
        adjustments["aspect_preference"] = "flexible"
        adjustments["min_foreground_ratio"] = 0.87
        adjustments["max_foreground_ratio"] = 0.93

    elif category == CategoryType.HUMAN_BUST:
        adjustments["aspect_preference"] = "square"
        adjustments["min_foreground_ratio"] = 0.82
        adjustments["max_foreground_ratio"] = 0.90

    elif category == CategoryType.ANIMAL_BUST:
        adjustments["aspect_preference"] = "square"
        adjustments["min_foreground_ratio"] = 0.82
        adjustments["max_foreground_ratio"] = 0.90

    elif category == CategoryType.GENERIC_OBJECT:
        adjustments["aspect_preference"] = "flexible"
        adjustments["min_foreground_ratio"] = 0.85
        adjustments["max_foreground_ratio"] = 0.95

    # Clamp foreground ratio to category limits
    if "min_foreground_ratio" in adjustments:
        foreground_ratio = max(foreground_ratio, adjustments["min_foreground_ratio"])
    if "max_foreground_ratio" in adjustments:
        foreground_ratio = min(foreground_ratio, adjustments["max_foreground_ratio"])

    # Apply centering and scaling
    square_output = (adjustments.get("aspect_preference") == "square")
    processed, metadata = center_and_scale(
        rgba_image,
        foreground_ratio=foreground_ratio,
        square_output=square_output
    )

    adjustments["applied_foreground_ratio"] = foreground_ratio
    metadata["category_adjustments"] = adjustments

    return processed, metadata


def enhance_alpha_quality(
    rgba_image: np.ndarray,
    category: CategoryType,
    aggressive: bool = False
) -> np.ndarray:
    """
    Enhance alpha channel quality based on category requirements.

    Args:
        rgba_image: Input RGBA image
        category: Category type
        aggressive: Use aggressive alpha cleanup (for TRELLIS.2)

    Returns:
        RGBA image with enhanced alpha
    """
    result = rgba_image.copy()
    alpha = result[:, :, 3]

    # Category-specific alpha handling
    if category in [CategoryType.ANIMAL_FULL_BODY, CategoryType.ANIMAL_BUST]:
        if aggressive:
            alpha = np.where(alpha > 240, 255, alpha)
            alpha = np.where(alpha < 15, 0, alpha)
        else:
            alpha = np.where(alpha > 250, 255, alpha)
            alpha = np.where(alpha < 5, 0, alpha)

    elif category in [CategoryType.HUMAN_FULL_BODY, CategoryType.HUMAN_BUST]:
        if aggressive:
            alpha = np.where(alpha > 230, 255, alpha)
            alpha = np.where(alpha < 25, 0, alpha)
        else:
            alpha = np.where(alpha > 245, 255, alpha)
            alpha = np.where(alpha < 10, 0, alpha)

    else:
        alpha = np.where(alpha > 240, 255, alpha)
        alpha = np.where(alpha < 15, 0, alpha)

    result[:, :, 3] = alpha
    return result


def detect_specular_regions(
    rgb_image: np.ndarray,
    threshold: int = 200
) -> np.ndarray:
    """
    Detect specular/reflective regions in the image.

    Args:
        rgb_image: Input RGB image
        threshold: Brightness threshold for specular detection

    Returns:
        Binary mask of specular regions
    """
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    _, specular_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return specular_mask


def apply_region_aware_processing(
    rgba_image: np.ndarray,
    category: CategoryType,
    enable_denoising: bool = True,
    enable_clahe: bool = True,
    enable_boundary_sharpening: bool = True,
    enable_highlight_compression: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Apply region-aware processing optimized for TripoSR.
    This is the core preprocessing function that applies different filters
    to different regions based on category.

    Args:
        rgba_image: Input RGBA image
        category: Category type
        enable_denoising: Apply bilateral filtering
        enable_clahe: Apply CLAHE for local contrast
        enable_boundary_sharpening: Apply selective edge sharpening
        enable_highlight_compression: Compress specular highlights

    Returns:
        Tuple of (processed RGBA image, processing metadata)
    """
    processing_metadata = {
        "category": category.value,
        "operations_applied": []
    }

    # Extract RGB and alpha
    rgb = rgba_image[:, :, :3].copy()
    alpha = rgba_image[:, :, 3].copy()

    # Get foreground mask
    fg_mask = (alpha > 10).astype(np.uint8) * 255

    # Category-specific processing
    if category in [CategoryType.HUMAN_BUST, CategoryType.HUMAN_FULL_BODY]:
        if enable_denoising:
            rgb_float = rgb.astype(np.float32)
            rgb_denoised = apply_bilateral_filter(rgb, d=9, sigma_color=75, sigma_space=75)
            fg_mask_3ch = np.stack([fg_mask] * 3, axis=-1).astype(np.float32) / 255.0
            rgb = (rgb_denoised * fg_mask_3ch + rgb_float * (1.0 - fg_mask_3ch)).astype(np.uint8)
            processing_metadata["operations_applied"].append("bilateral_filter_foreground")

        if enable_clahe:
            rgb_clahe = apply_clahe_luminance(rgb, clip_limit=2.0, tile_grid_size=(8, 8))
            fg_mask_3ch = np.stack([fg_mask] * 3, axis=-1).astype(np.float32) / 255.0
            rgb = (rgb_clahe * fg_mask_3ch * 0.7 + rgb.astype(np.float32) * (1.0 - fg_mask_3ch * 0.7)).astype(np.uint8)
            processing_metadata["operations_applied"].append("clahe_luminance_foreground")

    elif category in [CategoryType.ANIMAL_BUST, CategoryType.ANIMAL_FULL_BODY]:
        if enable_denoising:
            rgb_float = rgb.astype(np.float32)
            rgb_denoised = apply_bilateral_filter(rgb, d=7, sigma_color=50, sigma_space=50)
            fg_mask_3ch = np.stack([fg_mask] * 3, axis=-1).astype(np.float32) / 255.0
            rgb = (rgb_denoised * fg_mask_3ch + rgb_float * (1.0 - fg_mask_3ch)).astype(np.uint8)
            processing_metadata["operations_applied"].append("bilateral_filter_fur_aware")

        if enable_clahe:
            rgb_clahe = apply_clahe_luminance(rgb, clip_limit=1.5, tile_grid_size=(8, 8))
            fg_mask_3ch = np.stack([fg_mask] * 3, axis=-1).astype(np.float32) / 255.0
            rgb = (rgb_clahe * fg_mask_3ch * 0.5 + rgb.astype(np.float32) * (1.0 - fg_mask_3ch * 0.5)).astype(np.uint8)
            processing_metadata["operations_applied"].append("clahe_luminance_fur")

    elif category == CategoryType.GENERIC_OBJECT:
        if enable_highlight_compression:
            specular_mask = detect_specular_regions(rgb, threshold=200)
            if np.any(specular_mask):
                rgb = compress_highlights(rgb, threshold=200, compression_factor=0.7)
                processing_metadata["operations_applied"].append("highlight_compression")

        if enable_denoising:
            rgb_float = rgb.astype(np.float32)
            rgb_denoised = apply_bilateral_filter(rgb, d=9, sigma_color=75, sigma_space=75)
            fg_mask_3ch = np.stack([fg_mask] * 3, axis=-1).astype(np.float32) / 255.0
            rgb = (rgb_denoised * fg_mask_3ch + rgb_float * (1.0 - fg_mask_3ch)).astype(np.uint8)
            processing_metadata["operations_applied"].append("bilateral_filter_object")

        if enable_clahe:
            rgb_clahe = apply_clahe_luminance(rgb, clip_limit=2.5, tile_grid_size=(8, 8))
            fg_mask_3ch = np.stack([fg_mask] * 3, axis=-1).astype(np.float32) / 255.0
            rgb = (rgb_clahe * fg_mask_3ch * 0.8 + rgb.astype(np.float32) * (1.0 - fg_mask_3ch * 0.8)).astype(np.uint8)
            processing_metadata["operations_applied"].append("clahe_luminance_object")

    # Boundary sharpening (all categories)
    if enable_boundary_sharpening:
        rgb = apply_boundary_sharpening(rgb, alpha, amount=0.5, radius=1, ring_width=3)
        processing_metadata["operations_applied"].append("boundary_sharpening")

    # Recombine RGB and alpha
    result = np.dstack([rgb, alpha])

    return result, processing_metadata


# ---------------------------------------------------------------------------
# Main pipeline entry points
# ---------------------------------------------------------------------------

def preprocess_for_triposr(
    rgba_path: str,
    category: CategoryType,
    framing_mode: str = "auto_ffr",
    # --- Auto FFR (new default) ---
    auto_ffr: bool = True,
    ffr_target: float = FFR_TARGET_DEFAULT,
    ffr_min: float = FFR_MIN_DEFAULT,
    ffr_max: float = FFR_MAX_DEFAULT,
    output_size: int = MIN_OUTPUT_RESOLUTION,
    # --- Legacy foreground_ratio (ignored when auto_ffr=True) ---
    foreground_ratio: float = 0.85,
    # --- Region processing knobs ---
    enable_region_processing: bool = True,
    enable_denoising: bool = True,
    enable_clahe: bool = True,
    enable_boundary_sharpening: bool = True,
    enable_highlight_compression: bool = True,
    # --- Composite / output knobs ---
    composite_feather_radius: int = COMPOSITE_FEATHER_RADIUS_DEFAULT,
    background_color: Tuple[int, int, int] = TRIPOSR_BACKGROUND_COLOR,
    min_resolution: int = MIN_OUTPUT_RESOLUTION,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Complete TripoSR-optimized preprocessing pipeline.

    Output is **always RGB** — TripoSR is trained on RGB images and feeding
    RGBA causes domain shift.  The subject is composited onto a flat white
    background with a slight 1–2 px alpha feather to prevent floating
    artifact planes from hard mask cutouts.

    By default uses **automatic FFR framing** (auto_ffr=True):
    - Computes optimal square crop so bbox fills ~60 % of the frame area
    - Centers on mask centroid, pads with transparent where needed
    - Adjusts if FFR falls outside 0.45–0.75 range
    - Resizes to output_size × output_size (default 512)

    When auto_ffr=False, falls back to the legacy foreground_ratio system
    (adjust_for_category with static per-category ratio ranges).

    Other pipeline steps (applied regardless of framing mode):
    - Edge-preserving bilateral denoising
    - CLAHE on luminance for local contrast
    - Selective boundary sharpening (not global)
    - Category-specific region-aware processing
    - Highlight compression for specular surfaces
    - 1–2 px composite feather → flat white background → RGB

    Args:
        rgba_path: Path to input RGBA image
        category: Category type
        auto_ffr: Use automatic FFR framing (recommended, default True)
        ffr_target: Target FFR (~0.60)
        ffr_min: Minimum acceptable FFR (0.45)
        ffr_max: Maximum acceptable FFR (0.75)
        output_size: Square output size when auto_ffr=True (default 512)
        foreground_ratio: Legacy foreground ratio (ignored when auto_ffr=True)
        enable_region_processing: Enable region-aware processing
        enable_denoising: Apply bilateral filtering
        enable_clahe: Apply CLAHE for local contrast
        enable_boundary_sharpening: Apply selective edge sharpening
        enable_highlight_compression: Compress specular highlights
        composite_feather_radius: Alpha feather radius before compositing (1–2 px)
        background_color: Flat background RGB color (default: white)
        min_resolution: Minimum output dimension (legacy, used when auto_ffr=False)
        output_dir: Output directory (defaults to same as input)

    Returns:
        Dictionary with output_path and metadata
    """
    # Load RGBA image
    rgba_image = Image.open(rgba_path)
    if rgba_image.mode != 'RGBA':
        raise ValueError("Input image must be RGBA")

    rgba_array = np.array(rgba_image)

    metadata: Dict = {
        "input_path": rgba_path,
        "category": category.value,
        "output_format": "rgb",
        "auto_ffr": auto_ffr,
        "background_color": background_color,
        "composite_feather_radius": composite_feather_radius,
    }

    # Step 1: Region-aware processing (before framing, on full-res image)
    if enable_region_processing:
        rgba_array, processing_meta = apply_region_aware_processing(
            rgba_array,
            category,
            enable_denoising=enable_denoising,
            enable_clahe=enable_clahe,
            enable_boundary_sharpening=enable_boundary_sharpening,
            enable_highlight_compression=enable_highlight_compression
        )
        metadata["region_processing"] = processing_meta
    else:
        metadata["region_processing"] = {"operations_applied": []}

    # Step 2: Framing — official parity, auto FFR, or legacy
    fm = (framing_mode or ("auto_ffr" if auto_ffr else "legacy")).lower()

    if fm == "official":
        processed_rgba, framing_meta = official_frame_for_triposr(
            rgba_array,
            foreground_ratio=float(foreground_ratio),
            output_size=int(output_size),
        )
        metadata["framing_mode"] = "official"
        metadata["framing"] = framing_meta
        metadata["foreground_ratio"] = float(foreground_ratio)
        metadata["ffr_actual"] = framing_meta.get("ffr_area")

    elif fm == "auto_ffr":
        processed_rgba, framing_meta = auto_frame_for_triposr(
            rgba_array,
            ffr_target=ffr_target,
            ffr_min=ffr_min,
            ffr_max=ffr_max,
            output_size=output_size,
        )
        metadata["framing"] = framing_meta
        metadata["framing_mode"] = "auto_ffr"
        metadata["ffr_target"] = ffr_target
        metadata["ffr_actual"] = framing_meta.get("ffr_actual")
    else:
        # Legacy path: per-category static foreground_ratio
        processed_rgba, centering_meta = adjust_for_category(
            rgba_array,
            category,
            foreground_ratio
        )
        metadata.update(centering_meta)
        metadata["framing_mode"] = "legacy"
        metadata["foreground_ratio"] = foreground_ratio

    # Step 3: Composite feather — slight alpha softening before RGB conversion.
    # This prevents floating artifact planes from perfectly sharp mask edges.
    # Applied AFTER framing so the feather is at output resolution.
    if composite_feather_radius > 0:
        alpha_feathered = apply_composite_feather(
            processed_rgba[:, :, 3], radius=composite_feather_radius
        )
        processed_rgba[:, :, 3] = alpha_feathered
        metadata["region_processing"]["operations_applied"].append(
            f"composite_feather_r{composite_feather_radius}"
        )

    # Recompute mask metrics on the *actual* image that will be composited to RGB.
    # This ensures reported FFR/FAR reflect the final alpha used for TripoSR input.
    post_metrics = compute_mask_metrics(processed_rgba[:, :, 3])
    metadata["far_post_feather"] = float(post_metrics["far"])
    metadata["bbo_post_feather"] = float(post_metrics["bbo"])
    if auto_ffr:
        metadata["ffr_actual_post_feather"] = float(post_metrics["ffr"])

    # Step 4: Convert to RGB with flat background — ALWAYS.
    # TripoSR is trained on RGB; RGBA causes domain shift.
    final_image = rgba_to_rgb_with_background(processed_rgba, background_color)
    output_mode = "RGB"

    # Save output
    if output_dir is None:
        output_dir_path = Path(rgba_path).parent
    else:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

    input_name = Path(rgba_path).stem
    output_path = output_dir_path / f"{input_name}_triposr.png"

    output_image = Image.fromarray(final_image, mode=output_mode)
    output_image.save(output_path, 'PNG')

    metadata["output_path"] = str(output_path)
    metadata["output_size"] = final_image.shape[:2]

    return metadata


def preprocess_for_category(
    rgba_path: str,
    category: CategoryType,
    foreground_ratio: float = 0.85,
    enhance_alpha: bool = True,
    aggressive_alpha: bool = False,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Complete preprocessing pipeline for a category (legacy function).

    NOTE: For TripoSR, use preprocess_for_triposr() instead.
    This function is kept for backward compatibility.

    Args:
        rgba_path: Path to input RGBA image
        category: Category type
        foreground_ratio: Target foreground ratio
        enhance_alpha: Whether to enhance alpha channel
        aggressive_alpha: Use aggressive alpha cleanup
        output_dir: Output directory (defaults to same as input)

    Returns:
        Dictionary with output_path and metadata
    """
    # Load RGBA image
    rgba_image = Image.open(rgba_path)
    if rgba_image.mode != 'RGBA':
        raise ValueError("Input image must be RGBA")

    rgba_array = np.array(rgba_image)

    # Apply category-specific adjustments
    processed, metadata = adjust_for_category(rgba_array, category, foreground_ratio)

    # Enhance alpha if requested
    if enhance_alpha:
        processed = enhance_alpha_quality(processed, category, aggressive_alpha)
        metadata["alpha_enhanced"] = True
        metadata["aggressive_alpha"] = aggressive_alpha

    # Save output
    if output_dir is None:
        output_dir_path = Path(rgba_path).parent
    else:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

    input_name = Path(rgba_path).stem
    output_path = output_dir_path / f"{input_name}_preprocessed.png"

    output_image = Image.fromarray(processed, mode='RGBA')
    output_image.save(output_path, 'PNG')

    return {
        "output_path": str(output_path),
        "metadata": metadata,
        "category": category.value,
        "output_size": processed.shape[:2]
    }

