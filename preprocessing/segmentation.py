"""
Image segmentation module for preprocessing.
Supports multiple segmentation methods: rembg, SAM 2 (future).
Enhanced with region-aware segmentation optimized for TripoSR output.

Optimizations applied:
- Hard alpha edges (no feathering) for cleaner geometry inference
- Higher alpha threshold to remove semi-transparent fringe
- Edge decontamination to remove green/color spill
- Hair/fur enhancement disabled by default (soft alpha hurts TripoSR)
"""
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
from PIL import Image
from rembg import remove, new_session
import cv2


class SegmentationMethod(Enum):
    """Available segmentation methods."""
    REMBG = "rembg"
    REMBG_U2NET = "rembg_u2net"  # Default U2Net model
    REMBG_U2NET_HUMAN = "rembg_u2net_human_seg"  # Human segmentation
    REMBG_SILUETA = "rembg_silueta"  # Better for complex backgrounds
    SAM2 = "sam2"  # Future implementation


class Segmenter:
    """Main segmentation class with TripoSR-optimized segmentation."""

    def __init__(self, model_name: str = "u2net"):
        """
        Initialize segmentation models.

        Args:
            model_name: Model to use (u2net, u2net_human_seg, silueta, etc.)
        """
        # Initialize rembg session
        self.rembg_session = new_session(model_name)
        self.model_name = model_name

    def cleanup_gpu(self):
        """
        Clean up GPU memory used by segmentation models.
        Critical to call this after segmentation to free VRAM for TripoSR/Trellis.2.
        """
        try:
            import torch
            import gc

            # Clear the rembg session if it exists
            if hasattr(self, 'rembg_session') and self.rembg_session is not None:
                del self.rembg_session
                self.rembg_session = None

            # Force garbage collection
            gc.collect()

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(f"GPU memory cleared after segmentation.")

        except Exception as e:
            # Log but don't fail if cleanup has issues
            print(f"Warning: GPU cleanup encountered an issue: {e}")

    def segment_rembg(
        self,
        image_path: str,
        alpha_threshold: float = 0.5,
        post_process: bool = True
    ) -> np.ndarray:
        """
        Segment image using rembg (U2Net background removal).

        Args:
            image_path: Path to input image
            alpha_threshold: Threshold for alpha channel (0.0-1.0)
            post_process: Whether to apply post-processing to clean up mask

        Returns:
            RGBA numpy array
        """
        # Load image
        input_image = Image.open(image_path)

        # Convert to RGB if necessary
        if input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')

        # Remove background
        output_image = remove(
            input_image,
            session=self.rembg_session,
            post_process_mask=post_process
        )

        # Convert to numpy array
        rgba_array = np.array(output_image)

        # Apply alpha threshold if specified
        if alpha_threshold > 0:
            alpha_channel = rgba_array[:, :, 3]
            alpha_channel = np.where(
                alpha_channel > alpha_threshold * 255,
                alpha_channel,
                0
            )
            rgba_array[:, :, 3] = alpha_channel

        return rgba_array

    def segment_sam2(self, image_path: str) -> np.ndarray:
        """
        Segment image using SAM 2 (Segment Anything Model 2).
        Future implementation for multi-frame consistency.

        Args:
            image_path: Path to input image

        Returns:
            RGBA numpy array
        """
        raise NotImplementedError("SAM 2 segmentation not yet implemented")

    def refine_alpha_edges(
        self,
        rgba_array: np.ndarray,
        blur_kernel: int = 3,
        erode_iterations: int = 0,
        dilate_iterations: int = 0
    ) -> np.ndarray:
        """
        Refine alpha channel edges for cleaner masks.

        Args:
            rgba_array: Input RGBA image as numpy array
            blur_kernel: Kernel size for Gaussian blur (odd number)
            erode_iterations: Number of erosion iterations (shrinks mask)
            dilate_iterations: Number of dilation iterations (expands mask)

        Returns:
            Refined RGBA numpy array
        """
        alpha = rgba_array[:, :, 3]

        # Apply morphological operations
        if erode_iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            alpha = cv2.erode(alpha, kernel, iterations=erode_iterations)

        if dilate_iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            alpha = cv2.dilate(alpha, kernel, iterations=dilate_iterations)

        # Apply Gaussian blur for smooth edges
        if blur_kernel > 1:
            alpha = cv2.GaussianBlur(alpha, (blur_kernel, blur_kernel), 0)

        # Update alpha channel
        refined_rgba = rgba_array.copy()
        refined_rgba[:, :, 3] = alpha

        return refined_rgba

    def remove_soft_alpha(
        self,
        rgba_array: np.ndarray,
        threshold: float = 0.95
    ) -> np.ndarray:
        """
        Remove soft/semi-transparent alpha edges.
        Useful for fur/hair in TRELLIS.2 to avoid amputation.

        Args:
            rgba_array: Input RGBA image
            threshold: Alpha values above this become fully opaque (0.0-1.0)

        Returns:
            RGBA array with harder alpha edges
        """
        result = rgba_array.copy()
        alpha = result[:, :, 3].astype(float) / 255.0

        # Make alpha more binary
        alpha = np.where(alpha > threshold, 1.0, alpha)
        alpha = np.where(alpha < 0.05, 0.0, alpha)

        result[:, :, 3] = (alpha * 255).astype(np.uint8)
        return result

    def harden_alpha(
        self,
        rgba_array: np.ndarray,
        low_cutoff: float = 0.15,
        high_cutoff: float = 0.85
    ) -> np.ndarray:
        """
        Harden alpha channel to near-binary for TripoSR.
        Semi-transparent pixels create ambiguous depth boundaries that
        TripoSR interprets as thin inflated sheets or warped geometry.

        Args:
            rgba_array: Input RGBA image
            low_cutoff: Alpha below this becomes 0 (0.0-1.0)
            high_cutoff: Alpha above this becomes 255 (0.0-1.0)

        Returns:
            RGBA array with hardened alpha edges
        """
        result = rgba_array.copy()
        alpha = result[:, :, 3].astype(np.float32) / 255.0

        # Hard cutoffs: below low → 0, above high → 1, between → linear remap
        alpha = np.where(alpha < low_cutoff, 0.0, alpha)
        alpha = np.where(alpha > high_cutoff, 1.0, alpha)

        result[:, :, 3] = (alpha * 255).astype(np.uint8)
        return result

    def decontaminate_edges(
        self,
        rgba_array: np.ndarray,
        edge_width: int = 3
    ) -> np.ndarray:
        """
        Decontaminate edge colors (remove green-screen spill, background color bleed).
        Critical for TripoSR to avoid color artifacts at silhouette edges.

        Args:
            rgba_array: Input RGBA image
            edge_width: Width of edge region to process

        Returns:
            RGBA array with decontaminated edges
        """
        result = rgba_array.copy()
        rgb = result[:, :, :3].astype(np.float32)
        alpha = result[:, :, 3].astype(np.float32) / 255.0

        # Find edge regions (semi-transparent areas)
        edge_mask = (alpha > 0.05) & (alpha < 0.95)

        if not np.any(edge_mask):
            return result

        # Dilate edge mask to cover more area
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_width * 2 + 1, edge_width * 2 + 1))
        edge_mask_dilated = cv2.dilate(edge_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

        # For edge pixels, blend towards nearby opaque foreground colors
        # This removes background color contamination
        for c in range(3):
            channel = rgb[:, :, c]
            # Inpaint edge regions using nearby opaque foreground
            mask_for_inpaint = edge_mask_dilated.astype(np.uint8) * 255
            if np.any(mask_for_inpaint):
                channel_inpainted = cv2.inpaint(
                    channel.astype(np.uint8),
                    mask_for_inpaint,
                    inpaintRadius=edge_width,
                    flags=cv2.INPAINT_TELEA
                )
                # Blend based on alpha
                blend_factor = np.where(edge_mask_dilated, 1.0 - alpha, 0.0)
                rgb[:, :, c] = channel * (1.0 - blend_factor) + channel_inpainted * blend_factor

        result[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        return result

    def feather_edges(
        self,
        rgba_array: np.ndarray,
        feather_radius: int = 2
    ) -> np.ndarray:
        """
        Apply feathering to alpha edges for smoother transitions.
        NOTE: For TripoSR, feathering is generally HARMFUL. Hard edges
        produce better geometry. Only use for non-TripoSR workflows.

        Args:
            rgba_array: Input RGBA image
            feather_radius: Radius for feathering in pixels

        Returns:
            RGBA array with feathered edges
        """
        result = rgba_array.copy()
        alpha = result[:, :, 3]

        # Apply Gaussian blur to alpha channel
        if feather_radius > 0:
            kernel_size = feather_radius * 2 + 1
            alpha_feathered = cv2.GaussianBlur(alpha, (kernel_size, kernel_size), feather_radius / 2)
            result[:, :, 3] = alpha_feathered

        return result

    def enhance_hair_fur_matting(
        self,
        rgba_array: np.ndarray,
        preserve_detail: bool = True
    ) -> np.ndarray:
        """
        Enhance matting quality for hair/fur regions.
        Uses guided filter to preserve fine details while smoothing regions.

        NOTE: For TripoSR, this can create soft alpha that causes sheet-like
        meshes. Only enable when specifically needed (e.g. Trellis.2).

        Args:
            rgba_array: Input RGBA image
            preserve_detail: Whether to preserve fine hair/fur strands

        Returns:
            RGBA array with enhanced hair/fur matting
        """
        result = rgba_array.copy()
        rgb = result[:, :, :3]
        alpha = result[:, :, 3].astype(np.float32) / 255.0

        if preserve_detail:
            # Use bilateral filter on alpha to preserve edges
            alpha_8bit = (alpha * 255).astype(np.uint8)
            alpha_filtered = cv2.bilateralFilter(alpha_8bit, d=5, sigmaColor=50, sigmaSpace=50)
            alpha = alpha_filtered.astype(np.float32) / 255.0

        # Find semi-transparent regions (likely hair/fur)
        semi_transparent = (alpha > 0.1) & (alpha < 0.9)

        if np.any(semi_transparent):
            # Slightly expand alpha in these regions to avoid "disappearing" thin structures
            alpha_expanded = alpha.copy()
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            semi_transparent_mask = semi_transparent.astype(np.uint8) * 255
            dilated = cv2.dilate((alpha * 255).astype(np.uint8), kernel, iterations=1)

            # Blend expanded alpha only in semi-transparent regions
            blend_mask = semi_transparent.astype(np.float32)
            alpha_expanded = alpha * (1.0 - blend_mask * 0.3) + (dilated / 255.0) * (blend_mask * 0.3)
            alpha = np.clip(alpha_expanded, 0, 1)

        result[:, :, 3] = (alpha * 255).astype(np.uint8)
        return result


# Global segmenter instance
_segmenter = None


def get_segmenter() -> Segmenter:
    """Get or create global segmenter instance."""
    global _segmenter
    if _segmenter is None:
        _segmenter = Segmenter()
    return _segmenter


def clear_global_segmenter():
    """
    Clear the global segmenter instance and free GPU memory.
    CRITICAL: Call this after preprocessing completes to free VRAM used by rembg/U2Net models.
    """
    global _segmenter
    if _segmenter is not None:
        _segmenter.cleanup_gpu()
        _segmenter = None
        print("Global segmenter cleared")


def segment_image(
    image_path: str,
    method: SegmentationMethod = SegmentationMethod.REMBG,
    alpha_threshold: float = 0.5,
    refine_edges: bool = True,
    output_path: Optional[str] = None
) -> Dict:
    """
    Segment an image and return RGBA result (legacy function).

    Args:
        image_path: Path to input image
        method: Segmentation method to use
        alpha_threshold: Threshold for alpha channel
        refine_edges: Whether to refine alpha edges
        output_path: Optional path to save RGBA output

    Returns:
        Dictionary with rgba_array and rgba_path
    """
    segmenter = get_segmenter()

    # Perform segmentation
    if method == SegmentationMethod.REMBG:
        rgba_array = segmenter.segment_rembg(image_path, alpha_threshold)
    elif method == SegmentationMethod.SAM2:
        rgba_array = segmenter.segment_sam2(image_path)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")

    # Refine edges if requested
    if refine_edges:
        rgba_array = segmenter.refine_alpha_edges(rgba_array, blur_kernel=3)

    # Save output
    if output_path is None:
        input_path = Path(image_path)
        output_path = str(input_path.parent / f"{input_path.stem}_rgba.png")

    rgba_image = Image.fromarray(rgba_array, mode='RGBA')
    rgba_image.save(output_path, 'PNG')

    return {
        "rgba_array": rgba_array,
        "rgba_path": output_path,
        "shape": rgba_array.shape,
        "method": method.value
    }


def segment_for_triposr(
    image_path: str,
    category: str = "generic_object",
    model_name: str = "u2net",
    alpha_threshold: float = 0.1,
    enable_edge_decontamination: bool = True,
    enable_feathering: bool = False,
    enable_hair_fur_enhancement: bool = False,
    feather_radius: int = 0,
    output_path: Optional[str] = None
) -> Dict:
    """
    TripoSR-optimized segmentation pipeline with hard alpha edges.

    Key optimizations for TripoSR:
    - Hard alpha edges (no feathering) — soft edges create ambiguous depth
      boundaries that TripoSR interprets as thin sheets or warped geometry
    - Higher alpha threshold (0.1) to remove semi-transparent fringe
    - Edge decontamination to remove green/color spill at silhouette
    - Hair/fur enhancement DISABLED by default — soft alpha around fur
      causes amputation and sheet-like meshes in TripoSR

    Args:
        image_path: Path to input image
        category: Category type (human_bust, human_full_body, animal_bust,
                  animal_full_body, generic_object)
        model_name: Segmentation model (u2net, u2net_human_seg, silueta)
        alpha_threshold: Minimum alpha threshold (0.0-1.0). Higher = crisper.
                         Default 0.1 removes semi-transparent fringe.
        enable_edge_decontamination: Remove background color spill at edges
        enable_feathering: Apply edge feathering. DEFAULT FALSE for TripoSR —
                           hard edges produce better geometry inference.
        enable_hair_fur_enhancement: Enhance hair/fur matting. DEFAULT FALSE
                                     for TripoSR — soft alpha hurts geometry.
        feather_radius: Feathering radius in pixels (0 = no feathering)
        output_path: Optional output path

    Returns:
        Dictionary with rgba_array, rgba_path, and processing metadata
    """
    # Choose best model for category
    if category in ["human_bust", "human_full_body"] and model_name == "u2net":
        model_name = "u2net_human_seg"  # Better for humans

    # Create segmenter with appropriate model
    segmenter = Segmenter(model_name=model_name)

    metadata = {
        "category": category,
        "model_name": model_name,
        "operations_applied": []
    }

    # Step 1: Initial segmentation
    rgba_array = segmenter.segment_rembg(
        image_path,
        alpha_threshold=alpha_threshold,
        post_process=True
    )
    metadata["operations_applied"].append("background_removal")

    # Step 2: Edge decontamination (critical for TripoSR — removes green fringe)
    if enable_edge_decontamination:
        rgba_array = segmenter.decontaminate_edges(rgba_array, edge_width=3)
        metadata["operations_applied"].append("edge_decontamination")

    # Step 3: Harden alpha edges for TripoSR
    # Semi-transparent pixels create ambiguous depth boundaries
    rgba_array = segmenter.harden_alpha(
        rgba_array,
        low_cutoff=0.15,
        high_cutoff=0.85
    )
    metadata["operations_applied"].append("alpha_hardening")

    # Step 4: Hair/fur enhancement (DISABLED by default for TripoSR)
    # Only enable for non-TripoSR workflows (e.g. Trellis.2)
    if enable_hair_fur_enhancement and category in [
        "human_bust", "human_full_body", "animal_bust", "animal_full_body"
    ]:
        rgba_array = segmenter.enhance_hair_fur_matting(rgba_array, preserve_detail=True)
        metadata["operations_applied"].append("hair_fur_matting_enhancement")

    # Step 5: Edge feathering (DISABLED by default for TripoSR)
    # Hard edges produce better geometry inference
    if enable_feathering and feather_radius > 0:
        rgba_array = segmenter.feather_edges(rgba_array, feather_radius=feather_radius)
        metadata["operations_applied"].append(f"edge_feathering_{feather_radius}px")

    # CRITICAL: Clean up GPU memory after segmentation
    # This frees VRAM used by rembg/U2Net models before TripoSR loads
    segmenter.cleanup_gpu()
    metadata["operations_applied"].append("gpu_cleanup")

    # Save output
    if output_path is None:
        input_path = Path(image_path)
        output_path = str(input_path.parent / f"{input_path.stem}_triposr_rgba.png")

    rgba_image = Image.fromarray(rgba_array, mode='RGBA')
    rgba_image.save(output_path, 'PNG')

    return {
        "rgba_array": rgba_array,
        "rgba_path": output_path,
        "shape": rgba_array.shape,
        "metadata": metadata
    }

