"""
Preview image generator for preprocessing pipeline
Generates preview images at each stage for frontend display
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from PIL import Image
import json
import numpy as np


class PreviewGenerator:
    """Generates and manages preview images throughout the preprocessing pipeline"""

    def __init__(self, job_id: str, storage_root: str = "storage/previews"):
        self.job_id = job_id
        self.storage_root = Path(storage_root)
        self.job_dir = self.storage_root / job_id
        self.job_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.job_dir / "metadata.json"
        self.metadata = {
            "job_id": job_id,
            "created_at": datetime.utcnow().isoformat(),
            "stages": []
        }

    def save_preview(
        self,
        image: Image.Image,
        stage: str,
        filename: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Save a preview image for a specific stage

        Args:
            image: PIL Image to save
            stage: Stage name (raw, segmented, refined, final)
            filename: Optional custom filename (default: {stage}.png)

        Returns:
            Dict with stage info including path and timestamp
        """
        if filename is None:
            filename = f"{stage}.png"

        filepath = self.job_dir / filename

        # Save image
        image.save(filepath, "PNG")

        # Update metadata
        stage_info = {
            "stage": stage,
            "filename": filename,
            "path": str(filepath),
            "url": f"/api/preview/{self.job_id}/{filename}",
            "timestamp": datetime.utcnow().isoformat(),
            "size": {
                "width": image.width,
                "height": image.height
            }
        }

        self.metadata["stages"].append(stage_info)
        self._save_metadata()

        return stage_info

    def save_multiple_previews(
        self,
        images: List[Image.Image],
        stage: str,
        prefix: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Save multiple preview images for a stage (e.g., multi-view)

        Args:
            images: List of PIL Images
            stage: Stage name
            prefix: Optional filename prefix (default: stage name)

        Returns:
            List of stage info dicts
        """
        if prefix is None:
            prefix = stage

        stage_infos = []
        for i, image in enumerate(images):
            filename = f"{prefix}_{i}.png"
            stage_info = self.save_preview(image, f"{stage}_{i}", filename)
            stage_infos.append(stage_info)

        return stage_infos

    def get_previews(self) -> Dict:
        """Get all preview metadata for this job"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return self.metadata

    def get_preview_path(self, filename: str) -> Path:
        """Get the full path for a preview file"""
        return self.job_dir / filename

    def cleanup(self):
        """Remove all preview images and metadata for this job"""
        import shutil
        if self.job_dir.exists():
            shutil.rmtree(self.job_dir)

    def _save_metadata(self):
        """Save metadata to JSON file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)


class PreprocessingPipeline:
    """
    Complete TripoSR-optimized preprocessing pipeline with preview generation.
    Integrates advanced segmentation, region-aware processing, and TripoSR-specific optimizations.

    Output is always RGB with flat white background — TripoSR is trained on
    RGB images and feeding RGBA causes domain shift.
    """

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.preview_gen = PreviewGenerator(job_id)

    def process(
        self,
        input_image: Image.Image,
        category: str,
        aggressive_alpha: bool = False,
        use_triposr_optimization: bool = True,
        triposr_preprocess_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """
        Run complete TripoSR-optimized preprocessing pipeline with preview generation.

        This implements the full expert-recommended preprocessing:
        - Advanced segmentation with hard alpha edges (no feathering)
        - Edge decontamination to remove green/color fringe
        - Region-aware processing (bilateral filter, CLAHE, boundary sharpening)
        - Automatic FFR framing (bbox fills ~60% of square crop area)
        - 1-2 px composite feather on alpha edges (prevents floating artifact planes)
        - RGB output with flat white background (TripoSR is trained on RGB only)
        - Category-specific optimizations

        Args:
            input_image: Raw input image (can be RGB or RGBA)
            category: Category type (human_full_body, animal_bust, etc.)
            aggressive_alpha: Use aggressive alpha thresholding (for Trellis.2)
            use_triposr_optimization: Use TripoSR-optimized preprocessing (recommended)
            triposr_preprocess_overrides: Optional dict of preprocessing overrides.
                Supported keys:
                - auto_ffr (bool): Use automatic FFR framing (default True)
                - ffr_target (float): Target FFR (~0.60)
                - ffr_min (float): Minimum acceptable FFR (0.45)
                - ffr_max (float): Maximum acceptable FFR (0.75)
                - output_size (int): Square output size in pixels (512)
                - composite_feather_radius (int): Alpha edge feather (1-2 px)
                - foreground_ratio (float): Legacy ratio (only when auto_ffr=False)
                - enable_denoising, enable_clahe,
                  enable_boundary_sharpening, enable_highlight_compression

        Returns:
            Dict with preview metadata and final processed image path
        """
        from .segmentation import segment_for_triposr
        from .preprocessing import (
            preprocess_for_triposr, CategoryType,
            TRIPOSR_BACKGROUND_COLOR,
        )

        overrides = triposr_preprocess_overrides or {}

        # Stage 1: Save raw input
        self.preview_gen.save_preview(input_image, "raw", "raw.png")

        # Stage 2: TripoSR-optimized segmentation
        if input_image.mode != 'RGBA':
            # Need to segment - save input to temp file first
            temp_input_path = self.preview_gen.get_preview_path("temp_input.png")
            input_image.save(temp_input_path, "PNG")

            # Use TripoSR-optimized segmentation with hard alpha edges.
            # Feathering and hair/fur enhancement are OFF by default —
            # soft edges create ambiguous depth boundaries that cause
            # TripoSR to produce sheet/blob meshes.
            seg_result = segment_for_triposr(
                image_path=str(temp_input_path),
                category=category,
                enable_edge_decontamination=True,
            )

            # Extract rgba_array and convert to PIL Image
            rgba_array = seg_result["rgba_array"]
            segmented_image = Image.fromarray(rgba_array, mode='RGBA')
            self.preview_gen.save_preview(segmented_image, "segmented", "segmented.png")

            # Save segmented RGBA for next stage
            segmented_path = self.preview_gen.get_preview_path("segmented.png")
        else:
            # Already RGBA, just save as segmented
            segmented_image = input_image
            segmented_path = self.preview_gen.get_preview_path("segmented.png")
            segmented_image.save(segmented_path, "PNG")
            self.preview_gen.save_preview(segmented_image, "segmented", "segmented.png")

        # Stage 3 & 4: TripoSR-optimized preprocessing
        category_enum = CategoryType(category)

        # --- Resolve auto FFR vs legacy foreground_ratio ---
        # Auto FFR is the default when TripoSR optimization is enabled.
        # It replaces the old static per-category foreground_ratio maps.
        framing_mode = overrides.get("framing_mode", "auto_ffr")
        auto_ffr = overrides.get("auto_ffr", use_triposr_optimization)
        ffr_target = float(overrides.get("ffr_target", 0.60))
        ffr_min = float(overrides.get("ffr_min", 0.45))
        ffr_max = float(overrides.get("ffr_max", 0.75))
        output_size = int(overrides.get("output_size", 512))
        composite_feather_radius = int(overrides.get("composite_feather_radius", 1))

        # Legacy foreground_ratio — only used when auto_ffr is False
        foreground_ratio = float(overrides.get("foreground_ratio", 0.85))

        # Resolve per-operation overrides (fall back to use_triposr_optimization flag)
        enable_denoising = overrides.get("enable_denoising", use_triposr_optimization)
        # Default OFF for humans (tends to hurt geometric cues), still overrideable per-request
        enable_clahe = overrides.get("enable_clahe", (use_triposr_optimization and ("human" not in category)))
        enable_boundary_sharpening = overrides.get(
            "enable_boundary_sharpening", (use_triposr_optimization and ("human" not in category))
        )
        enable_highlight_compression = overrides.get(
            "enable_highlight_compression", (category == "generic_object")
        )

        # --- Final RGB output (what TripoSR actually receives) ---
        # Official parity: RGB with flat 50% gray background, 1–2 px composite feather.
        # TripoSR is trained on RGB images — RGBA causes domain shift.
        # preprocess_for_triposr() now always outputs RGB with feathering built in.
        final_rgb_result = preprocess_for_triposr(
            rgba_path=str(segmented_path),
            category=category_enum,
            framing_mode=framing_mode,
            auto_ffr=auto_ffr,
            ffr_target=ffr_target,
            ffr_min=ffr_min,
            ffr_max=ffr_max,
            output_size=output_size,
            foreground_ratio=foreground_ratio,  # legacy, ignored when auto_ffr=True
            enable_region_processing=use_triposr_optimization,
            enable_denoising=enable_denoising,
            enable_clahe=enable_clahe,
            enable_boundary_sharpening=enable_boundary_sharpening,
            enable_highlight_compression=enable_highlight_compression,
            composite_feather_radius=composite_feather_radius,
        )

        # Save the final RGB image as the preview TripoSR will consume
        final_rgb = Image.open(final_rgb_result["output_path"]).convert("RGB")
        self.preview_gen.save_preview(final_rgb, "final", "final.png")

        # Also create a preview with background overlay for visualization
        if use_triposr_optimization:
            preview_with_grid = self._create_background_preview(final_rgb)
            self.preview_gen.save_preview(preview_with_grid, "preview_with_bg", "preview_with_bg.png")

        # Add processing metadata to preview metadata
        preview_metadata = self.preview_gen.get_previews()
        preview_metadata["preprocessing_metadata"] = final_rgb_result
        preview_metadata["triposr_optimized"] = use_triposr_optimization
        preview_metadata["background_color"] = TRIPOSR_BACKGROUND_COLOR
        preview_metadata["output_format"] = "rgb"
        preview_metadata["composite_feather_radius"] = composite_feather_radius
        preview_metadata["auto_ffr"] = auto_ffr
        preview_metadata["framing_mode"] = framing_mode

        if auto_ffr:
            preview_metadata["ffr_target"] = ffr_target
            preview_metadata["ffr_actual"] = final_rgb_result.get("ffr_actual")
        else:
            preview_metadata["foreground_ratio"] = foreground_ratio

        # CRITICAL: Final GPU cleanup after all preprocessing is complete
        # This ensures VRAM is freed before the job is picked up by TripoSR worker
        self._cleanup_preprocessing_gpu()

        return preview_metadata

    def _cleanup_preprocessing_gpu(self):
        """
        Final GPU memory cleanup after preprocessing completes.
        Ensures all preprocessing models are cleared from VRAM before TripoSR loads.
        """
        try:
            import torch
            import gc

            # Force garbage collection
            gc.collect()

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("Final GPU cleanup after preprocessing complete.")

        except Exception as e:
            print(f"Warning: Final GPU cleanup encountered an issue: {e}")

    def _create_background_preview(self, final_image: Image.Image) -> Image.Image:
        """
        Create a preview image showing the background more clearly.
        Adds a checkerboard pattern to background areas for visualization.

        Args:
            final_image: Final RGB image with white background

        Returns:
            Preview image with enhanced background visualization
        """
        from .preprocessing import TRIPOSR_BACKGROUND_COLOR

        # Convert to numpy
        img_array = np.array(final_image)

        # Create checkerboard pattern for visualization
        h, w = img_array.shape[:2]
        checker_size = 20
        checker = np.zeros((h, w), dtype=np.uint8)

        for i in range(0, h, checker_size):
            for j in range(0, w, checker_size):
                if (i // checker_size + j // checker_size) % 2 == 0:
                    checker[i:i+checker_size, j:j+checker_size] = 100
                else:
                    checker[i:i+checker_size, j:j+checker_size] = 150

        # Detect background regions (close to TRIPOSR_BACKGROUND_COLOR)
        # Use int16 to avoid uint8 underflow in subtraction
        bg_r, bg_g, bg_b = TRIPOSR_BACKGROUND_COLOR
        bg_mask = (
            (np.abs(img_array[:, :, 0].astype(np.int16) - bg_r) < 10) &
            (np.abs(img_array[:, :, 1].astype(np.int16) - bg_g) < 10) &
            (np.abs(img_array[:, :, 2].astype(np.int16) - bg_b) < 10)
        )

        # Apply checkerboard only to background regions
        preview = img_array.copy()
        for c in range(3):
            preview[:, :, c] = np.where(bg_mask, checker, img_array[:, :, c])

        return Image.fromarray(preview)


def create_thumbnail(image: Image.Image, max_size: int = 512) -> Image.Image:
    """
    Create a thumbnail for faster preview loading

    Args:
        image: Source image
        max_size: Maximum dimension size

    Returns:
        Thumbnail image
    """
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image
