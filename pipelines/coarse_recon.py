"""
Coarse reconstruction via dense stereo / learned pointmap estimation.

This module implements the geometry-recovery stage of the pipeline.
It takes the **full images** (with background) and known camera poses
to produce a dense point cloud of the scene.

Supported backends:
  - DUSt3R  – pairwise dense pointmap prediction + global alignment
  - MASt3R  – improved matching + SfM-style refinement
  - VGGT    – direct multi-view geometry regression

The key principle (per expert guidance): **use background for pose
recovery / geometry estimation, then remove it later**.  The background
provides stable correspondences, stronger parallax cues, and better
global alignment—especially critical with only 3 images.

Pipeline:
  1. Load full (with-background) preprocessed images
  2. Run pairwise dense stereo (DUSt3R/MASt3R) or multi-view (VGGT)
  3. Global alignment of pairwise pointmaps
  4. Optionally refine with known camera priors
  5. Export fused point cloud (.ply)
  6. Return CoarseReconResult with aligned views and point cloud
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np

from api.models import (
    CoarseReconResult,
    PointCloud,
    ReconBackend,
    ResolvedView,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract backend protocol
# ---------------------------------------------------------------------------


class ReconstructionBackend(Protocol):
    """Protocol that all reconstruction backends must implement."""

    def reconstruct(
        self,
        image_paths: list[Path],
        resolved_views: list[ResolvedView],
        output_dir: Path,
        **kwargs: Any,
    ) -> CoarseReconResult:
        """
        Run dense reconstruction on the provided images.

        Parameters
        ----------
        image_paths : paths to preprocessed full (with-background) images
        resolved_views : views with known intrinsics and extrinsics
        output_dir : directory to write outputs (point clouds, etc.)

        Returns
        -------
        CoarseReconResult with point cloud and updated view info.
        """
        ...


# ---------------------------------------------------------------------------
# DUSt3R backend
# ---------------------------------------------------------------------------


class DUSt3RBackend:
    """
    Dense reconstruction using DUSt3R (Dense Unconstrained Stereo 3D Reconstruction).

    DUSt3R predicts dense pointmaps for image pairs and aligns them globally.
    It works well with few views because it uses a learned ViT-based architecture
    that provides strong geometric priors.

    The process:
    1. For each image pair, predict two aligned dense 3D pointmaps
    2. Build a global optimisation graph from all pairs
    3. Solve for globally consistent pointmaps
    4. Optionally incorporate known camera priors for better alignment
    5. Fuse into a single point cloud
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._model = None

    def _load_model(self) -> Any:
        """Lazy-load the DUSt3R model."""
        if self._model is not None:
            return self._model

        try:
            from dust3r.model import AsymmetricCroCo3DStereo
            from dust3r.utils.device import to_numpy

            logger.info("Loading DUSt3R model from %s", self.checkpoint_path)
            self._model = AsymmetricCroCo3DStereo.from_pretrained(
                self.checkpoint_path
            )
            self._model = self._model.to(self.device)
            self._model.eval()
            logger.info("DUSt3R model loaded successfully")
        except Exception as e:
            logger.exception("DUSt3R import/load failed: %r", e)
            raise

        return self._model

    def _prepare_pairs(
        self, image_paths: list[Path]
    ) -> list[tuple[int, int]]:
        """
        Generate all image pairs for pairwise reconstruction.

        With 3 images we get 3 pairs: (0,1), (0,2), (1,2).
        """
        n = len(image_paths)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))
        return pairs

    def reconstruct(
        self,
        image_paths: list[Path],
        resolved_views: list[ResolvedView],
        output_dir: Path,
        **kwargs: Any,
    ) -> CoarseReconResult:
        """Run DUSt3R pairwise reconstruction + global alignment."""
        output_dir.mkdir(parents=True, exist_ok=True)
        model = self._load_model()

        try:
            from dust3r.image_pairs import make_pairs
            from dust3r.inference import inference
            from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
            from dust3r.utils.device import to_numpy
        except ImportError as e:
            raise RuntimeError(
                "DUSt3R dependencies not available. "
                "Ensure dust3r is installed in the environment."
            ) from e

        # Load images in DUSt3R format
        from dust3r.utils.image import load_images

        imgs = load_images(
            [str(p) for p in image_paths],
            size=512,  # DUSt3R default input resolution
        )

        # Build all pairs
        pairs = make_pairs(
            imgs, scene_graph="complete", prefilter=None, symmetrize=True
        )

        # Run pairwise inference
        logger.info("Running DUSt3R pairwise inference on %d pairs", len(pairs))
        output = inference(pairs, model, self.device, batch_size=1)

        # Global alignment
        logger.info(
            "Running global alignment (PointCloudOptimizer mode for %d views)",
            len(imgs),
        )
        scene = global_aligner(
            output,
            device=self.device,
            mode=GlobalAlignerMode.PointCloudOptimizer,
        )

        # NOTE: We intentionally do NOT call scene.preset_focal().
        # Our intrinsics define fx=1700 for 2048×2048 images, but DUSt3R
        # internally resizes to ~512×384.  Presetting the unscaled focal
        # would give DUSt3R a ~17° FOV (extreme telephoto) instead of the
        # correct ~62°.  DUSt3R's own focal estimator is robust and will
        # recover a correct focal from the images themselves.

        # Build known pose matrices (cam-to-world) for prior injection.
        # DUSt3R expects cam-to-world poses; our extrinsics store w2c,
        # so we invert each 4×4 matrix.
        known_poses_c2w = []
        for v in resolved_views:
            R_w2c = np.array(v.extrinsics.R_w2c).reshape(3, 3)
            t_w2c = np.array(v.extrinsics.t_w2c)
            w2c = np.eye(4)
            w2c[:3, :3] = R_w2c
            w2c[:3, 3] = t_w2c
            # Invert to get cam-to-world (DUSt3R convention)
            c2w = np.linalg.inv(w2c)
            known_poses_c2w.append(c2w)

        # Inject known poses into the scene so DUSt3R treats them as fixed.
        # preset_pose sets requires_grad=False on im_poses and
        # norm_pw_scale=False, locking the cameras during optimisation.
        scene.preset_pose(known_poses_c2w)

        # Use MST initialisation instead of known_poses.  The
        # init_from_known_poses path requires ALL focals to also be
        # preset (asserts nkf == n_imgs), but we intentionally skip
        # preset_focal because our intrinsics are at 2048×2048 and
        # DUSt3R internally resizes to ~512×384 — presetting the
        # unscaled focal would produce a wildly wrong FOV.
        #
        # MST init still respects the preset poses: when nkp > 1 it
        # rigidly aligns the spanning-tree solution to the known
        # cameras, then the optimiser refines depth maps and focals
        # while the camera poses stay frozen.
        loss = scene.compute_global_alignment(
            init="mst",
            niter=300,
            schedule="cosine",
            lr=0.01,
        )
        logger.info("Global alignment loss: %.4f", loss)

        # Extract results
        pts3d = to_numpy(scene.get_pts3d())
        confidence = to_numpy(scene.get_masks())

        # Fuse point clouds from all views
        all_points = []
        all_colors = []
        all_conf = []

        for view_idx, (pts, conf, img_path) in enumerate(
            zip(pts3d, confidence, image_paths)
        ):
            # pts shape: (H, W, 3), conf shape: (H, W)
            mask = conf > 0.5  # confidence threshold
            valid_pts = pts[mask]
            all_points.append(valid_pts)
            all_conf.append(conf[mask])

            # Extract colours from original image
            from PIL import Image as PILImage

            img = PILImage.open(img_path).resize(
                (pts.shape[1], pts.shape[0])
            )
            colors = np.array(img, dtype=np.float64) / 255.0
            valid_colors = colors[mask]
            all_colors.append(valid_colors)

        fused_points = np.concatenate(all_points, axis=0)
        fused_colors = np.concatenate(all_colors, axis=0)
        fused_conf = np.concatenate(all_conf, axis=0)

        # Write PLY
        ply_path = output_dir / "coarse_pointcloud.ply"
        _write_ply(ply_path, fused_points, fused_colors)

        logger.info(
            "Coarse reconstruction complete: %d points, saved to %s",
            len(fused_points),
            ply_path,
        )

        return CoarseReconResult(
            point_cloud=PointCloud(
                num_points=len(fused_points),
                ply_path=str(ply_path),
                confidence_mean=float(fused_conf.mean()),
            ),
            views=resolved_views,
            backend_used=ReconBackend.DUST3R,
            alignment_error=float(loss),
        )


# ---------------------------------------------------------------------------
# MASt3R backend
# ---------------------------------------------------------------------------


class MASt3RBackend:
    """
    Dense reconstruction using MASt3R (Matching and Stereo 3D Reconstruction).

    MASt3R improves on DUSt3R with better matching robustness and a two-stage
    refinement: first a 3D matching loss, then a 2D reprojection loss.
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._model = None

    def _load_model(self) -> Any:
        """Lazy-load the MASt3R model."""
        if self._model is not None:
            return self._model

        try:
            from mast3r.model import AsymmetricMASt3R

            logger.info("Loading MASt3R model from %s", self.checkpoint_path)
            self._model = AsymmetricMASt3R.from_pretrained(
                self.checkpoint_path
            )
            self._model = self._model.to(self.device)
            self._model.eval()
            logger.info("MASt3R model loaded successfully")
        except Exception as e:
            logger.exception("MASt3R import/load failed: %r", e)
            raise

        return self._model

    def reconstruct(
        self,
        image_paths: list[Path],
        resolved_views: list[ResolvedView],
        output_dir: Path,
        **kwargs: Any,
    ) -> CoarseReconResult:
        """Run MASt3R-SfM style reconstruction."""
        output_dir.mkdir(parents=True, exist_ok=True)
        model = self._load_model()

        try:
            from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
            from mast3r.utils.misc import hash_md5
            from dust3r.image_pairs import make_pairs
            from dust3r.inference import inference
            from dust3r.utils.image import load_images
            from dust3r.utils.device import to_numpy
        except ImportError as e:
            raise RuntimeError(
                "MASt3R/DUSt3R dependencies not available."
            ) from e

        # Load images
        imgs = load_images(
            [str(p) for p in image_paths],
            size=512,
        )

        # Build pairs
        pairs = make_pairs(
            imgs, scene_graph="complete", prefilter=None, symmetrize=True
        )

        # Pairwise inference
        logger.info("Running MASt3R pairwise inference")
        output = inference(pairs, model, self.device, batch_size=1)

        # Sparse global alignment (MASt3R-SfM approach)
        # Uses 3D matching loss followed by 2D reprojection loss
        cache_dir = output_dir / "mast3r_cache"
        cache_dir.mkdir(exist_ok=True)

        scene = sparse_global_alignment(
            [str(p) for p in image_paths],
            pairs,
            cache_dir,
            model,
            lr1=0.07,
            niter1=500,
            lr2=0.014,
            niter2=200,
            device=self.device,
        )

        # Extract point cloud
        pts3d = to_numpy(scene.get_pts3d())
        masks = to_numpy(scene.get_masks())

        all_points = []
        all_colors = []
        for view_idx, (pts, mask, img_path) in enumerate(
            zip(pts3d, masks, image_paths)
        ):
            valid = mask > 0.5
            all_points.append(pts[valid])

            from PIL import Image as PILImage

            img = PILImage.open(img_path).resize(
                (pts.shape[1], pts.shape[0])
            )
            colors = np.array(img, dtype=np.float64) / 255.0
            all_colors.append(colors[valid])

        fused_points = np.concatenate(all_points, axis=0)
        fused_colors = np.concatenate(all_colors, axis=0)

        ply_path = output_dir / "coarse_pointcloud.ply"
        _write_ply(ply_path, fused_points, fused_colors)

        logger.info(
            "MASt3R reconstruction: %d points → %s",
            len(fused_points),
            ply_path,
        )

        return CoarseReconResult(
            point_cloud=PointCloud(
                num_points=len(fused_points),
                ply_path=str(ply_path),
            ),
            views=resolved_views,
            backend_used=ReconBackend.MAST3R,
        )


# ---------------------------------------------------------------------------
# VGGT backend (stub – direct multi-view geometry regression)
# ---------------------------------------------------------------------------


class VGGTBackend:
    """
    Dense reconstruction using VGGT (Visual Geometry Grounded Transformer).

    VGGT directly predicts camera extrinsics, intrinsics, depth, point maps,
    and tracks from a multi-view transformer in a single forward pass.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None

    def _load_model(self) -> Any:
        """Lazy-load the VGGT model."""
        if self._model is not None:
            return self._model

        try:
            from vggt.models.vggt import VGGT as VGGTModel

            logger.info("Loading VGGT model")
            self._model = VGGTModel.from_pretrained("facebook/VGGT-1B")
            self._model = self._model.to(self.device)
            self._model.eval()
            logger.info("VGGT model loaded successfully")
        except Exception as e:
            logger.exception("VGGT import/load failed: %r", e)
            raise

        return self._model

    def reconstruct(
        self,
        image_paths: list[Path],
        resolved_views: list[ResolvedView],
        output_dir: Path,
        **kwargs: Any,
    ) -> CoarseReconResult:
        """Run VGGT multi-view geometry prediction."""
        output_dir.mkdir(parents=True, exist_ok=True)
        model = self._load_model()

        try:
            import torch
            from vggt.utils.load_fn import load_and_preprocess_images
            from vggt.utils.geometry import unproject_depth_map
        except ImportError as e:
            raise RuntimeError("VGGT dependencies not available.") from e

        # Load and preprocess images for VGGT
        images = load_and_preprocess_images(
            [str(p) for p in image_paths]
        ).to(self.device)

        # Forward pass: predict everything at once
        with torch.no_grad():
            predictions = model(images)

        # Extract point maps (N, H, W, 3)
        point_maps = predictions["world_points"].cpu().numpy()
        depth_maps = predictions["depth"].cpu().numpy()
        confidence = predictions.get("confidence", None)

        all_points = []
        all_colors = []

        for idx, img_path in enumerate(image_paths):
            pts = point_maps[0, idx]  # (H, W, 3)

            if confidence is not None:
                conf = confidence[0, idx]
                mask = conf > 0.5
            else:
                # Use depth validity
                mask = depth_maps[0, idx] > 0

            all_points.append(pts[mask])

            from PIL import Image as PILImage

            img = PILImage.open(img_path).resize(
                (pts.shape[1], pts.shape[0])
            )
            colors = np.array(img, dtype=np.float64) / 255.0
            all_colors.append(colors[mask])

        fused_points = np.concatenate(all_points, axis=0)
        fused_colors = np.concatenate(all_colors, axis=0)

        ply_path = output_dir / "coarse_pointcloud.ply"
        _write_ply(ply_path, fused_points, fused_colors)

        logger.info(
            "VGGT reconstruction: %d points → %s",
            len(fused_points),
            ply_path,
        )

        return CoarseReconResult(
            point_cloud=PointCloud(
                num_points=len(fused_points),
                ply_path=str(ply_path),
            ),
            views=resolved_views,
            backend_used=ReconBackend.VGGT,
        )


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


def get_backend(
    backend: ReconBackend,
    device: str = "cuda",
    checkpoint_path: Optional[str] = None,
) -> ReconstructionBackend:
    """
    Instantiate the requested reconstruction backend.

    Parameters
    ----------
    backend : which backend to use
    device : torch device string
    checkpoint_path : path to model checkpoint (for DUSt3R/MASt3R)
    """
    from pipelines.config import DUST3R_CHECKPOINT, MAST3R_CHECKPOINT

    if backend == ReconBackend.DUST3R:
        ckpt = checkpoint_path or DUST3R_CHECKPOINT
        return DUSt3RBackend(checkpoint_path=ckpt, device=device)
    elif backend == ReconBackend.MAST3R:
        ckpt = checkpoint_path or MAST3R_CHECKPOINT
        return MASt3RBackend(checkpoint_path=ckpt, device=device)
    elif backend == ReconBackend.VGGT:
        return VGGTBackend(device=device)
    else:
        raise ValueError(f"Unknown reconstruction backend: {backend}")


def run_coarse_reconstruction(
    image_paths: list[Path],
    resolved_views: list[ResolvedView],
    output_dir: Path,
    backend: ReconBackend = ReconBackend.DUST3R,
    device: str = "cuda",
    **kwargs: Any,
) -> CoarseReconResult:
    """
    Top-level entry point for coarse reconstruction.

    This is the function called by the orchestrator.  It:
    1. Selects the appropriate backend
    2. Runs reconstruction on full (with-background) images
    3. Returns the CoarseReconResult

    Parameters
    ----------
    image_paths : paths to preprocessed full images (WITH background)
    resolved_views : views with known intrinsics and extrinsics
    output_dir : directory for output artifacts
    backend : which reconstruction method to use
    device : torch device
    """
    logger.info(
        "Starting coarse reconstruction with %s backend on %d images",
        backend.value,
        len(image_paths),
    )

    recon_backend = get_backend(backend, device=device)
    result = recon_backend.reconstruct(
        image_paths=image_paths,
        resolved_views=resolved_views,
        output_dir=output_dir,
        **kwargs,
    )

    logger.info(
        "Coarse reconstruction complete: %d points, backend=%s",
        result.point_cloud.num_points,
        result.backend_used.value,
    )

    return result


# ---------------------------------------------------------------------------
# PLY I/O
# ---------------------------------------------------------------------------


def _write_ply(
    path: Path,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> None:
    """
    Write a point cloud to a PLY file.

    Parameters
    ----------
    path : output file path
    points : (N, 3) float array of XYZ coordinates
    colors : optional (N, 3) float array of RGB in [0, 1]
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(points)
    has_color = colors is not None and len(colors) == n

    header_lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_color:
        header_lines.extend([
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ])
    header_lines.append("end_header")

    with open(path, "w") as f:
        f.write("\n".join(header_lines) + "\n")
        for i in range(n):
            line = f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}"
            if has_color:
                r = int(np.clip(colors[i, 0] * 255, 0, 255))
                g = int(np.clip(colors[i, 1] * 255, 0, 255))
                b = int(np.clip(colors[i, 2] * 255, 0, 255))
                line += f" {r} {g} {b}"
            f.write(line + "\n")

    logger.debug("Wrote PLY with %d points to %s", n, path)


def read_ply(path: Path) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Read a PLY file and return (points, colors).

    Returns
    -------
    points : (N, 3) float64 array
    colors : (N, 3) float64 array in [0, 1] or None if no colour data
    """
    with open(path, "r") as f:
        lines = f.readlines()

    # Parse header
    header_end = 0
    n_vertices = 0
    has_color = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("element vertex"):
            n_vertices = int(stripped.split()[-1])
        if "property" in stripped and "red" in stripped:
            has_color = True
        if stripped == "end_header":
            header_end = i + 1
            break

    data_lines = lines[header_end : header_end + n_vertices]
    points = np.zeros((n_vertices, 3), dtype=np.float64)
    colors = np.zeros((n_vertices, 3), dtype=np.float64) if has_color else None

    for i, line in enumerate(data_lines):
        parts = line.strip().split()
        points[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
        if has_color and len(parts) >= 6:
            colors[i] = [
                float(parts[3]) / 255.0,
                float(parts[4]) / 255.0,
                float(parts[5]) / 255.0,
            ]

    return points, colors

