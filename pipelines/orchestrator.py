"""
Pipeline orchestrator: coordinates the full reconstruction workflow.

Stages executed in order:
  1. PREPROCESSING  – validate, resize, normalise input images
  2. CAMERA_INIT    – resolve canonical view poses to COLMAP extrinsics
  3. COARSE_RECON   – dense geometry from full images (WITH background)
  4. ISOLATION      – remove background from images + filter 3D points
  5. TRELLIS        – generative completion to produce final 3D asset
  6. EXPORT         – package results for downstream consumption

The orchestrator updates the ReconJob model at each stage, persists
intermediate results, and handles errors gracefully so that partial
results are preserved.
"""

from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from api.models import (
    CoarseReconResult,
    IsolationResult,
    JobStatus,
    PipelineConfig,
    ReconBackend,
    ReconJob,
    TrellisResult,
    ViewLabel,
)
from pipelines.camera_init import (
    export_colmap_workspace,
    get_canonical_views,
    resolve_views,
)
from pipelines.config import STORAGE_ROOT, get_default_pipeline_config
from pipelines.preprocess import preprocess_views

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Raised when a pipeline stage fails."""

    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(f"[{stage}] {message}")


# Type alias for status-update callbacks (e.g., to push to Redis)
StatusCallback = Callable[[ReconJob], None]


def _noop_callback(job: ReconJob) -> None:
    """Default no-op status callback."""
    pass


class PipelineOrchestrator:
    """
    Coordinates the full multi-view reconstruction pipeline.

    Usage::

        orch = PipelineOrchestrator(job)
        result = orch.run()
    """

    def __init__(
        self,
        job: ReconJob,
        on_status_change: StatusCallback = _noop_callback,
        device: str = "cuda",
    ):
        self.job = job
        self.on_status_change = on_status_change
        self.device = device

        # Resolve storage directories
        self.job_dir = Path(job.storage_dir) if job.storage_dir else (
            STORAGE_ROOT / job.job_id
        )
        self.input_dir = self.job_dir / "input"
        self.preprocessed_dir = self.job_dir / "preprocessed"
        self.recon_dir = self.job_dir / "coarse_recon"
        self.isolation_dir = self.job_dir / "isolation"
        self.trellis_dir = self.job_dir / "trellis"
        self.export_dir = self.job_dir / "export"
        self.colmap_dir = self.job_dir / "colmap"

    def _update_status(self, status: JobStatus, error: str | None = None) -> None:
        """Update job status and notify via callback."""
        self.job.status = status
        self.job.updated_at = datetime.utcnow()
        if error:
            self.job.error_message = error
        self.on_status_change(self.job)
        logger.info("Job %s → %s", self.job.job_id, status.value)

    def run(self) -> ReconJob:
        """
        Execute the full pipeline.

        Returns the updated ReconJob with all stage results populated.
        """
        try:
            self._stage_preprocess()
            self._stage_camera_init()
            self._stage_coarse_recon()
            self._stage_isolation()

            if self.job.config.trellis_enabled:
                try:
                    self._stage_trellis()
                except PipelineError as e:
                    # TRELLIS.2 is optional – degrade gracefully so the
                    # pipeline still delivers coarse-recon + isolation output.
                    logger.warning(
                        "Trellis.2 completion unavailable, skipping: %s", e,
                    )

            self._stage_export()

            self._update_status(JobStatus.COMPLETED)

        except PipelineError as e:
            logger.error(
                "Pipeline failed at stage %s: %s\n%s",
                e.stage, str(e), traceback.format_exc(),
            )
            self._update_status(JobStatus.FAILED, error=str(e))
        except Exception as e:
            logger.exception("Unexpected pipeline error")
            self._update_status(JobStatus.FAILED, error=str(e))

        return self.job

    # ------------------------------------------------------------------
    # Stage 1: Preprocessing
    # ------------------------------------------------------------------

    def _stage_preprocess(self) -> None:
        """Validate and preprocess input images."""
        self._update_status(JobStatus.PREPROCESSING)
        logger.info("Stage 1: Preprocessing %d views", len(self.job.views))

        if not self.job.views:
            raise PipelineError(
                "preprocessing",
                "No views provided. Expected 3 views (front, side, top).",
            )

        if len(self.job.views) != 3:
            raise PipelineError(
                "preprocessing",
                f"Expected exactly 3 views, got {len(self.job.views)}.",
            )

        # Check all required labels are present
        labels = {v.label for v in self.job.views}
        required = {ViewLabel.FRONT, ViewLabel.SIDE, ViewLabel.TOP}
        if labels != required:
            raise PipelineError(
                "preprocessing",
                f"Missing view labels. Got {labels}, need {required}.",
            )

        try:
            updated_specs = preprocess_views(
                view_specs=self.job.views,
                input_dir=self.input_dir,
                output_dir=self.preprocessed_dir,
                target_size=self.job.config.image_size,
            )
            self.job.views = updated_specs
        except Exception as e:
            raise PipelineError("preprocessing", str(e) or repr(e)) from e

    # ------------------------------------------------------------------
    # Stage 2: Camera Initialisation
    # ------------------------------------------------------------------

    def _stage_camera_init(self) -> None:
        """Resolve canonical poses to full extrinsics."""
        self._update_status(JobStatus.CAMERA_INIT)
        logger.info("Stage 2: Camera initialisation")

        try:
            resolved = resolve_views(
                view_specs=self.job.views,
                intrinsics=self.job.config.intrinsics,
            )
            self.job.resolved_views = resolved

            # Export COLMAP workspace for optional downstream use
            export_colmap_workspace(resolved, self.colmap_dir)

        except Exception as e:
            raise PipelineError("camera_init", str(e) or repr(e)) from e

    # ------------------------------------------------------------------
    # Stage 3: Coarse Reconstruction
    # ------------------------------------------------------------------

    def _stage_coarse_recon(self) -> None:
        """
        Run dense geometry reconstruction using full (with-background) images.

        This is the key principle: use background for geometry estimation.
        """
        self._update_status(JobStatus.COARSE_RECON)
        logger.info(
            "Stage 3: Coarse reconstruction (backend=%s, with_background=%s)",
            self.job.config.recon_backend.value,
            self.job.config.use_background_for_pose,
        )

        # Use full (with-background) preprocessed images
        image_paths = [
            self.preprocessed_dir / v.image_filename
            for v in self.job.views
        ]

        # Verify all images exist
        for p in image_paths:
            if not p.exists():
                raise PipelineError(
                    "coarse_recon",
                    f"Preprocessed image not found: {p}",
                )

        try:
            from pipelines.coarse_recon import run_coarse_reconstruction

            result = run_coarse_reconstruction(
                image_paths=image_paths,
                resolved_views=self.job.resolved_views,
                output_dir=self.recon_dir,
                backend=self.job.config.recon_backend,
                device=self.device,
            )
            self.job.coarse_result = result

        except Exception as e:
            raise PipelineError("coarse_recon", str(e) or repr(e)) from e

    # ------------------------------------------------------------------
    # Stage 4: Subject Isolation
    # ------------------------------------------------------------------

    def _stage_isolation(self) -> None:
        """Remove background from images and filter 3D point cloud."""
        self._update_status(JobStatus.SUBJECT_ISOLATION)
        logger.info("Stage 4: Subject isolation (method=%s)", self.job.config.mask_method)

        # Get point cloud path from coarse reconstruction
        ply_path = None
        if self.job.coarse_result and self.job.coarse_result.point_cloud.ply_path:
            ply_path = Path(self.job.coarse_result.point_cloud.ply_path)

        try:
            from pipelines.subject_isolation import run_subject_isolation

            result = run_subject_isolation(
                image_dir=self.preprocessed_dir,
                views=self.job.resolved_views,
                point_cloud_path=ply_path,
                output_dir=self.isolation_dir,
                mask_method=self.job.config.mask_method,
            )
            self.job.isolation_result = result

        except Exception as e:
            raise PipelineError("subject_isolation", str(e) or repr(e)) from e

    # ------------------------------------------------------------------
    # Stage 5: Trellis.2 Completion
    # ------------------------------------------------------------------

    def _stage_trellis(self) -> None:
        """Run Trellis.2 generative completion on masked images."""
        self._update_status(JobStatus.TRELLIS_COMPLETION)
        logger.info("Stage 5: Trellis.2 completion")

        if not self.job.isolation_result or not self.job.isolation_result.masked_image_paths:
            raise PipelineError(
                "trellis_completion",
                "No masked images available from isolation stage.",
            )

        try:
            from pipelines.trellis_completion import run_trellis_completion

            result = run_trellis_completion(
                masked_image_paths=self.job.isolation_result.masked_image_paths,
                output_dir=self.trellis_dir,
                device=self.device,
            )
            self.job.trellis_result = result

        except Exception as e:
            raise PipelineError("trellis_completion", str(e) or repr(e)) from e


    # ------------------------------------------------------------------
    # Stage 6: Export
    # ------------------------------------------------------------------

    def _stage_export(self) -> None:
        """Export reconstruction results to GLB for downstream consumption."""
        self._update_status(JobStatus.EXPORTING)
        logger.info("Stage 6: Exporting GLB mesh")

        try:
            from pipelines.export import export_glb

            glb_path = export_glb(self.job_dir)
            logger.info("GLB exported: %s", glb_path)

        except Exception as e:
            # Export failure is non-fatal – the job still has useful
            # point cloud and isolation artifacts.
            logger.warning("GLB export failed (non-fatal): %s", e)


def run_pipeline(
    job: ReconJob,
    on_status_change: StatusCallback = _noop_callback,
    device: str = "cuda",
) -> ReconJob:
    """
    Convenience function to run the full pipeline.

    Parameters
    ----------
    job : the reconstruction job to process
    on_status_change : callback invoked on each status transition
    device : torch device string

    Returns
    -------
    The updated ReconJob with all results.
    """
    orchestrator = PipelineOrchestrator(
        job=job,
        on_status_change=on_status_change,
        device=device,
    )
    return orchestrator.run()

