"""
Pipeline orchestrator: coordinates the full reconstruction workflow.

Stages executed in order:
  1. PREPROCESSING  - validate, remove background, resize, normalise
  2. VIEW_NORM      - cross-view subject size normalization
  3. FIDUCIAL       - add reference geometry (orange squares + blue circle)
  4. CAMERA_INIT    - resolve canonical view poses to COLMAP extrinsics
  5. COARSE_RECON   - dense geometry from images WITH fiducial markers
  6. ISOLATION      - strip marker geometry from 3D point cloud
  7. TRELLIS        - generative completion on CLEAN images (no markers)
  8. EXPORT         - package results for downstream consumption

Key design decisions:
  - Background removal happens FIRST (stage 1) so that every downstream
    stage works on clean, gray-background images.
  - Fiducial markers are rendered onto COPIES of the preprocessed images.
    The originals (without markers) are preserved for Trellis.
  - DUSt3R / MASt3R receive images WITH markers for better correspondences.
  - After coarse recon, marker geometry is stripped from the point cloud.
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
        self.marked_dir = self.job_dir / "marked"       # images WITH fiducial markers
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
        logger.info("Job %s -> %s", self.job.job_id, status.value)

    def run(self) -> ReconJob:
        """
        Execute the full pipeline.

        Returns the updated ReconJob with all stage results populated.
        """
        try:
            self._stage_preprocess()
            self._stage_view_normalize()
            self._stage_fiducial_markers()
            self._stage_camera_init()
            self._stage_coarse_recon()
            self._stage_isolation()

            if self.job.config.trellis_enabled:
                try:
                    self._stage_trellis()
                except PipelineError as e:
                    # TRELLIS.2 is optional - degrade gracefully so the
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
    # Stage 1: Preprocessing (includes background removal)
    # ------------------------------------------------------------------

    def _stage_preprocess(self) -> None:
        """
        Validate and preprocess input images.

        Background removal happens here so every downstream stage
        works on clean, gray-background images.
        """
        self._update_status(JobStatus.PREPROCESSING)
        logger.info("Stage 1: Preprocessing %d views (bg removal included)", len(self.job.views))

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
    # Stage 2: Cross-view subject normalization
    # ------------------------------------------------------------------

    def _stage_view_normalize(self) -> None:
        """Normalize subject sizes across views for 3D consistency."""
        self._update_status(JobStatus.VIEW_NORMALIZATION)
        logger.info("Stage 2: Cross-view subject normalization")

        try:
            from pipelines.view_normalization import normalize_views

            normalize_views(
                view_specs=self.job.views,
                image_dir=self.preprocessed_dir,
            )
        except Exception as e:
            raise PipelineError("view_normalization", str(e) or repr(e)) from e

    # ------------------------------------------------------------------
    # Stage 3: Fiducial markers
    # ------------------------------------------------------------------

    def _stage_fiducial_markers(self) -> None:
        """
        Add reference geometry to view images for DUSt3R / MASt3R.

        Renders orange squares and a blue circle at known 3D positions
        onto copies of the preprocessed images.  The originals in
        preprocessed_dir are preserved (clean, no markers) for Trellis.
        """
        self._update_status(JobStatus.FIDUCIAL_MARKERS)
        logger.info("Stage 3: Adding fiducial markers for reconstruction")

        try:
            from pipelines.fiducial_markers import add_fiducial_markers

            add_fiducial_markers(
                view_specs=self.job.views,
                image_dir=self.preprocessed_dir,
                output_dir=self.marked_dir,
                intrinsics=self.job.config.intrinsics,
            )
        except Exception as e:
            raise PipelineError("fiducial_markers", str(e) or repr(e)) from e

    # ------------------------------------------------------------------
    # Stage 4: Camera Initialisation
    # ------------------------------------------------------------------

    def _stage_camera_init(self) -> None:
        """Resolve canonical poses to full extrinsics."""
        self._update_status(JobStatus.CAMERA_INIT)
        logger.info("Stage 4: Camera initialisation")

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
    # Stage 5: Coarse Reconstruction
    # ------------------------------------------------------------------

    def _stage_coarse_recon(self) -> None:
        """
        Run dense geometry reconstruction using images WITH fiducial markers.

        The markers provide strong multi-view correspondences that help
        DUSt3R / MASt3R produce better geometry with only 3 views.
        """
        self._update_status(JobStatus.COARSE_RECON)
        logger.info(
            "Stage 5: Coarse reconstruction (backend=%s, using marked images)",
            self.job.config.recon_backend.value,
        )

        # Use images WITH fiducial markers for reconstruction
        image_paths = [
            self.marked_dir / v.image_filename
            for v in self.job.views
        ]

        # Verify all images exist
        for p in image_paths:
            if not p.exists():
                raise PipelineError(
                    "coarse_recon",
                    f"Marked image not found: {p}",
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
    # Stage 6: Subject Isolation (strip marker geometry)
    # ------------------------------------------------------------------

    def _stage_isolation(self) -> None:
        """
        Strip fiducial marker geometry from the 3D point cloud.

        Since background was already removed in preprocessing, the main
        job here is removing the reconstructed marker points so the
        point cloud contains only the subject.
        """
        self._update_status(JobStatus.SUBJECT_ISOLATION)
        logger.info("Stage 6: Subject isolation (stripping marker geometry)")

        # Get point cloud path from coarse reconstruction
        ply_path = None
        if self.job.coarse_result and self.job.coarse_result.point_cloud.ply_path:
            ply_path = Path(self.job.coarse_result.point_cloud.ply_path)

        if ply_path is None or not ply_path.exists():
            logger.warning("No point cloud to filter - skipping isolation")
            return

        try:
            from pipelines.coarse_recon import read_ply, _write_ply
            from pipelines.fiducial_markers import strip_markers_from_pointcloud

            self.isolation_dir.mkdir(parents=True, exist_ok=True)

            points, colors = read_ply(ply_path)
            original_count = len(points)

            # Remove marker geometry
            filtered_points, filtered_colors = strip_markers_from_pointcloud(
                points, colors
            )

            # Write filtered point cloud
            filtered_ply = self.isolation_dir / "filtered_pointcloud.ply"
            _write_ply(filtered_ply, filtered_points, filtered_colors)

            num_retained = len(filtered_points)
            num_removed = original_count - num_retained

            self.job.isolation_result = IsolationResult(
                masked_image_paths=[
                    str(self.preprocessed_dir / v.image_filename)
                    for v in self.job.views
                ],
                filtered_ply_path=str(filtered_ply),
                num_points_retained=num_retained,
                num_points_removed=num_removed,
            )

            logger.info(
                "Point cloud filtered: %d retained, %d removed (markers stripped)",
                num_retained, num_removed,
            )

        except Exception as e:
            raise PipelineError("subject_isolation", str(e) or repr(e)) from e

    # ------------------------------------------------------------------
    # Stage 7: Trellis.2 Completion
    # ------------------------------------------------------------------

    def _stage_trellis(self) -> None:
        """
        Run Trellis.2 generative completion on CLEAN images (no markers).

        Uses the original preprocessed images (without fiducial markers)
        so Trellis generates clean geometry without marker artifacts.
        """
        self._update_status(JobStatus.TRELLIS_COMPLETION)
        logger.info("Stage 7: Trellis.2 completion (using clean images)")

        # Use clean images from preprocessed_dir (no markers)
        clean_image_paths = [
            str(self.preprocessed_dir / v.image_filename)
            for v in self.job.views
        ]

        # Verify clean images exist
        for p_str in clean_image_paths:
            p = Path(p_str)
            if not p.exists():
                raise PipelineError(
                    "trellis_completion",
                    f"Clean preprocessed image not found: {p}",
                )

        try:
            from pipelines.trellis_completion import run_trellis_completion

            result = run_trellis_completion(
                masked_image_paths=clean_image_paths,
                output_dir=self.trellis_dir,
                device=self.device,
            )
            self.job.trellis_result = result

        except Exception as e:
            raise PipelineError("trellis_completion", str(e) or repr(e)) from e

    # ------------------------------------------------------------------
    # Stage 8: Export
    # ------------------------------------------------------------------

    def _stage_export(self) -> None:
        """Export reconstruction results to GLB for downstream consumption."""
        self._update_status(JobStatus.EXPORTING)
        logger.info("Stage 8: Exporting GLB mesh")

        try:
            from pipelines.export import export_glb

            glb_path = export_glb(self.job_dir)
            logger.info("GLB exported: %s", glb_path)

        except Exception as e:
            # Export failure is non-fatal - the job still has useful
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

