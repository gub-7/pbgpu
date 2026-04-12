"""
Orchestrator for the canonical multi-view reconstruction pipeline.

Drives the job through all 10 stages in order, handling stage
transitions, error recovery, and artifact bookkeeping.

This is the entry point that the multi-view worker calls.
"""

import logging
from typing import Optional

from api.job_manager import JobManager
from api.storage import StorageManager
from api.models import (
    JobStatus,
    JobStage,
    StageStatus,
    MV_STAGE_ORDER,
)

from .config import CanonicalMVConfig

logger = logging.getLogger(__name__)


class CanonicalMVOrchestrator:
    """
    Drives a multi-view reconstruction job through all pipeline stages.

    Usage::

        orchestrator = CanonicalMVOrchestrator(job_id)
        success = orchestrator.run()
    """

    def __init__(
        self,
        job_id: str,
        job_manager: Optional[JobManager] = None,
        storage_manager: Optional[StorageManager] = None,
    ):
        self.job_id = job_id
        self.jm = job_manager or JobManager()
        self.sm = storage_manager or StorageManager()

        # Load job data and build config
        job_data = self.jm.get_job(job_id)
        if not job_data:
            raise ValueError(f"Job {job_id} not found")

        self.job_data = job_data
        self.config = CanonicalMVConfig.from_params(job_data.get("params", {}))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> bool:
        """
        Execute the pipeline from the current stage to completion.

        Returns True if the job completed successfully, False on failure.
        """
        logger.info(
            f"[{self.job_id}] Starting canonical MV pipeline "
            f"(pipeline={self.job_data.get('pipeline')})"
        )

        # Mark job as in-progress
        self.jm.update_job(
            self.job_id,
            status=JobStatus.GENERATING,
            current_stage=MV_STAGE_ORDER[0],
            stage_progress=0.0,
        )

        for stage_name in MV_STAGE_ORDER:
            # Reload job data to check current stage status
            self.job_data = self.jm.get_job(self.job_id) or self.job_data
            stage_status = self.job_data["stages"].get(stage_name)

            # Skip already-completed or explicitly-skipped stages
            if stage_status in (
                StageStatus.COMPLETED.value,
                StageStatus.SKIPPED.value,
            ):
                logger.info(f"[{self.job_id}] Stage '{stage_name}' already {stage_status}, skipping")
                continue

            # Check if stage should be skipped based on config
            if self._should_skip_stage(stage_name):
                logger.info(f"[{self.job_id}] Skipping stage '{stage_name}' (disabled by config)")
                self.jm.skip_stage(self.job_id, stage_name)
                continue

            # Mark stage as in-progress
            self.jm.update_job(
                self.job_id,
                stage_updates={stage_name: StageStatus.IN_PROGRESS.value},
                current_stage=stage_name,
                stage_progress=0.0,
            )

            try:
                logger.info(f"[{self.job_id}] Running stage: {stage_name}")
                self._run_stage(stage_name)
                self.jm.advance_stage(self.job_id, stage_name)
                logger.info(f"[{self.job_id}] Stage '{stage_name}' completed")

            except Exception as e:
                logger.error(
                    f"[{self.job_id}] Stage '{stage_name}' failed: {e}",
                    exc_info=True,
                )
                self.jm.fail_stage(self.job_id, stage_name, str(e))
                return False

        # All stages done
        self.jm.update_job(
            self.job_id,
            status=JobStatus.COMPLETED,
            progress=100,
        )
        logger.info(f"[{self.job_id}] Pipeline completed successfully")
        return True

    # ------------------------------------------------------------------
    # Stage dispatch
    # ------------------------------------------------------------------

    def _run_stage(self, stage_name: str):
        """
        Dispatch to the appropriate stage handler.

        Each stage is a separate module under pipelines/canonical_mv/.
        """
        handler = self._stage_handlers.get(stage_name)
        if handler is None:
            raise NotImplementedError(
                f"Stage '{stage_name}' is not yet implemented."
            )
        handler(self)

    def _run_ingest(self):
        """
        Stage: ingest

        Validates that all required view files are present and readable.
        Generates raw preview thumbnails for each view.
        """
        from .ingest import run_ingest
        run_ingest(self.job_id, self.config, self.jm, self.sm)

    def _run_preprocess_views(self):
        """
        Stage: preprocess_views

        Per-view segmentation, metric extraction, and cross-view
        consistent framing.
        """
        from .preprocess import run_preprocess_views
        run_preprocess_views(self.job_id, self.config, self.jm, self.sm)

    def _run_validate_views(self):
        """
        Stage: validate_views

        Cross-view consistency checks (silhouette area, sharpness,
        color consistency, mirror plausibility).
        """
        from .consistency import run_validate_views
        run_validate_views(self.job_id, self.config, self.jm, self.sm)

    def _run_initialize_cameras(self):
        """Stage: initialize_cameras — canonical camera rig setup."""
        from .camera_init import run_initialize_cameras
        run_initialize_cameras(self.job_id, self.config, self.jm, self.sm)

    def _run_reconstruct_coarse(self):
        """Stage: reconstruct_coarse — visual hull / sparse Gaussian init."""
        from .coarse_recon import run_reconstruct_coarse
        run_reconstruct_coarse(self.job_id, self.config, self.jm, self.sm)

    def _run_refine_joint(self):
        """Stage: refine_joint — joint mesh + Gaussian refinement."""
        from .refine import run_refine_joint
        run_refine_joint(self.job_id, self.config, self.jm, self.sm)

    def _run_complete_geometry(self):
        """Stage: complete_geometry — generative completion."""
        from .completion import run_complete_geometry
        run_complete_geometry(self.job_id, self.config, self.jm, self.sm)

    def _run_bake_texture(self):
        """Stage: bake_texture — multi-view texture projection."""
        from .texturing import run_bake_texture
        run_bake_texture(self.job_id, self.config, self.jm, self.sm)

    def _run_export(self):
        """Stage: export — mesh cleanup, decimation, GLB export."""
        from .export import run_export
        run_export(self.job_id, self.config, self.jm, self.sm)

    def _run_qa(self):
        """Stage: qa — quality scoring and diagnostics."""
        from .qa import run_qa
        run_qa(self.job_id, self.config, self.jm, self.sm)

    # Stage name -> handler method mapping
    _stage_handlers = {
        JobStage.INGEST.value: _run_ingest,
        JobStage.PREPROCESS_VIEWS.value: _run_preprocess_views,
        JobStage.VALIDATE_VIEWS.value: _run_validate_views,
        JobStage.INITIALIZE_CAMERAS.value: _run_initialize_cameras,
        JobStage.RECONSTRUCT_COARSE.value: _run_reconstruct_coarse,
        JobStage.REFINE_JOINT.value: _run_refine_joint,
        JobStage.COMPLETE_GEOMETRY.value: _run_complete_geometry,
        JobStage.BAKE_TEXTURE.value: _run_bake_texture,
        JobStage.EXPORT.value: _run_export,
        JobStage.QA.value: _run_qa,
    }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _should_skip_stage(self, stage_name: str) -> bool:
        """
        Check if a stage should be skipped based on pipeline config.

        Note: complete_geometry is NEVER skipped because symmetry and
        laplacian completion providers are always available as CPU
        fallbacks, even when GPU providers (TRELLIS, Hunyuan) are
        disabled.
        """
        if stage_name == JobStage.REFINE_JOINT.value:
            return not self.config.use_joint_refinement

        # complete_geometry always runs -- symmetry + laplacian fallbacks
        # are always available regardless of GPU provider config.

        return False

