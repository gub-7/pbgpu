"""
Job queue and state management for 3D reconstruction tasks.

Uses Redis for job queue and file-based storage for job metadata.
Supports both single-view (legacy) and multi-view canonical 5-view
reconstruction pipelines.
"""

import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import redis

from .models import (
    JobStatus,
    ModelEnum,
    CategoryEnum,
    PipelineEnum,
    JobStage,
    StageStatus,
    ViewStatus,
    MV_STAGE_ORDER,
    REQUIRED_VIEWS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MV_PIPELINES = frozenset({
    PipelineEnum.CANONICAL_MV_HYBRID.value,
    PipelineEnum.CANONICAL_MV_FAST.value,
    PipelineEnum.CANONICAL_MV_GENERATIVE.value,
})


def _is_multiview_pipeline(pipeline: Optional[str]) -> bool:
    """Return True if *pipeline* is one of the multi-view pipelines."""
    return pipeline in _MV_PIPELINES


def _build_stage_dict(stage_order: List[str]) -> Dict[str, str]:
    """Build an initial stage-status dict with every stage set to PENDING."""
    return {stage: StageStatus.PENDING.value for stage in stage_order}


def _build_views_dict(view_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """Build an initial per-view metadata dict."""
    return {
        vn: {
            "filename": f"{vn}.png",
            "status": ViewStatus.PENDING.value,
            "width": None,
            "height": None,
            "file_size": None,
            "segmentation_confidence": None,
            "sharpness_score": None,
            "warnings": [],
        }
        for vn in view_names
    }


def _current_stage(stages: Dict[str, str]) -> Optional[str]:
    """
    Return the name of the first stage that is IN_PROGRESS, or the first
    that is still PENDING (i.e. the next stage to run), or None if all
    are completed/skipped/failed.
    """
    for stage, status in stages.items():
        if status == StageStatus.IN_PROGRESS.value:
            return stage
    for stage, status in stages.items():
        if status == StageStatus.PENDING.value:
            return stage
    return None


def _overall_progress(stages: Dict[str, str]) -> int:
    """
    Compute an overall progress percentage (0-100) from stage statuses.
    Each completed/skipped stage counts equally.
    """
    total = len(stages)
    if total == 0:
        return 0
    done = sum(
        1 for s in stages.values()
        if s in (StageStatus.COMPLETED.value, StageStatus.SKIPPED.value)
    )
    return int(round(done / total * 100))


def _available_artifacts(stages: Dict[str, str]) -> List[str]:
    """
    Return a list of stage names that have completed and therefore
    are expected to have artifacts available.
    """
    return [
        stage for stage, status in stages.items()
        if status == StageStatus.COMPLETED.value
    ]


class JobManager:
    """Manages job lifecycle, queue, and state persistence."""

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        storage_root: str = "storage",
    ):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True,
        )
        self.storage_root = Path(storage_root)
        self.jobs_dir = self.storage_root / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Job creation
    # ------------------------------------------------------------------

    def create_job(
        self,
        category: CategoryEnum,
        model: ModelEnum,
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new single-view job (backward-compatible).

        Returns:
            Job ID (UUID).
        """
        job_id = str(uuid.uuid4())

        job_data: Dict[str, Any] = {
            "job_id": job_id,
            "category": category.value,
            "model": model.value,
            "pipeline": None,
            "status": JobStatus.QUEUED.value,
            "progress": 0,
            "current_stage": None,
            "stage_progress": None,
            "stages": {
                "preprocessing": StageStatus.PENDING.value,
                "generation": StageStatus.PENDING.value,
            },
            "views": None,
            "warnings": [],
            "artifacts_available": [],
            "params": params or {},
            "created_at": datetime.utcnow().isoformat(),
            "started_at": None,
            "completed_at": None,
            "error": None,
            "output_url": None,
        }

        self._save_job_metadata(job_id, job_data)
        self.redis_client.setex(f"job:{job_id}", 86400, json.dumps(job_data))
        return job_id

    def create_multiview_job(
        self,
        category: CategoryEnum,
        pipeline: PipelineEnum,
        views_received: List[str],
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new multi-view job.

        Args:
            category: Category type.
            pipeline: Multi-view pipeline to use.
            views_received: List of canonical view names that were uploaded.
            params: CanonicalMVParams as dict.

        Returns:
            Job ID (UUID).
        """
        job_id = str(uuid.uuid4())

        stages = _build_stage_dict(MV_STAGE_ORDER)
        views = _build_views_dict(views_received)

        for vn in views_received:
            if vn in views:
                views[vn]["status"] = ViewStatus.UPLOADED.value

        job_data: Dict[str, Any] = {
            "job_id": job_id,
            "category": category.value,
            "model": None,
            "pipeline": pipeline.value,
            "status": JobStatus.QUEUED.value,
            "progress": 0,
            "current_stage": JobStage.INGEST.value,
            "stage_progress": None,
            "stages": stages,
            "views": views,
            "warnings": [],
            "artifacts_available": [],
            "params": params or {},
            "created_at": datetime.utcnow().isoformat(),
            "started_at": None,
            "completed_at": None,
            "error": None,
            "output_url": None,
        }

        self._save_job_metadata(job_id, job_data)
        self.redis_client.setex(f"job:{job_id}", 86400, json.dumps(job_data))
        return job_id

    # ------------------------------------------------------------------
    # Job read / update
    # ------------------------------------------------------------------

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job metadata (Redis first, then file fallback)."""
        job_json = self.redis_client.get(f"job:{job_id}")
        if job_json:
            return json.loads(job_json)
        return self._load_job_metadata(job_id)

    def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        progress: Optional[int] = None,
        stage_updates: Optional[Dict[str, str]] = None,
        error: Optional[str] = None,
        output_url: Optional[str] = None,
        current_stage: Optional[str] = None,
        stage_progress: Optional[float] = None,
        warnings: Optional[List[str]] = None,
        view_updates: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Update job metadata.

        Accepts both legacy single-view fields and new multi-view fields.
        """
        job_data = self.get_job(job_id)
        if not job_data:
            raise ValueError(f"Job {job_id} not found")

        # Lifecycle status transitions
        if status:
            job_data["status"] = status.value

            if status == JobStatus.PREPROCESSING:
                job_data["started_at"] = datetime.utcnow().isoformat()
                if "preprocessing" in job_data["stages"]:
                    job_data["stages"]["preprocessing"] = StageStatus.IN_PROGRESS.value
            elif status == JobStatus.GENERATING:
                if "preprocessing" in job_data["stages"]:
                    job_data["stages"]["preprocessing"] = StageStatus.COMPLETED.value
                if "generation" in job_data["stages"]:
                    job_data["stages"]["generation"] = StageStatus.IN_PROGRESS.value
            elif status == JobStatus.COMPLETED:
                if "generation" in job_data["stages"]:
                    job_data["stages"]["generation"] = StageStatus.COMPLETED.value
                job_data["completed_at"] = datetime.utcnow().isoformat()
            elif status == JobStatus.FAILED:
                job_data["completed_at"] = datetime.utcnow().isoformat()

        if progress is not None:
            job_data["progress"] = progress

        if stage_updates:
            job_data["stages"].update(stage_updates)

        if error:
            job_data["error"] = error

        if output_url:
            job_data["output_url"] = output_url

        if current_stage is not None:
            job_data["current_stage"] = current_stage

        if stage_progress is not None:
            job_data["stage_progress"] = stage_progress

        if warnings is not None:
            existing = job_data.get("warnings", [])
            # Extend existing warnings, avoiding duplicates
            for w in warnings:
                if w not in existing:
                    existing.append(w)
            job_data["warnings"] = existing

        # Per-view metadata patches
        if view_updates and job_data.get("views"):
            for vn, patch in view_updates.items():
                if vn in job_data["views"]:
                    job_data["views"][vn].update(patch)

        # Recompute derived fields
        job_data["artifacts_available"] = _available_artifacts(job_data["stages"])
        if job_data.get("pipeline") and _is_multiview_pipeline(job_data["pipeline"]):
            job_data["current_stage"] = (
                current_stage
                if current_stage is not None
                else _current_stage(job_data["stages"])
            )
            if progress is None:
                job_data["progress"] = _overall_progress(job_data["stages"])

        # Persist
        self.redis_client.setex(f"job:{job_id}", 86400, json.dumps(job_data))
        self._save_job_metadata(job_id, job_data)

    # ------------------------------------------------------------------
    # Stage-level helpers (multi-view)
    # ------------------------------------------------------------------

    def advance_stage(
        self,
        job_id: str,
        completed_stage: str,
        *,
        next_stage: Optional[str] = None,
    ):
        """
        Mark *completed_stage* as COMPLETED, optionally set *next_stage*
        to IN_PROGRESS.  Recomputes progress automatically.
        """
        job_data = self.get_job(job_id)
        if not job_data:
            raise ValueError(f"Job {job_id} not found")

        stages = job_data["stages"]
        stages[completed_stage] = StageStatus.COMPLETED.value

        if next_stage is None:
            for s in MV_STAGE_ORDER:
                if stages.get(s) == StageStatus.PENDING.value:
                    next_stage = s
                    break

        if next_stage:
            stages[next_stage] = StageStatus.IN_PROGRESS.value

        self.update_job(
            job_id,
            stage_updates=stages,
            current_stage=next_stage,
            stage_progress=0.0,
        )

    def fail_stage(self, job_id: str, stage: str, error: str):
        """Mark a stage as FAILED and set the overall job to FAILED."""
        self.update_job(
            job_id,
            status=JobStatus.FAILED,
            stage_updates={stage: StageStatus.FAILED.value},
            current_stage=stage,
            error=error,
        )

    def skip_stage(self, job_id: str, stage: str):
        """Mark a stage as SKIPPED (e.g. when params disable it)."""
        self.update_job(
            job_id,
            stage_updates={stage: StageStatus.SKIPPED.value},
        )

    # ------------------------------------------------------------------
    # Queuing
    # ------------------------------------------------------------------

    def queue_job_for_generation(self, job_id: str):
        """
        Add job to the appropriate Redis queue for worker processing.

        Single-view jobs go to ``queue:{model}``.
        Multi-view jobs go to ``queue:{pipeline}``.
        """
        job_data = self.get_job(job_id)
        if not job_data:
            raise ValueError(f"Job {job_id} not found")

        pipeline = job_data.get("pipeline")
        if pipeline and _is_multiview_pipeline(pipeline):
            queue_name = f"queue:{pipeline}"
        else:
            model = job_data["model"]
            queue_name = f"queue:{model}"

        self.redis_client.rpush(queue_name, job_id)

    def get_next_job(self, model: ModelEnum) -> Optional[str]:
        """Get next single-view job from queue."""
        queue_name = f"queue:{model.value}"
        return self.redis_client.lpop(queue_name)

    def get_next_multiview_job(self, pipeline: PipelineEnum) -> Optional[str]:
        """Get next multi-view job from queue."""
        queue_name = f"queue:{pipeline.value}"
        return self.redis_client.lpop(queue_name)

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete_job(self, job_id: str):
        """Delete job metadata and associated files."""
        self.redis_client.delete(f"job:{job_id}")

        metadata_file = self.jobs_dir / f"{job_id}.json"
        if metadata_file.exists():
            metadata_file.unlink()

        from .storage import StorageManager
        storage = StorageManager()
        storage.cleanup_job(job_id)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_queue_length(self, model: ModelEnum) -> int:
        """Get current queue length for a single-view model."""
        queue_name = f"queue:{model.value}"
        return self.redis_client.llen(queue_name)

    def get_multiview_queue_length(self, pipeline: PipelineEnum) -> int:
        """Get current queue length for a multi-view pipeline."""
        queue_name = f"queue:{pipeline.value}"
        return self.redis_client.llen(queue_name)

    def get_active_jobs(self, model: Optional[ModelEnum] = None) -> list:
        """Get list of active (non-terminal) jobs, optionally filtered by model."""
        active_jobs = []
        for metadata_file in self.jobs_dir.glob("*.json"):
            with open(metadata_file, "r") as f:
                job_data = json.load(f)
            status = JobStatus(job_data["status"])
            if status in (JobStatus.QUEUED, JobStatus.PREPROCESSING, JobStatus.GENERATING):
                if model is None or job_data.get("model") == model.value:
                    active_jobs.append(job_data)
        return active_jobs

    def get_active_multiview_jobs(
        self, pipeline: Optional[PipelineEnum] = None
    ) -> list:
        """Get list of active multi-view jobs, optionally filtered by pipeline."""
        active_jobs = []
        for metadata_file in self.jobs_dir.glob("*.json"):
            with open(metadata_file, "r") as f:
                job_data = json.load(f)
            if not _is_multiview_pipeline(job_data.get("pipeline")):
                continue
            status = JobStatus(job_data["status"])
            if status in (JobStatus.QUEUED, JobStatus.PREPROCESSING, JobStatus.GENERATING):
                if pipeline is None or job_data.get("pipeline") == pipeline.value:
                    active_jobs.append(job_data)
        return active_jobs

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_job_metadata(self, job_id: str, job_data: Dict[str, Any]):
        """Save job metadata to file."""
        metadata_file = self.jobs_dir / f"{job_id}.json"
        with open(metadata_file, "w") as f:
            json.dump(job_data, f, indent=2)

    def _load_job_metadata(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Load job metadata from file."""
        metadata_file = self.jobs_dir / f"{job_id}.json"
        if not metadata_file.exists():
            return None
        with open(metadata_file, "r") as f:
            return json.load(f)

