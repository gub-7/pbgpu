"""
Job lifecycle management using Redis for state persistence.

Handles:
  - Creating new reconstruction jobs
  - Persisting and retrieving job state
  - Enqueueing jobs for worker processing
  - Status queries and updates
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

import redis

from api.models import (
    CreateJobRequest,
    JobStatus,
    PipelineConfig,
    ReconJob,
    ViewLabel,
    ViewSpec,
)
from api.storage import create_job_storage, get_job_dir
from pipelines.config import REDIS_URL, get_default_pipeline_config

logger = logging.getLogger(__name__)

# Redis key prefixes
JOB_KEY_PREFIX = "recon:job:"
JOB_QUEUE = "recon:queue"


class JobManager:
    """
    Manages reconstruction job lifecycle via Redis.

    Jobs are stored as JSON in Redis hashes and queued for
    worker processing via a Redis list (FIFO queue).
    """

    def __init__(self, redis_url: str = REDIS_URL):
        self.redis = redis.from_url(redis_url, decode_responses=True)

    def _job_key(self, job_id: str) -> str:
        return f"{JOB_KEY_PREFIX}{job_id}"

    def _save_job(self, job: ReconJob) -> None:
        """Persist job state to Redis."""
        key = self._job_key(job.job_id)
        data = job.model_dump_json()
        self.redis.set(key, data)
        # Set TTL of 24 hours for automatic cleanup
        self.redis.expire(key, 86400)

    def _load_job(self, job_id: str) -> Optional[ReconJob]:
        """Load job state from Redis."""
        key = self._job_key(job_id)
        data = self.redis.get(key)
        if data is None:
            return None
        return ReconJob.model_validate_json(data)

    def create_job(
        self,
        config: Optional[PipelineConfig] = None,
    ) -> ReconJob:
        """
        Create a new reconstruction job.

        1. Generate job ID
        2. Create storage directories
        3. Initialize job with canonical views
        4. Persist to Redis

        Parameters
        ----------
        config : optional pipeline configuration override
        """
        # Lazy import to avoid circular dependency:
        #   pipelines.camera_init → api.models → (triggers api.__init__)
        #   → api.job_manager → pipelines.camera_init (still initialising)
        from pipelines.camera_init import get_canonical_views

        cfg = config or get_default_pipeline_config()

        job = ReconJob(config=cfg)
        job_dir = create_job_storage(job.job_id)
        job.storage_dir = str(job_dir)

        # Initialize with canonical view specs (filenames TBD at upload)
        job.views = get_canonical_views()

        self._save_job(job)
        logger.info("Created job %s", job.job_id)
        return job

    def register_upload(
        self,
        job_id: str,
        label: ViewLabel,
        filename: str,
    ) -> ReconJob:
        """
        Register an uploaded image for a specific view.

        Updates the ViewSpec's image_filename for the given label.
        """
        job = self._load_job(job_id)
        if job is None:
            raise ValueError(f"Job not found: {job_id}")

        # Update the matching view spec
        for i, vs in enumerate(job.views):
            if vs.label == label:
                job.views[i] = vs.model_copy(
                    update={"image_filename": filename}
                )
                break
        else:
            raise ValueError(f"View label {label} not found in job")

        job.updated_at = datetime.utcnow()
        self._save_job(job)
        return job

    def enqueue_job(self, job_id: str) -> None:
        """
        Push a job onto the processing queue.

        Workers consume from this queue via BRPOP.
        """
        self.redis.lpush(JOB_QUEUE, job_id)
        logger.info("Enqueued job %s", job_id)

    def dequeue_job(self, timeout: int = 0) -> Optional[str]:
        """
        Pop the next job from the queue (blocking).

        Parameters
        ----------
        timeout : seconds to wait (0 = block forever)

        Returns
        -------
        job_id or None if timeout expired
        """
        result = self.redis.brpop(JOB_QUEUE, timeout=timeout)
        if result is None:
            return None
        _, job_id = result
        return job_id

    def get_job(self, job_id: str) -> Optional[ReconJob]:
        """Retrieve a job by ID."""
        return self._load_job(job_id)

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        error: Optional[str] = None,
    ) -> Optional[ReconJob]:
        """Update a job's status."""
        job = self._load_job(job_id)
        if job is None:
            return None

        job.status = status
        job.updated_at = datetime.utcnow()
        if error:
            job.error_message = error

        self._save_job(job)
        return job

    def save_job(self, job: ReconJob) -> None:
        """
        Save a full job object (used by the orchestrator to persist
        stage results).
        """
        job.updated_at = datetime.utcnow()
        self._save_job(job)

    def list_jobs(self, limit: int = 50) -> list[ReconJob]:
        """List recent jobs (scans Redis keys)."""
        pattern = f"{JOB_KEY_PREFIX}*"
        jobs = []

        for key in self.redis.scan_iter(match=pattern, count=100):
            data = self.redis.get(key)
            if data:
                try:
                    jobs.append(ReconJob.model_validate_json(data))
                except Exception:
                    logger.warning("Failed to parse job: %s", key)

        # Sort by creation time, newest first
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def delete_job(self, job_id: str) -> bool:
        """Delete a job from Redis."""
        key = self._job_key(job_id)
        return self.redis.delete(key) > 0

