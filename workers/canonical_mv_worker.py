"""
Worker process for the canonical multi-view reconstruction pipeline.

Polls the Redis queue for multi-view jobs and runs them through
the CanonicalMVOrchestrator.

Usage:
    python -m workers.canonical_mv_worker
"""

import logging
import time
import sys

from api.job_manager import JobManager
from api.models import JobStatus, PipelineEnum
from api.storage import StorageManager
from pipelines.canonical_mv import CanonicalMVOrchestrator

logger = logging.getLogger(__name__)

# Pipelines this worker handles
_HANDLED_PIPELINES = [
    PipelineEnum.CANONICAL_MV_HYBRID,
    PipelineEnum.CANONICAL_MV_FAST,
    PipelineEnum.CANONICAL_MV_GENERATIVE,
]


class CanonicalMVWorker:
    """
    Worker that polls Redis queues for canonical multi-view jobs
    and processes them through the full pipeline.
    """

    def __init__(self):
        self.jm = JobManager()
        self.sm = StorageManager()

    def run(self, poll_interval: float = 2.0):
        """
        Main loop: poll queues and process jobs.

        Args:
            poll_interval: Seconds to sleep between poll cycles.
        """
        logger.info("Canonical MV worker starting...")

        while True:
            job_id = self._poll_queues()
            if job_id:
                self._process_job(job_id)
            else:
                time.sleep(poll_interval)

    def _poll_queues(self):
        """Check all handled pipeline queues for a job."""
        for pipeline in _HANDLED_PIPELINES:
            job_id = self.jm.get_next_multiview_job(pipeline)
            if job_id:
                logger.info(
                    f"Picked up job {job_id} from queue:{pipeline.value}"
                )
                return job_id
        return None

    def _process_job(self, job_id: str):
        """Run a single job through the orchestrator."""
        try:
            orchestrator = CanonicalMVOrchestrator(
                job_id=job_id,
                job_manager=self.jm,
                storage_manager=self.sm,
            )
            success = orchestrator.run()

            if success:
                logger.info(f"Job {job_id} completed successfully")
            else:
                logger.warning(f"Job {job_id} failed (see job metadata for details)")

        except Exception as e:
            logger.error(f"Job {job_id} crashed: {e}", exc_info=True)
            try:
                self.jm.update_job(
                    job_id,
                    status=JobStatus.FAILED,
                    error=f"Worker crash: {str(e)}",
                )
            except Exception:
                logger.error(f"Failed to mark job {job_id} as failed")


def main():
    """Entry point for the worker process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    worker = CanonicalMVWorker()
    worker.run()


if __name__ == "__main__":
    main()

