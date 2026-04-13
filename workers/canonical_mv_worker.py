"""
Worker process for the canonical 3-view reconstruction pipeline.

Consumes jobs from the Redis queue and runs the full pipeline:
  1. Preprocessing
  2. Camera initialisation
  3. Coarse reconstruction (DUSt3R/MASt3R/VGGT) using full images
  4. Subject isolation (background removal + 3D filtering)
  5. Trellis.2 generative completion

Usage:
  python -m workers.canonical_mv_worker

Environment variables:
  REDIS_URL          – Redis connection string (default: redis://localhost:6379/0)
  CUDA_DEVICE        – CUDA device to use (default: cuda)
  WORKER_TIMEOUT     – Max seconds per job (default: 1800)
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.job_manager import JobManager
from api.models import ReconJob
from pipelines.config import REDIS_URL, ensure_directories
from pipelines.orchestrator import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("canonical_mv_worker")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CUDA_DEVICE = os.getenv("CUDA_DEVICE", "cuda")
WORKER_TIMEOUT = int(os.getenv("WORKER_TIMEOUT", "1800"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))

# Graceful shutdown flag
_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    logger.info("Received signal %d, shutting down gracefully...", signum)
    _shutdown = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# ---------------------------------------------------------------------------
# Status callback
# ---------------------------------------------------------------------------


def make_status_callback(jm: JobManager):
    """Create a callback that persists job state to Redis on each status change."""

    def callback(job: ReconJob) -> None:
        try:
            jm.save_job(job)
        except Exception:
            logger.exception("Failed to persist job status update")

    return callback


# ---------------------------------------------------------------------------
# Main worker loop
# ---------------------------------------------------------------------------


def process_job(jm: JobManager, job_id: str) -> None:
    """Process a single reconstruction job."""
    logger.info("Processing job %s", job_id)

    job = jm.get_job(job_id)
    if job is None:
        logger.error("Job %s not found in Redis, skipping", job_id)
        return

    callback = make_status_callback(jm)

    try:
        result = run_pipeline(
            job=job,
            on_status_change=callback,
            device=CUDA_DEVICE,
        )
        logger.info(
            "Job %s completed with status: %s",
            job_id,
            result.status.value,
        )
    except Exception:
        logger.exception("Unhandled error processing job %s", job_id)
        jm.update_status(job_id, status=job.status.FAILED, error="Worker crash")


def main() -> None:
    """Main worker entry point."""
    ensure_directories()

    logger.info("Canonical MV worker starting (device=%s)", CUDA_DEVICE)
    logger.info("Redis: %s", REDIS_URL)

    jm = JobManager(redis_url=REDIS_URL)

    # Verify Redis connectivity
    try:
        jm.redis.ping()
        logger.info("Redis connection OK")
    except Exception:
        logger.error("Cannot connect to Redis at %s", REDIS_URL)
        sys.exit(1)

    logger.info("Waiting for jobs...")

    while not _shutdown:
        try:
            job_id = jm.dequeue_job(timeout=POLL_INTERVAL)
            if job_id is not None:
                process_job(jm, job_id)
        except KeyboardInterrupt:
            break
        except Exception:
            logger.exception("Error in worker loop")
            time.sleep(POLL_INTERVAL)

    logger.info("Worker shutting down")


if __name__ == "__main__":
    main()

