"""
File storage layer for the GPU cluster API.

Manages job directories, image uploads, and artifact retrieval.
All job data lives under STORAGE_ROOT/<job_id>/.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

from pipelines.config import STORAGE_ROOT

logger = logging.getLogger(__name__)


class StorageError(Exception):
    """Raised on storage operation failures."""


def get_job_dir(job_id: str) -> Path:
    """Return the root directory for a job."""
    return STORAGE_ROOT / job_id


def create_job_storage(job_id: str) -> Path:
    """
    Create the directory structure for a new job.

    Layout:
      <job_id>/
        input/          – raw uploaded images
        preprocessed/   – resized/normalised images
        coarse_recon/   – point clouds from geometry stage
        isolation/      – masks and masked images
          masks/
          masked_images/
        trellis/        – Trellis.2 output
        colmap/         – COLMAP-format camera export
          sparse/0/
        export/         – final packaged output
    """
    job_dir = get_job_dir(job_id)

    if job_dir.exists():
        raise StorageError(f"Job directory already exists: {job_dir}")

    subdirs = [
        "input",
        "preprocessed",
        "coarse_recon",
        "isolation/masks",
        "isolation/masked_images",
        "trellis",
        "colmap/sparse/0",
        "export",
    ]

    for sub in subdirs:
        (job_dir / sub).mkdir(parents=True, exist_ok=True)

    logger.info("Created job storage: %s", job_dir)
    return job_dir


def save_upload(job_id: str, filename: str, data: bytes) -> Path:
    """
    Save an uploaded file to the job's input directory.

    Returns the path to the saved file.
    """
    job_dir = get_job_dir(job_id)
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    # Sanitise filename
    safe_name = Path(filename).name
    dst = input_dir / safe_name

    dst.write_bytes(data)
    logger.info("Saved upload: %s (%d bytes)", dst, len(data))
    return dst


def get_artifact_path(job_id: str, relative_path: str) -> Optional[Path]:
    """
    Resolve a relative path within a job's storage.

    Returns None if the file doesn't exist.
    """
    job_dir = get_job_dir(job_id)
    full_path = job_dir / relative_path

    # Security: prevent path traversal
    try:
        full_path.resolve().relative_to(job_dir.resolve())
    except ValueError:
        logger.warning("Path traversal attempt: %s", relative_path)
        return None

    if full_path.exists():
        return full_path
    return None


def list_artifacts(job_id: str, subdir: str = "") -> list[str]:
    """List files in a job subdirectory, returning relative paths."""
    job_dir = get_job_dir(job_id)
    target = job_dir / subdir if subdir else job_dir

    if not target.exists():
        return []

    results = []
    for p in sorted(target.rglob("*")):
        if p.is_file():
            results.append(str(p.relative_to(job_dir)))

    return results


def delete_job_storage(job_id: str) -> bool:
    """Delete all storage for a job. Returns True if deleted."""
    job_dir = get_job_dir(job_id)
    if job_dir.exists():
        shutil.rmtree(job_dir)
        logger.info("Deleted job storage: %s", job_dir)
        return True
    return False


def get_storage_usage(job_id: str) -> int:
    """Return total bytes used by a job's storage."""
    job_dir = get_job_dir(job_id)
    if not job_dir.exists():
        return 0

    total = 0
    for p in job_dir.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total

