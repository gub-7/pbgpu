"""
FastAPI application for the GPU cluster multi-view reconstruction service.

Endpoints:
  POST   /jobs                    – Create a new reconstruction job
  POST   /jobs/{id}/upload/{view} – Upload an image for a specific view
  POST   /jobs/{id}/start         – Start processing
  GET    /jobs/{id}               – Get job status and details
  GET    /jobs/{id}/artifacts     – List available artifacts
  GET    /jobs/{id}/artifacts/{path} – Download an artifact
  DELETE /jobs/{id}               – Delete a job
  GET    /jobs                    – List recent jobs
  GET    /health                  – Health check
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from api.job_manager import JobManager
from api.models import (
    CreateJobRequest,
    JobDetailResponse,
    JobStatusResponse,
    ViewLabel,
)
from api.storage import (
    delete_job_storage,
    get_artifact_path,
    list_artifacts,
    save_upload,
)
from pipelines.config import ensure_directories

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GPU Cluster – Multi-View 3D Reconstruction",
    description=(
        "3-image canonical multi-view reconstruction pipeline. "
        "Stages: preprocessing → camera init → coarse recon (DUSt3R/MASt3R) "
        "→ subject isolation → Trellis.2 completion."
    ),
    version="2.0.0",
)

# Lazy-initialised singleton (created on first request)
_job_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    global _job_manager
    if _job_manager is None:
        ensure_directories()
        _job_manager = JobManager()
    return _job_manager


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
async def health_check():
    """Service health check."""
    try:
        jm = get_job_manager()
        jm.redis.ping()
        redis_ok = True
    except Exception:
        redis_ok = False

    return {
        "status": "ok" if redis_ok else "degraded",
        "redis": redis_ok,
        "version": "2.0.0",
        "pipeline": "canonical_3view",
    }


# ---------------------------------------------------------------------------
# Job CRUD
# ---------------------------------------------------------------------------


@app.post("/jobs", response_model=JobStatusResponse, status_code=201)
async def create_job(request: CreateJobRequest = CreateJobRequest()):
    """
    Create a new reconstruction job.

    Returns the job ID and initial status. After creation, upload
    the three view images (front, side, top) then call /start.
    """
    jm = get_job_manager()
    job = jm.create_job(config=request.config)

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@app.post("/jobs/{job_id}/upload/{view_label}")
async def upload_view(
    job_id: str,
    view_label: ViewLabel,
    file: UploadFile = File(...),
):
    """
    Upload an image for a specific view (front, side, or top).

    The image will be validated and preprocessed when the job starts.
    """
    jm = get_job_manager()
    job = jm.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if file.filename is None:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Read and save the upload
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    # Use a standardised filename: {view_label}.{ext}
    ext = file.filename.rsplit(".", 1)[-1] if "." in file.filename else "png"
    filename = f"{view_label.value}.{ext}"

    save_upload(job_id, filename, data)

    # Register the upload in the job
    jm.register_upload(job_id, view_label, filename)

    return {
        "status": "uploaded",
        "view": view_label.value,
        "filename": filename,
        "size_bytes": len(data),
    }


@app.post("/jobs/{job_id}/start")
async def start_job(job_id: str):
    """
    Start processing a job.

    All three views must be uploaded before calling this endpoint.
    The job is enqueued for worker processing.
    """
    jm = get_job_manager()
    job = jm.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != "pending":
        raise HTTPException(
            status_code=409,
            detail=f"Job is already in state: {job.status}",
        )

    # Verify all views have been uploaded
    for vs in job.views:
        path = get_artifact_path(job_id, f"input/{vs.image_filename}")
        if path is None:
            raise HTTPException(
                status_code=400,
                detail=f"View '{vs.label.value}' has not been uploaded yet",
            )

    # Enqueue for processing
    jm.enqueue_job(job_id)

    return {"status": "queued", "job_id": job_id}


@app.get("/jobs/{job_id}", response_model=JobDetailResponse)
async def get_job(job_id: str):
    """Get full job details including stage results."""
    jm = get_job_manager()
    job = jm.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobDetailResponse(
        job_id=job.job_id,
        status=job.status,
        config=job.config,
        views=job.views,
        resolved_views=job.resolved_views,
        coarse_result=job.coarse_result,
        isolation_result=job.isolation_result,
        trellis_result=job.trellis_result,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@app.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get lightweight job status."""
    jm = get_job_manager()
    job = jm.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@app.get("/jobs/{job_id}/artifacts")
async def list_job_artifacts(job_id: str, subdir: str = ""):
    """List available artifacts for a job."""
    jm = get_job_manager()
    job = jm.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    artifacts = list_artifacts(job_id, subdir)
    return {"job_id": job_id, "artifacts": artifacts}


@app.head("/jobs/{job_id}/artifacts/{artifact_path:path}")
async def head_artifact(job_id: str, artifact_path: str):
    """Check whether a specific artifact file exists (HEAD only)."""
    path = get_artifact_path(job_id, artifact_path)
    if path is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return JSONResponse(content=None, status_code=200)


@app.get("/jobs/{job_id}/artifacts/{artifact_path:path}")
async def download_artifact(job_id: str, artifact_path: str):
    """Download a specific artifact file."""
    path = get_artifact_path(job_id, artifact_path)
    if path is None:
        raise HTTPException(status_code=404, detail="Artifact not found")

    return FileResponse(
        path=str(path),
        filename=path.name,
    )


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and all its data."""
    jm = get_job_manager()
    job = jm.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    jm.delete_job(job_id)
    delete_job_storage(job_id)

    return {"status": "deleted", "job_id": job_id}


@app.get("/jobs")
async def list_jobs(limit: int = 50):
    """List recent jobs."""
    jm = get_job_manager()
    jobs = jm.list_jobs(limit=limit)

    return {
        "jobs": [
            JobStatusResponse(
                job_id=j.job_id,
                status=j.status,
                error_message=j.error_message,
                created_at=j.created_at,
                updated_at=j.updated_at,
            ).model_dump()
            for j in jobs
        ]
    }

