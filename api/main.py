"""
FastAPI application for GPU-cluster 3D reconstruction service.

Supports both single-view (TripoSR / Trellis2) and multi-view
canonical 3-view reconstruction pipelines.
"""

import asyncio
import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .job_manager import JobManager
from .models import (
    ArtifactInfo,
    ArtifactListResponse,
    CanonicalMVParams,
    CategoryEnum,
    CategoriesResponse,
    CategoryInfo,
    ErrorResponse,
    JobStatus,
    JobStatusResponse,
    MetricsResponse,
    ModelEnum,
    MultiViewUploadResponse,
    PipelineEnum,
    PreviewResponse,
    PreviewStage,
    MultiViewPreviewResponse,
    QAWarning,
    REQUIRED_VIEWS,
    StageStatus,
    UploadRequest,
    UploadResponse,
    ViewName,
)
from .storage import StorageManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GPU Cluster 3D Reconstruction API",
    description="API for 3D model reconstruction from single or multi-view images",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service instances
job_manager = JobManager(
    redis_host=os.environ.get("REDIS_HOST", "localhost"),
    redis_port=int(os.environ.get("REDIS_PORT", "6379")),
    storage_root=os.environ.get("STORAGE_ROOT", "storage"),
)
storage_manager = StorageManager(
    storage_root=os.environ.get("STORAGE_ROOT", "storage"),
)

# ---------------------------------------------------------------------------
# Category guidance
# ---------------------------------------------------------------------------

CATEGORY_GUIDANCE = {
    "human_full_body": CategoryInfo(
        category="human_full_body",
        display_name="Human Full Body",
        description="Full body human figure reconstruction",
        best_model="triposr",
        foreground_ratio=0.85,
        texture_resolution=2048,
        expected_time_triposr="30-60s",
        expected_time_trellis2="60-120s",
        tips=[
            "Ensure the full body is visible",
            "Avoid extreme poses",
            "Good lighting on face and body",
        ],
    ),
    "animal_full_body": CategoryInfo(
        category="animal_full_body",
        display_name="Animal Full Body",
        description="Full body animal reconstruction",
        best_model="triposr",
        foreground_ratio=0.85,
        texture_resolution=2048,
        expected_time_triposr="30-60s",
        expected_time_trellis2="60-120s",
        tips=[
            "Capture the animal in a neutral pose",
            "Ensure good contrast with background",
        ],
    ),
    "human_bust": CategoryInfo(
        category="human_bust",
        display_name="Human Bust",
        description="Human head and shoulders reconstruction",
        best_model="trellis2",
        foreground_ratio=0.90,
        texture_resolution=2048,
        expected_time_triposr="20-40s",
        expected_time_trellis2="45-90s",
        tips=[
            "Frame from mid-chest up",
            "Ensure even lighting on face",
        ],
    ),
    "animal_bust": CategoryInfo(
        category="animal_bust",
        display_name="Animal Bust",
        description="Animal head and shoulders reconstruction",
        best_model="trellis2",
        foreground_ratio=0.90,
        texture_resolution=2048,
        expected_time_triposr="20-40s",
        expected_time_trellis2="45-90s",
        tips=[
            "Frame from mid-body up",
            "Avoid motion blur",
        ],
    ),
    "generic_object": CategoryInfo(
        category="generic_object",
        display_name="Generic Object",
        description="General object reconstruction",
        best_model="triposr",
        foreground_ratio=0.85,
        texture_resolution=2048,
        expected_time_triposr="20-45s",
        expected_time_trellis2="45-90s",
        tips=[
            "Place object on a plain background",
            "Ensure even lighting",
        ],
    ),
}


# ---------------------------------------------------------------------------
# Background preprocessing trigger (single-view legacy)
# ---------------------------------------------------------------------------


async def trigger_preprocessing(
    job_id: str, category: CategoryEnum, model: ModelEnum
):
    """Trigger preprocessing for a single-view job in the background."""
    try:
        job_manager.update_job(job_id, status=JobStatus.PREPROCESSING, progress=5)

        upload_dir = storage_manager.get_job_upload_dir(job_id)
        image_files = list(upload_dir.glob("*"))
        if not image_files:
            raise FileNotFoundError("No uploaded image found")

        image_path = str(image_files[0])

        from preprocessing import PreprocessingPipeline
        from PIL import Image

        pipeline = PreprocessingPipeline(job_id)
        input_image = Image.open(image_path)

        job_data = job_manager.get_job(job_id)
        params = job_data.get("params", {}) if job_data else {}
        use_triposr = params.get("use_triposr_optimization", True)
        preprocess_overrides = params.get("preprocess_overrides", None)

        result = pipeline.process(
            input_image,
            category.value,
            use_triposr_optimization=use_triposr,
            triposr_preprocess_overrides=preprocess_overrides,
        )

        job_manager.update_job(job_id, progress=30)
        job_manager.queue_job_for_generation(job_id)
        job_manager.update_job(job_id, status=JobStatus.GENERATING, progress=35)

        logger.info(f"Job {job_id}: preprocessing complete, queued for {model.value}")

    except Exception as e:
        logger.error(f"Preprocessing failed for job {job_id}: {e}")
        job_manager.update_job(
            job_id,
            status=JobStatus.FAILED,
            error=f"Preprocessing failed: {str(e)}",
        )


# ===================================================================
# ENDPOINTS: Single-view (legacy)
# ===================================================================


@app.post("/api/upload", response_model=UploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    category: str = Form(...),
    model: str = Form("triposr"),
    params: str = Form("{}"),
):
    """Upload a single image for 3D reconstruction (legacy endpoint)."""
    try:
        cat = CategoryEnum(category)
    except ValueError:
        raise HTTPException(400, f"Invalid category: {category}")
    try:
        mdl = ModelEnum(model)
    except ValueError:
        raise HTTPException(400, f"Invalid model: {model}")
    try:
        params_dict = json.loads(params)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid params JSON")

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 50MB)")

    job_id = job_manager.create_job(cat, mdl, params_dict)
    storage_manager.save_upload(job_id, file.filename or "input.png", content)
    asyncio.create_task(trigger_preprocessing(job_id, cat, mdl))

    return UploadResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        created_at=datetime.utcnow(),
    )


# ===================================================================
# ENDPOINTS: Multi-view canonical 3-view upload
# ===================================================================


@app.post("/api/upload_multiview", response_model=MultiViewUploadResponse)
async def upload_multiview(
    front: UploadFile = File(...),
    side: UploadFile = File(...),
    top: UploadFile = File(...),
    category: str = Form(...),
    pipeline: str = Form("canonical_mv_hybrid"),
    params: str = Form("{}"),
):
    """
    Upload 3 canonical view images for multi-view 3D reconstruction.

    Requires: front, side, top images.
    """
    try:
        cat = CategoryEnum(category)
    except ValueError:
        raise HTTPException(400, f"Invalid category: {category}")
    try:
        pipe = PipelineEnum(pipeline)
    except ValueError:
        raise HTTPException(400, f"Invalid pipeline: {pipeline}")

    mv_pipelines = {
        PipelineEnum.CANONICAL_MV_HYBRID,
        PipelineEnum.CANONICAL_MV_FAST,
        PipelineEnum.CANONICAL_MV_GENERATIVE,
    }
    if pipe not in mv_pipelines:
        raise HTTPException(
            400,
            f"Pipeline '{pipeline}' is not a multi-view pipeline. "
            f"Use one of: {[p.value for p in mv_pipelines]}",
        )

    try:
        params_dict = json.loads(params)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid params JSON")

    try:
        mv_params = CanonicalMVParams(**params_dict)
        params_dict = mv_params.model_dump()
    except Exception as e:
        raise HTTPException(400, f"Invalid multi-view params: {e}")

    view_files = {
        ViewName.FRONT.value: front,
        ViewName.SIDE.value: side,
        ViewName.TOP.value: top,
    }

    max_size = 50 * 1024 * 1024
    views_received = []

    job_id = job_manager.create_multiview_job(
        category=cat,
        pipeline=pipe,
        views_received=list(view_files.keys()),
        params=params_dict,
    )

    try:
        for view_name, upload_file in view_files.items():
            content = await upload_file.read()
            if len(content) > max_size:
                raise HTTPException(400, f"View '{view_name}' file too large (max 50MB)")
            if len(content) == 0:
                raise HTTPException(400, f"View '{view_name}' file is empty")

            original_name = upload_file.filename or f"{view_name}.png"
            ext = Path(original_name).suffix.lower()
            if ext not in (".png", ".jpg", ".jpeg", ".webp"):
                ext = ".png"

            storage_manager.save_view_upload(job_id, view_name, content, ext)
            views_received.append(view_name)

            job_manager.update_job(
                job_id,
                view_updates={
                    view_name: {
                        "status": "uploaded",
                        "filename": f"{view_name}{ext}",
                        "file_size": len(content),
                    }
                },
            )

    except HTTPException:
        job_manager.delete_job(job_id)
        raise
    except Exception as e:
        job_manager.delete_job(job_id)
        raise HTTPException(500, f"Failed to save views: {e}")

    job_manager.queue_job_for_generation(job_id)

    logger.info(
        f"Multi-view job {job_id} created: pipeline={pipe.value}, "
        f"views={views_received}"
    )

    return MultiViewUploadResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        pipeline=pipe,
        views_received=views_received,
        created_at=datetime.utcnow(),
    )


# ===================================================================
# ENDPOINTS: Job status & management
# ===================================================================


@app.get("/api/job/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get job status, progress, stages, and available artifacts."""
    job_data = job_manager.get_job(job_id)
    if not job_data:
        raise HTTPException(404, "Job not found")

    return JobStatusResponse(
        job_id=job_data["job_id"],
        status=JobStatus(job_data["status"]),
        model=ModelEnum(job_data["model"]) if job_data.get("model") else None,
        category=CategoryEnum(job_data["category"]),
        pipeline=PipelineEnum(job_data["pipeline"]) if job_data.get("pipeline") else None,
        progress=job_data.get("progress", 0),
        current_stage=job_data.get("current_stage"),
        stage_progress=job_data.get("stage_progress"),
        stages=job_data.get("stages", {}),
        views=job_data.get("views"),
        warnings=job_data.get("warnings", []),
        artifacts_available=job_data.get("artifacts_available", []),
        created_at=datetime.fromisoformat(job_data["created_at"]),
        started_at=datetime.fromisoformat(job_data["started_at"]) if job_data.get("started_at") else None,
        completed_at=datetime.fromisoformat(job_data["completed_at"]) if job_data.get("completed_at") else None,
        error=job_data.get("error"),
        output_url=job_data.get("output_url"),
    )


@app.delete("/api/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and all associated files."""
    job_data = job_manager.get_job(job_id)
    if not job_data:
        raise HTTPException(404, "Job not found")
    job_manager.delete_job(job_id)
    return {"status": "deleted", "job_id": job_id}


# ===================================================================
# ENDPOINTS: Previews
# ===================================================================


@app.get("/api/job/{job_id}/previews")
async def get_job_previews(job_id: str):
    """Get preview images for a job."""
    job_data = job_manager.get_job(job_id)
    if not job_data:
        raise HTTPException(404, "Job not found")

    if job_data.get("views") is not None:
        view_previews = storage_manager.list_view_previews(job_id)
        result: dict = {"job_id": job_id, "views": {}}
        for substage, views in view_previews.items():
            for view_name, filename in views.items():
                if view_name not in result["views"]:
                    result["views"][view_name] = []
                result["views"][view_name].append({
                    "stage": substage,
                    "filename": filename,
                    "url": f"/api/job/{job_id}/preview/{substage}/{view_name}",
                })
        return JSONResponse(content=result)

    preview_dir = storage_manager.get_job_preview_dir(job_id)
    if not preview_dir.exists():
        return PreviewResponse(job_id=job_id, previews=[])

    previews = []
    for p in sorted(preview_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            from PIL import Image as PILImage
            try:
                img = PILImage.open(p)
                size = {"width": img.width, "height": img.height}
            except Exception:
                size = {"width": 0, "height": 0}
            previews.append(
                PreviewStage(
                    stage=p.stem, filename=p.name,
                    url=f"/api/job/{job_id}/preview/{p.name}",
                    timestamp=datetime.fromtimestamp(p.stat().st_mtime),
                    size=size,
                )
            )
    return PreviewResponse(job_id=job_id, previews=previews)


@app.get("/api/job/{job_id}/preview/{filename:path}")
async def get_preview_file(job_id: str, filename: str):
    """Serve a preview image file."""
    job_data = job_manager.get_job(job_id)
    if not job_data:
        raise HTTPException(404, "Job not found")

    parts = filename.split("/")
    if len(parts) == 2:
        substage, view_name = parts
        path = storage_manager.get_view_preview_path(job_id, substage, view_name)
        if path and path.exists():
            return FileResponse(path, media_type=StorageManager.content_type_for(path.name))

    preview_path = storage_manager.get_preview_path(job_id, filename)
    if preview_path.exists():
        return FileResponse(preview_path, media_type=StorageManager.content_type_for(filename))

    raise HTTPException(404, "Preview not found")


# ===================================================================
# ENDPOINTS: Artifacts
# ===================================================================


@app.get("/api/job/{job_id}/artifacts", response_model=ArtifactListResponse)
async def get_job_artifacts(job_id: str):
    """List all available artifacts for a job."""
    job_data = job_manager.get_job(job_id)
    if not job_data:
        raise HTTPException(404, "Job not found")

    raw_artifacts = storage_manager.list_artifacts(job_id)
    artifacts = []
    for a in raw_artifacts:
        stage = _infer_stage_from_artifact(a["filename"])
        artifacts.append(
            ArtifactInfo(
                name=a["name"], stage=stage, filename=a["filename"],
                url=f"/api/job/{job_id}/download/{a['filename']}",
                file_size=a.get("file_size"), content_type=a.get("content_type"),
                created_at=datetime.fromisoformat(a["created_at"]) if a.get("created_at") else None,
            )
        )
    return ArtifactListResponse(job_id=job_id, artifacts=artifacts)


@app.get("/api/job/{job_id}/download/{artifact_path:path}")
async def download_artifact(job_id: str, artifact_path: str):
    """Download a specific artifact file."""
    job_data = job_manager.get_job(job_id)
    if not job_data:
        raise HTTPException(404, "Job not found")

    filepath = storage_manager.get_artifact_path(job_id, artifact_path)
    if not filepath:
        output_file = storage_manager.get_output_file(job_id)
        if output_file and output_file.name == artifact_path:
            filepath = output_file
        else:
            raise HTTPException(404, f"Artifact not found: {artifact_path}")

    return FileResponse(filepath, media_type=StorageManager.content_type_for(filepath.name), filename=filepath.name)


# ===================================================================
# ENDPOINTS: Metrics / QA
# ===================================================================


@app.get("/api/job/{job_id}/metrics", response_model=MetricsResponse)
async def get_job_metrics(job_id: str):
    """Get QA metrics for a job."""
    job_data = job_manager.get_job(job_id)
    if not job_data:
        raise HTTPException(404, "Job not found")

    metrics = storage_manager.load_metrics(job_id)
    if not metrics:
        return MetricsResponse(job_id=job_id)

    raw_warnings = metrics.get("warnings", [])
    warnings = []
    for w in raw_warnings:
        if isinstance(w, dict):
            warnings.append(QAWarning(**w))
        elif isinstance(w, str):
            warnings.append(QAWarning(code=w, message=w, severity="warning"))

    return MetricsResponse(
        job_id=job_id,
        quality_score=metrics.get("quality_score"),
        per_view_metrics=metrics.get("per_view_metrics", {}),
        mesh_metrics=metrics.get("mesh_metrics", {}),
        texture_metrics=metrics.get("texture_metrics", {}),
        warnings=warnings,
        recommended_retry=metrics.get("recommended_retry", []),
    )


# ===================================================================
# ENDPOINTS: Stage rerun
# ===================================================================


@app.post("/api/job/{job_id}/rerun_stage")
async def rerun_stage(job_id: str, stage: str = Form(...)):
    """Request a stage re-run for a multi-view job."""
    job_data = job_manager.get_job(job_id)
    if not job_data:
        raise HTTPException(404, "Job not found")
    if not job_data.get("pipeline"):
        raise HTTPException(400, "Stage rerun is only supported for multi-view jobs")

    stages = job_data.get("stages", {})
    if stage not in stages:
        raise HTTPException(400, f"Unknown stage: {stage}")

    from .models import MV_STAGE_ORDER
    found = False
    reset_stages = {}
    for s in MV_STAGE_ORDER:
        if s == stage:
            found = True
        if found and s in stages:
            reset_stages[s] = StageStatus.PENDING.value

    job_manager.update_job(
        job_id, status=JobStatus.QUEUED, stage_updates=reset_stages,
        current_stage=stage, stage_progress=0.0, error=None,
    )
    job_manager.queue_job_for_generation(job_id)

    return {"status": "requeued", "job_id": job_id, "rerun_from": stage, "stages_reset": list(reset_stages.keys())}


# ===================================================================
# ENDPOINTS: Camera calibration (fast preview)
# ===================================================================


@app.post("/api/job/{job_id}/calibrate_cameras")
async def calibrate_cameras(job_id: str, body: dict):
    """
    Run a fast camera calibration preview.

    Accepts camera parameter overrides and returns a visual hull
    preview image plus per-view depth maps and mask overlays,
    all as base64-encoded PNGs.

    Body JSON::

        {
            "cameras": {
                "front": {"yaw_deg": 0, "pitch_deg": 0, "distance": 2.5, "focal_length": 50},
                "side":  {"yaw_deg": 90, "pitch_deg": 0, "distance": 2.5, "focal_length": 50},
                "top":   {"yaw_deg": 0, "pitch_deg": -90, "distance": 2.5, "focal_length": 50}
            },
            "top_up_hint": [-1, 0, 0],
            "grid_resolution": 64,
            "consensus_ratio": 0.6,
            "mask_dilation": 15
        }
    """
    import base64

    job_data = job_manager.get_job(job_id)
    if not job_data:
        raise HTTPException(404, "Job not found")

    camera_overrides = body.get("cameras", {})
    top_up_hint = body.get("top_up_hint", None)
    grid_resolution = int(body.get("grid_resolution", 64))
    consensus_ratio = float(body.get("consensus_ratio", 0.6))
    mask_dilation = int(body.get("mask_dilation", 15))

    # Clamp grid resolution for speed
    grid_resolution = max(32, min(grid_resolution, 192))

    try:
        from pipelines.canonical_mv.calibrate import run_calibration_preview

        result = run_calibration_preview(
            job_id=job_id,
            sm=storage_manager,
            camera_overrides=camera_overrides,
            top_up_hint=top_up_hint,
            grid_resolution=grid_resolution,
            consensus_ratio=consensus_ratio,
            mask_dilation=mask_dilation,
        )

        # Encode images as base64
        response = {
            "preview": base64.b64encode(result["preview_png"]).decode("ascii"),
            "depth_maps": {
                vn: base64.b64encode(png).decode("ascii")
                for vn, png in result["depth_pngs"].items()
            },
            "overlays": {
                vn: base64.b64encode(png).decode("ascii")
                for vn, png in result["overlay_pngs"].items()
            },
            "n_occupied": result["n_occupied"],
            "occupancy_pct": result["occupancy_pct"],
        }

        return JSONResponse(content=response)

    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Calibration failed for job {job_id}: {e}", exc_info=True)
        raise HTTPException(500, f"Calibration failed: {e}")


# ===================================================================
# ENDPOINTS: Output download
# ===================================================================


@app.get("/api/job/{job_id}/output")
async def get_output(job_id: str):
    """Download the final output GLB file."""
    job_data = job_manager.get_job(job_id)
    if not job_data:
        raise HTTPException(404, "Job not found")
    if job_data["status"] != JobStatus.COMPLETED.value:
        raise HTTPException(400, "Job not yet completed")
    output_file = storage_manager.get_output_file(job_id)
    if not output_file:
        raise HTTPException(404, "Output file not found")
    return FileResponse(output_file, media_type="model/gltf-binary", filename=output_file.name)


# ===================================================================
# ENDPOINTS: Categories & health
# ===================================================================


@app.get("/api/categories", response_model=CategoriesResponse)
async def get_categories():
    """Get available categories with guidance."""
    return CategoriesResponse(categories=list(CATEGORY_GUIDANCE.values()))


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    try:
        job_manager.redis_client.ping()
        redis_ok = True
    except Exception:
        redis_ok = False
    return {
        "status": "healthy" if redis_ok else "degraded",
        "redis": "connected" if redis_ok else "disconnected",
        "version": "2.0.0",
        "pipelines": [p.value for p in PipelineEnum],
    }


@app.get("/api/queue/status")
async def queue_status():
    """Get queue lengths for all models and pipelines."""
    from .models import PipelineEnum as PE
    result = {"single_view": {}, "multi_view": {}}
    for model in ModelEnum:
        result["single_view"][model.value] = job_manager.get_queue_length(model)
    for pipe in [PE.CANONICAL_MV_HYBRID, PE.CANONICAL_MV_FAST, PE.CANONICAL_MV_GENERATIVE]:
        result["multi_view"][pipe.value] = job_manager.get_multiview_queue_length(pipe)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ARTIFACT_STAGE_MAP = {
    "camera_init": "initialize_cameras",
    "coarse_gaussians": "reconstruct_coarse",
    "coarse_mesh": "reconstruct_coarse",
    "coarse_voxel": "reconstruct_coarse",
    "coarse_visual_hull": "reconstruct_coarse",
    "coarse_depth": "reconstruct_coarse",
    "refined_mesh": "refine_joint",
    "metrics": "qa",
    "final": "export",
}


def _infer_stage_from_artifact(filename: str) -> str:
    """Infer the pipeline stage that produced an artifact from its filename."""
    stem = Path(filename).stem.lower()
    for pattern, stage in _ARTIFACT_STAGE_MAP.items():
        if pattern in stem:
            return stage
    parts = Path(filename).parts
    if len(parts) > 1 and parts[0].lower() == "textures":
        return "bake_texture"
    return "unknown"

