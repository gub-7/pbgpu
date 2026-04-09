"""
Ingest stage for the canonical multi-view pipeline.

Responsibilities:
    - Verify all 5 canonical view files are present and readable
    - Read basic image metadata (dimensions, file size)
    - Generate raw preview thumbnails for each view
    - Update per-view metadata in job state
    - Fail fast if any view is missing or corrupt
"""

import io
import logging
from typing import Optional

from PIL import Image

from api.job_manager import JobManager
from api.models import REQUIRED_VIEWS, ViewStatus
from api.storage import StorageManager

from .config import CanonicalMVConfig, CANONICAL_VIEW_ORDER

logger = logging.getLogger(__name__)

# Maximum dimension for raw preview thumbnails
_PREVIEW_MAX_SIZE = 512


def run_ingest(
    job_id: str,
    config: CanonicalMVConfig,
    jm: JobManager,
    sm: StorageManager,
) -> None:
    """
    Execute the ingest stage.

    For each canonical view:
        1. Locate the uploaded file
        2. Open and validate it as an image
        3. Record width, height, file_size in job metadata
        4. Generate a raw preview thumbnail
        5. Update view status to UPLOADED

    Raises:
        FileNotFoundError: if a required view is missing.
        ValueError: if a view file is not a valid image.
    """
    logger.info(f"[{job_id}] Ingest: starting validation of {len(REQUIRED_VIEWS)} views")

    warnings = []

    for idx, view_name in enumerate(CANONICAL_VIEW_ORDER):
        # Progress within stage
        progress = idx / len(CANONICAL_VIEW_ORDER)
        jm.update_job(job_id, stage_progress=progress)

        # 1. Locate uploaded file
        view_path = sm.get_view_upload_path(job_id, view_name)
        if view_path is None:
            raise FileNotFoundError(
                f"Required view '{view_name}' was not uploaded or file is missing"
            )

        # 2. Open and validate
        try:
            img = Image.open(view_path)
            img.verify()  # verify integrity without fully loading
            # Re-open after verify (verify can close the file)
            img = Image.open(view_path)
            img.load()  # force full decode to catch truncated files
        except Exception as e:
            raise ValueError(
                f"View '{view_name}' is not a valid image: {e}"
            )

        width, height = img.size
        file_size = view_path.stat().st_size

        logger.info(
            f"[{job_id}] Ingest: {view_name} = {width}x{height}, "
            f"{file_size / 1024:.1f} KB, mode={img.mode}"
        )

        # Warn on very small images
        if width < 256 or height < 256:
            w = f"{view_name}_low_resolution"
            warnings.append(w)
            logger.warning(f"[{job_id}] Ingest: {view_name} resolution is very low ({width}x{height})")

        # Warn on extreme aspect ratios
        aspect = max(width, height) / max(min(width, height), 1)
        if aspect > 2.5:
            w = f"{view_name}_extreme_aspect_ratio"
            warnings.append(w)
            logger.warning(f"[{job_id}] Ingest: {view_name} has extreme aspect ratio ({aspect:.2f})")

        # 3. Update per-view metadata
        jm.update_job(
            job_id,
            view_updates={
                view_name: {
                    "status": ViewStatus.UPLOADED.value,
                    "width": width,
                    "height": height,
                    "file_size": file_size,
                }
            },
        )

        # 4. Generate raw preview thumbnail
        _save_raw_preview(sm, job_id, view_name, img)

    # Store any warnings
    if warnings:
        jm.update_job(job_id, warnings=warnings)

    jm.update_job(job_id, stage_progress=1.0)
    logger.info(f"[{job_id}] Ingest: all {len(REQUIRED_VIEWS)} views validated successfully")


def _save_raw_preview(
    sm: StorageManager,
    job_id: str,
    view_name: str,
    img: Image.Image,
) -> None:
    """
    Generate and save a raw preview thumbnail for a view.

    Saves to: previews/{job_id}/raw/{view_name}.png
    """
    # Create thumbnail (preserves aspect ratio)
    thumb = img.copy()
    thumb.thumbnail((_PREVIEW_MAX_SIZE, _PREVIEW_MAX_SIZE), Image.LANCZOS)

    # Convert to RGB if needed (some inputs may be RGBA or palette)
    if thumb.mode not in ("RGB", "RGBA"):
        thumb = thumb.convert("RGB")

    # Save to bytes
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    buf.seek(0)

    sm.save_view_preview(job_id, "raw", view_name, buf.getvalue(), ".png")

