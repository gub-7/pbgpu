"""
Storage management for uploads, previews, artifacts, and outputs.

Supports both single-view (legacy) and multi-view canonical 3-view
reconstruction pipelines.

Multi-view storage layout per job:

    uploads/{job_id}/views/{view}.png          -- raw uploaded views
    previews/{job_id}/raw/{view}.png           -- raw thumbnails
    previews/{job_id}/segmented/{view}.png     -- post-segmentation
    previews/{job_id}/normalized/{view}.png    -- cross-view normalized
    artifacts/{job_id}/camera_init.json        -- canonical camera rig
    artifacts/{job_id}/coarse_gaussians.ply    -- coarse Gaussian cloud
    artifacts/{job_id}/coarse_mesh.glb         -- coarse mesh
    artifacts/{job_id}/refined_mesh.glb        -- joint-refined mesh
    artifacts/{job_id}/textures/*              -- baked texture maps
    artifacts/{job_id}/metrics.json            -- QA metrics
    outputs/{job_id}/final.glb                 -- production GLB
"""

import json
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime


# Canonical view names (must match ViewName enum values)
_CANONICAL_VIEWS = ("front", "side", "top")

# Preview sub-stages for multi-view jobs
_PREVIEW_SUBSTAGES = ("raw", "segmented", "normalized", "debug_recon")

# Well-known artifact filenames (stage -> filename)
_KNOWN_ARTIFACTS: Dict[str, str] = {
    "camera_init": "camera_init.json",
    "coarse_gaussians": "coarse_gaussians.ply",
    "coarse_mesh": "coarse_mesh.glb",
    "refined_mesh": "refined_mesh.glb",
    "metrics": "metrics.json",
}

# MIME types for common artifact extensions
_CONTENT_TYPES: Dict[str, str] = {
    ".json": "application/json",
    ".glb": "model/gltf-binary",
    ".ply": "application/octet-stream",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".npz": "application/octet-stream",
}


class StorageManager:
    """Manages file storage for the 3D reconstruction pipeline"""

    def __init__(self, storage_root: str = "storage"):
        self.storage_root = Path(storage_root)
        self.uploads_dir = self.storage_root / "uploads"
        self.previews_dir = self.storage_root / "previews"
        self.outputs_dir = self.storage_root / "outputs"
        self.artifacts_dir = self.storage_root / "artifacts"

        # Create top-level directories
        for d in (self.uploads_dir, self.previews_dir, self.outputs_dir, self.artifacts_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Single-view helpers (backward-compatible)
    # ------------------------------------------------------------------

    def get_upload_path(self, job_id: str, filename: str) -> Path:
        """Get path for uploaded file"""
        job_dir = self.uploads_dir / job_id
        job_dir.mkdir(exist_ok=True)
        return job_dir / filename

    def get_preview_path(self, job_id: str, filename: str) -> Path:
        """Get path for preview file"""
        job_dir = self.previews_dir / job_id
        job_dir.mkdir(exist_ok=True)
        return job_dir / filename

    def get_output_path(self, job_id: str, filename: str) -> Path:
        """Get path for output file"""
        job_dir = self.outputs_dir / job_id
        job_dir.mkdir(exist_ok=True)
        return job_dir / filename

    def save_upload(self, job_id: str, filename: str, content: bytes) -> Path:
        """Save uploaded file"""
        filepath = self.get_upload_path(job_id, filename)
        with open(filepath, 'wb') as f:
            f.write(content)
        return filepath

    def get_job_upload_dir(self, job_id: str) -> Path:
        """Get upload directory for a job"""
        return self.uploads_dir / job_id

    def get_job_preview_dir(self, job_id: str) -> Path:
        """Get preview directory for a job"""
        return self.previews_dir / job_id

    def get_job_output_dir(self, job_id: str) -> Path:
        """Get output directory for a job"""
        return self.outputs_dir / job_id

    def get_output_file(self, job_id: str) -> Optional[Path]:
        """Find the output GLB file for a job"""
        output_dir = self.get_job_output_dir(job_id)
        if not output_dir.exists():
            return None

        # Look for .glb file
        glb_files = list(output_dir.glob("*.glb"))
        if glb_files:
            return glb_files[0]

        return None

    # ------------------------------------------------------------------
    # Multi-view upload helpers
    # ------------------------------------------------------------------

    def get_views_dir(self, job_id: str) -> Path:
        """Get the views sub-directory under uploads for a multi-view job."""
        d = self.uploads_dir / job_id / "views"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_view_upload(
        self, job_id: str, view_name: str, content: bytes, extension: str = ".png"
    ) -> Path:
        """
        Save a single canonical view image.

        Args:
            job_id: Job identifier.
            view_name: One of front/side/top.
            content: Raw image bytes.
            extension: File extension (default .png).

        Returns:
            Path where the file was written.
        """
        views_dir = self.get_views_dir(job_id)
        filepath = views_dir / f"{view_name}{extension}"
        with open(filepath, "wb") as f:
            f.write(content)
        return filepath

    def get_view_upload_path(self, job_id: str, view_name: str) -> Optional[Path]:
        """
        Return the path to an uploaded view image, or None if not found.

        Checks common image extensions (.png, .jpg, .jpeg, .webp).
        """
        views_dir = self.uploads_dir / job_id / "views"
        if not views_dir.exists():
            return None
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            p = views_dir / f"{view_name}{ext}"
            if p.exists():
                return p
        return None

    def list_uploaded_views(self, job_id: str) -> List[str]:
        """
        Return the list of view names that have been uploaded for a job.

        E.g. ["front", "side", "top"]
        """
        views_dir = self.uploads_dir / job_id / "views"
        if not views_dir.exists():
            return []
        names = set()
        for p in views_dir.iterdir():
            if p.is_file():
                names.add(p.stem)
        # Return in canonical order
        return [v for v in _CANONICAL_VIEWS if v in names]

    # ------------------------------------------------------------------
    # Multi-view preview helpers
    # ------------------------------------------------------------------

    def get_view_preview_dir(self, job_id: str, substage: str) -> Path:
        """
        Get the preview sub-directory for a specific substage.

        Substages: raw, segmented, normalized.
        """
        d = self.previews_dir / job_id / substage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_view_preview(
        self,
        job_id: str,
        substage: str,
        view_name: str,
        content: bytes,
        extension: str = ".png",
    ) -> Path:
        """
        Save a preview image for a specific view and substage.

        Args:
            job_id: Job identifier.
            substage: One of raw/segmented/normalized.
            view_name: Canonical view name.
            content: Image bytes.
            extension: File extension.

        Returns:
            Path where the file was written.
        """
        d = self.get_view_preview_dir(job_id, substage)
        filepath = d / f"{view_name}{extension}"
        with open(filepath, "wb") as f:
            f.write(content)
        return filepath

    def get_view_preview_path(
        self, job_id: str, substage: str, view_name: str
    ) -> Optional[Path]:
        """Return the path to a view preview image, or None if not found."""
        d = self.previews_dir / job_id / substage
        if not d.exists():
            return None
        for ext in (".png", ".jpg", ".jpeg"):
            p = d / f"{view_name}{ext}"
            if p.exists():
                return p
        return None

    def list_view_previews(self, job_id: str) -> Dict[str, Dict[str, str]]:
        """
        Return a nested dict of available view previews.

        Structure: {substage: {view_name: filename, ...}, ...}
        """
        result: Dict[str, Dict[str, str]] = {}
        base = self.previews_dir / job_id
        if not base.exists():
            return result
        for substage in _PREVIEW_SUBSTAGES:
            sub_dir = base / substage
            if not sub_dir.exists():
                continue
            views: Dict[str, str] = {}
            for p in sub_dir.iterdir():
                if p.is_file() and p.stem in _CANONICAL_VIEWS:
                    views[p.stem] = p.name
            if views:
                result[substage] = views
        return result

    # ------------------------------------------------------------------
    # Artifact helpers
    # ------------------------------------------------------------------

    def get_artifact_dir(self, job_id: str) -> Path:
        """Get the artifacts directory for a job."""
        d = self.artifacts_dir / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_artifact_textures_dir(self, job_id: str) -> Path:
        """Get the textures sub-directory under artifacts."""
        d = self.artifacts_dir / job_id / "textures"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_artifact(
        self, job_id: str, filename: str, content: bytes
    ) -> Path:
        """
        Save an arbitrary artifact file.

        Args:
            job_id: Job identifier.
            filename: Filename (placed directly under artifacts/{job_id}/).
            content: Raw bytes.

        Returns:
            Path where the file was written.
        """
        d = self.get_artifact_dir(job_id)
        filepath = d / filename
        # Support sub-paths like "textures/diffuse.png"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(content)
        return filepath

    def save_artifact_json(
        self, job_id: str, filename: str, data: Any
    ) -> Path:
        """
        Save a JSON artifact (e.g. camera_init.json, metrics.json).

        Args:
            job_id: Job identifier.
            filename: Filename.
            data: JSON-serializable object.

        Returns:
            Path where the file was written.
        """
        d = self.get_artifact_dir(job_id)
        filepath = d / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return filepath

    def load_artifact_json(self, job_id: str, filename: str) -> Optional[Any]:
        """Load a JSON artifact, or return None if it doesn't exist."""
        filepath = self.artifacts_dir / job_id / filename
        if not filepath.exists():
            return None
        with open(filepath, "r") as f:
            return json.load(f)

    def get_artifact_path(self, job_id: str, filename: str) -> Optional[Path]:
        """Return the path to an artifact file, or None if not found."""
        filepath = self.artifacts_dir / job_id / filename
        if filepath.exists():
            return filepath
        return None

    def list_artifacts(self, job_id: str) -> List[Dict[str, Any]]:
        """
        List all artifacts for a job with metadata.

        Returns a list of dicts with keys:
            name, filename, file_size, content_type, created_at
        """
        d = self.artifacts_dir / job_id
        if not d.exists():
            return []

        artifacts = []
        for p in sorted(d.rglob("*")):
            if not p.is_file():
                continue
            # Build a relative name like "textures/diffuse.png" or "metrics.json"
            rel = p.relative_to(d)
            ext = p.suffix.lower()
            artifacts.append({
                "name": rel.stem,
                "filename": str(rel),
                "file_size": p.stat().st_size,
                "content_type": _CONTENT_TYPES.get(ext, "application/octet-stream"),
                "created_at": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
            })
        return artifacts

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------

    def save_metrics(self, job_id: str, metrics: Dict[str, Any]) -> Path:
        """Save QA metrics JSON for a job."""
        return self.save_artifact_json(job_id, "metrics.json", metrics)

    def load_metrics(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Load QA metrics JSON for a job, or None if not available."""
        return self.load_artifact_json(job_id, "metrics.json")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_job(self, job_id: str):
        """Remove all files for a job across all storage areas."""
        for directory in (
            self.uploads_dir,
            self.previews_dir,
            self.outputs_dir,
            self.artifacts_dir,
        ):
            job_dir = directory / job_id
            if job_dir.exists():
                shutil.rmtree(job_dir)

    def cleanup_old_jobs(self, days: int = 7):
        """Clean up jobs older than specified days"""
        cutoff = datetime.now().timestamp() - (days * 86400)

        for directory in (
            self.uploads_dir,
            self.previews_dir,
            self.outputs_dir,
            self.artifacts_dir,
        ):
            if not directory.exists():
                continue
            for job_dir in directory.iterdir():
                if job_dir.is_dir() and job_dir.stat().st_mtime < cutoff:
                    shutil.rmtree(job_dir)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def content_type_for(filename: str) -> str:
        """Return a MIME content-type for a filename based on extension."""
        ext = Path(filename).suffix.lower()
        return _CONTENT_TYPES.get(ext, "application/octet-stream")

