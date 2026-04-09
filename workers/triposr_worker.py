"""
TripoSR worker - processes 3D reconstruction jobs using TripoSR
Polls Redis queue for jobs and generates textured GLB models
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# Import torch and torchvision first to avoid circular import issues
# torchvision must be fully initialized before TripoSR imports it
import torch
import torchvision  # Explicit import to prevent circular import errors

import numpy as np
from PIL import Image
import trimesh
from scipy.spatial import cKDTree

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Add TripoSR to path (in case conda activation script didn't run)
triposr_path = Path(__file__).parent.parent.parent / "TripoSR"
if triposr_path.exists():
    sys.path.insert(0, str(triposr_path))

from api.job_manager import JobManager
from api.storage import StorageManager
from api.models import JobStatus, ModelEnum

# Configure logging FIRST before any logging calls
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("triposr_worker")

# Now log the TripoSR path status
if triposr_path.exists():
    logger.info(f"Added TripoSR to path: {triposr_path}")
else:
    logger.warning(f"TripoSR path not found: {triposr_path}")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# TripoSR normalizes geometry into roughly [-0.87, 0.87]^3.  The full cube
# diagonal is ~1.74.  A mesh whose bbox max-extent is near that value is
# almost certainly a "cube-filling fog shell" from a too-low threshold.
_CUBE_FILL_EXTENT = 1.65

# TripoSR normalizes geometry into roughly [-0.87, 0.87]^3.
# Full extent along any axis is ~1.74.
_TRIPOSR_COORD_EXTENT = 1.74
_TRIPOSR_COORD_HALF = 0.87


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def _mesh_health(tag: str, mesh):
    """
    Diagnostic helper to log mesh health metrics.
    Detects NaN/inf vertices, invalid faces, and coordinate explosions.
    """
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.faces)
    v_finite = np.isfinite(v).all(axis=1)
    f_inrange = ((f >= 0) & (f < len(v))).all(axis=1) if len(f) else np.array([], dtype=bool)

    msg = (
        f"[{tag}] verts={len(v)} faces={len(f)} "
        f"finite_verts={v_finite.mean() if len(v) else 0:.3f} "
        f"nonfinite_verts={(~v_finite).sum() if len(v) else 0} "
        f"faces_inrange={f_inrange.mean() if len(f) else 0:.3f} "
        f"bad_faces={(~f_inrange).sum() if len(f) else 0}"
    )
    logger.info(msg)

    # Sample bbox to detect explosions
    if len(v):
        vv = v[v_finite]
        if len(vv):
            mn = vv.min(axis=0)
            mx = vv.max(axis=0)
            logger.info(f"[{tag}] bbox min={mn} max={mx}")


def _color_health(tag: str, mesh):
    """
    Diagnostic helper to log vertex color health metrics.
    Detects NaN/inf colors, dtype issues, and range problems.
    """
    vc = getattr(mesh.visual, "vertex_colors", None) if hasattr(mesh, "visual") else None
    if vc is None:
        logger.info(f"[{tag}] vertex_colors: None")
        return
    vc = np.asarray(vc)
    finite = np.isfinite(vc).all() if np.issubdtype(vc.dtype, np.floating) else True
    logger.info(f"[{tag}] vertex_colors shape={vc.shape} dtype={vc.dtype} finite={finite} "
                f"min={vc.min() if vc.size else 'n/a'} max={vc.max() if vc.size else 'n/a'}")


def _debug_gpu_tensors(tag: str):
    """
    CRITICAL DEBUGGING: List all GPU tensors in memory and their sizes.
    This helps identify what's using VRAM and why cleanup isn't working.
    """
    import gc

    logger.info(f"========== GPU MEMORY DEBUG: {tag} ==========")

    # Get overall GPU memory stats
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - allocated

        logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, "
                     f"{free:.2f}GB free of {total:.2f}GB total")

        # Find all CUDA tensors in memory
        cuda_tensors = []
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    size_bytes = obj.element_size() * obj.nelement()
                    size_mb = size_bytes / 1024**2
                    cuda_tensors.append((size_mb, obj.shape, obj.dtype))
            except:
                pass

        # Sort by size (largest first)
        cuda_tensors.sort(reverse=True)

        # Log top 20 largest tensors
        logger.info(f"Found {len(cuda_tensors)} CUDA tensors in memory")
        if cuda_tensors:
            logger.info("Top 20 largest tensors:")
            for i, (size_mb, shape, dtype) in enumerate(cuda_tensors[:20]):
                logger.info(f"  #{i+1}: {size_mb:.2f}MB - shape={shape} dtype={dtype}")

            total_tensor_mb = sum(size_mb for size_mb, _, _ in cuda_tensors)
            logger.info(f"Total tensor memory: {total_tensor_mb / 1024:.2f}GB")
        else:
            logger.info("No CUDA tensors found in Python objects (memory may be in C++ objects)")

    logger.info(f"========== END GPU MEMORY DEBUG: {tag} ==========")


# ---------------------------------------------------------------------------
# Mesh scoring  (cube-filling penalty included)
# ---------------------------------------------------------------------------

def mesh_score(mesh: trimesh.Trimesh) -> float:
    """
    Score meshes to choose the highest-fidelity, most plausible output.
    Higher is better.

    Scoring factors:
    - Face count (more detail = better)
    - Volume (prefer solid objects over paper-thin junk)
    - Fragmentation penalty (fewer components = better)
    - Exploded bbox penalty (reasonable extents = better)
    - Cube-filling penalty (TripoSR normalizes to ~[-0.87,0.87]; meshes
      that fill this cube are fog shells from too-low thresholds)
    """
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return -1e18

    faces = float(len(mesh.faces))

    # Penalize fragmentation
    try:
        parts = mesh.split(only_watertight=False)
        comp = len(parts)
    except Exception:
        comp = 1

    # BBox sanity
    v = np.asarray(mesh.vertices)
    v = v[np.isfinite(v).all(axis=1)]
    if len(v) == 0:
        return -1e18

    ext = (v.max(axis=0) - v.min(axis=0))
    max_ext = float(ext.max()) if ext.size else 0.0

    # Cube-filling penalty for TripoSR normalized coords
    cube_fill_pen = 0.0
    if max_ext > _CUBE_FILL_EXTENT:
        cube_fill_pen = 1e7      # severe — almost certainly junk
    elif max_ext > 1.4:
        cube_fill_pen = 5e5      # suspicious

    # Legacy exploded-mesh penalty (unnormalized coords)
    exploded_pen = 0.0
    if max_ext > 50.0:
        exploded_pen = 5e6
    elif max_ext > 10.0:
        exploded_pen = 5e5

    comp_pen = float(max(comp - 1, 0)) * 2e5

    # Prefer some volume; helps reject paper-thin junk
    vol = 0.0
    try:
        vol = float(abs(mesh.volume))
    except Exception:
        vol = 0.0

    return faces + 1e3 * np.log1p(vol) - comp_pen - exploded_pen - cube_fill_pen


# ---------------------------------------------------------------------------
# Quick GPU helpers
# ---------------------------------------------------------------------------

def _log_gpu_memory(tag: str):
    """Quick GPU memory status log (lighter than full tensor debug)."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - allocated
        logger.info(f"[{tag}] GPU: {allocated:.2f}GB allocated, "
                     f"{reserved:.2f}GB reserved, {free:.2f}GB free of {total:.2f}GB")


def is_mesh_sheet_like(mesh) -> bool:
    """
    Check if mesh is sheet-like (flat/exploded) based on bounding box aspect ratio.
    Uses a relaxed heuristic: only reject meshes that are both suspiciously flat
    AND have an exploded bounding box (extents > 10 scene units).
    """
    v = np.asarray(mesh.vertices)
    if len(v) == 0:
        return True

    v_finite = v[np.isfinite(v).all(axis=1)]
    if len(v_finite) == 0:
        return True

    bbox_min = v_finite.min(axis=0)
    bbox_max = v_finite.max(axis=0)
    extents = bbox_max - bbox_min

    if extents.max() > 0:
        aspect_ratios = extents / extents.max()
        flat = (aspect_ratios < 0.03).any()
        if flat and extents.max() > 10.0:
            logger.warning(f"Mesh suspiciously flat+huge: extents={extents}, "
                           f"aspect={aspect_ratios}")
            return True

    if extents.max() > 50.0:
        logger.warning(f"Mesh is exploded: bbox extents={extents}")
        return True

    return False


def log_gpu_memory_detailed(label: str = ""):
    """Log detailed GPU memory statistics including all tensors in memory."""
    if not torch.cuda.is_available():
        return

    import gc
    gc.collect()
    torch.cuda.synchronize()

    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free = total - allocated

    logger.info("=" * 80)
    logger.info(f"GPU MEMORY SNAPSHOT: {label}")
    logger.info("=" * 80)
    logger.info(f"Allocated: {allocated:.2f} GB")
    logger.info(f"Reserved:  {reserved:.2f} GB")
    logger.info(f"Free:      {free:.2f} GB")
    logger.info(f"Total:     {total:.2f} GB")
    logger.info(f"Utilization: {(allocated/total)*100:.1f}%")

    logger.info("-" * 80)
    logger.info("GPU TENSORS IN MEMORY:")

    total_tensor_memory = 0
    tensor_count = 0

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if obj.is_cuda:
                    tensor_size = obj.element_size() * obj.nelement() / 1024**3
                    if tensor_size > 0.01:
                        logger.info(f"  Tensor: shape={tuple(obj.shape)}, "
                                     f"dtype={obj.dtype}, size={tensor_size:.3f} GB")
                        total_tensor_memory += tensor_size
                        tensor_count += 1
        except Exception:
            pass

    logger.info("-" * 80)
    logger.info(f"Total GPU tensors: {tensor_count}")
    logger.info(f"Total tensor memory: {total_tensor_memory:.2f} GB")
    logger.info(f"Unaccounted memory: {allocated - total_tensor_memory:.2f} GB "
                 f"(model weights, buffers, etc.)")
    logger.info("=" * 80)


# ---------------------------------------------------------------------------
# GLB post-processing: force unlit material for vertex-color exports
# ---------------------------------------------------------------------------

def force_unlit_glb(path: str):
    """
    Post-process a GLB file to apply KHR_materials_unlit extension.

    In glTF, vertex colors (COLOR_0) are NOT final color — they are multiplied
    by the PBR material pipeline (lighting, normals, metallic/roughness, IBL).
    This causes vertex colors to appear faint, uneven, or washed out depending
    on the viewer's lighting environment.

    Setting KHR_materials_unlit ensures vertex colors are displayed directly
    as final color with no lighting influence — bold, uniform, and consistent
    across all viewers.
    """
    try:
        from pygltflib import GLTF2
    except ImportError:
        logger.warning(
            "pygltflib not installed — cannot apply KHR_materials_unlit. "
            "Vertex colors may appear faint/uneven in PBR viewers. "
            "Install with: pip install pygltflib"
        )
        return

    try:
        gltf = GLTF2().load(path)

        # Ensure extension is listed in extensionsUsed
        if gltf.extensionsUsed is None:
            gltf.extensionsUsed = []
        if "KHR_materials_unlit" not in gltf.extensionsUsed:
            gltf.extensionsUsed.append("KHR_materials_unlit")

        # Apply unlit extension to all materials (or create one if missing)
        if gltf.materials is None or len(gltf.materials) == 0:
            from pygltflib import Material, PbrMetallicRoughness
            gltf.materials = [Material(pbrMetallicRoughness=PbrMetallicRoughness())]

        for mat in gltf.materials:
            if mat.extensions is None:
                mat.extensions = {}
            mat.extensions["KHR_materials_unlit"] = {}

            # Set sane PBR defaults — some viewers still read these factors
            # even with unlit, so ensure they don't darken vertex colors
            if mat.pbrMetallicRoughness is None:
                from pygltflib import PbrMetallicRoughness
                mat.pbrMetallicRoughness = PbrMetallicRoughness()

            mat.pbrMetallicRoughness.baseColorFactor = [1, 1, 1, 1]
            mat.pbrMetallicRoughness.metallicFactor = 0.0
            mat.pbrMetallicRoughness.roughnessFactor = 1.0

        gltf.save(path)
        logger.info(
            "Applied KHR_materials_unlit to %d material(s) in %s",
            len(gltf.materials), path,
        )

    except Exception as e:
        logger.warning("Failed to apply KHR_materials_unlit to GLB: %s", e)


# ---------------------------------------------------------------------------
# Worker class
# ---------------------------------------------------------------------------

class TripoSRWorker:
    """Worker for processing TripoSR 3D reconstruction jobs"""

    def __init__(self):
        self.job_manager = JobManager()
        self.storage_manager = StorageManager()
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._cached_field_regime: Optional[str] = None

        logger.info(f"TripoSR Worker initialized on device: {self.device}")

        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"PyTorch Version: {torch.__version__}")

        self.verify_triposr()

    def verify_triposr(self):
        """Verify that TripoSR is installed and can be imported"""
        try:
            from tsr.system import TSR
            logger.info("TripoSR module verified and ready")
        except ImportError as e:
            logger.error(f"TripoSR import failed: {e}")
            logger.error("TripoSR is not installed. Please run setup/setup_triposr.sh")
            logger.error(f"Expected TripoSR at: {triposr_path}")
            raise RuntimeError("TripoSR not installed. Cannot start worker.") from e

    def load_model(self):
        """Load TripoSR model"""
        if self.model is not None:
            return

        logger.info("Loading TripoSR model...")
        _debug_gpu_tensors("BEFORE_MODEL_LOAD")
        self._clear_gpu_memory()

        try:
            from tsr.system import TSR

            self.model = TSR.from_pretrained(
                "stabilityai/TripoSR",
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
            self.model.to(self.device)
            self.model.eval()

            # Reset cached regime when model is (re)loaded
            self._cached_field_regime = None

            logger.info("TripoSR model loaded successfully")
            _debug_gpu_tensors("AFTER_MODEL_LOAD")

        except Exception as e:
            logger.error(f"Failed to load TripoSR model: {e}")
            _debug_gpu_tensors("MODEL_LOAD_FAILED")
            raise

    def _clear_gpu_memory(self):
        """Clear GPU memory before loading TripoSR model."""
        try:
            import gc
            logger.info("Clearing GPU memory before loading TripoSR model...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                logger.info(f"GPU memory status: {allocated:.2f}GB allocated, "
                             f"{reserved:.2f}GB reserved")
        except Exception as e:
            logger.warning(f"GPU memory cleanup warning: {e}")

    # ------------------------------------------------------------------
    # Field regime probing — picks tightest bbox, NOT max faces
    # ------------------------------------------------------------------

    def _probe_density_field(
        self,
        scene_codes,
        has_vc: bool = False,
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Probe the scalar density field to determine the correct threshold regime.

        TripoSR's ``extract_mesh`` runs marching cubes on a density field whose
        numeric range depends on the model variant / checkpoint:

        * **density regime** -- values roughly 0...100+, official default ~ 25.0
        * **probability regime** -- values roughly 0...1, typical threshold ~ 0.04

        **Key insight:** the OLD version picked the threshold with the *most faces*.
        For probability-regime fields the *lowest* thresholds produce a giant
        "fog shell" that fills the entire marching-cubes cube -- maximum faces --
        so the probe chose a near-zero threshold -> cube-filling junk.

        The NEW version computes **bbox extents** for every probe mesh and picks
        the threshold whose bbox is **tightest** (smallest max-extent) while still
        having enough faces.  This pushes the center *away* from cube-filling
        thresholds and toward the iso-surface that wraps the actual subject.

        Returns:
            (regime, center_threshold, diagnostics_dict)
        """
        probe_resolution = 64  # Low resolution for speed
        probe_thresholds = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0]

        # Each result: (threshold, n_verts, n_faces, bbox_max_extent)
        results: List[Tuple[float, int, int, float]] = []

        logger.info("=" * 60)
        logger.info("FIELD REGIME PROBE -- extract_mesh at res=%d with %d thresholds",
                     probe_resolution, len(probe_thresholds))
        logger.info("=" * 60)

        for thr in probe_thresholds:
            try:
                meshes = self.model.extract_mesh(
                    scene_codes,
                    has_vertex_color=False,
                    resolution=probe_resolution,
                    threshold=float(thr),
                )
                m = meshes[0]
                n_verts = len(m.vertices)
                n_faces = len(m.faces)

                # Compute bbox max-extent for cube-filling detection
                bbox_max_ext = 0.0
                if n_verts > 0:
                    vv = np.asarray(m.vertices)
                    vv = vv[np.isfinite(vv).all(axis=1)]
                    if len(vv) > 0:
                        ext = vv.max(axis=0) - vv.min(axis=0)
                        bbox_max_ext = float(ext.max())

                results.append((thr, n_verts, n_faces, bbox_max_ext))
                cube_flag = " *** CUBE-FILL" if bbox_max_ext > _CUBE_FILL_EXTENT else ""
                logger.info(
                    "  probe thr=%8.4f -> verts=%6d  faces=%6d  max_ext=%.3f%s",
                    thr, n_verts, n_faces, bbox_max_ext, cube_flag,
                )
                del m
            except Exception as e:
                logger.warning("  probe thr=%8.4f -> FAILED: %s", thr, e)
                results.append((thr, 0, 0, 0.0))

        # ---- Analyse results ----
        # "valid_tight" = enough faces AND not cube-filling
        valid_tight = [
            (t, v, f, ext) for t, v, f, ext in results
            if f > 50 and ext < _CUBE_FILL_EXTENT
        ]
        # "any_geom" = enough faces (even if cube-filling)
        any_geom = [(t, v, f, ext) for t, v, f, ext in results if f > 50]

        if not any_geom:
            logger.warning("Field probe: NO threshold produced >50 faces at res=%d. "
                           "Using full multi-regime sweep as fallback.", probe_resolution)
            return "unknown", 25.0, {"probes": results}

        # Separate low-regime (<=1.0) and high-regime (>=5.0)
        low_max_faces = max((f for t, v, f, ext in results if t <= 1.0), default=0)
        high_max_faces = max((f for t, v, f, ext in results if t >= 5.0), default=0)

        # Determine regime
        if high_max_faces > 0 and low_max_faces == 0:
            regime = "density"
        elif low_max_faces > 0 and high_max_faces == 0:
            regime = "probability"
        elif high_max_faces >= low_max_faces * 0.3:
            regime = "density"
        else:
            regime = "probability"

        # ---- Pick center threshold ----
        # Among non-cube-filling probes, pick the tightest bbox.
        # Tie-break: prefer more faces.
        if valid_tight:
            valid_sorted = sorted(valid_tight, key=lambda x: (x[3], -x[2]))
            center = valid_sorted[0][0]
            logger.info(
                "  Selected center from tightest-bbox probe: thr=%.4f "
                "(max_ext=%.3f, faces=%d)",
                valid_sorted[0][0], valid_sorted[0][3], valid_sorted[0][2],
            )
        else:
            # All probes are cube-filling -- pick HIGHEST threshold with geometry
            any_geom_sorted = sorted(any_geom, key=lambda x: x[0], reverse=True)
            center = any_geom_sorted[0][0]
            logger.warning(
                "  All probes cube-filling! Highest-thr fallback: "
                "thr=%.4f (max_ext=%.3f, faces=%d)",
                any_geom_sorted[0][0], any_geom_sorted[0][3], any_geom_sorted[0][2],
            )

        logger.info("-" * 60)
        logger.info(
            "FIELD PROBE RESULT: regime=%s  center=%.4f  "
            "low_max_faces=%d  high_max_faces=%d  "
            "tight_probes=%d  cube_fill_probes=%d",
            regime, center, low_max_faces, high_max_faces,
            len(valid_tight), len(any_geom) - len(valid_tight),
        )
        logger.info("=" * 60)

        self._cached_field_regime = regime

        return regime, center, {
            "probes": results,
            "low_max_faces": low_max_faces,
            "high_max_faces": high_max_faces,
            "tight_probes": len(valid_tight),
            "cube_fill_probes": len(any_geom) - len(valid_tight),
        }

    # ------------------------------------------------------------------
    # Adaptive thresholds — biased UPWARD for probability regime
    # ------------------------------------------------------------------

    def _compute_adaptive_thresholds(
        self,
        regime: str,
        center: float,
        user_threshold: Optional[float] = None,
    ) -> List[float]:
        """
        Compute an adaptive threshold sweep based on the detected field regime.

        **Key fix:** for probability-regime fields the sweep is biased *upward*
        from center.  Low thresholds cause cube-filling fog shells; we push
        toward the iso-surface that wraps the actual subject.
        """
        thresholds: List[float] = []

        if user_threshold is not None and user_threshold > 0:
            base = user_threshold
            if regime == "probability" or base < 2.0:
                offsets = [-0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04, 0.08, 0.15]
            else:
                offsets = [-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0]
            thresholds = [max(1e-4, base + off) for off in offsets]

        elif regime == "probability":
            # Bias UPWARD -- low thresholds cause cube-filling shells.
            # Do NOT go below center (that direction is fog).
            thresholds = sorted(set([
                center,
                center * 1.3,
                center * 1.7,
                center * 2.2,
                center * 3.0,
                center * 4.0,
            ]))
            # Clamp to [1e-4, 0.5]
            thresholds = [max(1e-4, min(0.5, t)) for t in thresholds]

        elif regime == "density":
            thresholds = sorted(set([
                max(0.5, center - 8.0),
                max(0.5, center - 4.0),
                max(0.5, center - 2.0),
                center,
                center + 2.0,
                center + 4.0,
                center + 8.0,
            ]))

        else:
            # Unknown -- broad multi-regime sweep
            thresholds = [0.01, 0.04, 0.08, 0.15, 0.5, 1.0,
                          5.0, 15.0, 25.0, 35.0, 50.0]

        thresholds = sorted(set(round(t, 6) for t in thresholds))

        logger.info(
            "Adaptive thresholds (regime=%s, center=%.4f, user=%s): %s",
            regime, center,
            f"{user_threshold:.4f}" if user_threshold is not None else "auto",
            [f"{t:.4f}" for t in thresholds],
        )
        return thresholds

    # ------------------------------------------------------------------
    # Main job processing
    # ------------------------------------------------------------------

    def process_job(self, job_id: str) -> bool:
        """Process a single TripoSR job."""
        try:
            logger.info(f"Processing job {job_id}")

            job_data = self.job_manager.get_job(job_id)
            if not job_data:
                logger.error(f"Job {job_id} not found")
                return False

            self.job_manager.update_job(
                job_id, status=JobStatus.GENERATING, progress=60
            )

            preview_dir = self.storage_manager.get_job_preview_dir(job_id)
            final_image_path = preview_dir / "final.png"

            logger.info(f"Looking for preprocessed image at: {final_image_path}")

            if not final_image_path.exists():
                error_msg = f"Preprocessed image not found at: {final_image_path}\n"
                if preview_dir.exists():
                    files = list(preview_dir.glob("*"))
                    error_msg += f"Files found: {[f.name for f in files]}\n"
                else:
                    error_msg += "Preview directory does not exist!\n"
                error_msg += "Preprocessing hasn't completed yet or failed."
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            # ----------------------------------------------------------
            # Parameters
            # ----------------------------------------------------------
            params = job_data.get("params", {})
            foreground_ratio = params.get("foreground_ratio", 0.85)

            # Sweep resolution (moderate so the sweep is fast)
            mc_resolution = int(params.get("mc_resolution", 128))
            # Final hi-res re-extraction after selecting best threshold
            mc_resolution_final = int(params.get("mc_resolution_final", 256))

            _raw_mc_threshold = params.get("mc_threshold", None)
            user_mc_threshold: Optional[float] = (
                float(_raw_mc_threshold) if _raw_mc_threshold is not None else None
            )

            # Export mode: "vertex_colors" (default) or "texture"
            # vertex_colors = GLB with per-vertex RGBA, no texture image
            # texture = GLB with xatlas UV unwrap + baked texture image
            export_mode = params.get("export_mode", "vertex_colors")
            bake_texture = bool(params.get("bake_texture", False))
            texture_resolution = int(params.get("texture_resolution", 1024))
            reproject_colors = bool(params.get("reproject_colors", True))
            projection_axis = params.get("projection_axis", "auto")
            chunk_size = int(params.get("chunk_size", 8192))
            model_input_size = int(params.get("model_input_size", 512))
            keep_largest_component = bool(params.get("keep_largest_component", True))

            # ----------------------------------------------------------
            # Load image -- MUST be RGB, never RGBA
            # ----------------------------------------------------------
            logger.info("Loading preprocessed image (already framed by preprocessing)...")
            raw_image = Image.open(final_image_path)
            logger.info(
                "Raw image: mode=%s size=%s -- converting to RGB %dx%d",
                raw_image.mode, raw_image.size, model_input_size, model_input_size,
            )
            input_image = (
                raw_image
                .convert("RGB")  # CRITICAL: TripoSR expects 3-channel RGB
                .resize((model_input_size, model_input_size), Image.BICUBIC)
            )
            logger.info(
                "Model input image: mode=%s size=%s (confirmed RGB)",
                input_image.mode, input_image.size,
            )

            logger.info(
                f"Parameters: foreground_ratio={foreground_ratio}, "
                f"mc_resolution={mc_resolution}, mc_resolution_final={mc_resolution_final}, "
                f"texture_res={texture_resolution}, "
                f"user_mc_threshold={user_mc_threshold}, chunk_size={chunk_size}, "
                f"export_mode={export_mode}, bake_texture={bake_texture}"
            )

            logger.info(f"Reprojection: reproject_colors={reproject_colors}, "
                        f"projection_axis={projection_axis}")

            self.load_model()

            if hasattr(self.model, "renderer") and hasattr(self.model.renderer, "set_chunk_size"):
                self.model.renderer.set_chunk_size(int(chunk_size))
                logger.info(f"Set renderer chunk_size to {chunk_size}")

            self.job_manager.update_job(job_id, progress=70)
            _debug_gpu_tensors("BEFORE_INFERENCE")

            logger.info("Running TripoSR inference...")

            with torch.no_grad():
                scene_codes = self.model(input_image, device=self.device)
                _log_gpu_memory("AFTER_SCENE_CODES")
                self.job_manager.update_job(job_id, progress=80)

                has_vc = True

                # ----------------------------------------------------------
                # Step 1: Probe density field to detect threshold regime
                # ----------------------------------------------------------
                regime, probe_center, probe_diag = self._probe_density_field(
                    scene_codes, has_vc=False,
                )

                # ----------------------------------------------------------
                # Step 2: Compute adaptive threshold sweep
                # ----------------------------------------------------------
                adaptive_thresholds = self._compute_adaptive_thresholds(
                    regime=regime,
                    center=probe_center,
                    user_threshold=user_mc_threshold,
                )

                # ----------------------------------------------------------
                # Step 3: Sweep at mc_resolution, score, pick best
                # ----------------------------------------------------------
                logger.info(
                    "Extracting meshes (sweep_res=%d, has_vc=%s, regime=%s, %d thresholds)...",
                    mc_resolution, has_vc, regime, len(adaptive_thresholds),
                )

                all_candidates: List[Tuple[float, trimesh.Trimesh, float, Optional[str]]] = []
                best_mesh: Optional[trimesh.Trimesh] = None
                best_score = -float("inf")
                best_thr = 0.0

                for thr in adaptive_thresholds:
                    logger.info("--- Trying threshold=%.4f ---", thr)
                    try:
                        extracted = self.model.extract_mesh(
                            scene_codes,
                            has_vertex_color=has_vc,
                            resolution=mc_resolution,
                            threshold=float(thr),
                        )
                        m_raw = extracted[0]
                    except Exception as e:
                        logger.warning("extract_mesh FAILED at thr=%.4f: %s", thr, e)
                        continue

                    _mesh_health(f"thr={thr:.4f}_raw", m_raw)
                    _color_health(f"thr={thr:.4f}_raw_colors", m_raw)

                    m_processed, rejection_reason = self._postprocess_and_diagnose(
                        label=f"thr={thr:.4f}",
                        mesh=m_raw,
                        keep_largest=keep_largest_component,
                    )
                    del m_raw

                    score = mesh_score(m_processed)
                    n_v = len(m_processed.vertices)
                    n_f = len(m_processed.faces)

                    try:
                        parts = m_processed.split(only_watertight=False)
                        n_comp = len(parts)
                    except Exception:
                        n_comp = -1

                    v_arr = np.asarray(m_processed.vertices)
                    v_fin = v_arr[np.isfinite(v_arr).all(axis=1)] if len(v_arr) else v_arr
                    bbox_ext = (
                        (v_fin.max(axis=0) - v_fin.min(axis=0)).tolist()
                        if len(v_fin) else [0, 0, 0]
                    )

                    logger.info(
                        "CANDIDATE thr=%.4f | verts=%d faces=%d components=%d "
                        "bbox=[%.2f, %.2f, %.2f] score=%.1f | %s",
                        thr, n_v, n_f, n_comp,
                        bbox_ext[0], bbox_ext[1], bbox_ext[2], score,
                        f"REJECTED: {rejection_reason}" if rejection_reason else "ACCEPTED",
                    )

                    all_candidates.append((thr, m_processed, score, rejection_reason))

                    if rejection_reason is None and score > best_score:
                        best_mesh = m_processed
                        best_score = score
                        best_thr = thr
                        logger.info(
                            "  * New best at thr=%.4f (score=%.1f, v=%d, f=%d)",
                            thr, score, n_v, n_f,
                        )

                # ----------------------------------------------------------
                # Step 4: Fallback -- never hard-fail
                # ----------------------------------------------------------
                if best_mesh is None:
                    logger.warning("=" * 70)
                    logger.warning(
                        "ALL %d CANDIDATES REJECTED -- picking least-bad fallback",
                        len(all_candidates),
                    )
                    logger.warning("=" * 70)

                    for thr_fb, m_fb, score_fb, reason_fb in sorted(
                        all_candidates, key=lambda x: x[2], reverse=True,
                    ):
                        if len(m_fb.faces) > 0 and len(m_fb.vertices) > 0:
                            best_mesh = m_fb
                            best_score = score_fb
                            best_thr = thr_fb
                            logger.warning(
                                "FALLBACK: thr=%.4f  v=%d  f=%d  score=%.1f  reason=%s",
                                thr_fb, len(m_fb.vertices), len(m_fb.faces),
                                score_fb, reason_fb,
                            )
                            break

                if best_mesh is None:
                    raise RuntimeError(
                        "Mesh extraction failed: all candidates produced empty meshes. "
                        f"Regime={regime}, thresholds={adaptive_thresholds}"
                    )

                # Free non-best candidates
                for _tc, mc, _sc, _rr in all_candidates:
                    if mc is not best_mesh:
                        del mc
                del all_candidates

                logger.info(
                    "Sweep winner: %d verts, %d faces "
                    "(score=%.1f, thr=%.4f, regime=%s, sweep_res=%d)",
                    len(best_mesh.vertices), len(best_mesh.faces),
                    best_score, best_thr, regime, mc_resolution,
                )
                _color_health("sweep_winner", best_mesh)

                # ----------------------------------------------------------
                # Step 5: Two-pass -- re-extract at higher resolution
                # ----------------------------------------------------------
                if mc_resolution_final > mc_resolution:
                    logger.info(
                        "Two-pass: re-extracting thr=%.4f at res=%d (was %d)...",
                        best_thr, mc_resolution_final, mc_resolution,
                    )
                    try:
                        hi_meshes = self.model.extract_mesh(
                            scene_codes,
                            has_vertex_color=has_vc,
                            resolution=mc_resolution_final,
                            threshold=float(best_thr),
                        )
                        hi_raw = hi_meshes[0]
                        _mesh_health("hi_res_raw", hi_raw)
                        _color_health("hi_res_raw_colors", hi_raw)

                        hi_mesh, hi_reason = self._postprocess_and_diagnose(
                            label="hi_res", mesh=hi_raw,
                            keep_largest=keep_largest_component,
                        )
                        del hi_raw

                        if hi_reason is None:
                            hi_score = mesh_score(hi_mesh)
                            logger.info(
                                "Hi-res accepted: %d v, %d f, score=%.1f",
                                len(hi_mesh.vertices), len(hi_mesh.faces), hi_score,
                            )
                            del best_mesh
                            best_mesh = hi_mesh
                            best_score = hi_score
                        else:
                            logger.warning("Hi-res rejected (%s), keeping sweep mesh", hi_reason)
                            del hi_mesh
                    except Exception as e:
                        logger.warning("Hi-res re-extraction failed: %s. Keeping sweep mesh.", e)

                mesh = best_mesh
                logger.info(
                    "Final mesh: %d verts, %d faces (score=%.1f, thr=%.4f, regime=%s)",
                    len(mesh.vertices), len(mesh.faces),
                    best_score, best_thr, regime,
                )
                _color_health("final_mesh_pre_decimate", mesh)

                # Optional decimation
                target_faces = int(params.get("decimate_target_faces", 250_000))
                if target_faces > 0 and len(mesh.faces) > target_faces:
                    mesh = self.decimate_mesh(mesh, target_faces=target_faces)
                    logger.info(
                        "After decimation: %d verts, %d faces (target=%d)",
                        len(mesh.vertices), len(mesh.faces), target_faces,
                    )
                    _color_health("post_decimate", mesh)

                # Free scene_codes (xatlas bake doesn't need them)
                del scene_codes
                torch.cuda.empty_cache()
                logger.info("Scene codes deleted and GPU cache cleared")

                # ----------------------------------------------------------
                # Step 6: Reproject input image colors onto mesh
                # ----------------------------------------------------------
                if reproject_colors:
                    logger.info("Reprojecting input image colors onto mesh...")
                    try:
                        # Prefer a true RGBA cutout for reprojection mask (segmented.png / final_rgba.png)
                        preview_dir = self.storage_manager.get_job_preview_dir(job_id)
                        candidate_rgba = [
                            preview_dir / "segmented.png",
                            preview_dir / "final_rgba.png",
                            preview_dir / "temp_input_triposr_rgba.png",
                            preview_dir / "final.png",  # fallback only
                        ]

                        source_rgba_path = None
                        for p in candidate_rgba:
                            if p.exists():
                                source_rgba_path = str(p)
                                break

                        logger.info("Reprojection source image: %s", source_rgba_path)

                        mesh = self._reproject_image_onto_mesh(
                            mesh=mesh,
                            source_image_path=source_rgba_path,
                            projection_axis=projection_axis,
                        )
                        _color_health("post_reproject", mesh)
                        logger.info(
                            "Image reprojection complete: %d verts, %d faces",
                            len(mesh.vertices), len(mesh.faces),
                        )
                    except Exception as e:
                        logger.warning(
                            "Image reprojection failed: %s. "
                            "Falling back to TripoSR native vertex colors.",
                            e, exc_info=True,
                        )


                self.job_manager.update_job(job_id, progress=85)

                # ----------------------------------------------------------
                # Export path: vertex colors (default) or texture bake
                # ----------------------------------------------------------
                if export_mode == "texture" and bake_texture:
                    logger.info(f"Baking texture at {texture_resolution}x{texture_resolution} "
                                 f"using xatlas...")
                    mesh_with_texture = self.bake_texture(
                        mesh=mesh, texture_resolution=texture_resolution,
                    )
                    self.job_manager.update_job(job_id, progress=95)
                else:
                    # Vertex-color-only export (default path)
                    # Keep vertex colors intact — no xatlas, no TextureVisuals
                    logger.info("Exporting with vertex colors (no texture bake)...")
                    mesh_with_texture = self.normalize_vertex_colors(mesh)
                    mesh_with_texture = self._ensure_vertex_color_material(mesh_with_texture)
                    self.job_manager.update_job(job_id, progress=95)

            _color_health("pre_export", mesh_with_texture)

            output_path = self.storage_manager.get_output_path(job_id, "output.glb")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving output to {output_path}")
            mesh_with_texture.export(str(output_path), file_type="glb")

            # Apply KHR_materials_unlit so vertex colors render as final color
            # (bypasses PBR lighting that causes faint/uneven appearance).
            # Only for vertex-color exports — texture-baked exports use PBR.
            if export_mode != "texture" or not bake_texture:
                force_unlit_glb(str(output_path))

            self.job_manager.update_job(
                job_id, status=JobStatus.COMPLETED, progress=100,
                output_url=f"/api/job/{job_id}/download",
            )
            logger.info(f"Job {job_id} completed successfully")
            self._cleanup_after_job()
            return True

        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}", exc_info=True)
            self.job_manager.update_job(
                job_id, status=JobStatus.FAILED, error=str(e),
            )
            self._cleanup_after_job()
            return False

    # ------------------------------------------------------------------
    # Post-processing with cube-filling rejection
    # ------------------------------------------------------------------

    def _postprocess_and_diagnose(
        self,
        label: str,
        mesh: trimesh.Trimesh,
        keep_largest: bool = True,
    ) -> Tuple[trimesh.Trimesh, Optional[str]]:
        """
        Clean a mesh, optionally keep largest component, and check quality.

        Always returns (processed_mesh, rejection_reason).
        rejection_reason is None when the mesh passes all quality checks.
        """
        m_clean = self.clean_mesh(mesh)
        _mesh_health(f"{label}_clean", m_clean)
        _color_health(f"{label}_clean_colors", m_clean)

        if keep_largest:
            m_clean = self.keep_largest_component(m_clean)
            _mesh_health(f"{label}_largest", m_clean)
            _color_health(f"{label}_largest_colors", m_clean)

        n_verts = len(m_clean.vertices)
        n_faces = len(m_clean.faces)

        v = np.asarray(m_clean.vertices)
        v_finite = v[np.isfinite(v).all(axis=1)] if len(v) else v
        bbox_extents = (
            (v_finite.max(axis=0) - v_finite.min(axis=0))
            if len(v_finite) else np.zeros(3)
        )

        # --- Rejection checks (most to least severe) ---

        if n_faces == 0 or n_verts == 0:
            return m_clean, f"empty mesh (verts={n_verts}, faces={n_faces})"

        # Cube-filling rejection: bbox near the full TripoSR cube (~1.74)
        if float(bbox_extents.max()) > _CUBE_FILL_EXTENT:
            return m_clean, (
                f"cube-filling geometry -- threshold too low "
                f"(max_ext={float(bbox_extents.max()):.3f}, "
                f"extents=[{bbox_extents[0]:.2f}, {bbox_extents[1]:.2f}, "
                f"{bbox_extents[2]:.2f}])"
            )

        if is_mesh_sheet_like(m_clean):
            return m_clean, (
                f"sheet-like geometry "
                f"(extents=[{bbox_extents[0]:.2f}, {bbox_extents[1]:.2f}, "
                f"{bbox_extents[2]:.2f}])"
            )

        if n_faces < 500 or n_verts < 250:
            return m_clean, (
                f"too few primitives (verts={n_verts}, faces={n_faces}, "
                f"min_verts=250, min_faces=500)"
            )

        return m_clean, None  # Accepted

    # ------------------------------------------------------------------
    # Job cleanup
    # ------------------------------------------------------------------

    def _cleanup_after_job(self):
        """Clean up GPU memory after job processing completes."""
        try:
            import gc
            logger.info("Cleaning up GPU memory after job completion...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                free = total - allocated
                logger.info(f"GPU after cleanup: {allocated:.2f}GB alloc, "
                             f"{reserved:.2f}GB reserved, {free:.2f}GB free of {total:.2f}GB")
        except Exception as e:
            logger.warning(f"GPU cleanup warning: {e}")

    # ------------------------------------------------------------------
    # Mesh utilities
    # ------------------------------------------------------------------

    def clean_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Clean mesh by removing NaN/inf values and invalid geometry."""
        logger.info("Cleaning mesh (removing NaN/inf values and invalid geometry)...")

        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        bad_v = ~np.isfinite(vertices).all(axis=1)

        if bad_v.any():
            logger.warning(f"Found {bad_v.sum()} vertices with NaN/inf, removing...")
            valid_vertices = ~bad_v
            vertex_map = np.full(len(vertices), -1, dtype=int)
            vertex_map[valid_vertices] = np.arange(valid_vertices.sum())
            valid_faces = np.all(valid_vertices[faces], axis=1)
            faces = faces[valid_faces]
            faces = vertex_map[faces]
            vertices = vertices[valid_vertices]
            if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                vertex_colors = np.asarray(mesh.visual.vertex_colors)[valid_vertices]
            else:
                vertex_colors = None
        else:
            vertex_colors = (
                np.asarray(mesh.visual.vertex_colors)
                if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None
                else None
            )

        valid_face_idx = ((faces >= 0) & (faces < len(vertices))).all(axis=1)
        if not valid_face_idx.all():
            logger.warning(f"Found {(~valid_face_idx).sum()} faces with invalid indices, removing...")
            faces = faces[valid_face_idx]

        deg = ((faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) |
               (faces[:, 0] == faces[:, 2]))
        if deg.any():
            logger.warning(f"Found {deg.sum()} degenerate faces, removing...")
            faces = faces[~deg]

        if vertex_colors is not None and len(vertex_colors) > 0:
            color_bad_mask = ~np.isfinite(vertex_colors[:, :3]).all(axis=1)
            if color_bad_mask.any():
                logger.warning(f"Found {color_bad_mask.sum()} bad vertex colors, replacing with gray...")
                vertex_colors[color_bad_mask] = (
                    [128, 128, 128, 255] if vertex_colors.shape[1] == 4 else [128, 128, 128]
                )

        cleaned_mesh = trimesh.Trimesh(
            vertices=vertices, faces=faces,
            vertex_colors=vertex_colors, process=False,
        )
        try:
            cleaned_mesh.remove_unreferenced_vertices()
            cleaned_mesh.remove_degenerate_faces()
        except Exception as e:
            logger.warning(f"Cleanup ops failed: {e}")

        logger.info(f"Mesh cleaned: {len(cleaned_mesh.vertices)} verts, "
                     f"{len(cleaned_mesh.faces)} faces")
        return cleaned_mesh

    def keep_largest_component(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Keep only the largest connected component of the mesh."""
        try:
            parts = mesh.split(only_watertight=False)
            if not parts:
                return mesh
            main = max(parts, key=lambda p: len(p.faces))
            return self.clean_mesh(main)
        except Exception as e:
            logger.warning(f"keep_largest_component failed: {e}")
            return mesh

    def decimate_mesh(self, mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
        """
        Decimate mesh to a target face count (quadratic decimation when available).

        Uses nearest-neighbor color transfer to preserve vertex colors after
        decimation, since vertex order is NOT preserved by simplification.
        """
        if target_faces <= 0:
            return mesh
        try:
            if len(mesh.faces) <= target_faces:
                return mesh
            if hasattr(mesh, "simplify_quadratic_decimation"):
                logger.info(f"Decimating: {len(mesh.faces)} -> ~{target_faces} faces")
                m2 = mesh.simplify_quadratic_decimation(int(target_faces))

                # Transfer vertex colors via nearest-neighbor lookup
                # (vertex order changes during decimation, so slicing is wrong)
                if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
                    try:
                        from scipy.spatial import cKDTree
                        original_colors = np.asarray(mesh.visual.vertex_colors)
                        original_verts = np.asarray(mesh.vertices)
                        decimated_verts = np.asarray(m2.vertices)
                        tree = cKDTree(original_verts)
                        _, indices = tree.query(decimated_verts)
                        m2.visual.vertex_colors = original_colors[indices]
                        logger.info(f"Transferred vertex colors via nearest-neighbor "
                                     f"({len(decimated_verts)} verts)")
                    except ImportError:
                        logger.warning(
                            "scipy not available for nearest-neighbor color transfer. "
                            "Skipping decimation to preserve vertex color integrity. "
                            "Install scipy to enable decimation with correct color transfer."
                        )
                        return mesh
                    except Exception as e:
                        logger.warning(f"Color transfer during decimation failed: {e}")

                return self.clean_mesh(m2)
            logger.warning("Decimation not available (missing simplify_quadratic_decimation).")
            return mesh
        except Exception as e:
            logger.warning(f"Decimation failed: {e}")
            return mesh

    def normalize_vertex_colors(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Normalize vertex colors to uint8 RGBA format for GLB export."""
        if not hasattr(mesh.visual, 'vertex_colors') or mesh.visual.vertex_colors is None:
            logger.warning("normalize_vertex_colors: no vertex colors found on mesh!")
            return mesh

        vc = np.asarray(mesh.visual.vertex_colors)
        if vc.ndim == 2 and vc.shape[1] == 3:
            alpha = np.full((vc.shape[0], 1), 255, dtype=vc.dtype)
            vc = np.concatenate([vc, alpha], axis=1)

        if np.issubdtype(vc.dtype, np.floating):
            vc = np.nan_to_num(vc, nan=0.5, posinf=1.0, neginf=0.0)
            if vc.max() <= 1.0:
                vc = vc * 255.0

        vc = np.clip(vc, 0, 255).astype(np.uint8)
        mesh.visual.vertex_colors = vc
        logger.info(f"Normalized vertex colors to uint8 RGBA: shape={vc.shape}")
        return mesh

    def _ensure_vertex_color_material(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Ensure the mesh has a proper vertex-color visual so GLB viewers
        render colors correctly (not black).

        Sets an explicit ColorVisuals with vertex_colors in uint8 RGBA.
        This avoids the TextureVisuals path that strips vertex colors.
        Some viewers need the visual type to be ColorVisuals for per-vertex
        color to be encoded in the glTF vertex attributes.
        """
        vc = None
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            vc = np.asarray(mesh.visual.vertex_colors)

        if vc is None or len(vc) == 0:
            logger.warning("_ensure_vertex_color_material: no vertex colors found, "
                            "assigning default gray")
            vc = np.full((len(mesh.vertices), 4), 128, dtype=np.uint8)
            vc[:, 3] = 255

        # Ensure uint8 RGBA
        if vc.ndim == 2 and vc.shape[1] == 3:
            alpha = np.full((vc.shape[0], 1), 255, dtype=np.uint8)
            vc = np.concatenate([vc, alpha], axis=1)

        if np.issubdtype(vc.dtype, np.floating):
            vc = np.nan_to_num(vc, nan=0.5, posinf=1.0, neginf=0.0)
            if vc.max() <= 1.0:
                vc = (vc * 255.0)
            vc = np.clip(vc, 0, 255).astype(np.uint8)
        else:
            vc = np.clip(vc, 0, 255).astype(np.uint8)

        # Set vertex colors via ColorVisuals (NOT TextureVisuals!)
        # This ensures trimesh encodes COLOR_0 vertex attribute in the GLB
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh,
            vertex_colors=vc,
        )

        logger.info(f"Vertex color material set: {len(vc)} colors, "
                     f"dtype={vc.dtype}, range=[{vc.min()}, {vc.max()}]")
        return mesh


    # ------------------------------------------------------------------
    # Image reprojection — project source image onto mesh surface
    # ------------------------------------------------------------------

    def _reproject_image_onto_mesh(
        self,
        mesh: trimesh.Trimesh,
        source_image_path: str,
        projection_axis: str = "auto",
        front_threshold: float = 0.1,
        background_alpha_threshold: int = 128,
    ) -> trimesh.Trimesh:
        """
        Replace TripoSR's low-quality vertex colors with colors sampled
        directly from the source image via orthographic reprojection.

        TripoSR's ``extract_mesh(has_vertex_color=True)`` produces a
        low-frequency, smeared color field that does not faithfully map
        the input image's appearance.  This method fixes that by:

        1. Loading the source image (with alpha for foreground masking).
        2. Auto-detecting (or using a specified) forward axis.
        3. Computing per-vertex normals to identify front-facing vertices.
        4. Projecting front-facing vertices into image UV space and
           sampling the source image color at each projected position.
        5. Filling back-facing / background vertices via KDTree
           nearest-neighbor from successfully-projected front vertices.

        Args:
            mesh: The extracted trimesh with geometry.
            source_image_path: Path to the preprocessed source image
                (``final.png``).  Loaded as RGBA if possible for
                foreground masking; falls back to RGB.
            projection_axis: One of ``"auto"``, ``"+z"``, ``"-z"``,
                ``"+x"``, ``"-x"``, ``"+y"``, ``"-y"``.
            front_threshold: Dot-product threshold for front-facing
                normals (vertices with ``dot(normal, forward) > threshold``
                are considered front-facing).
            background_alpha_threshold: Alpha value below which a pixel
                is considered background (only used when the image has
                an alpha channel).

        Returns:
            The mesh with reprojected vertex colors applied.
        """
        vertices = np.asarray(mesh.vertices)
        n_verts = len(vertices)
        if n_verts == 0:
            logger.warning("Reprojection: mesh has no vertices, skipping.")
            return mesh

        # ---- Load source image (RGBA preferred for foreground mask) ----
        raw_img = Image.open(source_image_path)
        has_alpha = raw_img.mode in ("RGBA", "LA", "PA")
        img_rgba = raw_img.convert("RGBA")
        img_arr = np.asarray(img_rgba)  # (H, W, 4) uint8
        H, W = img_arr.shape[:2]
        img_rgb = img_arr[:, :, :3]
        img_alpha = img_arr[:, :, 3]

        # Build foreground mask
        if has_alpha:
            fg_mask = img_alpha >= background_alpha_threshold
        else:
            # No alpha channel — assume all pixels are foreground
            fg_mask = np.ones((H, W), dtype=bool)

        logger.info(
            "Reprojection alpha stats: alpha_min=%d alpha_max=%d fg_pct=%.1f%%",
            int(img_alpha.min()), int(img_alpha.max()),
            100.0 * fg_mask.mean()
        )
        fg_count = int(fg_mask.sum())
        logger.info(
            "Reprojection: image %dx%d, has_alpha=%s, fg_pixels=%d/%d (%.1f%%)",
            W, H, has_alpha, fg_count, H * W,
            100.0 * fg_count / max(1, H * W),
        )

        # ---- Compute vertex normals ----
        normals = self._compute_robust_vertex_normals(mesh)

        # ---- Determine forward axis ----
        if projection_axis == "auto":
            forward_axis = self._auto_detect_forward_axis(
                vertices, normals, img_rgb, fg_mask,
            )
        else:
            forward_axis = self._parse_axis_string(projection_axis)

        logger.info("Reprojection: using forward axis = %s", forward_axis)

        # ---- Identify front-facing vertices ----
        dots = normals @ forward_axis
        front_mask = dots > front_threshold
        n_front = int(front_mask.sum())
        logger.info(
            "Reprojection: %d/%d vertices are front-facing (threshold=%.2f)",
            n_front, n_verts, front_threshold,
        )

        if n_front == 0:
            logger.warning(
                "Reprojection: no front-facing vertices found! "
                "Trying with relaxed threshold (0.0)...",
            )
            front_mask = dots > 0.0
            n_front = int(front_mask.sum())
            if n_front == 0:
                logger.warning(
                    "Reprojection: still no front-facing vertices. "
                    "Skipping reprojection.",
                )
                return mesh

        # ---- Project front-facing vertices to image UV ----
        abs_fwd = np.abs(forward_axis)
        fwd_dim = int(np.argmax(abs_fwd))
        plane_dims = [d for d in range(3) if d != fwd_dim]
        fwd_sign = float(forward_axis[fwd_dim])
        d0, d1 = plane_dims

        front_verts = vertices[front_mask]
        # --- Compute foreground bbox in image space (from true alpha mask) ---
        ys, xs = np.where(fg_mask)
        if len(xs) < 10 or len(ys) < 10:
            raise RuntimeError("Foreground mask too small/empty; use a true RGBA cutout image.")

        xmin, xmax = int(xs.min()), int(xs.max())
        ymin, ymax = int(ys.min()), int(ys.max())

        # Add a small padding so edges still catch color
        pad = 2
        xmin = max(0, xmin - pad)
        ymin = max(0, ymin - pad)
        xmax = min(W - 1, xmax + pad)
        ymax = min(H - 1, ymax + pad)

        # --- Compute mesh bbox in the chosen projection plane (front-facing verts) ---
        front_verts = vertices[front_mask]
        plane_xy = front_verts[:, [d0, d1]]

        mx0, mx1 = float(plane_xy[:, 0].min()), float(plane_xy[:, 0].max())
        my0, my1 = float(plane_xy[:, 1].min()), float(plane_xy[:, 1].max())

        # Avoid division by zero
        eps = 1e-8
        sx = max(mx1 - mx0, eps)
        sy = max(my1 - my0, eps)

        # Normalize mesh plane coords to [0,1]
        u0 = (front_verts[:, d0] - mx0) / sx
        v0 = (front_verts[:, d1] - my0) / sy

        # Try both mirror modes and choose the one with the most foreground hits
        def project_and_count(u, v, flip_u: bool):
            if flip_u:
                u = 1.0 - u
            # v axis: image y increases downward, so invert v
            px_f = xmin + u * (xmax - xmin)
            py_f = ymin + (1.0 - v) * (ymax - ymin)
            px_i = np.clip(px_f.astype(np.int32), 0, W - 1)
            py_i = np.clip(py_f.astype(np.int32), 0, H - 1)
            hits = int(fg_mask[py_i, px_i].sum())
            return px_i, py_i, hits

        px_a, py_a, hits_a = project_and_count(u0, v0, flip_u=False)
        px_b, py_b, hits_b = project_and_count(u0, v0, flip_u=True)

        if hits_b > hits_a:
            logger.info("Reprojection: using mirrored U (hits %d > %d)", hits_b, hits_a)
            px, py = px_b, py_b
        else:
            logger.info("Reprojection: using normal U (hits %d >= %d)", hits_a, hits_b)
            px, py = px_a, py_a

        # ---- Sample colors and apply foreground mask ----
        sampled_rgb = img_rgb[py, px]  # (n_front, 3)
        sampled_fg = fg_mask[py, px]   # (n_front,) bool

        n_fg_hit = int(sampled_fg.sum())
        logger.info(
            "Reprojection: %d/%d front verts project to foreground pixels",
            n_fg_hit, n_front,
        )

        # ---- Build final color array ----
        # Start with existing vertex colors as fallback (or gray)
        if (
            hasattr(mesh.visual, "vertex_colors")
            and mesh.visual.vertex_colors is not None
        ):
            existing_vc = np.asarray(mesh.visual.vertex_colors)
            if existing_vc.shape[0] == n_verts:
                final_colors = existing_vc[:, :3].copy()
                if np.issubdtype(final_colors.dtype, np.floating):
                    if final_colors.max() <= 1.0:
                        final_colors = (final_colors * 255).astype(np.uint8)
                    else:
                        final_colors = np.clip(final_colors, 0, 255).astype(np.uint8)
                else:
                    final_colors = np.clip(final_colors, 0, 255).astype(np.uint8)
            else:
                final_colors = np.full((n_verts, 3), 128, dtype=np.uint8)
        else:
            final_colors = np.full((n_verts, 3), 128, dtype=np.uint8)

        # Write reprojected colors for front-facing + foreground vertices
        good_mask_local = sampled_fg  # within front_mask subset
        front_indices = np.where(front_mask)[0]
        good_global_indices = front_indices[good_mask_local]

        final_colors[good_global_indices] = np.clip(
            sampled_rgb[good_mask_local], 0, 255,
        ).astype(np.uint8)

        n_good = len(good_global_indices)
        logger.info("Reprojection: %d vertices got direct image colors", n_good)

        # ---- Fill remaining vertices from nearest good vertex ----
        if n_good > 0 and n_good < n_verts:
            bad_mask = np.ones(n_verts, dtype=bool)
            bad_mask[good_global_indices] = False
            n_bad = int(bad_mask.sum())

            logger.info(
                "Reprojection: filling %d remaining vertices via "
                "nearest-neighbor from %d good vertices...",
                n_bad, n_good,
            )

            good_verts = vertices[good_global_indices]
            bad_verts = vertices[bad_mask]
            tree = cKDTree(good_verts)
            _, nn_indices = tree.query(bad_verts)
            final_colors[bad_mask] = final_colors[good_global_indices[nn_indices]]

            logger.info("Reprojection: nearest-neighbor fill complete")

        # ---- Apply to mesh ----
        alpha_col = np.full((n_verts, 1), 255, dtype=np.uint8)
        rgba = np.concatenate([final_colors, alpha_col], axis=1)

        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh,
            vertex_colors=rgba,
        )

        logger.info(
            "Reprojection complete: applied %d direct + %d filled colors "
            "to %d vertices",
            n_good, n_verts - n_good, n_verts,
        )
        return mesh


    def _compute_robust_vertex_normals(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Compute per-vertex normals robustly, handling degenerate meshes.
        Falls back to a default direction if trimesh normals fail.
        """
        try:
            normals = np.asarray(mesh.vertex_normals)
            if normals.shape[0] == len(mesh.vertices):
                # Check for NaN/zero normals
                norms = np.linalg.norm(normals, axis=1, keepdims=True)
                bad = (norms.squeeze() < 1e-8) | ~np.isfinite(normals).all(axis=1)
                if bad.any():
                    logger.warning(
                        "Reprojection: %d/%d vertex normals are degenerate, "
                        "replacing with [0,0,1]",
                        int(bad.sum()), len(normals),
                    )
                    normals[bad] = [0.0, 0.0, 1.0]
                    norms[bad.reshape(-1)] = 1.0
                # Re-normalize
                normals = normals / np.maximum(norms, 1e-8)
                return normals
        except Exception as e:
            logger.warning("Reprojection: vertex_normals failed: %s", e)

        # Fallback: all normals point toward +Z (front)
        logger.warning(
            "Reprojection: using fallback normals [0,0,1] for all vertices",
        )
        return np.tile([0.0, 0.0, 1.0], (len(mesh.vertices), 1))

    def _auto_detect_forward_axis(
        self,
        vertices: np.ndarray,
        normals: np.ndarray,
        img_rgb: np.ndarray,
        fg_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Auto-detect the forward axis by trying all 6 axis directions and
        picking the one where the most front-facing vertices project onto
        foreground pixels in the source image.
        """
        H, W = fg_mask.shape
        candidates = [
            ("+z", np.array([0.0, 0.0, 1.0])),
            ("-z", np.array([0.0, 0.0, -1.0])),
            ("+x", np.array([1.0, 0.0, 0.0])),
            ("-x", np.array([-1.0, 0.0, 0.0])),
            ("+y", np.array([0.0, 1.0, 0.0])),
            ("-y", np.array([0.0, -1.0, 0.0])),
        ]

        best_name = "+z"
        best_axis = candidates[0][1]
        best_hits = -1

        for name, axis in candidates:
            dots = normals @ axis
            front = dots > 0.1
            if front.sum() == 0:
                continue

            fwd_dim = int(np.argmax(np.abs(axis)))
            fwd_sign = float(axis[fwd_dim])
            plane_dims = [d for d in range(3) if d != fwd_dim]
            d0, d1 = plane_dims

            fv = vertices[front]
            u = (fv[:, d0] + _TRIPOSR_COORD_HALF) / _TRIPOSR_COORD_EXTENT
            v = (_TRIPOSR_COORD_HALF - fv[:, d1]) / _TRIPOSR_COORD_EXTENT
            if fwd_sign < 0:
                u = 1.0 - u

            px = np.clip((u * (W - 1)).astype(np.int32), 0, W - 1)
            py = np.clip((v * (H - 1)).astype(np.int32), 0, H - 1)

            hits = int(fg_mask[py, px].sum())
            logger.info(
                "Axis probe %s: %d front verts, %d fg hits",
                name, int(front.sum()), hits,
            )

            if hits > best_hits:
                best_hits = hits
                best_axis = axis
                best_name = name

        logger.info(
            "Auto-detected forward axis: %s (%d foreground hits)",
            best_name, best_hits,
        )
        return best_axis

    @staticmethod
    def _parse_axis_string(axis_str: str) -> np.ndarray:
        """Parse an axis string like '+z', '-x' into a unit vector."""
        axis_map = {
            "+x": [1, 0, 0], "-x": [-1, 0, 0],
            "+y": [0, 1, 0], "-y": [0, -1, 0],
            "+z": [0, 0, 1], "-z": [0, 0, -1],
        }
        key = axis_str.lower().strip()
        if key in axis_map:
            return np.array(axis_map[key], dtype=float)
        logger.warning("Unknown axis '%s', defaulting to +z", axis_str)
        return np.array([0.0, 0.0, 1.0])


    # ------------------------------------------------------------------
    # Texture baking — vectorized UV splat (no Python for-loop)
    # ------------------------------------------------------------------

    def bake_texture(
        self,
        mesh: trimesh.Trimesh,
        texture_resolution: int,
    ) -> trimesh.Trimesh:
        """
        Bake texture using xatlas UV unwrapping + vertex color sampling.
        Uses vectorized numpy operations for the UV splat (~100x faster
        than the old Python for-loop).

        NOTE: This path is only used when export_mode="texture".
        The default export_mode="vertex_colors" skips this entirely.
        """
        try:
            import xatlas

            logger.info(f"Generating UV map with xatlas (res={texture_resolution})...")
            mesh = self.clean_mesh(mesh)

            vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
            logger.info(f"xatlas: vmapping={len(vmapping)} indices={indices.shape} "
                         f"uvs={uvs.shape}")

            if np.isnan(uvs).any():
                logger.warning("xatlas produced NaN UVs, cleaning...")
                uvs = np.nan_to_num(uvs, nan=0.5)
            if not np.isfinite(uvs).all():
                raise RuntimeError("Non-finite UVs after xatlas.")

            has_colors = (hasattr(mesh.visual, 'vertex_colors') and
                          mesh.visual.vertex_colors is not None)
            if has_colors:
                vertex_colors = np.asarray(mesh.visual.vertex_colors)[vmapping]
            else:
                vertex_colors = None

            mesh.vertices = mesh.vertices[vmapping]
            mesh.faces = indices

            # Vectorized UV splat (replaces slow Python for-loop)
            R = texture_resolution
            texture_array = np.ones((R, R, 3), dtype=np.uint8) * 128

            if vertex_colors is not None:
                uvs_clamped = np.clip(uvs, 0.0, 1.0)
                tx = (uvs_clamped[:, 0] * (R - 1)).astype(np.int32)
                ty = (uvs_clamped[:, 1] * (R - 1)).astype(np.int32)
                valid = (tx >= 0) & (tx < R) & (ty >= 0) & (ty < R)

                # Ensure uint8 RGB
                vc_rgb = np.asarray(vertex_colors[:, :3])
                if np.issubdtype(vc_rgb.dtype, np.floating):
                    if vc_rgb.max() <= 1.0:
                        vc_rgb = vc_rgb * 255.0
                    vc_rgb = np.clip(vc_rgb, 0, 255).astype(np.uint8)
                else:
                    vc_rgb = np.clip(vc_rgb, 0, 255).astype(np.uint8)

                # Vectorized scatter (last-write-wins, same as old loop)
                texture_array[ty[valid], tx[valid]] = vc_rgb[valid]
                logger.info(f"Texture splat: {int(valid.sum())}/{len(valid)} verts mapped")

            texture = Image.fromarray(texture_array)
            material = trimesh.visual.material.PBRMaterial(
                baseColorTexture=texture,
                baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            )
            mesh.visual = trimesh.visual.TextureVisuals(
                uv=uvs, material=material, image=texture,
            )

            logger.info(f"Texture bake complete: {len(mesh.vertices)} verts, "
                         f"{len(mesh.faces)} faces, texture {R}x{R}")
            return mesh

        except Exception as e:
            logger.warning(f"Texture bake failed: {e}. Returning mesh with vertex colors.")
            return self.clean_mesh(mesh)

    # ------------------------------------------------------------------
    # Main worker loop
    # ------------------------------------------------------------------

    def run(self, poll_interval: float = 2.0):
        """Main worker loop - polls queue and processes jobs."""
        logger.info("TripoSR worker started. Polling for jobs...")

        while True:
            try:
                job_id = self.job_manager.get_next_job(ModelEnum.TRIPOSR)
                if job_id:
                    logger.info(f"Picked up job: {job_id}")
                    self.process_job(job_id)
                else:
                    time.sleep(poll_interval)
            except KeyboardInterrupt:
                logger.info("Worker stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in worker loop: {e}", exc_info=True)
                time.sleep(poll_interval)


def main():
    """Main entry point"""
    worker = TripoSRWorker()
    try:
        worker.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    main()

