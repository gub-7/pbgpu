"""
Trellis.2 worker - processes 3D reconstruction jobs using Trellis.2
Polls Redis queue for jobs and generates high-quality textured GLB models with PBR materials

NOTE: This worker uses the stock DINOv3 backbone (facebook/dinov3-vitl16-pretrain-lvd1689m)
which is a gated model. You must set HF_TOKEN in your environment and have accepted the
model license on Hugging Face before running this worker.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from PIL import Image
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from api.job_manager import JobManager
from api.storage import StorageManager
from api.models import JobStatus, ModelEnum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("trellis2_worker")

class Trellis2Worker:
    """Worker for processing Trellis.2 3D reconstruction jobs"""

    def __init__(self):
        self.job_manager = JobManager(
            redis_host=os.environ.get("REDIS_HOST", "localhost"),
            redis_port=int(os.environ.get("REDIS_PORT", "6379")),
            storage_root=os.environ.get("STORAGE_ROOT", "storage"),
        )
        self.storage_manager = StorageManager(
            storage_root=os.environ.get("STORAGE_ROOT", "storage"),
        )
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Trellis.2 Worker initialized on device: {self.device}")

        # Check CUDA/GPU info
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"PyTorch Version: {torch.__version__}")

            # Check for sm_120 support (RTX 5090)
            gpu_name = torch.cuda.get_device_name(0)
            if "5090" in gpu_name or "50" in gpu_name:
                logger.info("RTX 5090 detected - sm_120 support enabled")

    def load_model(self):
        """Load Trellis.2 pipeline with stock DINOv3 backbone"""
        if self.pipeline is not None:
            return

        logger.info("Loading Trellis.2 pipeline...")

        # Verify HF_TOKEN is set (required for gated DINOv3 model)
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            logger.warning(
                "HF_TOKEN not set. The DINOv3 backbone "
                "(facebook/dinov3-vitl16-pretrain-lvd1689m) is a gated model. "
                "Set HF_TOKEN or run 'huggingface-cli login' before starting "
                "this worker."
            )

        try:
            # Import TRELLIS modules
            from trellis2.pipelines import TrellisImageTo3DPipeline
            from trellis2.representations import Gaussian, MeshExtractResult
            from trellis2.utils import render_utils, postprocessing_utils

            # Store utils for later use
            self.render_utils = render_utils
            self.postprocessing_utils = postprocessing_utils
            self.Gaussian = Gaussian
            self.MeshExtractResult = MeshExtractResult

            # Load pipeline (uses DINOv3 backbone from upstream config)
            self.pipeline = TrellisImageTo3DPipeline.from_pretrained(
                "JeffreyXiang/TRELLIS-image-large"
            )
            self.pipeline.to(self.device)

            logger.info("Trellis.2 pipeline loaded successfully (DINOv3 backbone)")

        except ImportError as e:
            logger.error(f"Failed to import TRELLIS modules: {e}")
            logger.error("Make sure TRELLIS.2 is properly installed via setup.sh")
            raise
        except Exception as e:
            logger.error(f"Failed to load Trellis.2 pipeline: {e}")
            raise

    def process_job(self, job_id: str) -> bool:
        """
        Process a single Trellis.2 job

        Args:
            job_id: Job ID to process

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Processing job {job_id}")

            # Get job data
            job_data = self.job_manager.get_job(job_id)
            if not job_data:
                logger.error(f"Job {job_id} not found")
                return False

            # Update status
            self.job_manager.update_job(
                job_id,
                status=JobStatus.GENERATING,
                progress=60
            )

            # Load preprocessed image (final stage)
            preview_dir = self.storage_manager.get_job_preview_dir(job_id)
            final_image_path = preview_dir / "final.png"

            logger.info(f"Looking for preprocessed image at: {final_image_path}")
            logger.info(f"Preview directory: {preview_dir}")

            if not final_image_path.exists():
                # Provide detailed error message for debugging
                error_msg = f"Preprocessed image not found at: {final_image_path}\n"
                error_msg += f"Preview directory: {preview_dir}\n"

                # List what files actually exist in the directory
                if preview_dir.exists():
                    files = list(preview_dir.glob("*"))
                    error_msg += f"Files found in directory: {[f.name for f in files]}\n"
                else:
                    error_msg += "Preview directory does not exist!\n"

                error_msg += "This usually means preprocessing hasn't completed yet or failed."
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            # Load image and ensure it's RGB (not RGBA)
            # Trellis2 expects 3-channel RGB images
            input_image = Image.open(final_image_path).convert('RGB')

            # Get parameters
            params = job_data.get("params", {})
            resolution = params.get("resolution", 1024)
            texture_size = params.get("texture_size", 2048)
            ss_guidance_strength = params.get("ss_guidance_strength", 7.5)
            ss_sampling_steps = params.get("ss_sampling_steps", 12)
            shape_guidance_strength = params.get("shape_guidance_strength", 7.5)
            shape_sampling_steps = params.get("shape_sampling_steps", 12)
            tex_guidance_strength = params.get("tex_guidance_strength", 1.0)
            tex_sampling_steps = params.get("tex_sampling_steps", 12)
            decimation_target = params.get("decimation_target", 500000)

            logger.info(f"Parameters: resolution={resolution}, texture_size={texture_size}, "
                       f"ss_guidance={ss_guidance_strength}, tex_guidance={tex_guidance_strength}")

            # Ensure model is loaded
            self.load_model()

            # Update progress
            self.job_manager.update_job(job_id, progress=65)

            # Run Trellis.2 pipeline
            logger.info("Running Trellis.2 pipeline...")

            # Stage 1: Sparse structure generation
            logger.info("Stage 1: Generating sparse structure...")
            outputs = self.pipeline(
                image=input_image,
                seed=42,
                sparse_structure_sampler_params={
                    "steps": ss_sampling_steps,
                    "cfg_strength": ss_guidance_strength,
                },
                slat_sampler_params={
                    "steps": shape_sampling_steps,
                    "cfg_strength": shape_guidance_strength,
                }
            )

            # Update progress
            self.job_manager.update_job(job_id, progress=75)

            # Stage 2: Extract Gaussian representation
            logger.info("Stage 2: Extracting Gaussian representation...")
            gaussian = outputs['gaussian'][0]

            # Update progress
            self.job_manager.update_job(job_id, progress=80)

            # Stage 3: Extract mesh
            logger.info("Stage 3: Extracting mesh...")
            mesh_result = self.pipeline.extract_mesh(
                gaussian,
                resolution=resolution,
                threshold=0.0
            )[0]

            # Update progress
            self.job_manager.update_job(job_id, progress=85)

            # Stage 4: Bake texture with PBR materials
            logger.info(f"Stage 4: Baking PBR texture at {texture_size}x{texture_size}...")
            mesh_with_texture = self.pipeline.bake_texture(
                mesh_result,
                gaussian,
                texture_size=texture_size,
                texture_bake_steps=tex_sampling_steps,
                texture_cfg_strength=tex_guidance_strength
            )

            # Update progress
            self.job_manager.update_job(job_id, progress=92)

            # Stage 5: Post-processing (decimation, cleanup)
            logger.info("Stage 5: Post-processing mesh...")
            if decimation_target and mesh_result.vertices.shape[0] > decimation_target:
                logger.info(f"Decimating mesh to {decimation_target} vertices...")
                mesh_with_texture = self.postprocessing_utils.decimate_mesh(
                    mesh_with_texture,
                    target=decimation_target
                )

            # Update progress
            self.job_manager.update_job(job_id, progress=97)

            # Save output
            output_path = self.storage_manager.get_output_path(job_id, "output.glb")
            logger.info(f"Saving output to {output_path}")

            # Export as GLB with PBR materials
            self.export_glb(mesh_with_texture, output_path)

            # Update job as completed
            self.job_manager.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                progress=100,
                output_url=f"/api/job/{job_id}/download"
            )

            logger.info(f"Job {job_id} completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}", exc_info=True)

            self.job_manager.update_job(
                job_id,
                status=JobStatus.FAILED,
                error=str(e)
            )

            return False

    def export_glb(self, mesh_result, output_path: Path):
        """
        Export mesh with PBR materials to GLB format

        Args:
            mesh_result: TRELLIS mesh extraction result with PBR textures
            output_path: Output file path
        """
        try:
            import trimesh
            from PIL import Image

            # Extract mesh data
            vertices = mesh_result.vertices.cpu().numpy()
            faces = mesh_result.faces.cpu().numpy()

            # Create trimesh object
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces
            )

            # Extract PBR textures if available
            if hasattr(mesh_result, 'textures'):
                textures = mesh_result.textures

                # Base color texture
                if 'base_color' in textures:
                    base_color = textures['base_color'].cpu().numpy()
                    base_color = (base_color * 255).astype(np.uint8)
                    base_color_image = Image.fromarray(base_color)

                    # Create PBR material
                    material = trimesh.visual.material.PBRMaterial(
                        baseColorTexture=base_color_image,
                        baseColorFactor=[1.0, 1.0, 1.0, 1.0]
                    )

                    # Add metallic/roughness if available
                    if 'metallic' in textures and 'roughness' in textures:
                        metallic = textures['metallic'].cpu().numpy()
                        roughness = textures['roughness'].cpu().numpy()

                        # Combine into single texture (metallic in B, roughness in G)
                        mr_texture = np.zeros((*metallic.shape[:2], 3), dtype=np.uint8)
                        mr_texture[:, :, 1] = (roughness[:, :, 0] * 255).astype(np.uint8)
                        mr_texture[:, :, 2] = (metallic[:, :, 0] * 255).astype(np.uint8)

                        material.metallicRoughnessTexture = Image.fromarray(mr_texture)

                    # Extract UV coordinates
                    if hasattr(mesh_result, 'uv'):
                        uv = mesh_result.uv.cpu().numpy()
                    else:
                        # Generate simple UV mapping if not available
                        logger.warning("No UV coordinates found, generating simple mapping")
                        uv = self._generate_simple_uv(vertices)

                    # Apply material
                    mesh.visual = trimesh.visual.TextureVisuals(
                        uv=uv,
                        material=material,
                        image=base_color_image
                    )

            # Export to GLB
            mesh.export(str(output_path), file_type="glb")

            logger.info(f"Exported GLB with {len(vertices)} vertices and {len(faces)} faces")

        except Exception as e:
            logger.error(f"Error exporting GLB: {e}")
            raise

    def _generate_simple_uv(self, vertices: np.ndarray) -> np.ndarray:
        """Generate simple UV coordinates via spherical projection"""
        # Normalize vertices
        vertices_norm = vertices - vertices.mean(axis=0)
        max_dist = np.abs(vertices_norm).max()
        if max_dist > 0:
            vertices_norm /= max_dist

        # Spherical projection
        x, y, z = vertices_norm[:, 0], vertices_norm[:, 1], vertices_norm[:, 2]

        u = 0.5 + np.arctan2(x, z) / (2 * np.pi)
        v = 0.5 - np.arcsin(np.clip(y, -1, 1)) / np.pi

        return np.stack([u, v], axis=1)

    def run(self, poll_interval: float = 2.0):
        """
        Main worker loop - polls queue and processes jobs

        Args:
            poll_interval: Seconds between queue polls
        """
        logger.info("Trellis.2 worker started. Polling for jobs...")

        while True:
            try:
                # Get next job from queue
                job_id = self.job_manager.get_next_job(ModelEnum.TRELLIS2)

                if job_id:
                    logger.info(f"Picked up job: {job_id}")
                    self.process_job(job_id)
                else:
                    # No jobs, wait
                    time.sleep(poll_interval)

            except KeyboardInterrupt:
                logger.info("Worker stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in worker loop: {e}", exc_info=True)
                time.sleep(poll_interval)


def main():
    """Main entry point"""
    worker = Trellis2Worker()

    try:
        worker.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    main()

