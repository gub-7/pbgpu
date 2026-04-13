"""
Trellis.2 generative completion stage.

This is stage C of the pipeline:
  A. Camera/geometry solved using full images (with background)
  B. Subject isolation – background removed from images and point cloud
  C. **Generative completion** – Trellis.2 produces a clean, complete 3D asset

TRELLIS.2 is a large image-to-3D generative model using an O-Voxel sparse
voxel representation.  It excels at producing plausible, high-quality 3D
assets with complex topology and materials.

Important distinction (per expert guidance):
  - Trellis.2 is a *generative* model, not strict metric photogrammetry
  - Its strength is plausibility and asset quality
  - The cleaner the geometric prior from stages A+B, the less it hallucinates

Inputs:
  - Masked (background-removed) images from subject isolation
  - Optionally: estimated cameras and coarse point cloud / mesh
  - Pipeline configuration

Outputs:
  - High-quality textured mesh (.glb / .obj)
  - Optional voxel representation
  - Metadata about the generation
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from api.models import TrellisResult, ResolvedView

logger = logging.getLogger(__name__)


class TrellisCompletionError(Exception):
    """Raised when Trellis.2 completion fails."""


class TrellisRunner:
    """
    Manages interaction with the TRELLIS.2 model.

    TRELLIS.2 runs in its own micromamba environment (Python 3.10 + cu124),
    so we invoke it as a subprocess to avoid environment conflicts with the
    main pipeline (which may use a different Python/CUDA version).
    """

    def __init__(
        self,
        repo_dir: Path,
        env_name: str = "trellis2",
        device: str = "cuda",
    ):
        self.repo_dir = repo_dir
        self.env_name = env_name
        self.device = device

    def _build_command(
        self,
        image_paths: list[Path],
        output_dir: Path,
        seed: int = 42,
        sparse_structure_steps: int = 12,
        slat_steps: int = 12,
    ) -> list[str]:
        """
        Build the shell command to invoke Trellis.2.

        We use micromamba run to execute in the correct environment.
        """
        # Build a small inline Python script that:
        # 1. Loads the Trellis pipeline
        # 2. Processes the masked images
        # 3. Exports the result
        image_args = " ".join(f'"{p}"' for p in image_paths)

        script = f"""
import sys
sys.path.insert(0, "{self.repo_dir}")

from pathlib import Path
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline

# Load pipeline
pipeline = TrellisImageTo3DPipeline.from_pretrained(
    "JeffreyXiang/TRELLIS-image-large"
)
pipeline.to("{self.device}")

# Load images
images = []
for p in [{', '.join(f'"{p}"' for p in image_paths)}]:
    images.append(Image.open(p))

# Run generation (multi-image mode)
outputs = pipeline.run_multi_image(
    images,
    seed={seed},
    sparse_structure_sampler_params={{
        "steps": {sparse_structure_steps},
    }},
    slat_sampler_params={{
        "steps": {slat_steps},
    }},
)

# Export mesh
output_dir = Path("{output_dir}")
output_dir.mkdir(parents=True, exist_ok=True)

# Export as GLB
glb = outputs.export_glb()
glb_path = output_dir / "trellis_output.glb"
with open(glb_path, "wb") as f:
    f.write(glb)
print(f"TRELLIS_GLB:{{glb_path}}")

# Export as OBJ if available
try:
    obj_data = outputs.export_obj()
    obj_path = output_dir / "trellis_output.obj"
    with open(obj_path, "w") as f:
        f.write(obj_data)
    print(f"TRELLIS_OBJ:{{obj_path}}")
except Exception:
    pass

print("TRELLIS_DONE")
"""
        return [
            "micromamba",
            "run",
            "-n",
            self.env_name,
            "python",
            "-c",
            script,
        ]

    def run(
        self,
        masked_image_paths: list[Path],
        output_dir: Path,
        seed: int = 42,
        sparse_structure_steps: int = 12,
        slat_steps: int = 12,
        timeout: int = 600,
    ) -> TrellisResult:
        """
        Run Trellis.2 completion on masked images.

        Parameters
        ----------
        masked_image_paths : paths to background-removed images
        output_dir : directory for output artifacts
        seed : random seed for reproducibility
        sparse_structure_steps : number of sparse structure sampling steps
        slat_steps : number of SLAT sampling steps
        timeout : maximum seconds to wait for completion
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = self._build_command(
            masked_image_paths,
            output_dir,
            seed=seed,
            sparse_structure_steps=sparse_structure_steps,
            slat_steps=slat_steps,
        )

        logger.info("Running Trellis.2 completion with %d images", len(masked_image_paths))
        logger.debug("Command: %s", " ".join(cmd[:4]) + " ...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.repo_dir),
            )
        except subprocess.TimeoutExpired:
            raise TrellisCompletionError(
                f"Trellis.2 timed out after {timeout}s"
            )
        except FileNotFoundError:
            raise TrellisCompletionError(
                "micromamba not found. Ensure Trellis.2 environment is set up. "
                "Run setup/setup_trellis.sh first."
            )

        if result.returncode != 0:
            logger.error("Trellis.2 stderr: %s", result.stderr)
            raise TrellisCompletionError(
                f"Trellis.2 failed (exit {result.returncode}): {result.stderr[:500]}"
            )

        # Parse output for file paths
        mesh_path = None
        texture_path = None
        metadata: dict[str, Any] = {}

        for line in result.stdout.splitlines():
            if line.startswith("TRELLIS_GLB:"):
                mesh_path = line.split(":", 1)[1].strip()
            elif line.startswith("TRELLIS_OBJ:"):
                metadata["obj_path"] = line.split(":", 1)[1].strip()

        if "TRELLIS_DONE" not in result.stdout:
            raise TrellisCompletionError(
                "Trellis.2 did not complete successfully. "
                f"stdout: {result.stdout[:500]}"
            )

        metadata["seed"] = seed
        metadata["sparse_structure_steps"] = sparse_structure_steps
        metadata["slat_steps"] = slat_steps
        metadata["num_input_images"] = len(masked_image_paths)

        logger.info("Trellis.2 completion successful: %s", mesh_path)

        return TrellisResult(
            mesh_path=mesh_path,
            texture_path=texture_path,
            metadata=metadata,
        )


def run_trellis_completion(
    masked_image_paths: list[str],
    output_dir: Path,
    repo_dir: Optional[Path] = None,
    env_name: str = "trellis2",
    device: str = "cuda",
    seed: int = 42,
    timeout: int = 600,
) -> TrellisResult:
    """
    Top-level entry point for Trellis.2 completion.

    Called by the orchestrator after subject isolation.

    Parameters
    ----------
    masked_image_paths : paths to masked (background-removed) images
    output_dir : directory for output artifacts
    repo_dir : path to TRELLIS.2 repository
    env_name : micromamba environment name
    device : torch device
    seed : random seed
    timeout : max seconds for the subprocess
    """
    from pipelines.config import TRELLIS_REPO_DIR

    repo = repo_dir or TRELLIS_REPO_DIR

    if not repo.exists():
        raise TrellisCompletionError(
            f"TRELLIS.2 repository not found at {repo}. "
            "Run setup/setup_trellis.sh first."
        )

    runner = TrellisRunner(repo_dir=repo, env_name=env_name, device=device)

    image_paths = [Path(p) for p in masked_image_paths]

    return runner.run(
        masked_image_paths=image_paths,
        output_dir=output_dir,
        seed=seed,
        timeout=timeout,
    )

