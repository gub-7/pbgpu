#!/usr/bin/env python3
"""
Apply all reprojection patches to workers/triposr_worker.py

Usage:
    python patches/apply_patches.py

This script reads the original file, applies all insertions in order,
and writes the result back. It is idempotent — running it twice will
detect that patches are already applied and skip them.
"""
import re
import sys
from pathlib import Path

WORKER = Path(__file__).parent.parent / "workers" / "triposr_worker.py"
PATCHES_DIR = Path(__file__).parent


def read(path):
    return path.read_text(encoding="utf-8")


def already_has(src, marker):
    return marker in src


def insert_after(src, anchor, block):
    """Insert block after the first occurrence of anchor line."""
    idx = src.find(anchor)
    if idx == -1:
        raise ValueError(f"Anchor not found: {anchor!r}")
    # Find end of anchor line
    end = src.index("\n", idx) + 1
    return src[:end] + block + src[end:]


def insert_before(src, anchor, block):
    """Insert block before the first occurrence of anchor line."""
    idx = src.find(anchor)
    if idx == -1:
        raise ValueError(f"Anchor not found: {anchor!r}")
    return src[:idx] + block + src[idx:]


def main():
    if not WORKER.exists():
        print(f"ERROR: {WORKER} not found")
        sys.exit(1)

    src = read(WORKER)
    changed = False

    # ---- PATCH 1: Add scipy import ----
    marker1 = "from scipy.spatial import cKDTree"
    if not already_has(src, marker1):
        print("Applying: add scipy import...")
        src = insert_after(src, "import numpy as np\n", "from scipy.spatial import cKDTree\n")
        changed = True
    else:
        print("Skip: scipy import already present")

    # ---- PATCH 2: Add coordinate constants ----
    marker2 = "_TRIPOSR_COORD_EXTENT"
    if not already_has(src, marker2):
        print("Applying: add coordinate constants...")
        block = (
            "\n"
            "# TripoSR normalizes geometry into roughly [-0.87, 0.87]^3.\n"
            "# Full extent along any axis is ~1.74.\n"
            "_TRIPOSR_COORD_EXTENT = 1.74\n"
            "_TRIPOSR_COORD_HALF = 0.87\n"
        )
        src = insert_after(src, "_CUBE_FILL_EXTENT = 1.65\n", block)
        changed = True
    else:
        print("Skip: coordinate constants already present")

    # ---- PATCH 3: Add params in process_job ----
    marker3 = "reproject_colors = bool(params.get"
    if not already_has(src, marker3):
        print("Applying: add reproject params...")
        block = (
            "            reproject_colors = bool(params.get(\"reproject_colors\", True))\n"
            "            projection_axis = params.get(\"projection_axis\", \"auto\")\n"
        )
        src = insert_after(
            src,
            "            texture_resolution = int(params.get(\"texture_resolution\", 1024))\n",
            block,
        )
        changed = True
    else:
        print("Skip: reproject params already present")

    # ---- PATCH 4: Add param logging ----
    marker4 = "Reprojection: reproject_colors="
    if not already_has(src, marker4):
        print("Applying: add reprojection param logging...")
        block = (
            "            logger.info(f\"Reprojection: reproject_colors={reproject_colors}, \"\n"
            "                        f\"projection_axis={projection_axis}\")\n"
        )
        # Insert after the existing parameter logging block, before self.load_model()
        src = insert_before(
            src,
            "            self.load_model()\n",
            block + "\n",
        )
        changed = True
    else:
        print("Skip: reprojection param logging already present")

    # ---- PATCH 5: Add reprojection call in process_job ----
    marker5 = "Step 6: Reproject input image colors"
    if not already_has(src, marker5):
        print("Applying: add reprojection call in process_job...")
        block = (
            "\n"
            "                # ----------------------------------------------------------\n"
            "                # Step 6: Reproject input image colors onto mesh\n"
            "                # ----------------------------------------------------------\n"
            "                if reproject_colors:\n"
            "                    logger.info(\"Reprojecting input image colors onto mesh...\")\n"
            "                    try:\n"
            "                        mesh = self._reproject_image_onto_mesh(\n"
            "                            mesh=mesh,\n"
            "                            source_image_path=str(final_image_path),\n"
            "                            projection_axis=projection_axis,\n"
            "                        )\n"
            "                        _color_health(\"post_reproject\", mesh)\n"
            "                        logger.info(\n"
            "                            \"Image reprojection complete: %d verts, %d faces\",\n"
            "                            len(mesh.vertices), len(mesh.faces),\n"
            "                        )\n"
            "                    except Exception as e:\n"
            "                        logger.warning(\n"
            "                            \"Image reprojection failed: %s. \"\n"
            "                            \"Falling back to TripoSR native vertex colors.\",\n"
            "                            e, exc_info=True,\n"
            "                        )\n"
            "\n"
        )
        src = insert_after(
            src,
            "                logger.info(\"Scene codes deleted and GPU cache cleared\")\n",
            block,
        )
        changed = True
    else:
        print("Skip: reprojection call already present")

    # ---- PATCH 6: Add reprojection methods to class ----
    marker6 = "def _reproject_image_onto_mesh("
    if not already_has(src, marker6):
        print("Applying: add reprojection methods...")
        reproject_code = read(PATCHES_DIR / "snippet_reproject_methods.py")
        helper_code = read(PATCHES_DIR / "snippet_helper_methods.py")
        block = (
            "\n"
            + reproject_code
            + "\n"
            + helper_code
            + "\n"
        )
        # Insert before the texture baking section
        anchor = "    # ------------------------------------------------------------------\n    # Texture baking"
        src = insert_before(src, anchor, block)
        changed = True
    else:
        print("Skip: reprojection methods already present")

    if changed:
        WORKER.write_text(src, encoding="utf-8")
        print(f"\nPatched: {WORKER}")
    else:
        print("\nNo changes needed — all patches already applied.")

    # Verify syntax
    print("\nVerifying syntax...")
    import py_compile
    try:
        py_compile.compile(str(WORKER), doraise=True)
        print("SYNTAX OK")
    except py_compile.PyCompileError as e:
        print(f"SYNTAX ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

