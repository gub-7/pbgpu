"""
Export stage: convert reconstruction artifacts to downloadable formats.

This module converts the coarse point cloud (PLY) into a GLB mesh so that
downstream consumers (the backend, the frontend 3D viewer) have a standard
file to work with even when TRELLIS.2 is not available.

Pipeline position:
  … → coarse_recon → subject_isolation → **export** → completed

The export is deliberately lightweight – it uses marching cubes on a
voxelised occupancy grid to produce a watertight mesh, then maps vertex
colours from the nearest source points.  If a TRELLIS-produced GLB already
exists it is left untouched and the export stage simply copies/links it
into the canonical ``export/model.glb`` location.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ExportError(Exception):
    """Raised when export fails."""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def export_glb(
    job_dir: Path,
    output_path: Optional[Path] = None,
    voxel_resolution: int = 128,
) -> Path:
    """
    Produce ``export/model.glb`` for a completed job.

    Strategy (in priority order):
      1. If a TRELLIS GLB already exists → copy it to export/
      2. Otherwise build a mesh from the coarse point cloud via
         voxel marching-cubes and export as GLB.

    Parameters
    ----------
    job_dir : root storage directory for the job
    output_path : override for the output GLB location
    voxel_resolution : grid resolution for marching cubes (higher = finer)

    Returns
    -------
    Path to the written GLB file.
    """
    export_dir = job_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    glb_path = output_path or (export_dir / "model.glb")

    # --- Strategy 1: reuse TRELLIS output if available --------------------
    trellis_glb = job_dir / "trellis" / "trellis_output.glb"
    if trellis_glb.exists():
        import shutil

        shutil.copy2(trellis_glb, glb_path)
        logger.info("Copied TRELLIS GLB → %s", glb_path)
        return glb_path

    # --- Strategy 2: build from coarse point cloud ------------------------
    ply_path = job_dir / "coarse_recon" / "coarse_pointcloud.ply"
    if not ply_path.exists():
        raise ExportError(
            f"No point cloud found at {ply_path} and no TRELLIS GLB available."
        )

    logger.info("Building GLB from coarse point cloud: %s", ply_path)
    points, colors = _read_ply(ply_path)

    if len(points) < 100:
        raise ExportError(
            f"Point cloud too small ({len(points)} points) to build a mesh."
        )

    # Build mesh via voxel marching cubes
    verts, faces, vert_colors = _pointcloud_to_mesh(
        points, colors, resolution=voxel_resolution,
    )

    # Export as GLB
    _write_glb(glb_path, verts, faces, vert_colors)
    logger.info(
        "Exported GLB: %d verts, %d faces → %s",
        len(verts), len(faces), glb_path,
    )
    return glb_path


# ---------------------------------------------------------------------------
# Point cloud → mesh conversion
# ---------------------------------------------------------------------------


def _pointcloud_to_mesh(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    resolution: int = 128,
    pad_cells: int = 2,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Convert a point cloud to a triangle mesh using voxel marching cubes.

    1. Voxelise the point cloud into a binary occupancy grid.
    2. Gaussian-smooth the grid for nicer iso-surfaces.
    3. Run marching cubes to extract the iso-surface.
    4. Map vertex colours from the nearest source points.

    Returns (vertices, faces, vertex_colors).
    """
    from scipy import ndimage
    from scipy.spatial import cKDTree

    try:
        from skimage.measure import marching_cubes
    except ImportError:
        from scipy.ndimage import uniform_filter
        # Fallback: use scipy's own marching cubes (available since 0.19)
        # Actually skimage is the standard; let's try importing differently
        raise ExportError(
            "scikit-image is required for mesh export. "
            "Install it with: pip install scikit-image"
        )

    # --- Compute bounding box with padding --------------------------------
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    extent = pmax - pmin
    # Avoid degenerate axes
    extent = np.maximum(extent, extent.max() * 0.01)

    # Add padding so the mesh isn't clipped at boundaries
    pad = extent * (pad_cells / resolution)
    pmin -= pad
    pmax += pad
    extent = pmax - pmin

    # --- Voxelise -----------------------------------------------------------
    voxel_size = extent / resolution
    # Map points to grid indices
    indices = ((points - pmin) / voxel_size).astype(np.int32)
    indices = np.clip(indices, 0, resolution - 1)

    grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1.0

    # Gaussian smooth for nicer iso-surface
    grid = ndimage.gaussian_filter(grid, sigma=1.0)

    # --- Marching cubes -----------------------------------------------------
    # Use a threshold that captures the shape well
    threshold = grid.max() * 0.15
    if threshold < 1e-6:
        raise ExportError("Voxel grid is empty after smoothing.")

    verts_grid, faces, normals, _ = marching_cubes(
        grid, level=threshold, step_size=1,
    )

    # Convert grid coords back to world coords
    verts = verts_grid * voxel_size + pmin

    # --- Map vertex colours from nearest source points --------------------
    vert_colors = None
    if colors is not None and len(colors) == len(points):
        tree = cKDTree(points)
        _, idx = tree.query(verts, k=1)
        vert_colors = colors[idx]

    return verts.astype(np.float32), faces.astype(np.int32), vert_colors


# ---------------------------------------------------------------------------
# GLB writer (minimal, no trimesh dependency)
# ---------------------------------------------------------------------------


def _write_glb(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> None:
    """
    Write a mesh to GLB format.

    Uses trimesh if available, otherwise falls back to a minimal
    manual GLB writer.
    """
    try:
        import trimesh

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if colors is not None:
            # trimesh expects uint8 RGBA
            rgba = np.zeros((len(colors), 4), dtype=np.uint8)
            rgba[:, :3] = np.clip(colors * 255, 0, 255).astype(np.uint8)
            rgba[:, 3] = 255
            mesh.visual.vertex_colors = rgba

        mesh.export(str(path), file_type="glb")
        return
    except ImportError:
        logger.info("trimesh not available, using minimal GLB writer")

    # --- Minimal manual GLB writer (no external deps) ---------------------
    _write_glb_manual(path, vertices, faces, colors)


def _write_glb_manual(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> None:
    """
    Minimal GLB 2.0 writer – produces a valid binary glTF file with a
    single mesh primitive.  No external dependencies required.
    """
    import json
    import struct

    verts = vertices.astype(np.float32)
    tris = faces.astype(np.uint32)

    # Build binary buffer
    vert_bytes = verts.tobytes()
    idx_bytes = tris.tobytes()

    buffer_views = []
    accessors = []
    attributes = {}

    # Vertex positions
    bv_pos = {
        "buffer": 0,
        "byteOffset": 0,
        "byteLength": len(vert_bytes),
        "target": 34962,  # ARRAY_BUFFER
    }
    buffer_views.append(bv_pos)
    acc_pos = {
        "bufferView": 0,
        "componentType": 5126,  # FLOAT
        "count": len(verts),
        "type": "VEC3",
        "max": verts.max(axis=0).tolist(),
        "min": verts.min(axis=0).tolist(),
    }
    accessors.append(acc_pos)
    attributes["POSITION"] = 0

    offset = len(vert_bytes)

    # Vertex colours (if provided)
    color_bytes = b""
    if colors is not None:
        col_u8 = np.clip(colors * 255, 0, 255).astype(np.uint8)
        # Pad to RGBA
        rgba = np.zeros((len(col_u8), 4), dtype=np.uint8)
        rgba[:, :3] = col_u8
        rgba[:, 3] = 255
        color_bytes = rgba.tobytes()

        bv_col = {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(color_bytes),
            "target": 34962,
        }
        buffer_views.append(bv_col)
        acc_col = {
            "bufferView": len(buffer_views) - 1,
            "componentType": 5121,  # UNSIGNED_BYTE
            "normalized": True,
            "count": len(rgba),
            "type": "VEC4",
        }
        accessors.append(acc_col)
        attributes["COLOR_0"] = len(accessors) - 1
        offset += len(color_bytes)

    # Indices
    bv_idx = {
        "buffer": 0,
        "byteOffset": offset,
        "byteLength": len(idx_bytes),
        "target": 34963,  # ELEMENT_ARRAY_BUFFER
    }
    buffer_views.append(bv_idx)
    acc_idx = {
        "bufferView": len(buffer_views) - 1,
        "componentType": 5125,  # UNSIGNED_INT
        "count": tris.size,
        "type": "SCALAR",
    }
    accessors.append(acc_idx)
    idx_accessor = len(accessors) - 1

    total_buf_len = len(vert_bytes) + len(color_bytes) + len(idx_bytes)

    gltf = {
        "asset": {"version": "2.0", "generator": "brickedup-gpu-cluster"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": attributes,
                        "indices": idx_accessor,
                    }
                ]
            }
        ],
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": total_buf_len}],
    }

    json_str = json.dumps(gltf, separators=(",", ":"))
    # Pad JSON to 4-byte boundary
    while len(json_str) % 4 != 0:
        json_str += " "
    json_bytes = json_str.encode("utf-8")

    # Pad binary buffer to 4-byte boundary
    bin_data = vert_bytes + color_bytes + idx_bytes
    while len(bin_data) % 4 != 0:
        bin_data += b"\x00"

    # GLB header
    total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
    header = struct.pack("<III", 0x46546C67, 2, total_length)  # magic, version, length

    # JSON chunk
    json_chunk_header = struct.pack("<II", len(json_bytes), 0x4E4F534A)  # JSON

    # BIN chunk
    bin_chunk_header = struct.pack("<II", len(bin_data), 0x004E4942)  # BIN

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(json_chunk_header)
        f.write(json_bytes)
        f.write(bin_chunk_header)
        f.write(bin_data)


# ---------------------------------------------------------------------------
# PLY reader (reuses logic from coarse_recon but self-contained)
# ---------------------------------------------------------------------------


def _read_ply(path: Path) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Read a PLY file, return (points, colors)."""
    with open(path, "r") as f:
        lines = f.readlines()

    header_end = 0
    n_vertices = 0
    has_color = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("element vertex"):
            n_vertices = int(stripped.split()[-1])
        if "property" in stripped and "red" in stripped:
            has_color = True
        if stripped == "end_header":
            header_end = i + 1
            break

    data_lines = lines[header_end: header_end + n_vertices]
    points = np.zeros((n_vertices, 3), dtype=np.float64)
    colors = np.zeros((n_vertices, 3), dtype=np.float64) if has_color else None

    for i, line in enumerate(data_lines):
        parts = line.strip().split()
        points[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
        if has_color and colors is not None and len(parts) >= 6:
            colors[i] = [
                float(parts[3]) / 255.0,
                float(parts[4]) / 255.0,
                float(parts[5]) / 255.0,
            ]

    return points, colors

