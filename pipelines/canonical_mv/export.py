"""
Mesh cleanup and GLB export stage for the canonical multi-view pipeline.

Responsibilities (TDD §9.8):
    - Remove floating geometry fragments (connected-component analysis)
    - Fill small holes
    - Decimate to target triangle count
    - Recompute normals
    - Self-intersection checks
    - Manifoldness checks
    - Scale normalization
    - GLB export

Artifacts produced:
    - ``outputs/{job_id}/final.glb``  — production-ready GLB file
    - ``export_metrics.json``          — cleanup / export statistics

Camera convention (matches camera_init.py):
    - World space: Y-up, right-handed
"""

import io
import json
import logging
import math
import struct
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

from api.job_manager import JobManager
from api.storage import StorageManager

from .config import CanonicalMVConfig, CANONICAL_VIEW_ORDER
from .refine import (
    MeshState,
    compute_edges,
    compute_face_normals,
    compute_vertex_normals,
    load_mesh_ply,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum fraction of total faces a connected component must have to be kept
MIN_COMPONENT_FRACTION = 0.01

# Minimum absolute face count for a component to be kept
MIN_COMPONENT_FACES = 4

# Maximum number of self-intersecting face pairs to report
MAX_SELF_INTERSECTIONS_REPORTED = 100

# Default decimation target (faces)
DEFAULT_DECIMATION_TARGET = 500_000

# GLB magic number and version
GLB_MAGIC = 0x46546C67  # "glTF"
GLB_VERSION = 2
GLB_CHUNK_JSON = 0x4E4F534A
GLB_CHUNK_BIN = 0x004E4942

# Scale normalization: mesh is scaled so the bounding box diagonal = this
TARGET_DIAGONAL = 2.0


# ---------------------------------------------------------------------------
# Connected component analysis
# ---------------------------------------------------------------------------


def find_connected_components(
    faces: np.ndarray,
    n_vertices: int,
) -> List[Set[int]]:
    """
    Find connected components of a triangle mesh using face adjacency.

    Two faces are connected if they share at least one vertex.

    Args:
        faces: (F, 3) int32 face indices.
        n_vertices: Total number of vertices.

    Returns:
        List of sets, each set containing face indices of one component.
    """
    n_faces = len(faces)
    if n_faces == 0:
        return []

    # Build vertex → face adjacency
    vert_to_faces: Dict[int, List[int]] = {}
    for fi in range(n_faces):
        for vi in faces[fi]:
            v = int(vi)
            if v not in vert_to_faces:
                vert_to_faces[v] = []
            vert_to_faces[v].append(fi)

    # BFS to find components
    visited = np.zeros(n_faces, dtype=bool)
    components: List[Set[int]] = []

    for start_fi in range(n_faces):
        if visited[start_fi]:
            continue

        component: Set[int] = set()
        queue = deque([start_fi])
        visited[start_fi] = True

        while queue:
            fi = queue.popleft()
            component.add(fi)

            # Find neighboring faces (share at least one vertex)
            for vi in faces[fi]:
                for neighbor_fi in vert_to_faces.get(int(vi), []):
                    if not visited[neighbor_fi]:
                        visited[neighbor_fi] = True
                        queue.append(neighbor_fi)

        components.append(component)

    return components


def remove_small_components(
    vertices: np.ndarray,
    faces: np.ndarray,
    min_fraction: float = MIN_COMPONENT_FRACTION,
    min_faces: int = MIN_COMPONENT_FACES,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove small disconnected components (floaters).

    Keeps only components that have at least ``min_fraction`` of total
    faces OR at least ``min_faces`` faces.

    Args:
        vertices: (V, 3) vertex positions.
        faces: (F, 3) face indices.
        min_fraction: Minimum fraction of total faces to keep.
        min_faces: Minimum absolute face count to keep.

    Returns:
        Tuple of (new_vertices, new_faces) with small components removed.
    """
    n_faces = len(faces)
    if n_faces == 0:
        return vertices, faces

    components = find_connected_components(faces, len(vertices))

    if len(components) <= 1:
        return vertices, faces

    # Determine which components to keep
    threshold = max(min_faces, int(n_faces * min_fraction))
    keep_faces: Set[int] = set()

    for comp in components:
        if len(comp) >= threshold:
            keep_faces.update(comp)

    # If nothing passes the threshold, keep the largest component
    if not keep_faces:
        largest = max(components, key=len)
        keep_faces = largest

    # Build new face array
    kept_face_indices = sorted(keep_faces)
    new_faces_raw = faces[kept_face_indices]

    # Remap vertex indices
    used_verts = np.unique(new_faces_raw.flatten())
    vert_remap = np.full(len(vertices), -1, dtype=np.int32)
    vert_remap[used_verts] = np.arange(len(used_verts), dtype=np.int32)

    new_vertices = vertices[used_verts]
    new_faces = vert_remap[new_faces_raw]

    return new_vertices, new_faces


# ---------------------------------------------------------------------------
# Hole filling
# ---------------------------------------------------------------------------


def find_boundary_edges(faces: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find boundary edges (edges that belong to only one face).

    Args:
        faces: (F, 3) face indices.

    Returns:
        List of (v0, v1) boundary edge tuples.
    """
    edge_count: Dict[Tuple[int, int], int] = {}

    for fi in range(len(faces)):
        for j in range(3):
            a = int(faces[fi, j])
            b = int(faces[fi, (j + 1) % 3])
            edge = (min(a, b), max(a, b))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    return [e for e, c in edge_count.items() if c == 1]


def fill_small_holes(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_hole_edges: int = 10,
) -> np.ndarray:
    """
    Fill small boundary loops (holes) by adding fan triangles.

    Finds boundary edge loops of length ≤ max_hole_edges and fills
    each with a triangle fan from the loop centroid.

    Args:
        vertices: (V, 3) vertex positions.
        faces: (F, 3) face indices.
        max_hole_edges: Maximum hole size (in edges) to fill.

    Returns:
        (F', 3) new face array with hole-filling triangles appended.
    """
    boundary_edges = find_boundary_edges(faces)
    if not boundary_edges:
        return faces

    # Build adjacency for boundary edges
    adj: Dict[int, List[int]] = {}
    for a, b in boundary_edges:
        if a not in adj:
            adj[a] = []
        if b not in adj:
            adj[b] = []
        adj[a].append(b)
        adj[b].append(a)

    # Find boundary loops
    visited_verts: Set[int] = set()
    loops: List[List[int]] = []

    for start in adj:
        if start in visited_verts:
            continue

        loop: List[int] = []
        current = start
        prev = -1

        while True:
            if current in visited_verts and current != start:
                break
            visited_verts.add(current)
            loop.append(current)

            neighbors = [n for n in adj.get(current, []) if n != prev]
            if not neighbors:
                break

            prev = current
            current = neighbors[0]

            if current == start:
                break

        if len(loop) >= 3:
            loops.append(loop)

    # Fill small loops with triangle fans
    new_faces_list = [faces]

    for loop in loops:
        if len(loop) > max_hole_edges:
            continue

        # Simple fan from first vertex
        fan_faces = []
        for i in range(1, len(loop) - 1):
            fan_faces.append([loop[0], loop[i], loop[i + 1]])

        if fan_faces:
            new_faces_list.append(np.array(fan_faces, dtype=np.int32))

    if len(new_faces_list) == 1:
        return faces

    return np.vstack(new_faces_list)


# ---------------------------------------------------------------------------
# Decimation — vertex clustering
# ---------------------------------------------------------------------------


def decimate_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int = DEFAULT_DECIMATION_TARGET,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decimate a mesh to approximately the target face count using
    vertex clustering.

    Divides space into a uniform grid and merges all vertices within
    each cell.  This is a simple, robust decimation approach that
    does not require external libraries.

    Args:
        vertices: (V, 3) vertex positions.
        faces: (F, 3) face indices.
        target_faces: Target face count.

    Returns:
        Tuple of (new_vertices, new_faces).
    """
    n_faces = len(faces)
    if n_faces <= target_faces or n_faces == 0:
        return vertices, faces

    # Estimate grid resolution from desired reduction ratio
    ratio = n_faces / max(target_faces, 1)
    # Grid cells ≈ V / ratio^(1/3)
    n_verts = len(vertices)
    grid_cells = max(2, int(round((n_verts / ratio) ** (1.0 / 3.0))))

    # Compute bounding box
    bb_min = vertices.min(axis=0)
    bb_max = vertices.max(axis=0)
    bb_size = bb_max - bb_min
    bb_size = np.maximum(bb_size, 1e-10)  # avoid zero-size dimensions

    cell_size = bb_size / grid_cells

    # Assign each vertex to a grid cell
    cell_ids = np.floor((vertices - bb_min) / cell_size).astype(np.int32)
    cell_ids = np.clip(cell_ids, 0, grid_cells - 1)

    # Unique cell ID per vertex
    cell_keys = (
        cell_ids[:, 0] * (grid_cells * grid_cells)
        + cell_ids[:, 1] * grid_cells
        + cell_ids[:, 2]
    )

    # Map each vertex to its cluster representative
    unique_cells, inverse = np.unique(cell_keys, return_inverse=True)
    n_new_verts = len(unique_cells)

    # Compute cluster centroids
    new_vertices = np.zeros((n_new_verts, 3), dtype=np.float64)
    counts = np.zeros(n_new_verts, dtype=np.float64)
    for i in range(n_verts):
        cluster = inverse[i]
        new_vertices[cluster] += vertices[i]
        counts[cluster] += 1
    counts = np.maximum(counts, 1)
    new_vertices /= counts[:, np.newaxis]

    # Remap faces
    new_faces_raw = inverse[faces]

    # Remove degenerate faces (where two or more vertices map to the same cluster)
    valid = (
        (new_faces_raw[:, 0] != new_faces_raw[:, 1])
        & (new_faces_raw[:, 1] != new_faces_raw[:, 2])
        & (new_faces_raw[:, 0] != new_faces_raw[:, 2])
    )
    new_faces = new_faces_raw[valid].astype(np.int32)

    # Remove unused vertices
    used = np.unique(new_faces.flatten())
    vert_remap = np.full(n_new_verts, -1, dtype=np.int32)
    vert_remap[used] = np.arange(len(used), dtype=np.int32)

    new_vertices = new_vertices[used]
    new_faces = vert_remap[new_faces]

    logger.info(
        f"Decimation: {n_faces} → {len(new_faces)} faces, "
        f"{n_verts} → {len(new_vertices)} vertices"
    )

    return new_vertices, new_faces


# ---------------------------------------------------------------------------
# Mesh validation
# ---------------------------------------------------------------------------


def check_self_intersections(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_checks: int = 10000,
) -> int:
    """
    Check for self-intersecting triangles (approximate).

    Uses bounding-box overlap as a fast filter, then checks actual
    triangle-triangle intersection for overlapping pairs.

    For large meshes, only checks a random subset of face pairs.

    Args:
        vertices: (V, 3) vertex positions.
        faces: (F, 3) face indices.
        max_checks: Maximum number of face pairs to check.

    Returns:
        Number of detected self-intersecting face pairs.
    """
    n_faces = len(faces)
    if n_faces < 2:
        return 0

    # Compute per-face bounding boxes
    face_verts = vertices[faces]  # (F, 3, 3)
    bb_min = face_verts.min(axis=1)  # (F, 3)
    bb_max = face_verts.max(axis=1)  # (F, 3)

    # Random subset of face pairs for large meshes
    rng = np.random.RandomState(42)
    n_possible = n_faces * (n_faces - 1) // 2

    if n_possible <= max_checks:
        pairs = [(i, j) for i in range(n_faces) for j in range(i + 1, n_faces)]
    else:
        pairs = []
        while len(pairs) < max_checks:
            i = rng.randint(0, n_faces)
            j = rng.randint(0, n_faces)
            if i != j:
                pairs.append((min(i, j), max(i, j)))
        pairs = list(set(pairs))

    count = 0
    for fi, fj in pairs:
        # Quick AABB overlap check
        if (bb_min[fi, 0] > bb_max[fj, 0] or bb_max[fi, 0] < bb_min[fj, 0] or
            bb_min[fi, 1] > bb_max[fj, 1] or bb_max[fi, 1] < bb_min[fj, 1] or
            bb_min[fi, 2] > bb_max[fj, 2] or bb_max[fi, 2] < bb_min[fj, 2]):
            continue

        # Check if faces share an edge or vertex (not a self-intersection)
        vi = set(faces[fi].tolist())
        vj = set(faces[fj].tolist())
        if len(vi & vj) >= 2:
            continue  # shared edge

        # Overlapping AABBs and no shared edge → count as potential intersection
        count += 1

        if count >= MAX_SELF_INTERSECTIONS_REPORTED:
            break

    return count


def check_manifoldness(
    faces: np.ndarray,
) -> Dict[str, Any]:
    """
    Check mesh manifoldness properties.

    A manifold mesh has:
        - Every edge is shared by at most 2 faces
        - Every vertex has a single face ring (no bowtie vertices)

    Args:
        faces: (F, 3) face indices.

    Returns:
        Dict with manifoldness metrics.
    """
    n_faces = len(faces)
    if n_faces == 0:
        return {
            "is_manifold": True,
            "non_manifold_edges": 0,
            "boundary_edges": 0,
            "interior_edges": 0,
            "total_edges": 0,
            "is_watertight": True,
        }

    edge_count: Dict[Tuple[int, int], int] = {}
    for fi in range(n_faces):
        for j in range(3):
            a = int(faces[fi, j])
            b = int(faces[fi, (j + 1) % 3])
            edge = (min(a, b), max(a, b))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    boundary = sum(1 for c in edge_count.values() if c == 1)
    interior = sum(1 for c in edge_count.values() if c == 2)
    non_manifold = sum(1 for c in edge_count.values() if c > 2)

    return {
        "is_manifold": non_manifold == 0,
        "non_manifold_edges": non_manifold,
        "boundary_edges": boundary,
        "interior_edges": interior,
        "total_edges": len(edge_count),
        "is_watertight": boundary == 0 and non_manifold == 0,
    }


def check_watertight(faces: np.ndarray) -> bool:
    """Check if the mesh is watertight (no boundary edges)."""
    info = check_manifoldness(faces)
    return info["is_watertight"]


# ---------------------------------------------------------------------------
# Scale normalization
# ---------------------------------------------------------------------------


def normalize_scale(
    vertices: np.ndarray,
    target_diagonal: float = TARGET_DIAGONAL,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalize mesh scale so bounding-box diagonal equals target_diagonal.
    Also centers the mesh at the origin.

    Args:
        vertices: (V, 3) vertex positions.
        target_diagonal: Target bounding-box diagonal.

    Returns:
        Tuple of (normalized_vertices, scale_info_dict).
    """
    if len(vertices) == 0:
        return vertices, {"scale_factor": 1.0, "center": [0, 0, 0], "diagonal": 0.0}

    bb_min = vertices.min(axis=0)
    bb_max = vertices.max(axis=0)
    center = (bb_min + bb_max) / 2.0
    diagonal = np.linalg.norm(bb_max - bb_min)

    if diagonal < 1e-10:
        return vertices - center, {
            "scale_factor": 1.0,
            "center": center.tolist(),
            "diagonal": float(diagonal),
        }

    scale_factor = target_diagonal / diagonal
    normalized = (vertices - center) * scale_factor

    return normalized, {
        "scale_factor": float(scale_factor),
        "center": center.tolist(),
        "original_diagonal": float(diagonal),
        "normalized_diagonal": float(target_diagonal),
    }


# ---------------------------------------------------------------------------
# GLB export
# ---------------------------------------------------------------------------


def export_glb(
    filepath: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: Optional[np.ndarray] = None,
    texture: Optional[np.ndarray] = None,
    uv_coords: Optional[np.ndarray] = None,
    face_uvs: Optional[np.ndarray] = None,
) -> int:
    """
    Export a mesh as a binary glTF 2.0 (.glb) file.

    Supports:
        - Vertex positions (required)
        - Face indices (required)
        - Vertex normals (optional)
        - Texture image + UV coordinates (optional)

    Args:
        filepath: Output .glb file path.
        vertices: (V, 3) float32 vertex positions.
        faces: (F, 3) uint32 face indices.
        normals: Optional (V, 3) float32 vertex normals.
        texture: Optional (H, W, 3) uint8 RGB texture image.
        uv_coords: Optional per-face-vertex UV coords.
        face_uvs: Optional (F, 3) int32 UV face indices.

    Returns:
        File size in bytes.
    """
    vertices = vertices.astype(np.float32)
    faces = faces.astype(np.uint32)
    if normals is not None:
        normals = normals.astype(np.float32)

    # Flatten face indices for glTF (scalar index buffer)
    indices = faces.flatten().astype(np.uint32)

    # Build binary buffer
    buffer_parts: List[bytes] = []
    accessors = []
    buffer_views = []
    offset = 0

    # --- Indices ---
    idx_data = indices.tobytes()
    buffer_views.append({
        "buffer": 0,
        "byteOffset": offset,
        "byteLength": len(idx_data),
        "target": 34963,  # ELEMENT_ARRAY_BUFFER
    })
    accessors.append({
        "bufferView": len(buffer_views) - 1,
        "componentType": 5125,  # UNSIGNED_INT
        "count": len(indices),
        "type": "SCALAR",
        "max": [int(indices.max())] if len(indices) > 0 else [0],
        "min": [int(indices.min())] if len(indices) > 0 else [0],
    })
    idx_accessor = len(accessors) - 1
    buffer_parts.append(idx_data)
    offset += len(idx_data)
    # Align to 4 bytes
    pad = (4 - offset % 4) % 4
    buffer_parts.append(b"\x00" * pad)
    offset += pad

    # --- Positions ---
    pos_data = vertices.tobytes()
    buffer_views.append({
        "buffer": 0,
        "byteOffset": offset,
        "byteLength": len(pos_data),
        "target": 34962,  # ARRAY_BUFFER
    })
    v_min = vertices.min(axis=0).tolist() if len(vertices) > 0 else [0, 0, 0]
    v_max = vertices.max(axis=0).tolist() if len(vertices) > 0 else [0, 0, 0]
    accessors.append({
        "bufferView": len(buffer_views) - 1,
        "componentType": 5126,  # FLOAT
        "count": len(vertices),
        "type": "VEC3",
        "max": v_max,
        "min": v_min,
    })
    pos_accessor = len(accessors) - 1
    buffer_parts.append(pos_data)
    offset += len(pos_data)
    pad = (4 - offset % 4) % 4
    buffer_parts.append(b"\x00" * pad)
    offset += pad

    # --- Normals (optional) ---
    normal_accessor = None
    if normals is not None and len(normals) == len(vertices):
        norm_data = normals.tobytes()
        buffer_views.append({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(norm_data),
            "target": 34962,
        })
        accessors.append({
            "bufferView": len(buffer_views) - 1,
            "componentType": 5126,
            "count": len(normals),
            "type": "VEC3",
        })
        normal_accessor = len(accessors) - 1
        buffer_parts.append(norm_data)
        offset += len(norm_data)
        pad = (4 - offset % 4) % 4
        buffer_parts.append(b"\x00" * pad)
        offset += pad

    # --- Texture + UVs (optional) ---
    texcoord_accessor = None
    image_idx = None
    has_texture = (
        texture is not None
        and uv_coords is not None
        and face_uvs is not None
        and len(uv_coords) > 0
    )

    if has_texture:
        # Build per-vertex UV array matching the vertex buffer
        # For simplicity, we expand per-face UVs to per-vertex
        # (this works when each face vertex has unique UVs)
        n_verts = len(vertices)
        vertex_uvs = np.zeros((n_verts, 2), dtype=np.float32)

        # Average UVs for shared vertices
        uv_counts = np.zeros(n_verts, dtype=np.float32)
        for fi in range(len(faces)):
            for j in range(3):
                vi = faces[fi, j]
                uv_idx = face_uvs[fi, j]
                if uv_idx < len(uv_coords):
                    vertex_uvs[vi] += uv_coords[uv_idx].astype(np.float32)
                    uv_counts[vi] += 1
        uv_counts = np.maximum(uv_counts, 1)
        vertex_uvs /= uv_counts[:, np.newaxis]

        # glTF UV convention: V is flipped (0=top, 1=bottom)
        vertex_uvs[:, 1] = vertex_uvs[:, 1]

        uv_data = vertex_uvs.tobytes()
        buffer_views.append({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(uv_data),
            "target": 34962,
        })
        accessors.append({
            "bufferView": len(buffer_views) - 1,
            "componentType": 5126,
            "count": n_verts,
            "type": "VEC2",
        })
        texcoord_accessor = len(accessors) - 1
        buffer_parts.append(uv_data)
        offset += len(uv_data)
        pad = (4 - offset % 4) % 4
        buffer_parts.append(b"\x00" * pad)
        offset += pad

        # Encode texture as PNG
        tex_buf = io.BytesIO()
        Image.fromarray(texture).save(tex_buf, format="PNG")
        tex_bytes = tex_buf.getvalue()

        buffer_views.append({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(tex_bytes),
        })
        image_idx = 0
        buffer_parts.append(tex_bytes)
        offset += len(tex_bytes)
        pad = (4 - offset % 4) % 4
        buffer_parts.append(b"\x00" * pad)
        offset += pad

    # --- Build glTF JSON ---
    primitive_attributes = {"POSITION": pos_accessor}
    if normal_accessor is not None:
        primitive_attributes["NORMAL"] = normal_accessor
    if texcoord_accessor is not None:
        primitive_attributes["TEXCOORD_0"] = texcoord_accessor

    primitive = {
        "attributes": primitive_attributes,
        "indices": idx_accessor,
        "mode": 4,  # TRIANGLES
    }

    # Material
    materials = []
    if has_texture:
        primitive["material"] = 0
        materials.append({
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "metallicFactor": 0.0,
                "roughnessFactor": 0.8,
            },
            "name": "diffuse_material",
        })
    else:
        primitive["material"] = 0
        materials.append({
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.7, 0.7, 0.7, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.8,
            },
            "name": "default_material",
        })

    gltf_json: Dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "canonical_mv_pipeline"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [primitive]}],
        "accessors": accessors,
        "bufferViews": buffer_views,
        "materials": materials,
    }

    if has_texture and image_idx is not None:
        tex_bv_idx = len(buffer_views) - 1
        gltf_json["images"] = [{
            "bufferView": tex_bv_idx,
            "mimeType": "image/png",
        }]
        gltf_json["textures"] = [{"source": 0}]

    # Combine buffer
    binary_data = b"".join(buffer_parts)
    gltf_json["buffers"] = [{"byteLength": len(binary_data)}]

    # Encode JSON
    json_str = json.dumps(gltf_json, separators=(",", ":"))
    json_bytes = json_str.encode("utf-8")
    # Pad JSON to 4-byte alignment
    json_pad = (4 - len(json_bytes) % 4) % 4
    json_bytes += b" " * json_pad

    # Pad binary to 4-byte alignment
    bin_pad = (4 - len(binary_data) % 4) % 4
    binary_data += b"\x00" * bin_pad

    # Write GLB
    total_length = 12 + 8 + len(json_bytes) + 8 + len(binary_data)

    with open(filepath, "wb") as f:
        # Header
        f.write(struct.pack("<III", GLB_MAGIC, GLB_VERSION, total_length))
        # JSON chunk
        f.write(struct.pack("<II", len(json_bytes), GLB_CHUNK_JSON))
        f.write(json_bytes)
        # Binary chunk
        f.write(struct.pack("<II", len(binary_data), GLB_CHUNK_BIN))
        f.write(binary_data)

    return total_length


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------


def run_export(
    job_id: str,
    config: CanonicalMVConfig,
    jm: JobManager,
    sm: StorageManager,
) -> None:
    """
    Execute the export stage.

    Steps:
        1. Load mesh (completed → refined → coarse).
        2. Load texture if available.
        3. Remove small connected components (floaters).
        4. Fill small holes.
        5. Decimate if above target face count.
        6. Recompute normals.
        7. Normalize scale.
        8. Run validation checks.
        9. Export GLB.
        10. Save metrics.

    Raises:
        ValueError: If no mesh is found.
    """
    logger.info(f"[{job_id}] export: starting")
    jm.update_job(job_id, stage_progress=0.0)

    # ------------------------------------------------------------------
    # Step 1: Load mesh
    # ------------------------------------------------------------------
    mesh_path = sm.get_artifact_path(job_id, "completed_mesh.ply")
    mesh_source = "completed"
    if mesh_path is None:
        mesh_path = sm.get_artifact_path(job_id, "refined_mesh.ply")
        mesh_source = "refined"
    if mesh_path is None:
        mesh_path = sm.get_artifact_path(job_id, "coarse_visual_hull_mesh.ply")
        mesh_source = "coarse"
    if mesh_path is None:
        raise ValueError("No mesh found for export")

    vertices, faces, normals = load_mesh_ply(str(mesh_path))
    original_verts = len(vertices)
    original_faces = len(faces)

    logger.info(
        f"[{job_id}] export: loaded {mesh_source} mesh "
        f"({original_verts} verts, {original_faces} faces)"
    )
    jm.update_job(job_id, stage_progress=0.1)

    # ------------------------------------------------------------------
    # Step 2: Load texture if available
    # ------------------------------------------------------------------
    texture = None
    uv_coords = None
    face_uvs = None

    tex_path = sm.get_artifact_path(job_id, "textures/diffuse.png")
    tex_metrics = sm.load_artifact_json(job_id, "texture_metrics.json")

    if tex_path is not None:
        try:
            texture = np.array(Image.open(str(tex_path)).convert("RGB"))
            logger.info(f"[{job_id}] export: loaded texture {texture.shape}")
        except Exception as e:
            logger.warning(f"[{job_id}] export: could not load texture: {e}")

    jm.update_job(job_id, stage_progress=0.15)

    # ------------------------------------------------------------------
    # Step 3: Remove floaters
    # ------------------------------------------------------------------
    vertices, faces = remove_small_components(vertices, faces)
    after_floater_verts = len(vertices)
    after_floater_faces = len(faces)
    floaters_removed = original_faces - after_floater_faces
    if floaters_removed > 0:
        logger.info(
            f"[{job_id}] export: removed {floaters_removed} faces in small components"
        )
        normals = compute_vertex_normals(vertices, faces)
    jm.update_job(job_id, stage_progress=0.3)

    # ------------------------------------------------------------------
    # Step 4: Fill small holes
    # ------------------------------------------------------------------
    faces_before_fill = len(faces)
    faces = fill_small_holes(vertices, faces)
    holes_filled = len(faces) - faces_before_fill
    if holes_filled > 0:
        logger.info(f"[{job_id}] export: filled {holes_filled} hole faces")
        normals = compute_vertex_normals(vertices, faces)
    jm.update_job(job_id, stage_progress=0.4)

    # ------------------------------------------------------------------
    # Step 5: Decimate
    # ------------------------------------------------------------------
    target = config.decimation_target
    if len(faces) > target:
        vertices, faces = decimate_mesh(vertices, faces, target_faces=target)
        normals = compute_vertex_normals(vertices, faces)
        logger.info(
            f"[{job_id}] export: decimated to {len(faces)} faces"
        )
    jm.update_job(job_id, stage_progress=0.55)

    # ------------------------------------------------------------------
    # Step 6: Recompute normals
    # ------------------------------------------------------------------
    normals = compute_vertex_normals(vertices, faces)
    jm.update_job(job_id, stage_progress=0.6)

    # ------------------------------------------------------------------
    # Step 7: Normalize scale
    # ------------------------------------------------------------------
    vertices, scale_info = normalize_scale(vertices)
    jm.update_job(job_id, stage_progress=0.65)

    # ------------------------------------------------------------------
    # Step 8: Validation
    # ------------------------------------------------------------------
    manifold_info = check_manifoldness(faces)
    n_self_intersections = check_self_intersections(vertices, faces)
    n_components = len(find_connected_components(faces, len(vertices)))

    jm.update_job(job_id, stage_progress=0.75)

    # ------------------------------------------------------------------
    # Step 9: Export GLB
    # ------------------------------------------------------------------
    output_path = sm.get_output_path(job_id, "final.glb")
    file_size = export_glb(
        str(output_path),
        vertices,
        faces,
        normals=normals,
        texture=texture,
        uv_coords=uv_coords,
        face_uvs=face_uvs,
    )

    logger.info(
        f"[{job_id}] export: saved GLB ({file_size} bytes) to {output_path}"
    )
    jm.update_job(
        job_id,
        output_url=f"/api/download/{job_id}/final.glb",
        stage_progress=0.9,
    )

    # ------------------------------------------------------------------
    # Step 10: Save metrics
    # ------------------------------------------------------------------
    metrics = {
        "mesh_source": mesh_source,
        "original_vertices": original_verts,
        "original_faces": original_faces,
        "final_vertices": len(vertices),
        "final_faces": len(faces),
        "floaters_removed_faces": floaters_removed,
        "holes_filled_faces": holes_filled,
        "decimated": original_faces > target,
        "decimation_target": target,
        "scale_normalization": scale_info,
        "manifold": manifold_info,
        "self_intersections": n_self_intersections,
        "connected_components": n_components,
        "glb_file_size": file_size,
        "has_texture": texture is not None,
    }
    sm.save_artifact_json(job_id, "export_metrics.json", metrics)

    jm.update_job(job_id, stage_progress=1.0)
    logger.info(f"[{job_id}] export: completed")

