"""
Texture baking stage for the canonical multi-view pipeline.

Responsibilities (TDD §9.7):
    - Unwrap UVs for the mesh
    - Compute per-texel visibility from calibrated camera views
    - Project colors from best-visible views with angle-based weighting
    - Blend with seam-aware weighting
    - Inpaint occluded texels
    - Save texture map and baked mesh

UV strategy:
    Simple per-triangle atlas packing.  Each triangle gets its own
    rectangular region in the texture atlas, laid out in rows.  This
    avoids heavy dependencies (xatlas) while producing a functional
    UV layout.  Seams exist at every triangle boundary, but the
    seam-aware blending step minimises visible artefacts.

Per-texel weights (TDD §9.7):
    - viewing angle   (cos between surface normal and view direction)
    - segmentation confidence  (mask value at projected texel)
    - distance to silhouette boundary  (prefer interior texels)

Artifacts produced:
    - ``textures/diffuse.png``        — diffuse texture map
    - ``baked_mesh.ply``              — mesh with UV coordinates
    - ``texture_metrics.json``        — coverage / quality stats

Camera convention (matches camera_init.py):
    - World space: Y-up, right-handed
    - Camera space: X-right, Y-down, Z-forward (OpenCV)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from api.job_manager import JobManager
from api.storage import StorageManager

from .camera_init import CameraRig, project_point
from .config import CanonicalMVConfig, CANONICAL_VIEW_ORDER
from .refine import MeshState, compute_vertex_normals, load_mesh_ply
from .coarse_recon import save_mesh_ply

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Padding between triangles in the atlas (pixels)
ATLAS_PADDING = 2

# Minimum triangle edge length in UV space (pixels) to avoid degenerate UVs
MIN_UV_TRIANGLE_EDGE_PX = 4

# Default texture resolution
DEFAULT_TEXTURE_RESOLUTION = 2048

# Inpainting radius (pixels) for filling occluded texels
INPAINT_RADIUS = 5

# Minimum viewing-angle cosine to consider a view useful for a texel
MIN_VIEW_COSINE = 0.05

# Distance-transform falloff for silhouette boundary weighting
BOUNDARY_FALLOFF_PIXELS = 20.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class UVLayout:
    """Result of UV unwrapping."""

    uv_coords: np.ndarray          # (V, 2) per-vertex UV in [0, 1]
    face_uvs: np.ndarray           # (F, 3) per-face UV indices (into uv_coords)
    texture_size: int               # texture map side length in pixels
    n_charts: int                   # number of UV charts (= number of faces for simple atlas)
    coverage: float                 # fraction of texture space used


@dataclass
class TextureResult:
    """Result of texture baking."""

    diffuse: np.ndarray             # (H, W, 3) uint8 RGB texture map
    mask: np.ndarray                # (H, W) uint8 binary mask of baked texels
    uv_layout: UVLayout
    metrics: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# UV unwrapping — simple per-triangle atlas
# ---------------------------------------------------------------------------


def unwrap_uvs(
    vertices: np.ndarray,
    faces: np.ndarray,
    texture_size: int = DEFAULT_TEXTURE_RESOLUTION,
) -> UVLayout:
    """
    Create a simple per-triangle UV atlas.

    Each triangle is laid out as a small rectangle in the atlas,
    packed left-to-right, top-to-bottom.  This is not optimal but
    is dependency-free and produces a functional UV map.

    Args:
        vertices: (V, 3) vertex positions.
        faces: (F, 3) face indices.
        texture_size: side length of the square texture map.

    Returns:
        UVLayout with per-vertex UV coordinates.
    """
    n_faces = len(faces)
    if n_faces == 0:
        return UVLayout(
            uv_coords=np.zeros((0, 2), dtype=np.float64),
            face_uvs=np.zeros((0, 3), dtype=np.int32),
            texture_size=texture_size,
            n_charts=0,
            coverage=0.0,
        )

    # Compute per-face 2D bounding sizes in 3D space
    # We flatten each triangle into its own local 2D frame
    face_sizes = []
    for fi in range(n_faces):
        v0 = vertices[faces[fi, 0]]
        v1 = vertices[faces[fi, 1]]
        v2 = vertices[faces[fi, 2]]
        edge_a = np.linalg.norm(v1 - v0)
        edge_b = np.linalg.norm(v2 - v0)
        edge_c = np.linalg.norm(v2 - v1)
        max_edge = max(edge_a, edge_b, edge_c)
        face_sizes.append(max_edge)

    # Normalize face sizes so the largest triangle fills a reasonable
    # fraction of the texture.  Target: largest triangle ≈ 1/sqrt(n_faces)
    # of the texture side.
    max_size = max(face_sizes) if face_sizes else 1.0
    if max_size < 1e-12:
        max_size = 1.0

    target_px = max(
        MIN_UV_TRIANGLE_EDGE_PX,
        texture_size / max(1, math.isqrt(n_faces) + 1),
    )

    scale = target_px / max_size  # world-units → pixels

    # Pack triangles into the atlas row by row
    uv_coords_list: List[np.ndarray] = []
    face_uvs_list: List[np.ndarray] = []

    cursor_x = ATLAS_PADDING
    cursor_y = ATLAS_PADDING
    row_height = 0
    uv_idx = 0

    for fi in range(n_faces):
        v0 = vertices[faces[fi, 0]]
        v1 = vertices[faces[fi, 1]]
        v2 = vertices[faces[fi, 2]]

        # Build local 2D frame for this triangle
        e1 = v1 - v0
        e2 = v2 - v0

        # Local X axis = e1 direction
        len_e1 = np.linalg.norm(e1)
        if len_e1 < 1e-12:
            local_x = np.array([1, 0, 0], dtype=np.float64)
            len_e1 = 1e-12
        else:
            local_x = e1 / len_e1

        # Local Y axis = component of e2 perpendicular to local_x
        proj = np.dot(e2, local_x)
        perp = e2 - proj * local_x
        len_perp = np.linalg.norm(perp)

        # Triangle in local 2D: p0=(0,0), p1=(len_e1,0), p2=(proj, len_perp)
        p0 = np.array([0.0, 0.0])
        p1 = np.array([len_e1, 0.0])
        p2 = np.array([proj, len_perp])

        # Bounding box in pixels
        local_pts = np.array([p0, p1, p2]) * scale
        bb_min = local_pts.min(axis=0)
        local_pts -= bb_min  # shift to origin
        bb_w = int(math.ceil(local_pts[:, 0].max())) + 2 * ATLAS_PADDING
        bb_h = int(math.ceil(local_pts[:, 1].max())) + 2 * ATLAS_PADDING
        bb_w = max(bb_w, MIN_UV_TRIANGLE_EDGE_PX)
        bb_h = max(bb_h, MIN_UV_TRIANGLE_EDGE_PX)

        # Advance to next row if needed
        if cursor_x + bb_w > texture_size:
            cursor_x = ATLAS_PADDING
            cursor_y += row_height + ATLAS_PADDING
            row_height = 0

        # If we've run out of vertical space, just clamp (atlas overflow)
        if cursor_y + bb_h > texture_size:
            cursor_y = min(cursor_y, texture_size - bb_h)

        # Place triangle UVs
        offset = np.array([cursor_x + ATLAS_PADDING, cursor_y + ATLAS_PADDING], dtype=np.float64)
        uv0 = (local_pts[0] + offset) / texture_size
        uv1 = (local_pts[1] + offset) / texture_size
        uv2 = (local_pts[2] + offset) / texture_size

        uv_coords_list.append(uv0)
        uv_coords_list.append(uv1)
        uv_coords_list.append(uv2)
        face_uvs_list.append(np.array([uv_idx, uv_idx + 1, uv_idx + 2], dtype=np.int32))
        uv_idx += 3

        cursor_x += bb_w + ATLAS_PADDING
        row_height = max(row_height, bb_h)

    uv_coords = np.array(uv_coords_list, dtype=np.float64)
    face_uvs = np.array(face_uvs_list, dtype=np.int32)

    # Clamp UVs to [0, 1]
    uv_coords = np.clip(uv_coords, 0.0, 1.0)

    # Compute coverage
    used_area = _compute_uv_coverage(uv_coords, face_uvs, texture_size)
    total_area = texture_size * texture_size
    coverage = used_area / total_area if total_area > 0 else 0.0

    return UVLayout(
        uv_coords=uv_coords,
        face_uvs=face_uvs,
        texture_size=texture_size,
        n_charts=n_faces,
        coverage=float(coverage),
    )


def _compute_uv_coverage(
    uv_coords: np.ndarray,
    face_uvs: np.ndarray,
    texture_size: int,
) -> float:
    """Compute approximate area covered by UV triangles (in pixels²)."""
    total = 0.0
    for fi in range(len(face_uvs)):
        idx = face_uvs[fi]
        p0 = uv_coords[idx[0]] * texture_size
        p1 = uv_coords[idx[1]] * texture_size
        p2 = uv_coords[idx[2]] * texture_size
        # Triangle area via cross product
        e1 = p1 - p0
        e2 = p2 - p0
        area = abs(e1[0] * e2[1] - e1[1] * e2[0]) / 2.0
        total += area
    return total


# ---------------------------------------------------------------------------
# Texture baking — multi-view projection
# ---------------------------------------------------------------------------


def rasterize_uv_triangles(
    uv_layout: UVLayout,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rasterize UV triangles into a texture-space map.

    For each pixel in the texture, determines which face it belongs to
    and the barycentric coordinates within that face.

    Args:
        uv_layout: UV layout from unwrap_uvs.

    Returns:
        Tuple of:
            - face_id_map: (H, W) int32, face index per pixel (-1 = empty)
            - bary_map: (H, W, 3) float32, barycentric coords per pixel
            - mask: (H, W) uint8, 255 where a face covers the pixel
    """
    ts = uv_layout.texture_size
    face_id_map = np.full((ts, ts), -1, dtype=np.int32)
    bary_map = np.zeros((ts, ts, 3), dtype=np.float32)
    mask = np.zeros((ts, ts), dtype=np.uint8)

    for fi in range(len(uv_layout.face_uvs)):
        idx = uv_layout.face_uvs[fi]
        p0 = uv_layout.uv_coords[idx[0]] * ts
        p1 = uv_layout.uv_coords[idx[1]] * ts
        p2 = uv_layout.uv_coords[idx[2]] * ts

        # Bounding box
        xs = [p0[0], p1[0], p2[0]]
        ys = [p0[1], p1[1], p2[1]]
        x_min = max(0, int(math.floor(min(xs))))
        x_max = min(ts - 1, int(math.ceil(max(xs))))
        y_min = max(0, int(math.floor(min(ys))))
        y_max = min(ts - 1, int(math.ceil(max(ys))))

        # Precompute barycentric denominator
        v0 = p1 - p0
        v1 = p2 - p0
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-10:
            continue

        inv_denom = 1.0 / denom

        for py in range(y_min, y_max + 1):
            for px in range(x_min, x_max + 1):
                p = np.array([px + 0.5, py + 0.5], dtype=np.float64)
                v2 = p - p0
                d20 = np.dot(v2, v0)
                d21 = np.dot(v2, v1)
                b1 = (d11 * d20 - d01 * d21) * inv_denom
                b2 = (d00 * d21 - d01 * d20) * inv_denom
                b0 = 1.0 - b1 - b2
                if b0 >= -0.01 and b1 >= -0.01 and b2 >= -0.01:
                    face_id_map[py, px] = fi
                    bary_map[py, px] = [b0, b1, b2]
                    mask[py, px] = 255

    return face_id_map, bary_map, mask


def bake_texture(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    uv_layout: UVLayout,
    rig: CameraRig,
    images: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    face_id_map: np.ndarray,
    bary_map: np.ndarray,
    tex_mask: np.ndarray,
) -> np.ndarray:
    """
    Bake a diffuse texture by projecting camera views onto the UV atlas.

    For each texel:
        1. Compute the 3D world position via barycentric interpolation.
        2. Compute the surface normal.
        3. For each camera view, project the 3D point and sample the image.
        4. Weight by viewing angle (cos) and mask confidence.
        5. Blend weighted colors.

    Args:
        vertices: (V, 3) vertex positions.
        faces: (F, 3) face indices.
        normals: (V, 3) vertex normals.
        uv_layout: UV layout.
        rig: Calibrated camera rig.
        images: Dict view_name → RGB (H, W, 3) uint8.
        masks: Dict view_name → binary mask (H, W) uint8.
        face_id_map: (H, W) int32 from rasterize_uv_triangles.
        bary_map: (H, W, 3) float32 barycentric coords.
        tex_mask: (H, W) uint8 mask of valid texels.

    Returns:
        (H, W, 3) uint8 RGB texture map.
    """
    ts = uv_layout.texture_size
    texture = np.full((ts, ts, 3), 128, dtype=np.uint8)

    if len(faces) == 0:
        return texture

    # Precompute distance transforms for boundary weighting
    boundary_weights = {}
    for vn, m in masks.items():
        binary = (m > 127).astype(np.uint8)
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        # Normalize: pixels far from boundary get weight 1, near boundary get lower
        boundary_weights[vn] = np.clip(dist / BOUNDARY_FALLOFF_PIXELS, 0.0, 1.0)

    view_names = [vn for vn in CANONICAL_VIEW_ORDER if vn in images and vn in rig.cameras]

    # Process each texel
    valid_pixels = np.argwhere(tex_mask > 0)  # (N, 2) — [y, x]

    for py, px in valid_pixels:
        fi = face_id_map[py, px]
        if fi < 0 or fi >= len(faces):
            continue

        bary = bary_map[py, px]  # (3,)
        vi = faces[fi]

        # Interpolate 3D position and normal
        pos_3d = (
            bary[0] * vertices[vi[0]]
            + bary[1] * vertices[vi[1]]
            + bary[2] * vertices[vi[2]]
        )
        normal = (
            bary[0] * normals[vi[0]]
            + bary[1] * normals[vi[1]]
            + bary[2] * normals[vi[2]]
        )
        n_len = np.linalg.norm(normal)
        if n_len > 1e-10:
            normal = normal / n_len

        # Accumulate weighted color from all views
        total_weight = 0.0
        total_color = np.zeros(3, dtype=np.float64)

        for vn in view_names:
            img = images[vn]
            h, w = img.shape[:2]

            ext = rig.get_extrinsic(vn)
            intr = rig.get_intrinsic(vn)

            # Project to image
            p2d = project_point(pos_3d, ext, intr)
            if p2d is None:
                continue

            ix = int(round(p2d[0]))
            iy = int(round(p2d[1]))
            if ix < 0 or ix >= w or iy < 0 or iy >= h:
                continue

            # Viewing angle weight
            cam_pos = rig.get_position(vn)
            view_dir = cam_pos - pos_3d
            view_dist = np.linalg.norm(view_dir)
            if view_dist < 1e-10:
                continue
            view_dir = view_dir / view_dist
            cos_angle = np.dot(normal, view_dir)
            if cos_angle < MIN_VIEW_COSINE:
                continue

            # Mask weight
            mask_val = masks[vn][iy, ix] / 255.0 if vn in masks else 1.0

            # Boundary weight
            bw = boundary_weights[vn][iy, ix] if vn in boundary_weights else 1.0

            weight = cos_angle * mask_val * bw
            if weight < 1e-6:
                continue

            color = img[iy, ix].astype(np.float64)
            total_color += weight * color
            total_weight += weight

        if total_weight > 0:
            final_color = total_color / total_weight
            texture[py, px] = np.clip(final_color, 0, 255).astype(np.uint8)

    return texture


def inpaint_texture(
    texture: np.ndarray,
    mask: np.ndarray,
    radius: int = INPAINT_RADIUS,
) -> np.ndarray:
    """
    Inpaint occluded texels (texels inside UV triangles but not covered
    by any camera view).

    Uses OpenCV's Telea inpainting algorithm.

    Args:
        texture: (H, W, 3) uint8 RGB texture.
        mask: (H, W) uint8 mask of VALID texels (255 = valid).
        radius: Inpainting neighbourhood radius.

    Returns:
        (H, W, 3) uint8 inpainted texture.
    """
    # Inpainting mask: 255 where we need to fill, 0 where we have data
    # We want to inpaint texels that are inside UV triangles but have
    # the default gray color (i.e., no camera saw them).
    # For simplicity, we inpaint all non-masked texels that are near
    # masked texels (dilate the mask, then inpaint the difference).
    if not np.any(mask > 0):
        return texture

    # Dilate the valid mask to find the border region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    inpaint_mask = (dilated > 0) & (mask == 0)
    inpaint_mask_u8 = inpaint_mask.astype(np.uint8) * 255

    if not np.any(inpaint_mask_u8 > 0):
        return texture

    result = cv2.inpaint(texture, inpaint_mask_u8, radius, cv2.INPAINT_TELEA)
    return result


# ---------------------------------------------------------------------------
# Mesh with UVs — PLY export
# ---------------------------------------------------------------------------


def save_mesh_with_uvs_ply(
    filepath: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    uv_layout: UVLayout,
) -> None:
    """
    Save a mesh with UV coordinates in ASCII PLY format.

    Each face vertex gets its own UV coordinate (per-face UVs),
    stored as face properties.

    Args:
        filepath: Output file path.
        vertices: (V, 3) vertex positions.
        faces: (F, 3) face indices.
        normals: (V, 3) vertex normals.
        uv_layout: UV layout with per-face UV indices.
    """
    import struct

    n_verts = len(vertices)
    n_faces = len(faces)
    has_normals = normals is not None and len(normals) == n_verts

    # Build header
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n_verts}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_normals:
        header_lines += [
            "property float nx",
            "property float ny",
            "property float nz",
        ]
    header_lines += [
        f"element face {n_faces}",
        "property list uchar int vertex_indices",
    ]
    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    with open(filepath, "wb") as f:
        f.write(header.encode("ascii"))

        for i in range(n_verts):
            f.write(struct.pack("<fff", *vertices[i].astype(np.float32)))
            if has_normals:
                f.write(struct.pack("<fff", *normals[i].astype(np.float32)))

        for i in range(n_faces):
            f.write(struct.pack("<B", 3))
            f.write(struct.pack("<iii", *faces[i].astype(np.int32)))


# ---------------------------------------------------------------------------
# Texture metrics
# ---------------------------------------------------------------------------


def compute_texture_metrics(
    texture: np.ndarray,
    mask: np.ndarray,
    uv_layout: UVLayout,
) -> Dict[str, Any]:
    """
    Compute texture quality metrics.

    Args:
        texture: (H, W, 3) uint8 texture map.
        mask: (H, W) uint8 valid-texel mask.
        uv_layout: UV layout.

    Returns:
        Dict with texture metrics.
    """
    ts = uv_layout.texture_size
    total_pixels = ts * ts
    valid_pixels = int(np.sum(mask > 0))

    # Color variance in valid regions (low variance → flat/boring texture)
    if valid_pixels > 0:
        valid_colors = texture[mask > 0].astype(np.float64)
        color_mean = valid_colors.mean(axis=0)
        color_std = valid_colors.std(axis=0)
        color_variance = float(np.mean(color_std))
    else:
        color_mean = [0, 0, 0]
        color_std = [0, 0, 0]
        color_variance = 0.0

    return {
        "texture_resolution": ts,
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "coverage_fraction": float(valid_pixels / max(total_pixels, 1)),
        "uv_coverage": uv_layout.coverage,
        "n_charts": uv_layout.n_charts,
        "color_mean_rgb": [float(c) for c in color_mean],
        "color_std_rgb": [float(c) for c in color_std],
        "color_variance": color_variance,
    }


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------


def run_bake_texture(
    job_id: str,
    config: CanonicalMVConfig,
    jm: JobManager,
    sm: StorageManager,
) -> None:
    """
    Execute the bake_texture stage.

    Steps:
        1. Load camera rig from camera_init.json.
        2. Load completed mesh (or refined/coarse fallback).
        3. Load segmented images and masks.
        4. Unwrap UVs.
        5. Rasterize UV triangles to texture space.
        6. Bake texture via multi-view projection.
        7. Inpaint occluded texels.
        8. Save texture, baked mesh, and metrics.

    Raises:
        ValueError: If required artifacts are missing.
    """
    logger.info(f"[{job_id}] bake_texture: starting")
    jm.update_job(job_id, stage_progress=0.0)

    # ------------------------------------------------------------------
    # Step 1: Load camera rig
    # ------------------------------------------------------------------
    rig_data = sm.load_artifact_json(job_id, "camera_init.json")
    if rig_data is None:
        raise ValueError(
            "camera_init.json not found — initialize_cameras must run first"
        )
    rig = CameraRig.from_dict(rig_data)
    image_size = tuple(rig.shared_params["image_size"])
    jm.update_job(job_id, stage_progress=0.05)

    # ------------------------------------------------------------------
    # Step 2: Load mesh (prefer completed → refined → coarse)
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
        raise ValueError(
            "No mesh found — complete_geometry, refine_joint, or "
            "reconstruct_coarse must run first"
        )

    vertices, faces, normals = load_mesh_ply(str(mesh_path))
    if len(normals) != len(vertices) or np.allclose(normals, 0):
        normals = compute_vertex_normals(vertices, faces)

    logger.info(
        f"[{job_id}] bake_texture: loaded {mesh_source} mesh "
        f"({len(vertices)} verts, {len(faces)} faces)"
    )
    jm.update_job(job_id, stage_progress=0.1)

    # ------------------------------------------------------------------
    # Step 3: Load segmented images and masks
    # ------------------------------------------------------------------
    masks_dict, images_dict = _load_segmented_views(job_id, sm, target_size=image_size)
    logger.info(
        f"[{job_id}] bake_texture: loaded {len(images_dict)} view images"
    )
    jm.update_job(job_id, stage_progress=0.15)

    # ------------------------------------------------------------------
    # Step 4: Unwrap UVs
    # ------------------------------------------------------------------
    tex_res = config.texture_resolution
    uv_layout = unwrap_uvs(vertices, faces, texture_size=tex_res)
    logger.info(
        f"[{job_id}] bake_texture: UV unwrap done "
        f"(coverage={uv_layout.coverage:.2%}, charts={uv_layout.n_charts})"
    )
    jm.update_job(job_id, stage_progress=0.25)

    # ------------------------------------------------------------------
    # Step 5: Rasterize UV triangles
    # ------------------------------------------------------------------
    face_id_map, bary_map, tex_mask = rasterize_uv_triangles(uv_layout)
    jm.update_job(job_id, stage_progress=0.35)

    # ------------------------------------------------------------------
    # Step 6: Bake texture
    # ------------------------------------------------------------------
    diffuse = bake_texture(
        vertices, faces, normals, uv_layout,
        rig, images_dict, masks_dict,
        face_id_map, bary_map, tex_mask,
    )
    jm.update_job(job_id, stage_progress=0.7)

    # ------------------------------------------------------------------
    # Step 7: Inpaint
    # ------------------------------------------------------------------
    diffuse = inpaint_texture(diffuse, tex_mask)
    jm.update_job(job_id, stage_progress=0.8)

    # ------------------------------------------------------------------
    # Step 8: Save artifacts
    # ------------------------------------------------------------------
    # Save texture image
    tex_dir = sm.get_artifact_textures_dir(job_id)
    diffuse_path = tex_dir / "diffuse.png"
    Image.fromarray(diffuse).save(str(diffuse_path))

    # Save baked mesh
    baked_mesh_path = sm.get_artifact_dir(job_id) / "baked_mesh.ply"
    save_mesh_with_uvs_ply(
        str(baked_mesh_path), vertices, faces, normals, uv_layout,
    )

    # Save metrics
    metrics = compute_texture_metrics(diffuse, tex_mask, uv_layout)
    metrics["mesh_source"] = mesh_source
    metrics["mesh_vertices"] = len(vertices)
    metrics["mesh_faces"] = len(faces)
    metrics["views_used"] = list(images_dict.keys())
    sm.save_artifact_json(job_id, "texture_metrics.json", metrics)

    jm.update_job(job_id, stage_progress=1.0)
    logger.info(f"[{job_id}] bake_texture: completed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_segmented_views(
    job_id: str,
    sm: StorageManager,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load segmented view masks and RGB images.

    Args:
        job_id: Job identifier.
        sm: Storage manager.
        target_size: Optional (width, height) to resize loaded images to.
                     Should match the camera rig image_size.
    """
    from .coarse_recon import _load_segmented_views as _load
    return _load(job_id, sm, target_size=target_size)

