"""
Coarse reconstruction stage for the canonical multi-view pipeline.

Orthographic voxel-carving approach
====================================

This module reconstructs a coarse 3D volume from three dead-on
orthographic views (front, top, side) using shape-from-silhouettes.

The three views are not arbitrary perspective photographs. They are
axis-aligned orthographic projections with these exact adjacencies:

    * bottom of top matches top of front
    * right of side matches left of front
    * left of top matches top of side

The reconstruction formula is simply:

    V[z, y, x] = Front[z, x] AND Top[y, x] AND Side[z, y]

where x runs left to right (front width / top width),
y runs back to front (top height / side width),
z runs top to bottom (front height / side height).

After voxel carving the volume is cleaned with morphological operations,
converted to a triangle mesh via marching cubes, and optionally a coarse
Gaussian point cloud is sampled on the surface for downstream stages.

Coordinate mapping (voxel to world, Y-up right-handed):

    world_x =  (x - X/2) / max_dim * 2 * extent
    world_y = -(z - Z/2) / max_dim * 2 * extent   (Z=0 is top -> +Y)
    world_z =  (y - Y/2) / max_dim * 2 * extent   (Y_max is front -> +Z)

Artifacts produced:
    - coarse_voxel.npz             raw occupancy grid + metadata
    - coarse_visual_hull_mesh.ply  triangle mesh from marching cubes
    - coarse_gaussians.ply         Gaussian point cloud on surface
    - coarse_depth_{view}.png      per-view depth maps (orthographic)
"""

import io
import logging
import struct
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    from scipy import ndimage
except ImportError:
    ndimage = None

from api.job_manager import JobManager
from api.storage import StorageManager

from .config import CanonicalMVConfig, CANONICAL_VIEW_ORDER

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_GRID_RESOLUTION = 256
DEFAULT_GRID_HALF_EXTENT = 1.0
DEFAULT_N_SURFACE_POINTS = 50_000
DEFAULT_GAUSSIAN_SCALE_FACTOR = 1.5
MIN_OCCUPIED_VOXELS = 10
MASK_FILL_HOLES = True
MASK_OPEN_ITERS = 1
MASK_CLOSE_ITERS = 2
VOLUME_CLOSE_ITERS = 1
VOLUME_OPEN_ITERS = 1
BG_THRESH = 28
DEPTH_MODE = "average"


def extract_mask(img_rgba, use_alpha=True, bg_thresh=BG_THRESH):
    """Extract a binary subject mask from an RGBA or RGB image."""
    rgb = img_rgba[:, :, :3]
    alpha = img_rgba[:, :, 3]
    if use_alpha and np.any(alpha < 250):
        return alpha > 10
    h, w = rgb.shape[:2]
    corners = np.array([
        rgb[0, 0], rgb[0, w - 1], rgb[h - 1, 0], rgb[h - 1, w - 1],
    ], dtype=np.float32)
    bg = corners.mean(axis=0)
    dist = np.linalg.norm(rgb.astype(np.float32) - bg[np.newaxis, np.newaxis, :], axis=2)
    return dist > bg_thresh


def clean_mask(mask, fill_holes=MASK_FILL_HOLES, open_iters=MASK_OPEN_ITERS, close_iters=MASK_CLOSE_ITERS):
    """Clean a binary mask with morphological operations."""
    m = mask.astype(bool)
    if ndimage is not None:
        if fill_holes:
            m = ndimage.binary_fill_holes(m)
        if open_iters > 0:
            m = ndimage.binary_opening(m, iterations=open_iters)
        if close_iters > 0:
            m = ndimage.binary_closing(m, iterations=close_iters)
        m = ndimage.binary_fill_holes(m)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mu8 = m.astype(np.uint8) * 255
        if close_iters > 0:
            mu8 = cv2.morphologyEx(mu8, cv2.MORPH_CLOSE, kernel, iterations=close_iters)
        if open_iters > 0:
            mu8 = cv2.morphologyEx(mu8, cv2.MORPH_OPEN, kernel, iterations=open_iters)
        m = mu8 > 127
    return m


def resize_mask(mask, size_wh):
    """Resize a boolean mask to (width, height) using nearest-neighbor."""
    pil = Image.fromarray((mask.astype(np.uint8) * 255))
    pil = pil.resize(size_wh, Image.NEAREST)
    return np.array(pil) > 127


def prepare_views(front_mask, top_mask, side_mask, grid_resolution=DEFAULT_GRID_RESOLUTION, depth_mode=DEPTH_MODE):
    """Resize the three orthographic masks to a common voxel grid."""
    fz, fx = front_mask.shape
    ty, tx = top_mask.shape
    sz, sy = side_mask.shape
    X_raw = fx
    Z_raw = fz
    if depth_mode == "top":
        Y_raw = ty
    elif depth_mode == "side":
        Y_raw = sy
    else:
        Y_raw = int(round((ty + sy) / 2))
    max_dim = max(X_raw, Y_raw, Z_raw)
    sc = grid_resolution / max(max_dim, 1)
    X = max(int(round(X_raw * sc)), 1)
    Y = max(int(round(Y_raw * sc)), 1)
    Z = max(int(round(Z_raw * sc)), 1)
    front_r = resize_mask(front_mask, (X, Z))
    top_r = resize_mask(top_mask, (X, Y))
    side_r = resize_mask(side_mask, (Y, Z))
    return front_r, top_r, side_r, X, Y, Z


def reconstruct_volume(front_mask, top_mask, side_mask):
    """Build occupancy volume: V[z, y, x] = Front[z, x] AND Top[y, x] AND Side[z, y]"""
    Z, X = front_mask.shape
    Y, X2 = top_mask.shape
    Z2, Y2 = side_mask.shape
    assert X == X2, f"X mismatch: {X} != {X2}"
    assert Z == Z2, f"Z mismatch: {Z} != {Z2}"
    assert Y == Y2, f"Y mismatch: {Y} != {Y2}"
    front_3d = front_mask[:, np.newaxis, :]
    top_3d = top_mask[np.newaxis, :, :]
    side_3d = side_mask[:, :, np.newaxis]
    return front_3d & top_3d & side_3d


def cleanup_volume(volume):
    """Morphologically clean the occupancy volume."""
    v = volume.astype(bool)
    if ndimage is not None:
        if VOLUME_CLOSE_ITERS > 0:
            v = ndimage.binary_closing(v, iterations=VOLUME_CLOSE_ITERS)
        v = ndimage.binary_fill_holes(v)
        if VOLUME_OPEN_ITERS > 0:
            v = ndimage.binary_opening(v, iterations=VOLUME_OPEN_ITERS)
    return v


def voxel_to_world(voxel_coords, X, Y, Z, half_extent=DEFAULT_GRID_HALF_EXTENT):
    """Convert voxel (z, y, x) indices to world (wx, wy, wz) positions."""
    max_dim = max(X, Y, Z)
    sc = 2.0 * half_extent / max(max_dim, 1)
    z = voxel_coords[:, 0].astype(np.float64)
    y = voxel_coords[:, 1].astype(np.float64)
    x = voxel_coords[:, 2].astype(np.float64)
    wx = (x - X / 2.0 + 0.5) * sc
    wy = -(z - Z / 2.0 + 0.5) * sc
    wz = (y - Y / 2.0 + 0.5) * sc
    return np.stack([wx, wy, wz], axis=1)


def extract_surface_mesh(volume, X, Y, Z, half_extent=DEFAULT_GRID_HALF_EXTENT, level=0.5):
    """Extract a triangle mesh from the occupancy volume using marching cubes."""
    vol = volume.astype(np.float32)
    if vol.max() < level:
        raise ValueError("Occupancy volume is empty -- check segmentation masks.")
    mc_level = level - 1e-6 if vol.max() <= level else level
    try:
        from skimage.measure import marching_cubes
        verts_vox, faces, normals_vox, _ = marching_cubes(vol, level=mc_level)
        max_dim = max(X, Y, Z)
        sc = 2.0 * half_extent / max(max_dim, 1)
        wx = (verts_vox[:, 2] - X / 2.0 + 0.5) * sc
        wy = -(verts_vox[:, 0] - Z / 2.0 + 0.5) * sc
        wz = (verts_vox[:, 1] - Y / 2.0 + 0.5) * sc
        vertices = np.stack([wx, wy, wz], axis=1).astype(np.float64)
        nx = normals_vox[:, 2]
        ny = -normals_vox[:, 0]
        nz = normals_vox[:, 1]
        normals = np.stack([nx, ny, nz], axis=1).astype(np.float64)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(norms, 1e-12)
        logger.info(f"Mesh extracted: {len(vertices)} vertices, {len(faces)} faces")
        return vertices, faces.astype(np.int32), normals
    except ImportError:
        logger.warning("scikit-image not available -- fallback to point cloud")
        return _fallback_surface_extraction(volume, X, Y, Z, half_extent, level)


def _fallback_surface_extraction(volume, X, Y, Z, half_extent=DEFAULT_GRID_HALF_EXTENT, level=0.5):
    """Fallback: return surface voxel centers as a point cloud."""
    binary = volume >= level
    N_z, N_y, N_x = binary.shape
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    has_empty = np.zeros_like(binary)
    for dz, dy, dx in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
        nbr = padded[1+dz:N_z+1+dz, 1+dy:N_y+1+dy, 1+dx:N_x+1+dx]
        has_empty |= ~nbr
    surface = binary & has_empty
    coords = np.argwhere(surface)
    if len(coords) == 0:
        raise ValueError("No surface voxels found.")
    vertices = voxel_to_world(coords, X, Y, Z, half_extent)
    faces = np.zeros((0, 3), dtype=np.int32)
    grad = np.gradient(volume.astype(np.float64))
    normals = np.zeros_like(vertices)
    for i, (iz, iy, ix) in enumerate(coords):
        n = np.array([grad[2][iz, iy, ix], -grad[0][iz, iy, ix], grad[1][iz, iy, ix]])
        nm = np.linalg.norm(n)
        if nm > 1e-10:
            normals[i] = n / nm
    return vertices, faces, normals


def compute_depth_maps(volume, X, Y, Z, image_size=(512, 512)):
    """Render per-view depth maps from the occupancy volume."""
    binary = volume.astype(bool)
    w, h = image_size
    depth_maps = {}
    front_depth = np.zeros((Z, X), dtype=np.float32)
    for z in range(Z):
        for x in range(X):
            col = binary[z, :, x]
            occ = np.where(col)[0]
            if len(occ) > 0:
                front_depth[z, x] = float(Y - occ[-1]) / Y
    depth_maps["front"] = cv2.resize(front_depth, (w, h), interpolation=cv2.INTER_LINEAR)
    top_depth = np.zeros((Y, X), dtype=np.float32)
    for y in range(Y):
        for x in range(X):
            col = binary[:, y, x]
            occ = np.where(col)[0]
            if len(occ) > 0:
                top_depth[y, x] = float(occ[0] + 1) / Z
    depth_maps["top"] = cv2.resize(top_depth, (w, h), interpolation=cv2.INTER_LINEAR)
    side_depth = np.zeros((Z, Y), dtype=np.float32)
    for z in range(Z):
        for y in range(Y):
            col = binary[z, y, :]
            occ = np.where(col)[0]
            if len(occ) > 0:
                side_depth[z, y] = float(occ[0] + 1) / X
    depth_maps["side"] = cv2.resize(side_depth, (w, h), interpolation=cv2.INTER_LINEAR)
    return depth_maps


def sample_surface_points(volume, X, Y, Z, half_extent=DEFAULT_GRID_HALF_EXTENT, n_points=DEFAULT_N_SURFACE_POINTS, seed=None):
    """Sample points on the volume surface for Gaussian initialization."""
    rng = np.random.RandomState(seed)
    binary = volume.astype(bool)
    N_z, N_y, N_x = binary.shape
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    has_empty = np.zeros_like(binary)
    for dz, dy, dx in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
        nbr = padded[1+dz:N_z+1+dz, 1+dy:N_y+1+dy, 1+dx:N_x+1+dx]
        has_empty |= ~nbr
    surface = binary & has_empty
    surface_indices = np.argwhere(surface)
    n_surface = len(surface_indices)
    if n_surface == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    if n_points <= n_surface:
        chosen = rng.choice(n_surface, size=n_points, replace=False)
    else:
        chosen = rng.choice(n_surface, size=n_points, replace=True)
    chosen_vox = surface_indices[chosen]
    max_dim = max(X, Y, Z)
    voxel_size = 2.0 * half_extent / max(max_dim, 1)
    jitter = rng.uniform(-0.4, 0.4, size=(n_points, 3)) * voxel_size
    points = voxel_to_world(chosen_vox, X, Y, Z, half_extent) + jitter
    grad = np.gradient(volume.astype(np.float64))
    normals = np.zeros((n_points, 3), dtype=np.float64)
    for i, (iz, iy, ix) in enumerate(chosen_vox):
        n = np.array([grad[2][iz, iy, ix], -grad[0][iz, iy, ix], grad[1][iz, iy, ix]])
        nm = np.linalg.norm(n)
        if nm > 1e-10:
            normals[i] = n / nm
    return points, normals


def color_surface_points(points, normals, images, masks, X, Y, Z, half_extent=DEFAULT_GRID_HALF_EXTENT):
    """Assign colors to surface points by projecting into the best view."""
    n_pts = len(points)
    colors = np.full((n_pts, 3), 128, dtype=np.uint8)
    if n_pts == 0:
        return colors
    max_dim = max(X, Y, Z)
    sc = 2.0 * half_extent / max(max_dim, 1)
    best_scores = np.full(n_pts, -1.0)
    for vn in ["front", "top", "side"]:
        if vn not in images:
            continue
        img = images[vn]
        ih, iw = img.shape[:2]
        wx, wy, wz = points[:, 0], points[:, 1], points[:, 2]
        if vn == "front":
            img_x = wx / sc + X / 2.0 - 0.5
            img_z = -wy / sc + Z / 2.0 - 0.5
            px = img_x / X * iw
            py = img_z / Z * ih
            scores = np.abs(normals[:, 2])
        elif vn == "top":
            img_x = wx / sc + X / 2.0 - 0.5
            img_y = wz / sc + Y / 2.0 - 0.5
            px = img_x / X * iw
            py = img_y / Y * ih
            scores = np.abs(normals[:, 1])
        elif vn == "side":
            img_y = wz / sc + Y / 2.0 - 0.5
            img_z = -wy / sc + Z / 2.0 - 0.5
            px = img_y / Y * iw
            py = img_z / Z * ih
            scores = np.abs(normals[:, 0])
        else:
            continue
        valid = (px >= 0) & (px < iw) & (py >= 0) & (py < ih)
        better = valid & (scores > best_scores)
        idx = np.where(better)[0]
        if len(idx) > 0:
            px_i = np.clip(px[idx].astype(np.int32), 0, iw - 1)
            py_i = np.clip(py[idx].astype(np.int32), 0, ih - 1)
            colors[idx] = img[py_i, px_i]
            best_scores[idx] = scores[idx]
    return colors


def build_coarse_gaussians(points, colors, normals, voxel_size, scale_factor=DEFAULT_GAUSSIAN_SCALE_FACTOR):
    """Build a coarse Gaussian point cloud from surface samples."""
    n = len(points)
    gs = voxel_size * scale_factor
    return {
        "positions": points.astype(np.float32),
        "colors": colors.astype(np.uint8),
        "scales": np.full((n, 3), gs, dtype=np.float32),
        "opacities": np.ones((n, 1), dtype=np.float32),
        "normals": normals.astype(np.float32),
    }


def save_mesh_ply(filepath, vertices, faces, normals=None):
    """Save a triangle mesh in binary PLY format."""
    n_verts = len(vertices)
    n_faces = len(faces)
    has_normals = normals is not None and len(normals) == n_verts
    header_lines = ["ply", "format binary_little_endian 1.0", f"element vertex {n_verts}",
                    "property float x", "property float y", "property float z"]
    if has_normals:
        header_lines += ["property float nx", "property float ny", "property float nz"]
    if n_faces > 0:
        header_lines += [f"element face {n_faces}", "property list uchar int vertex_indices"]
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


def save_gaussians_ply(filepath, gaussians):
    """Save a Gaussian point cloud in binary PLY format."""
    positions = gaussians["positions"]
    colors = gaussians["colors"]
    scales = gaussians["scales"]
    opacities = gaussians["opacities"]
    norms = gaussians["normals"]
    n = len(positions)
    header_lines = ["ply", "format binary_little_endian 1.0", f"element vertex {n}",
                    "property float x", "property float y", "property float z",
                    "property float nx", "property float ny", "property float nz",
                    "property uchar red", "property uchar green", "property uchar blue",
                    "property float scale_x", "property float scale_y", "property float scale_z",
                    "property float opacity", "end_header"]
    header = "\n".join(header_lines) + "\n"
    with open(filepath, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(n):
            f.write(struct.pack("<fff", *positions[i]))
            f.write(struct.pack("<fff", *norms[i]))
            f.write(struct.pack("<BBB", *colors[i]))
            f.write(struct.pack("<fff", *scales[i]))
            f.write(struct.pack("<f", opacities[i, 0]))


def save_depth_map_png(filepath, depth_map, max_depth=None):
    """Save a depth map as a 16-bit grayscale PNG."""
    if max_depth is None:
        max_depth = float(depth_map[depth_map > 0].max()) if np.any(depth_map > 0) else 1.0
    normalized = np.zeros_like(depth_map, dtype=np.uint16)
    valid = depth_map > 0
    if np.any(valid) and max_depth > 0:
        normalized[valid] = (depth_map[valid] / max_depth * 65535).astype(np.uint16)
    img = Image.fromarray(normalized, mode="I;16")
    img.save(filepath)


def render_visual_hull_preview(volume, X, Y, Z, half_extent=DEFAULT_GRID_HALF_EXTENT, image_size=(512, 512), azimuth_deg=35.0, elevation_deg=25.0, distance=3.5):
    """Render a preview image of the visual hull from a 3/4 angle."""
    import math
    binary = volume.astype(bool)
    surface_coords = np.argwhere(binary)
    w, h = image_size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (30, 30, 40)
    if len(surface_coords) == 0:
        return img
    surface_world = voxel_to_world(surface_coords, X, Y, Z, half_extent)
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    cam_x = distance * math.cos(el) * math.sin(az)
    cam_y = distance * math.sin(el)
    cam_z = distance * math.cos(el) * math.cos(az)
    eye = np.array([cam_x, cam_y, cam_z], dtype=np.float64)
    target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = -np.cross(fwd, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-10:
        world_up = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        right = -np.cross(fwd, world_up)
        right_norm = np.linalg.norm(right)
    right = right / right_norm
    up = np.cross(right, fwd)
    up = up / np.linalg.norm(up)
    R_view = np.zeros((3, 3), dtype=np.float64)
    R_view[0, :] = right
    R_view[1, :] = up
    R_view[2, :] = -fwd
    t_view = -R_view @ eye
    cam_coords = (R_view @ surface_world.T).T + t_view
    focal = w * 0.8
    cx, cy = w / 2.0, h / 2.0
    z_cam = cam_coords[:, 2]
    valid = z_cam > 0.01
    with np.errstate(divide="ignore", invalid="ignore"):
        px = focal * cam_coords[:, 0] / z_cam + cx
        py = -focal * cam_coords[:, 1] / z_cam + cy
    in_bounds = valid & (px >= 0) & (px < w) & (py >= 0) & (py < h)
    valid_idx = np.where(in_bounds)[0]
    if len(valid_idx) == 0:
        return img
    depth_buf = np.full((h, w), np.inf, dtype=np.float64)
    color_buf = np.zeros((h, w, 3), dtype=np.uint8)
    depths = z_cam[valid_idx]
    min_d, max_d = depths.min(), depths.max()
    depth_range = max(max_d - min_d, 0.001)
    px_int = px[valid_idx].astype(np.int32)
    py_int = py[valid_idx].astype(np.int32)
    for i in range(len(valid_idx)):
        x_i, y_i, d = px_int[i], py_int[i], depths[i]
        if d < depth_buf[y_i, x_i]:
            depth_buf[y_i, x_i] = d
            t = 1.0 - (d - min_d) / depth_range
            r = int(40 + 80 * t)
            g = int(120 + 135 * t)
            b = int(180 + 75 * t)
            color_buf[y_i, x_i] = (r, g, b)
    vis_mask = depth_buf < np.inf
    img[vis_mask] = color_buf[vis_mask]
    return img


def run_incremental_debug_recon(job_id, config, jm, sm, masks, images, grid_res):
    """Run incremental voxel carving for debugging."""
    logger.info(f"[{job_id}] debug: running incremental reconstruction")
    view_order = [vn for vn in CANONICAL_VIEW_ORDER if vn in masks]
    if not view_order:
        return {}
    preview_paths = {}
    artifact_dir = sm.get_artifact_dir(job_id)
    for pass_idx in range(len(view_order)):
        active_views = view_order[:pass_idx + 1]
        view_label = " + ".join(active_views)
        pass_name = f"debug_recon_{pass_idx + 1}view"
        try:
            active_masks = {vn: masks[vn] for vn in active_views}
            cleaned = {}
            for vn, m in active_masks.items():
                bm = m > 127 if m.dtype == np.uint8 else m.astype(bool)
                cleaned[vn] = clean_mask(bm)
            f_mask = cleaned.get("front", np.ones((grid_res, grid_res), dtype=bool))
            t_mask = cleaned.get("top", np.ones((grid_res, grid_res), dtype=bool))
            s_mask = cleaned.get("side", np.ones((grid_res, grid_res), dtype=bool))
            f_r, t_r, s_r, gX, gY, gZ = prepare_views(f_mask, t_mask, s_mask, grid_resolution=grid_res)
            vol = reconstruct_volume(f_r, t_r, s_r)
            vol = cleanup_volume(vol)
            n_occupied = int(vol.sum())
            preview = render_visual_hull_preview(vol, gX, gY, gZ, image_size=(512, 512))
            preview_path = artifact_dir / f"{pass_name}.png"
            Image.fromarray(preview).save(str(preview_path))
            preview_paths[active_views[-1]] = f"{pass_name}.png"
            logger.info(f"[{job_id}] debug pass {pass_idx + 1}: {n_occupied} occupied voxels")
        except Exception as e:
            logger.warning(f"[{job_id}] debug pass {pass_idx + 1} failed: {e}", exc_info=True)
            preview_paths[active_views[-1]] = f"{pass_name}.png"
    sm.save_artifact_json(job_id, "debug_incremental_recon.json", {
        "passes": len(view_order),
        "views_per_pass": [view_order[:i+1] for i in range(len(view_order))],
        "preview_files": preview_paths,
    })
    return preview_paths


def run_reconstruct_coarse(job_id, config, jm, sm):
    """Execute the reconstruct_coarse stage using orthographic voxel carving."""
    logger.info(f"[{job_id}] reconstruct_coarse: starting")
    jm.update_job(job_id, stage_progress=0.0)

    raw_masks, images = _load_segmented_views(job_id, sm, target_size=None)
    if len(raw_masks) < 3:
        raise ValueError(f"Only {len(raw_masks)} segmented views found, need 3")
    logger.info(f"[{job_id}] reconstruct_coarse: loaded {len(raw_masks)} views")
    jm.update_job(job_id, stage_progress=0.1)

    cleaned_masks = {}
    for vn in CANONICAL_VIEW_ORDER:
        if vn not in raw_masks:
            continue
        m = raw_masks[vn]
        bm = m > 127 if m.dtype == np.uint8 else m.astype(bool)
        cleaned_masks[vn] = clean_mask(bm)
    for vn, cm in cleaned_masks.items():
        mask_path = sm.get_artifact_dir(job_id) / f"clean_mask_{vn}.png"
        Image.fromarray((cm.astype(np.uint8) * 255)).save(str(mask_path))
    jm.update_job(job_id, stage_progress=0.15)

    grid_res = _grid_resolution_from_config(config)
    if getattr(config, "debug_incremental_recon", False):
        run_incremental_debug_recon(job_id, config, jm, sm, cleaned_masks, images, grid_res)
        jm.update_job(job_id, stage_progress=0.2)

    front_mask = cleaned_masks.get("front")
    top_mask = cleaned_masks.get("top")
    side_mask = cleaned_masks.get("side")
    if front_mask is None or top_mask is None or side_mask is None:
        raise ValueError("Missing one or more required views (front, top, side)")

    front_r, top_r, side_r, X, Y, Z = prepare_views(front_mask, top_mask, side_mask, grid_resolution=grid_res)
    logger.info(f"[{job_id}] reconstruct_coarse: aligned grid X={X}, Y={Y}, Z={Z}")
    for name, mask_arr in [("front", front_r), ("top", top_r), ("side", side_r)]:
        p = sm.get_artifact_dir(job_id) / f"aligned_mask_{name}.png"
        Image.fromarray((mask_arr.astype(np.uint8) * 255)).save(str(p))

    volume = reconstruct_volume(front_r, top_r, side_r)
    n_raw = int(volume.sum())
    logger.info(f"[{job_id}] reconstruct_coarse: raw intersection {n_raw} voxels ({n_raw / volume.size * 100:.1f}%)")

    volume = cleanup_volume(volume)
    n_occupied = int(volume.sum())
    if n_occupied < MIN_OCCUPIED_VOXELS:
        raise ValueError(f"Visual hull has only {n_occupied} occupied voxels (minimum {MIN_OCCUPIED_VOXELS})")
    logger.info(f"[{job_id}] reconstruct_coarse: cleaned volume {n_occupied} voxels ({n_occupied / volume.size * 100:.1f}%)")
    jm.update_job(job_id, stage_progress=0.35)

    voxel_path = sm.get_artifact_dir(job_id) / "coarse_voxel.npz"
    np.savez_compressed(str(voxel_path), occupancy=volume.astype(np.float32),
                        grid_dims=np.array([X, Y, Z]), grid_resolution=np.array([grid_res]),
                        half_extent=np.array([DEFAULT_GRID_HALF_EXTENT]))
    jm.update_job(job_id, stage_progress=0.4)

    try:
        vertices, faces, normals = extract_surface_mesh(volume, X, Y, Z, half_extent=DEFAULT_GRID_HALF_EXTENT)
        mesh_path = sm.get_artifact_dir(job_id) / "coarse_visual_hull_mesh.ply"
        save_mesh_ply(str(mesh_path), vertices, faces, normals)
        logger.info(f"[{job_id}] reconstruct_coarse: mesh saved ({len(vertices)} verts, {len(faces)} faces)")
    except ValueError as e:
        logger.warning(f"[{job_id}] reconstruct_coarse: mesh extraction failed: {e}")
        vertices = np.zeros((0, 3))
        faces = np.zeros((0, 3), dtype=np.int32)
        normals = np.zeros((0, 3))
    jm.update_job(job_id, stage_progress=0.55)

    depth_maps = compute_depth_maps(volume, X, Y, Z)
    max_depth = 0.0
    for dm in depth_maps.values():
        if np.any(dm > 0):
            max_depth = max(max_depth, float(dm.max()))
    if max_depth == 0:
        max_depth = 1.0
    for vn, dm in depth_maps.items():
        depth_path = sm.get_artifact_dir(job_id) / f"coarse_depth_{vn}.png"
        save_depth_map_png(str(depth_path), dm, max_depth=max_depth)
    jm.update_job(job_id, stage_progress=0.7)

    n_points = DEFAULT_N_SURFACE_POINTS
    surface_points, surface_normals = sample_surface_points(
        volume, X, Y, Z, half_extent=DEFAULT_GRID_HALF_EXTENT, n_points=n_points, seed=config.seed)
    if len(surface_points) > 0:
        surface_colors = color_surface_points(surface_points, surface_normals, images, cleaned_masks, X, Y, Z, half_extent=DEFAULT_GRID_HALF_EXTENT)
    else:
        surface_colors = np.zeros((0, 3), dtype=np.uint8)
    jm.update_job(job_id, stage_progress=0.85)

    max_dim = max(X, Y, Z)
    voxel_size = 2.0 * DEFAULT_GRID_HALF_EXTENT / max(max_dim, 1)
    gaussians = build_coarse_gaussians(surface_points, surface_colors, surface_normals, voxel_size)
    gauss_path = sm.get_artifact_dir(job_id) / "coarse_gaussians.ply"
    save_gaussians_ply(str(gauss_path), gaussians)
    logger.info(f"[{job_id}] reconstruct_coarse: {len(surface_points)} Gaussians saved")
    jm.update_job(job_id, stage_progress=0.95)

    recon_metrics = {
        "method": "orthographic_voxel_carving",
        "grid_resolution": grid_res,
        "grid_dims": {"X": X, "Y": Y, "Z": Z},
        "half_extent": DEFAULT_GRID_HALF_EXTENT,
        "voxel_size": float(voxel_size),
        "n_raw_voxels": n_raw,
        "n_occupied_voxels": n_occupied,
        "occupancy_fraction": float(n_occupied / volume.size),
        "mesh_vertices": len(vertices),
        "mesh_faces": len(faces),
        "n_gaussians": len(surface_points),
        "depth_maps_generated": list(depth_maps.keys()),
        "max_depth": float(max_depth),
        "depth_mode": DEPTH_MODE,
    }
    sm.save_artifact_json(job_id, "coarse_recon_metrics.json", recon_metrics)
    jm.update_job(job_id, stage_progress=1.0)
    logger.info(f"[{job_id}] reconstruct_coarse: completed")


def _load_segmented_views(job_id, sm, target_size=None):
    """Load segmented view masks and RGB images from the preprocess stage."""
    masks = {}
    images = {}
    for vn in CANONICAL_VIEW_ORDER:
        path = sm.get_view_preview_path(job_id, "segmented", vn)
        if path is None:
            path = sm.get_view_preview_path(job_id, "normalized", vn)
        if path is None:
            logger.warning(f"No segmented/normalized preview found for view \'{vn}\'")
            continue
        img = Image.open(path)
        if target_size is not None:
            tw, th = target_size
            if img.size != (tw, th):
                img = img.resize((tw, th), Image.Resampling.LANCZOS)
        if img.mode == "RGBA":
            rgba = np.array(img)
            masks[vn] = rgba[:, :, 3]
            images[vn] = rgba[:, :, :3]
        elif img.mode == "RGB":
            rgb = np.array(img)
            images[vn] = rgb
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            masks[vn] = (gray > 10).astype(np.uint8) * 255
        else:
            img_rgb = img.convert("RGB")
            images[vn] = np.array(img_rgb)
            masks[vn] = np.full((img.height, img.width), 255, dtype=np.uint8)
    return masks, images


def _grid_resolution_from_config(config):
    """Determine grid resolution from config mesh_resolution."""
    mr = config.mesh_resolution
    if mr <= 128:
        return 128
    elif mr <= 256:
        return 256
    elif mr <= 384:
        return 384
    else:
        return 512
