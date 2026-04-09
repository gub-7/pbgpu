"""
Coarse reconstruction stage for the canonical multi-view pipeline.

Responsibilities:
    - Compute a visual hull from silhouette masks + calibrated cameras
    - Extract a coarse triangle mesh via marching cubes
    - Generate per-view depth maps from the visual hull
    - Initialize a coarse Gaussian point cloud on the surface
    - Save all coarse artifacts for downstream refinement

Algorithm (Option B from TDD — safer engineering path):
    1. Load segmented masks and camera rig from previous stages.
    2. Build a 3D occupancy grid (voxel volume) centered at the origin.
    3. For each voxel, project into all camera views; mark as occupied
       only if the projection falls inside the silhouette in ALL views.
    4. Extract an isosurface from the occupancy grid using marching cubes.
    5. Sample surface points and assign colors from the best-visible view.
    6. Build a coarse Gaussian point cloud from surface samples.
    7. Render per-view depth maps by projecting the volume.

Artifacts produced:
    - ``coarse_voxel.npz``           — raw occupancy grid + metadata
    - ``coarse_visual_hull_mesh.ply`` — extracted triangle mesh
    - ``coarse_gaussians.ply``        — Gaussian point cloud
    - ``coarse_depth_{view}.png``     — per-view depth maps

Camera convention (matches camera_init.py):
    - World space: Y-up, right-handed
    - Camera space: X-right, Y-down, Z-forward (OpenCV)
"""

import io
import logging
import struct
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from api.job_manager import JobManager
from api.storage import StorageManager

from .camera_init import CameraRig, project_point
from .config import CanonicalMVConfig, CANONICAL_VIEW_ORDER

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default voxel grid resolution (per axis). Higher = more detail but slower.
# 128 is a good balance for coarse reconstruction.
DEFAULT_GRID_RESOLUTION = 128

# World-space half-extent of the voxel grid.
# The grid spans [-GRID_HALF_EXTENT, +GRID_HALF_EXTENT] in each axis.
DEFAULT_GRID_HALF_EXTENT = 1.0

# Number of surface points to sample for Gaussian initialization.
DEFAULT_N_SURFACE_POINTS = 50_000

# Default Gaussian scale relative to voxel size.
DEFAULT_GAUSSIAN_SCALE_FACTOR = 1.5

# Minimum number of occupied voxels to consider the visual hull valid.
MIN_OCCUPIED_VOXELS = 10

# Minimum fraction of views that must see a voxel as foreground.
# 1.0 = strict intersection (all views must agree).
# Lower values are more tolerant of segmentation errors
# and AI-generated view misalignment.
# 0.6 = at least 2 out of 3 views must agree.
DEFAULT_CONSENSUS_RATIO = 0.6

# Pixels to dilate masks before visual hull projection.
# Adds tolerance for slight misalignment between AI-generated views.
DEFAULT_MASK_DILATION = 15


# ---------------------------------------------------------------------------
# Visual hull computation
# ---------------------------------------------------------------------------


def compute_visual_hull(
    masks: Dict[str, np.ndarray],
    rig: CameraRig,
    grid_resolution: int = DEFAULT_GRID_RESOLUTION,
    grid_half_extent: float = DEFAULT_GRID_HALF_EXTENT,
    consensus_ratio: float = DEFAULT_CONSENSUS_RATIO,
    mask_dilation: int = DEFAULT_MASK_DILATION,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the visual hull from silhouette masks and calibrated cameras.

    For each voxel in a 3D grid, projects its center into every camera view
    and checks whether it falls inside the corresponding silhouette mask.
    A voxel is marked as occupied if at least ``consensus_ratio`` fraction
    of views agree.

    Args:
        masks: Dict mapping view name → binary mask (H, W), uint8 0/255 or bool.
        rig: Calibrated camera rig from the camera_init stage.
        grid_resolution: Number of voxels per axis (e.g. 128 → 128³ grid).
        grid_half_extent: Half-extent of the grid in world units.
        consensus_ratio: Fraction of views that must agree (0.0–1.0).

    Returns:
        Tuple of:
            - occupancy: (N, N, N) float32 array, values in [0, 1] representing
              the fraction of views that see each voxel as foreground.
            - grid_origin: (3,) array, world-space position of voxel [0,0,0].
            - voxel_size: float, side length of each voxel in world units.
    """
    N = grid_resolution
    voxel_size = (2.0 * grid_half_extent) / N
    grid_origin = np.array(
        [-grid_half_extent, -grid_half_extent, -grid_half_extent],
        dtype=np.float64,
    )

    # Pre-compute voxel center coordinates (N, N, N, 3)
    coords_1d = np.linspace(
        -grid_half_extent + voxel_size / 2,
        grid_half_extent - voxel_size / 2,
        N,
    )
    # Build meshgrid: x varies along axis 0, y along axis 1, z along axis 2
    gx, gy, gz = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    voxel_centers = np.stack([gx, gy, gz], axis=-1)  # (N, N, N, 3)

    # Flatten for batch projection
    flat_centers = voxel_centers.reshape(-1, 3)  # (N³, 3)
    n_voxels = flat_centers.shape[0]

    # Accumulate votes from each view
    vote_count = np.zeros(n_voxels, dtype=np.float32)
    n_views = 0

    view_names = [vn for vn in CANONICAL_VIEW_ORDER if vn in masks and vn in rig.cameras]

    for vn in view_names:
        mask = masks[vn]
        if mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255

        # Ensure binary
        binary_mask = (mask > 127).astype(np.uint8)

        # Dilate mask to add tolerance for AI-generated view misalignment
        if mask_dilation > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (mask_dilation * 2 + 1, mask_dilation * 2 + 1),
            )
            binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

        h, w = binary_mask.shape[:2]

        ext = rig.get_extrinsic(vn)
        intr = rig.get_intrinsic(vn)

        # Batch project: transform to camera space
        ones = np.ones((n_voxels, 1), dtype=np.float64)
        homo = np.hstack([flat_centers, ones])  # (N³, 4)
        cam_coords = (ext @ homo.T).T  # (N³, 4) → keep first 3

        # Filter: z > 0 (in front of camera)
        z_cam = cam_coords[:, 2]
        in_front = z_cam > 0

        # Project to image plane
        p_img = (intr @ cam_coords[:, :3].T).T  # (N³, 3)
        # Normalize by z
        with np.errstate(divide="ignore", invalid="ignore"):
            px = p_img[:, 0] / p_img[:, 2]
            py = p_img[:, 1] / p_img[:, 2]

        # Check bounds
        in_bounds = in_front & (px >= 0) & (px < w) & (py >= 0) & (py < h)

        # Sample mask at projected positions
        inside_mask = np.zeros(n_voxels, dtype=bool)
        valid_idx = np.where(in_bounds)[0]
        if len(valid_idx) > 0:
            px_int = np.clip(px[valid_idx].astype(np.int32), 0, w - 1)
            py_int = np.clip(py[valid_idx].astype(np.int32), 0, h - 1)
            sampled = binary_mask[py_int, px_int]
            inside_mask[valid_idx] = sampled > 0

        vote_count[inside_mask] += 1.0
        n_views += 1

    # Compute occupancy as fraction of agreeing views
    if n_views > 0:
        occupancy_flat = vote_count / n_views
    else:
        occupancy_flat = np.zeros(n_voxels, dtype=np.float32)

    occupancy = occupancy_flat.reshape(N, N, N)

    logger.info(
        f"Visual hull: resolution={N}, voxel_size={voxel_size:.4f}, "
        f"views={n_views}, occupied={int((occupancy >= consensus_ratio).sum())}"
    )

    return occupancy, grid_origin, voxel_size


def threshold_occupancy(
    occupancy: np.ndarray,
    consensus_ratio: float = DEFAULT_CONSENSUS_RATIO,
) -> np.ndarray:
    """
    Threshold the occupancy grid to produce a binary volume.

    Args:
        occupancy: (N, N, N) float array with vote fractions.
        consensus_ratio: Minimum fraction to consider occupied.

    Returns:
        (N, N, N) bool array.
    """
    return occupancy >= consensus_ratio


# ---------------------------------------------------------------------------
# Mesh extraction
# ---------------------------------------------------------------------------


def extract_surface_mesh(
    occupancy: np.ndarray,
    grid_origin: np.ndarray,
    voxel_size: float,
    level: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract a triangle mesh from the occupancy grid using marching cubes.

    Args:
        occupancy: (N, N, N) float32 occupancy grid.
        grid_origin: (3,) world-space position of voxel [0,0,0].
        voxel_size: Side length of each voxel in world units.
        level: Isosurface level for marching cubes.

    Returns:
        Tuple of:
            - vertices: (V, 3) float64 array of vertex positions in world space.
            - faces: (F, 3) int32 array of triangle face indices.
            - normals: (V, 3) float64 array of vertex normals.

    Raises:
        ValueError: if the occupancy grid has no surface to extract.
    """
    # Check for empty volume
    if occupancy.max() < level:
        raise ValueError(
            "Occupancy grid has no voxels above the isosurface level — "
            "visual hull is empty. Check segmentation masks and camera calibration."
        )

    # When the maximum occupancy exactly equals the level (e.g. all views
    # agree → occupancy=1.0, consensus_ratio=1.0), marching cubes cannot
    # find a transition.  Use a slightly lower level so the boundary
    # between occupied (>=level) and unoccupied (<level) voxels is detected.
    mc_level = level - 1e-6 if occupancy.max() <= level else level

    try:
        from skimage.measure import marching_cubes

        vertices, faces, normals, _ = marching_cubes(
            occupancy,
            level=mc_level,
            spacing=(voxel_size, voxel_size, voxel_size),
        )

        # Transform vertices from grid space to world space
        vertices = vertices + grid_origin

        logger.info(
            f"Mesh extracted: {len(vertices)} vertices, {len(faces)} faces"
        )

        return (
            vertices.astype(np.float64),
            faces.astype(np.int32),
            normals.astype(np.float64),
        )

    except ImportError:
        logger.warning(
            "scikit-image not available — falling back to surface voxel extraction"
        )
        return _fallback_surface_extraction(occupancy, grid_origin, voxel_size, level)


def _fallback_surface_extraction(
    occupancy: np.ndarray,
    grid_origin: np.ndarray,
    voxel_size: float,
    level: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fallback surface extraction when scikit-image is not available.

    Finds surface voxels (occupied voxels with at least one empty neighbor)
    and creates a simple point-based representation. Each surface voxel
    produces 8 vertices (cube corners) and 12 faces (2 triangles per face).

    This is less smooth than marching cubes but functional.
    """
    binary = occupancy >= level
    N = binary.shape[0]

    # Find surface voxels: occupied with at least one empty 6-connected neighbor
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    surface_mask = binary.copy()

    for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
        neighbor = padded[1+dx:N+1+dx, 1+dy:N+1+dy, 1+dz:N+1+dz]
        surface_mask &= ~(binary & neighbor) | binary
    # Simpler: surface = occupied AND has at least one empty neighbor
    has_empty_neighbor = np.zeros_like(binary)
    for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
        neighbor = padded[1+dx:N+1+dx, 1+dy:N+1+dy, 1+dz:N+1+dz]
        has_empty_neighbor |= ~neighbor
    surface_mask = binary & has_empty_neighbor

    surface_coords = np.argwhere(surface_mask)  # (S, 3) indices

    if len(surface_coords) == 0:
        raise ValueError("No surface voxels found in the occupancy grid.")

    # Convert to world coordinates (voxel centers)
    vertices = surface_coords.astype(np.float64) * voxel_size + grid_origin + voxel_size / 2
    # Create dummy faces (empty — this is a point cloud fallback)
    faces = np.zeros((0, 3), dtype=np.int32)
    normals = np.zeros_like(vertices)

    # Estimate normals from gradient
    grad = np.gradient(occupancy.astype(np.float64))
    for i, (ix, iy, iz) in enumerate(surface_coords):
        n = np.array([grad[0][ix, iy, iz], grad[1][ix, iy, iz], grad[2][ix, iy, iz]])
        norm = np.linalg.norm(n)
        if norm > 1e-10:
            normals[i] = n / norm

    logger.info(
        f"Fallback surface extraction: {len(vertices)} surface points"
    )

    return vertices, faces, normals


# ---------------------------------------------------------------------------
# Depth map rendering
# ---------------------------------------------------------------------------


def compute_depth_maps(
    occupancy: np.ndarray,
    rig: CameraRig,
    grid_origin: np.ndarray,
    voxel_size: float,
    image_size: Tuple[int, int],
    level: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    Render per-view depth maps from the visual hull.

    For each pixel in each camera view, casts a ray through the occupancy
    grid and records the depth of the first occupied voxel hit.

    Uses a simplified approach: projects all occupied surface voxels into
    each view and records the minimum depth (closest surface point) per pixel.

    Args:
        occupancy: (N, N, N) float32 occupancy grid.
        rig: Calibrated camera rig.
        grid_origin: (3,) world-space origin of the grid.
        voxel_size: Voxel side length in world units.
        image_size: (width, height) of the output depth maps.
        level: Occupancy threshold for surface detection.

    Returns:
        Dict mapping view name → depth map (H, W) float32 array.
        Depth is in world units; 0 = no surface visible.
    """
    binary = occupancy >= level
    surface_coords = np.argwhere(binary)  # (S, 3) grid indices

    if len(surface_coords) == 0:
        w, h = image_size
        return {vn: np.zeros((h, w), dtype=np.float32) for vn in rig.cameras}

    # Convert to world coordinates
    surface_world = (
        surface_coords.astype(np.float64) * voxel_size
        + grid_origin
        + voxel_size / 2
    )

    n_points = len(surface_world)
    ones = np.ones((n_points, 1), dtype=np.float64)
    homo = np.hstack([surface_world, ones])  # (S, 4)

    w, h = image_size
    depth_maps = {}

    for vn in CANONICAL_VIEW_ORDER:
        if vn not in rig.cameras:
            continue

        ext = rig.get_extrinsic(vn)
        intr = rig.get_intrinsic(vn)

        # Transform to camera space
        cam_coords = (ext @ homo.T).T  # (S, 4)
        z_cam = cam_coords[:, 2]

        # Project to image plane
        p_img = (intr @ cam_coords[:, :3].T).T
        with np.errstate(divide="ignore", invalid="ignore"):
            px = p_img[:, 0] / p_img[:, 2]
            py = p_img[:, 1] / p_img[:, 2]

        # Filter valid projections
        valid = (z_cam > 0) & (px >= 0) & (px < w) & (py >= 0) & (py < h)
        valid_idx = np.where(valid)[0]

        depth_map = np.zeros((h, w), dtype=np.float32)

        if len(valid_idx) > 0:
            px_int = px[valid_idx].astype(np.int32)
            py_int = py[valid_idx].astype(np.int32)
            depths = z_cam[valid_idx].astype(np.float32)

            # For each pixel, keep the minimum depth (closest surface)
            for i in range(len(valid_idx)):
                y, x, d = py_int[i], px_int[i], depths[i]
                if depth_map[y, x] == 0 or d < depth_map[y, x]:
                    depth_map[y, x] = d

        depth_maps[vn] = depth_map

    return depth_maps


# ---------------------------------------------------------------------------
# Surface point sampling
# ---------------------------------------------------------------------------


def sample_surface_points(
    occupancy: np.ndarray,
    grid_origin: np.ndarray,
    voxel_size: float,
    n_points: int = DEFAULT_N_SURFACE_POINTS,
    level: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample points on the visual hull surface for Gaussian initialization.

    Finds surface voxels (occupied with at least one empty neighbor) and
    samples points uniformly within those voxels, with slight jitter
    toward the surface.

    Args:
        occupancy: (N, N, N) float32 occupancy grid.
        grid_origin: (3,) world-space origin.
        voxel_size: Voxel side length.
        n_points: Number of points to sample.
        level: Occupancy threshold.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of:
            - points: (P, 3) float64 array of surface point positions.
            - normals: (P, 3) float64 array of estimated surface normals.
    """
    rng = np.random.RandomState(seed)

    binary = occupancy >= level
    N = binary.shape[0]

    # Find surface voxels
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    has_empty_neighbor = np.zeros_like(binary)
    for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
        neighbor = padded[1+dx:N+1+dx, 1+dy:N+1+dy, 1+dz:N+1+dz]
        has_empty_neighbor |= ~neighbor
    surface_mask = binary & has_empty_neighbor

    surface_indices = np.argwhere(surface_mask)  # (S, 3)
    n_surface = len(surface_indices)

    if n_surface == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

    # Sample with replacement if n_points > n_surface
    if n_points <= n_surface:
        chosen = rng.choice(n_surface, size=n_points, replace=False)
    else:
        chosen = rng.choice(n_surface, size=n_points, replace=True)

    chosen_indices = surface_indices[chosen]  # (P, 3)

    # Convert to world coordinates with jitter within the voxel
    jitter = rng.uniform(-0.4, 0.4, size=(n_points, 3)) * voxel_size
    points = (
        chosen_indices.astype(np.float64) * voxel_size
        + grid_origin
        + voxel_size / 2
        + jitter
    )

    # Estimate normals from occupancy gradient
    grad = np.gradient(occupancy.astype(np.float64))
    normals = np.zeros((n_points, 3), dtype=np.float64)
    for i, (ix, iy, iz) in enumerate(chosen_indices):
        n = np.array([
            grad[0][ix, iy, iz],
            grad[1][ix, iy, iz],
            grad[2][ix, iy, iz],
        ])
        norm = np.linalg.norm(n)
        if norm > 1e-10:
            normals[i] = n / norm

    return points, normals


def color_surface_points(
    points: np.ndarray,
    normals: np.ndarray,
    rig: CameraRig,
    images: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Assign colors to surface points by projecting into camera views.

    For each point, selects the view with the best visibility (based on
    normal-to-view-direction alignment and mask confidence) and samples
    the color from that view.

    Args:
        points: (P, 3) surface point positions.
        normals: (P, 3) surface normals.
        rig: Calibrated camera rig.
        images: Dict mapping view name → RGB image (H, W, 3) uint8.
        masks: Dict mapping view name → binary mask (H, W) uint8.

    Returns:
        (P, 3) uint8 array of RGB colors.
    """
    n_points = len(points)
    colors = np.full((n_points, 3), 128, dtype=np.uint8)  # default gray

    if n_points == 0:
        return colors

    view_names = [vn for vn in CANONICAL_VIEW_ORDER if vn in images and vn in rig.cameras]

    if not view_names:
        return colors

    # Pre-compute per-view projections and scores
    ones = np.ones((n_points, 1), dtype=np.float64)
    homo = np.hstack([points, ones])

    best_scores = np.full(n_points, -1.0, dtype=np.float64)

    for vn in view_names:
        img = images[vn]
        mask = masks.get(vn)
        h, w = img.shape[:2]

        ext = rig.get_extrinsic(vn)
        intr = rig.get_intrinsic(vn)
        cam_pos = rig.get_position(vn)

        # Project points
        cam_coords = (ext @ homo.T).T
        z_cam = cam_coords[:, 2]

        p_img = (intr @ cam_coords[:, :3].T).T
        with np.errstate(divide="ignore", invalid="ignore"):
            px = p_img[:, 0] / p_img[:, 2]
            py = p_img[:, 1] / p_img[:, 2]

        valid = (z_cam > 0) & (px >= 0) & (px < w) & (py >= 0) & (py < h)

        # Compute view direction and score (dot product with normal)
        view_dirs = cam_pos - points  # (P, 3)
        view_dists = np.linalg.norm(view_dirs, axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            view_dirs_norm = view_dirs / np.maximum(view_dists, 1e-10)

        # Score: cosine of angle between normal and view direction
        # Higher = surface faces toward this camera
        dot = np.sum(normals * view_dirs_norm, axis=1)
        scores = np.maximum(dot, 0.0)  # ignore back-facing

        # Check mask if available
        if mask is not None:
            binary_mask = (mask > 127).astype(np.uint8)
            valid_idx = np.where(valid)[0]
            if len(valid_idx) > 0:
                px_int = np.clip(px[valid_idx].astype(np.int32), 0, w - 1)
                py_int = np.clip(py[valid_idx].astype(np.int32), 0, h - 1)
                in_mask = binary_mask[py_int, px_int] > 0
                # Zero out scores for points not in mask
                mask_valid = np.zeros(n_points, dtype=bool)
                mask_valid[valid_idx[in_mask]] = True
                scores[~mask_valid] = 0.0

        # Update colors where this view has the best score
        better = valid & (scores > best_scores)
        better_idx = np.where(better)[0]

        if len(better_idx) > 0:
            px_int = np.clip(px[better_idx].astype(np.int32), 0, w - 1)
            py_int = np.clip(py[better_idx].astype(np.int32), 0, h - 1)
            colors[better_idx] = img[py_int, px_int]
            best_scores[better_idx] = scores[better_idx]

    return colors


# ---------------------------------------------------------------------------
# Gaussian point cloud construction
# ---------------------------------------------------------------------------


def build_coarse_gaussians(
    points: np.ndarray,
    colors: np.ndarray,
    normals: np.ndarray,
    voxel_size: float,
    scale_factor: float = DEFAULT_GAUSSIAN_SCALE_FACTOR,
) -> Dict[str, np.ndarray]:
    """
    Build a coarse Gaussian point cloud from surface samples.

    Each Gaussian has:
        - position (3,)
        - color (3,) RGB uint8
        - scale (3,) — isotropic, based on voxel size
        - opacity (1,) — 1.0 for surface points
        - normal (3,) — estimated surface normal

    Args:
        points: (P, 3) surface point positions.
        colors: (P, 3) RGB colors.
        normals: (P, 3) surface normals.
        voxel_size: Voxel side length (used to set Gaussian scale).
        scale_factor: Multiplier for Gaussian scale.

    Returns:
        Dict with keys: positions, colors, scales, opacities, normals.
    """
    n = len(points)
    scale = voxel_size * scale_factor

    return {
        "positions": points.astype(np.float32),
        "colors": colors.astype(np.uint8),
        "scales": np.full((n, 3), scale, dtype=np.float32),
        "opacities": np.ones((n, 1), dtype=np.float32),
        "normals": normals.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# PLY I/O
# ---------------------------------------------------------------------------


def save_mesh_ply(
    filepath: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: Optional[np.ndarray] = None,
) -> None:
    """
    Save a triangle mesh in binary PLY format.

    Args:
        filepath: Output file path.
        vertices: (V, 3) float64 vertex positions.
        faces: (F, 3) int32 face indices.
        normals: Optional (V, 3) float64 vertex normals.
    """
    n_verts = len(vertices)
    n_faces = len(faces)
    has_normals = normals is not None and len(normals) == n_verts

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
    if n_faces > 0:
        header_lines += [
            f"element face {n_faces}",
            "property list uchar int vertex_indices",
        ]
    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    with open(filepath, "wb") as f:
        f.write(header.encode("ascii"))

        # Write vertices
        for i in range(n_verts):
            f.write(struct.pack("<fff", *vertices[i].astype(np.float32)))
            if has_normals:
                f.write(struct.pack("<fff", *normals[i].astype(np.float32)))

        # Write faces
        for i in range(n_faces):
            f.write(struct.pack("<B", 3))
            f.write(struct.pack("<iii", *faces[i].astype(np.int32)))


def save_gaussians_ply(
    filepath: str,
    gaussians: Dict[str, np.ndarray],
) -> None:
    """
    Save a Gaussian point cloud in binary PLY format.

    Attributes per point: x, y, z, nx, ny, nz, red, green, blue,
    scale_x, scale_y, scale_z, opacity.

    Args:
        filepath: Output file path.
        gaussians: Dict from build_coarse_gaussians().
    """
    positions = gaussians["positions"]
    colors = gaussians["colors"]
    scales = gaussians["scales"]
    opacities = gaussians["opacities"]
    normals = gaussians["normals"]

    n = len(positions)

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property float scale_x",
        "property float scale_y",
        "property float scale_z",
        "property float opacity",
        "end_header",
    ]
    header = "\n".join(header_lines) + "\n"

    with open(filepath, "wb") as f:
        f.write(header.encode("ascii"))

        for i in range(n):
            f.write(struct.pack("<fff", *positions[i]))
            f.write(struct.pack("<fff", *normals[i]))
            f.write(struct.pack("<BBB", *colors[i]))
            f.write(struct.pack("<fff", *scales[i]))
            f.write(struct.pack("<f", opacities[i, 0]))


def save_depth_map_png(
    filepath: str,
    depth_map: np.ndarray,
    max_depth: Optional[float] = None,
) -> None:
    """
    Save a depth map as a 16-bit grayscale PNG.

    Depth values are normalized to [0, 65535] range.
    Zero depth (no surface) remains zero.

    Args:
        filepath: Output file path.
        depth_map: (H, W) float32 depth map.
        max_depth: Maximum depth for normalization. If None, uses the
                   maximum non-zero depth in the map.
    """
    if max_depth is None:
        max_depth = float(depth_map[depth_map > 0].max()) if np.any(depth_map > 0) else 1.0

    normalized = np.zeros_like(depth_map, dtype=np.uint16)
    valid = depth_map > 0
    if np.any(valid) and max_depth > 0:
        normalized[valid] = (depth_map[valid] / max_depth * 65535).astype(np.uint16)

    img = Image.fromarray(normalized, mode="I;16")
    img.save(filepath)



# ---------------------------------------------------------------------------
# Debug: 3/4-angle preview renderer
# ---------------------------------------------------------------------------


def render_visual_hull_preview(
    occupancy: np.ndarray,
    grid_origin: np.ndarray,
    voxel_size: float,
    level: float = 0.5,
    image_size: Tuple[int, int] = (512, 512),
    azimuth_deg: float = 35.0,
    elevation_deg: float = 25.0,
    distance: float = 3.5,
) -> np.ndarray:
    """
    Render a preview image of the visual hull from a 3/4 angle.

    Creates a depth-colored rendering of the occupied voxels projected
    through a virtual camera at the specified azimuth/elevation.

    Args:
        occupancy: (N, N, N) float32 occupancy grid.
        grid_origin: (3,) world-space origin.
        voxel_size: Voxel side length.
        level: Occupancy threshold.
        image_size: (width, height) of the output image.
        azimuth_deg: Camera azimuth in degrees (0 = front).
        elevation_deg: Camera elevation in degrees.
        distance: Camera distance from origin.

    Returns:
        (H, W, 3) uint8 RGB image.
    """
    import math

    binary = occupancy >= level
    surface_coords = np.argwhere(binary)

    w, h = image_size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (30, 30, 40)  # dark background

    if len(surface_coords) == 0:
        # Draw "no data" text area
        cv2.putText(img, "No occupied voxels", (w // 6, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        return img

    # Convert to world coordinates
    surface_world = (
        surface_coords.astype(np.float64) * voxel_size
        + grid_origin
        + voxel_size / 2
    )

    # Build a virtual camera at 3/4 angle
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    cam_x = distance * math.cos(el) * math.sin(az)
    cam_y = distance * math.sin(el)
    cam_z = distance * math.cos(el) * math.cos(az)
    eye = np.array([cam_x, cam_y, cam_z], dtype=np.float64)
    target = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    # Build view matrix (simplified look-at for rendering only)
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

    # Camera-space rotation: X=right, Y=up (for rendering, not OpenCV), Z=-forward
    R_view = np.zeros((3, 3), dtype=np.float64)
    R_view[0, :] = right
    R_view[1, :] = up  # Y=up for rendering (flip later for image coords)
    R_view[2, :] = -fwd

    t_view = -R_view @ eye

    # Project points
    n_pts = len(surface_world)
    cam_coords = (R_view @ surface_world.T).T + t_view  # (N, 3)

    # Perspective projection
    focal = w * 0.8
    cx, cy = w / 2.0, h / 2.0

    z_cam = cam_coords[:, 2]
    valid = z_cam > 0.01

    with np.errstate(divide="ignore", invalid="ignore"):
        px = focal * cam_coords[:, 0] / z_cam + cx
        # Flip Y for image coords (Y-down in image)
        py = -focal * cam_coords[:, 1] / z_cam + cy

    in_bounds = valid & (px >= 0) & (px < w) & (py >= 0) & (py < h)
    valid_idx = np.where(in_bounds)[0]

    if len(valid_idx) == 0:
        cv2.putText(img, "No visible voxels", (w // 6, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        return img

    # Z-buffer rendering with depth coloring
    depth_buf = np.full((h, w), np.inf, dtype=np.float64)
    color_buf = np.zeros((h, w, 3), dtype=np.uint8)

    depths = z_cam[valid_idx]
    min_d, max_d = depths.min(), depths.max()
    depth_range = max(max_d - min_d, 0.001)

    px_int = px[valid_idx].astype(np.int32)
    py_int = py[valid_idx].astype(np.int32)

    for i in range(len(valid_idx)):
        x, y, d = px_int[i], py_int[i], depths[i]
        if d < depth_buf[y, x]:
            depth_buf[y, x] = d
            # Color by depth: close = bright cyan, far = dark blue
            t = 1.0 - (d - min_d) / depth_range
            r = int(40 + 80 * t)
            g = int(120 + 135 * t)
            b = int(180 + 75 * t)
            color_buf[y, x] = (r, g, b)

    # Composite onto background
    mask = depth_buf < np.inf
    img[mask] = color_buf[mask]

    # Dilate slightly to fill gaps between voxels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for c in range(3):
        channel = img[:, :, c].copy()
        dilated = cv2.dilate(channel, kernel, iterations=1)
        # Only fill where the original was background
        bg_mask = ~mask
        channel[bg_mask] = dilated[bg_mask]
        img[:, :, c] = channel

    return img


def _save_debug_preview(
    filepath: str,
    image: np.ndarray,
    label: str,
    n_occupied: int,
) -> None:
    """Save a debug preview image with an overlay label."""
    img = image.copy()
    h, w = img.shape[:2]

    # Draw label background
    cv2.rectangle(img, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.putText(img, label, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw voxel count
    count_text = f"{n_occupied:,} voxels"
    cv2.putText(img, count_text, (w - 180, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 1)

    Image.fromarray(img).save(filepath)


# ---------------------------------------------------------------------------
# Debug: Incremental reconstruction
# ---------------------------------------------------------------------------


def run_incremental_debug_recon(
    job_id: str,
    config: CanonicalMVConfig,
    jm: JobManager,
    sm: StorageManager,
    masks: Dict[str, np.ndarray],
    images: Dict[str, np.ndarray],
    rig: "CameraRig",
    grid_res: int,
) -> Dict[str, str]:
    """
    Run incremental visual hull reconstruction for debugging camera alignment.

    Performs 3 passes with increasing numbers of views:
        1. Front only → preview
        2. Front + Side → preview
        3. Front + Side + Top → preview

    This lets the user see how each additional view carves the visual hull,
    making it easy to spot if a view's camera angle is misaligned.

    Args:
        job_id: Job identifier.
        config: Pipeline config.
        jm: Job manager.
        sm: Storage manager.
        masks: Dict mapping view name → binary mask.
        images: Dict mapping view name → RGB image.
        rig: Calibrated camera rig.
        grid_res: Voxel grid resolution.

    Returns:
        Dict mapping view name → preview artifact path (relative).
    """
    logger.info(f"[{job_id}] debug: running incremental reconstruction")

    view_order = [vn for vn in CANONICAL_VIEW_ORDER if vn in masks]
    if not view_order:
        logger.warning(f"[{job_id}] debug: no views available for incremental recon")
        return {}

    preview_paths: Dict[str, str] = {}
    artifact_dir = sm.get_artifact_dir(job_id)

    for pass_idx in range(len(view_order)):
        # Use views 0..pass_idx (inclusive)
        active_views = view_order[: pass_idx + 1]
        active_masks = {vn: masks[vn] for vn in active_views}
        view_label = " + ".join(active_views)
        pass_name = f"debug_recon_{pass_idx + 1}view"

        logger.info(
            f"[{job_id}] debug pass {pass_idx + 1}/{len(view_order)}: "
            f"views=[{view_label}]"
        )

        try:
            # Run visual hull with subset of views
            # Use lower consensus for single-view (everything that view sees)
            consensus = 1.0 / max(len(active_views), 1)

            occupancy, grid_origin, voxel_size = compute_visual_hull(
                masks=active_masks,
                rig=rig,
                grid_resolution=grid_res,
                grid_half_extent=DEFAULT_GRID_HALF_EXTENT,
                consensus_ratio=consensus,
                mask_dilation=DEFAULT_MASK_DILATION,
            )

            binary = threshold_occupancy(occupancy, consensus)
            n_occupied = int(binary.sum())

            # Render preview from 3/4 angle
            preview = render_visual_hull_preview(
                occupancy, grid_origin, voxel_size,
                level=consensus,
                image_size=(512, 512),
            )

            # Save preview
            preview_filename = f"{pass_name}.png"
            preview_path = artifact_dir / preview_filename
            _save_debug_preview(
                str(preview_path),
                preview,
                f"Pass {pass_idx + 1}: {view_label}",
                n_occupied,
            )

            # Also save as a view preview so the existing preview system picks it up
            view_name = active_views[-1]  # the "new" view in this pass
            # Convert rendered image to PNG bytes for the storage API
            from io import BytesIO as _BytesIO
            _buf = _BytesIO()
            Image.fromarray(preview).save(_buf, format="PNG")
            sm.save_view_preview(job_id, "debug_recon", view_name, _buf.getvalue())

            preview_paths[view_name] = preview_filename

            logger.info(
                f"[{job_id}] debug pass {pass_idx + 1}: "
                f"{n_occupied} occupied voxels, preview saved"
            )

        except Exception as e:
            logger.warning(
                f"[{job_id}] debug pass {pass_idx + 1} failed: {e}",
                exc_info=True,
            )
            # Create error preview
            err_img = np.zeros((512, 512, 3), dtype=np.uint8)
            err_img[:] = (40, 20, 20)
            cv2.putText(err_img, f"Pass {pass_idx + 1} failed", (30, 256),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
            cv2.putText(err_img, str(e)[:60], (30, 290),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 150, 150), 1)
            preview_filename = f"{pass_name}.png"
            preview_path = artifact_dir / preview_filename
            Image.fromarray(err_img).save(str(preview_path))
            view_name = active_views[-1]
            preview_paths[view_name] = preview_filename

    # Save debug metadata
    sm.save_artifact_json(job_id, "debug_incremental_recon.json", {
        "passes": len(view_order),
        "views_per_pass": [view_order[:i+1] for i in range(len(view_order))],
        "preview_files": preview_paths,
    })

    logger.info(f"[{job_id}] debug: incremental reconstruction complete")
    return preview_paths

# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------


def run_reconstruct_coarse(
    job_id: str,
    config: CanonicalMVConfig,
    jm: JobManager,
    sm: StorageManager,
) -> None:
    """
    Execute the reconstruct_coarse stage.

    Steps:
        1. Load camera rig from camera_init.json.
        2. Load segmented masks from the preprocess stage.
        3. Compute visual hull from masks + cameras.
        4. Extract coarse mesh via marching cubes.
        5. Render per-view depth maps.
        6. Sample surface points and assign colors.
        7. Build coarse Gaussian point cloud.
        8. Save all artifacts.

    Raises:
        ValueError: if required artifacts are missing or visual hull is empty.
    """
    logger.info(f"[{job_id}] reconstruct_coarse: starting")
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

    image_size = tuple(rig.shared_params["image_size"])  # (W, H)
    jm.update_job(job_id, stage_progress=0.05)

    # ------------------------------------------------------------------
    # Step 2: Load segmented masks
    # ------------------------------------------------------------------
    masks, images = _load_segmented_views(job_id, sm, target_size=image_size)
    if len(masks) < 3:
        raise ValueError(
            f"Only {len(masks)} segmented views found, need at least 3 "
            f"for visual hull reconstruction"
        )

    logger.info(
        f"[{job_id}] reconstruct_coarse: loaded {len(masks)} views"
    )
    jm.update_job(job_id, stage_progress=0.1)

    # ------------------------------------------------------------------
    # Debug: Incremental reconstruction (if enabled)
    # ------------------------------------------------------------------
    grid_res = _grid_resolution_from_config(config)

    if config.debug_incremental_recon:
        logger.info(f"[{job_id}] debug_incremental_recon enabled — running incremental passes")
        debug_previews = run_incremental_debug_recon(
            job_id, config, jm, sm, masks, images, rig, grid_res,
        )
        # Store debug preview paths in job metadata so the frontend can fetch them
        # Debug previews are saved via sm.save_view_preview() and will be
        jm.update_job(job_id, stage_progress=0.2)


    # ------------------------------------------------------------------
    # Step 3: Compute visual hull
    # ------------------------------------------------------------------

    occupancy, grid_origin, voxel_size = compute_visual_hull(
        masks=masks,
        rig=rig,
        grid_resolution=grid_res,
        grid_half_extent=DEFAULT_GRID_HALF_EXTENT,
        consensus_ratio=DEFAULT_CONSENSUS_RATIO,
    )

    binary = threshold_occupancy(occupancy, DEFAULT_CONSENSUS_RATIO)
    n_occupied = int(binary.sum())

    if n_occupied < MIN_OCCUPIED_VOXELS:
        raise ValueError(
            f"Visual hull has only {n_occupied} occupied voxels "
            f"(minimum {MIN_OCCUPIED_VOXELS}). Check masks and cameras."
        )

    logger.info(
        f"[{job_id}] reconstruct_coarse: visual hull has {n_occupied} "
        f"occupied voxels ({n_occupied / binary.size * 100:.1f}%)"
    )
    jm.update_job(job_id, stage_progress=0.35)

    # Save voxel grid
    voxel_path = sm.get_artifact_dir(job_id) / "coarse_voxel.npz"
    np.savez_compressed(
        str(voxel_path),
        occupancy=occupancy,
        grid_origin=grid_origin,
        voxel_size=np.array([voxel_size]),
        grid_resolution=np.array([grid_res]),
    )
    jm.update_job(job_id, stage_progress=0.4)

    # ------------------------------------------------------------------
    # Step 4: Extract coarse mesh
    # ------------------------------------------------------------------
    try:
        vertices, faces, normals = extract_surface_mesh(
            occupancy, grid_origin, voxel_size, level=DEFAULT_CONSENSUS_RATIO
        )
        mesh_path = sm.get_artifact_dir(job_id) / "coarse_visual_hull_mesh.ply"
        save_mesh_ply(str(mesh_path), vertices, faces, normals)
        logger.info(
            f"[{job_id}] reconstruct_coarse: mesh saved "
            f"({len(vertices)} verts, {len(faces)} faces)"
        )
    except ValueError as e:
        logger.warning(f"[{job_id}] reconstruct_coarse: mesh extraction failed: {e}")
        vertices = np.zeros((0, 3))
        faces = np.zeros((0, 3), dtype=np.int32)
        normals = np.zeros((0, 3))

    jm.update_job(job_id, stage_progress=0.55)

    # ------------------------------------------------------------------
    # Step 5: Render depth maps
    # ------------------------------------------------------------------
    depth_maps = compute_depth_maps(
        occupancy, rig, grid_origin, voxel_size, image_size,
        level=DEFAULT_CONSENSUS_RATIO,
    )

    # Find global max depth for consistent normalization
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

    # ------------------------------------------------------------------
    # Step 6 & 7: Sample surface points, color them, build Gaussians
    # ------------------------------------------------------------------
    n_points = DEFAULT_N_SURFACE_POINTS
    if config.generate_gaussian_debug:
        n_points = min(n_points, 100_000)

    surface_points, surface_normals = sample_surface_points(
        occupancy, grid_origin, voxel_size,
        n_points=n_points,
        level=DEFAULT_CONSENSUS_RATIO,
        seed=config.seed,
    )

    if len(surface_points) > 0:
        surface_colors = color_surface_points(
            surface_points, surface_normals, rig, images, masks,
        )
    else:
        surface_colors = np.zeros((0, 3), dtype=np.uint8)

    jm.update_job(job_id, stage_progress=0.85)

    gaussians = build_coarse_gaussians(
        surface_points, surface_colors, surface_normals, voxel_size,
    )

    gauss_path = sm.get_artifact_dir(job_id) / "coarse_gaussians.ply"
    save_gaussians_ply(str(gauss_path), gaussians)
    logger.info(
        f"[{job_id}] reconstruct_coarse: {len(surface_points)} Gaussians saved"
    )

    jm.update_job(job_id, stage_progress=0.95)

    # ------------------------------------------------------------------
    # Step 8: Save summary metrics
    # ------------------------------------------------------------------
    recon_metrics = {
        "grid_resolution": grid_res,
        "voxel_size": float(voxel_size),
        "grid_half_extent": DEFAULT_GRID_HALF_EXTENT,
        "n_occupied_voxels": n_occupied,
        "occupancy_fraction": float(n_occupied / binary.size),
        "mesh_vertices": len(vertices),
        "mesh_faces": len(faces),
        "n_gaussians": len(surface_points),
        "depth_maps_generated": list(depth_maps.keys()),
        "max_depth": float(max_depth),
    }
    sm.save_artifact_json(job_id, "coarse_recon_metrics.json", recon_metrics)

    jm.update_job(job_id, stage_progress=1.0)
    logger.info(f"[{job_id}] reconstruct_coarse: completed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_segmented_views(
    job_id: str,
    sm: StorageManager,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load segmented view masks and RGB images from the preprocess stage.

    Preview images are stored as thumbnails (max 512px) but the camera
    rig intrinsics are built for the full canvas size (e.g. 1024×1024).
    When ``target_size`` is provided, all loaded images and masks are
    resized to match so that pixel coordinates from the camera projection
    align correctly with the image data.

    Args:
        job_id: Job identifier.
        sm: Storage manager for loading artifacts.
        target_size: Optional (width, height) to resize all loaded images
                     to.  Should match the camera rig's image_size so that
                     intrinsic projections land on the correct pixels.

    Returns:
        Tuple of (masks_dict, images_dict) where:
            masks_dict maps view_name → binary mask (H, W) uint8
            images_dict maps view_name → RGB image (H, W, 3) uint8
    """
    masks = {}
    images = {}

    for vn in CANONICAL_VIEW_ORDER:
        # Try segmented preview first (from preprocess stage)
        path = sm.get_view_preview_path(job_id, "segmented", vn)
        if path is None:
            # Try normalized preview as fallback
            path = sm.get_view_preview_path(job_id, "normalized", vn)
        if path is None:
            logger.warning(f"No segmented/normalized preview found for view '{vn}'")
            continue

        img = Image.open(path)

        # Resize to target dimensions before converting to numpy so that
        # mask/image pixel coordinates match the camera intrinsics.
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
            # Without alpha, create a simple mask (non-black pixels)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            masks[vn] = (gray > 10).astype(np.uint8) * 255
        else:
            img_rgb = img.convert("RGB")
            images[vn] = np.array(img_rgb)
            masks[vn] = np.full(
                (img.height, img.width), 255, dtype=np.uint8
            )

    return masks, images


def _grid_resolution_from_config(config: CanonicalMVConfig) -> int:
    """
    Determine grid resolution from config mesh_resolution.

    Maps the mesh_resolution parameter (128–512) to an appropriate
    voxel grid resolution for the visual hull.
    """
    # mesh_resolution is the marching cubes resolution target.
    # For visual hull, we use a somewhat lower resolution since
    # this is only the coarse stage.
    mr = config.mesh_resolution
    if mr <= 128:
        return 64
    elif mr <= 256:
        return 128
    elif mr <= 384:
        return 192
    else:
        return 256

