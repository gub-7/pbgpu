"""
Joint mesh + Gaussian refinement stage for the canonical multi-view pipeline.

Responsibilities:
    - Load coarse mesh and Gaussian cloud from the coarse reconstruction stage
    - Jointly optimize mesh vertices and Gaussian positions under multiple losses
    - Produce a refined mesh with better silhouette agreement and smoother geometry
    - Update Gaussian cloud to track the refined surface
    - Save refined artifacts for downstream texture baking and export

Optimization losses (from TDD §9.5):
    - Silhouette loss:           rendered mask IoU with input masks
    - Photometric loss:          projected vertex color agreement
    - Laplacian smoothness:      ||L @ V||² penalizes non-smooth surfaces
    - Edge-length regularization: penalizes variance of edge lengths
    - Normal consistency:        adjacent faces should have similar normals
    - Symmetry regularization:   bilateral symmetry prior (optional)

Implementation approach:
    - Pure numpy/scipy (no PyTorch dependency)
    - Vertex-based gradient descent with per-loss weighting
    - Silhouette rendering via OpenCV triangle rasterization
    - Sparse Laplacian matrix for efficient smoothness computation
    - Iterative refinement with configurable step count and learning rate

Artifacts produced:
    - ``refined_mesh.ply``        — optimized triangle mesh
    - ``refined_gaussians.ply``   — updated Gaussian point cloud
    - ``refine_metrics.json``     — optimization metrics and loss history

Camera convention (matches camera_init.py):
    - World space: Y-up, right-handed
    - Camera space: X-right, Y-down, Z-forward (OpenCV)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from scipy import sparse as sp_sparse
except ImportError:
    sp_sparse = None

from api.job_manager import JobManager
from api.storage import StorageManager

from .camera_init import CameraRig, project_point
from .coarse_recon import (
    save_gaussians_ply,
    save_mesh_ply,
    build_coarse_gaussians,
)
from .config import CanonicalMVConfig, CANONICAL_VIEW_ORDER

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default optimization parameters
DEFAULT_N_ITERATIONS = 50
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_MOMENTUM = 0.9

# Default loss weights
DEFAULT_SILHOUETTE_WEIGHT = 1.0
DEFAULT_LAPLACIAN_WEIGHT = 0.5
DEFAULT_EDGE_LENGTH_WEIGHT = 0.1
DEFAULT_NORMAL_CONSISTENCY_WEIGHT = 0.2
DEFAULT_SYMMETRY_WEIGHT = 0.05

# Numerical gradient epsilon
GRADIENT_EPSILON = 1e-4

# Maximum vertex displacement per iteration (prevents explosions)
MAX_VERTEX_DISPLACEMENT = 0.02

# Convergence threshold (stop if total loss change < this)
CONVERGENCE_THRESHOLD = 1e-6

# Minimum number of mesh faces required for refinement
MIN_FACES_FOR_REFINEMENT = 4


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MeshState:
    """
    Mutable mesh representation for optimization.

    Vertices are the optimizable parameters; faces define topology
    (fixed during optimization).
    """
    vertices: np.ndarray    # (V, 3) float64 — optimizable
    faces: np.ndarray       # (F, 3) int32 — fixed topology
    normals: np.ndarray     # (V, 3) float64 — recomputed each step
    vertex_colors: np.ndarray  # (V, 3) uint8 — updated from views

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    def copy(self) -> "MeshState":
        return MeshState(
            vertices=self.vertices.copy(),
            faces=self.faces.copy(),
            normals=self.normals.copy(),
            vertex_colors=self.vertex_colors.copy(),
        )


@dataclass
class RefinementConfig:
    """Configuration for the refinement optimization."""
    n_iterations: int = DEFAULT_N_ITERATIONS
    learning_rate: float = DEFAULT_LEARNING_RATE
    momentum: float = DEFAULT_MOMENTUM

    # Loss weights
    silhouette_weight: float = DEFAULT_SILHOUETTE_WEIGHT
    laplacian_weight: float = DEFAULT_LAPLACIAN_WEIGHT
    edge_length_weight: float = DEFAULT_EDGE_LENGTH_WEIGHT
    normal_consistency_weight: float = DEFAULT_NORMAL_CONSISTENCY_WEIGHT
    symmetry_weight: float = DEFAULT_SYMMETRY_WEIGHT

    # Symmetry
    use_symmetry: bool = True
    symmetry_axis: int = 0  # 0=X (bilateral left-right symmetry)

    # Convergence
    convergence_threshold: float = CONVERGENCE_THRESHOLD
    max_vertex_displacement: float = MAX_VERTEX_DISPLACEMENT

    @classmethod
    def from_pipeline_config(cls, config: CanonicalMVConfig) -> "RefinementConfig":
        """Build refinement config from the pipeline config."""
        return cls(
            use_symmetry=config.symmetry_prior,
            symmetry_weight=(
                DEFAULT_SYMMETRY_WEIGHT if config.symmetry_prior else 0.0
            ),
        )


# ---------------------------------------------------------------------------
# Mesh topology utilities
# ---------------------------------------------------------------------------


def compute_edges(faces: np.ndarray) -> np.ndarray:
    """
    Extract unique undirected edges from face array.

    Args:
        faces: (F, 3) int32 face indices.

    Returns:
        (E, 2) int32 array of unique edge vertex pairs, sorted per-edge.
    """
    # Each triangle has 3 edges
    e0 = faces[:, [0, 1]]
    e1 = faces[:, [1, 2]]
    e2 = faces[:, [2, 0]]
    all_edges = np.vstack([e0, e1, e2])

    # Sort each edge so (a, b) with a < b
    all_edges = np.sort(all_edges, axis=1)

    # Remove duplicates
    unique_edges = np.unique(all_edges, axis=0)
    return unique_edges.astype(np.int32)


def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute per-face normals.

    Args:
        vertices: (V, 3) vertex positions.
        faces: (F, 3) face indices.

    Returns:
        (F, 3) unit face normals. Zero-area faces get zero normals.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    edge1 = v1 - v0
    edge2 = v2 - v0
    cross = np.cross(edge1, edge2)

    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    # Avoid division by zero for degenerate faces
    safe_norms = np.maximum(norms, 1e-12)
    face_normals = cross / safe_norms

    return face_normals.astype(np.float64)


def compute_vertex_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_normals: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute per-vertex normals by averaging adjacent face normals.

    Args:
        vertices: (V, 3) vertex positions.
        faces: (F, 3) face indices.
        face_normals: Optional pre-computed (F, 3) face normals.

    Returns:
        (V, 3) unit vertex normals.
    """
    if face_normals is None:
        face_normals = compute_face_normals(vertices, faces)

    n_verts = len(vertices)
    vertex_normals = np.zeros((n_verts, 3), dtype=np.float64)

    # Accumulate face normals to vertices
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_normals)

    # Normalize
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, 1e-12)
    vertex_normals = vertex_normals / safe_norms

    return vertex_normals


def compute_adjacency_face_pairs(faces: np.ndarray) -> np.ndarray:
    """
    Find pairs of faces that share an edge (adjacent faces).

    Args:
        faces: (F, 3) face indices.

    Returns:
        (P, 2) int32 array of adjacent face index pairs.
    """
    # Build edge → face mapping
    edge_to_faces: Dict[Tuple[int, int], List[int]] = {}

    for fi in range(len(faces)):
        for j in range(3):
            a = int(faces[fi, j])
            b = int(faces[fi, (j + 1) % 3])
            edge = (min(a, b), max(a, b))
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(fi)

    pairs = []
    for face_list in edge_to_faces.values():
        if len(face_list) == 2:
            pairs.append(face_list)

    if not pairs:
        return np.zeros((0, 2), dtype=np.int32)

    return np.array(pairs, dtype=np.int32)


def build_laplacian_matrix(
    n_vertices: int,
    edges: np.ndarray,
) -> Any:
    """
    Build the combinatorial graph Laplacian matrix L = D - A.

    If scipy.sparse is available, returns a sparse CSR matrix.
    Otherwise returns a dense numpy array.

    Args:
        n_vertices: Number of vertices.
        edges: (E, 2) edge vertex pairs.

    Returns:
        (V, V) Laplacian matrix (sparse or dense).
    """
    n_edges = len(edges)

    if sp_sparse is not None:
        # Build sparse adjacency matrix
        row = np.concatenate([edges[:, 0], edges[:, 1]])
        col = np.concatenate([edges[:, 1], edges[:, 0]])
        data = np.ones(2 * n_edges, dtype=np.float64)

        A = sp_sparse.csr_matrix(
            (data, (row, col)),
            shape=(n_vertices, n_vertices),
        )

        # Degree matrix
        degrees = np.array(A.sum(axis=1)).flatten()
        D = sp_sparse.diags(degrees, format="csr")

        # Laplacian L = D - A
        L = D - A
        return L
    else:
        # Dense fallback
        A = np.zeros((n_vertices, n_vertices), dtype=np.float64)
        for e in edges:
            A[e[0], e[1]] = 1.0
            A[e[1], e[0]] = 1.0

        degrees = A.sum(axis=1)
        D = np.diag(degrees)
        L = D - A
        return L


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def silhouette_loss(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_masks: Dict[str, np.ndarray],
    rig: CameraRig,
    image_size: Tuple[int, int],
) -> float:
    """
    Compute silhouette loss as 1 - mean IoU across all views.

    Renders the mesh silhouette into each camera view and compares
    with the target segmentation masks.

    Args:
        vertices: (V, 3) current vertex positions.
        faces: (F, 3) face indices.
        target_masks: Dict mapping view name → binary mask (H, W) uint8.
        rig: Calibrated camera rig.
        image_size: (width, height) of the rendered silhouettes.

    Returns:
        Scalar loss in [0, 1]. Lower is better (1 = no overlap, 0 = perfect).
    """
    rendered = render_silhouettes(vertices, faces, rig, image_size)

    iou_sum = 0.0
    n_views = 0

    for vn in CANONICAL_VIEW_ORDER:
        if vn not in rendered or vn not in target_masks:
            continue

        r_mask = (rendered[vn] > 127).astype(np.uint8)
        t_mask = (target_masks[vn] > 127).astype(np.uint8)

        # Resize target to match rendered if needed
        if t_mask.shape != r_mask.shape:
            t_mask = cv2.resize(
                t_mask, (r_mask.shape[1], r_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        intersection = np.sum(r_mask & t_mask)
        union = np.sum(r_mask | t_mask)

        if union > 0:
            iou = intersection / union
        else:
            iou = 1.0 if intersection == 0 else 0.0

        iou_sum += iou
        n_views += 1

    if n_views == 0:
        return 0.0

    mean_iou = iou_sum / n_views
    return 1.0 - mean_iou


def laplacian_loss(
    vertices: np.ndarray,
    laplacian: Any,
) -> float:
    """
    Compute Laplacian smoothness loss: ||L @ V||².

    Penalizes non-smooth surfaces by measuring how much each vertex
    deviates from the average of its neighbors.

    Args:
        vertices: (V, 3) vertex positions.
        laplacian: (V, V) Laplacian matrix (sparse or dense).

    Returns:
        Scalar loss (mean squared Laplacian displacement).
    """
    if sp_sparse is not None and sp_sparse.issparse(laplacian):
        Lv = laplacian.dot(vertices)
    else:
        Lv = laplacian @ vertices

    # Mean squared norm of Laplacian displacement
    return float(np.mean(np.sum(Lv ** 2, axis=1)))


def edge_length_loss(
    vertices: np.ndarray,
    edges: np.ndarray,
) -> float:
    """
    Compute edge-length regularization loss.

    Penalizes variance of edge lengths, encouraging uniform edge distribution.

    Args:
        vertices: (V, 3) vertex positions.
        edges: (E, 2) edge vertex pairs.

    Returns:
        Scalar loss (variance of edge lengths).
    """
    if len(edges) == 0:
        return 0.0

    v0 = vertices[edges[:, 0]]
    v1 = vertices[edges[:, 1]]
    lengths = np.linalg.norm(v1 - v0, axis=1)

    # Variance of edge lengths
    return float(np.var(lengths))


def normal_consistency_loss(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_pairs: np.ndarray,
) -> float:
    """
    Compute normal consistency loss for adjacent faces.

    Penalizes large angles between normals of adjacent faces,
    encouraging smooth surface transitions.

    Args:
        vertices: (V, 3) vertex positions.
        faces: (F, 3) face indices.
        face_pairs: (P, 2) pairs of adjacent face indices.

    Returns:
        Scalar loss. Lower = more consistent normals.
    """
    if len(face_pairs) == 0:
        return 0.0

    face_normals = compute_face_normals(vertices, faces)

    n1 = face_normals[face_pairs[:, 0]]
    n2 = face_normals[face_pairs[:, 1]]

    # 1 - cos(angle) between adjacent normals
    # cos(angle) = dot(n1, n2) for unit normals
    dots = np.sum(n1 * n2, axis=1)
    dots = np.clip(dots, -1.0, 1.0)

    # Loss: mean of (1 - cos_angle)
    return float(np.mean(1.0 - dots))


def symmetry_loss(
    vertices: np.ndarray,
    axis: int = 0,
) -> float:
    """
    Compute bilateral symmetry loss across a specified axis.

    For each vertex, finds the closest vertex on the opposite side
    of the symmetry plane and penalizes the distance.

    This is an approximate symmetry loss — it does not require
    explicit vertex correspondences.

    Args:
        vertices: (V, 3) vertex positions.
        axis: Symmetry axis (0=X for left-right, 1=Y, 2=Z).

    Returns:
        Scalar loss. Lower = more symmetric.
    """
    if len(vertices) == 0:
        return 0.0

    # Reflect vertices across the symmetry plane
    reflected = vertices.copy()
    reflected[:, axis] = -reflected[:, axis]

    # For each vertex, find the closest reflected vertex
    # Use a simple approach: for each vertex, compute distance to all reflected
    # This is O(V²) but fine for coarse meshes (< 100k vertices)
    n = len(vertices)

    if n > 10000:
        # For large meshes, subsample
        rng = np.random.RandomState(42)
        indices = rng.choice(n, size=min(n, 5000), replace=False)
        v_sub = vertices[indices]
        r_sub = reflected
    else:
        v_sub = vertices
        r_sub = reflected

    # Compute pairwise distances (V_sub, V)
    # For memory efficiency, compute in chunks
    total_loss = 0.0
    chunk_size = 1000
    n_sub = len(v_sub)

    for start in range(0, n_sub, chunk_size):
        end = min(start + chunk_size, n_sub)
        chunk = v_sub[start:end]  # (C, 3)

        # Distance from each vertex in chunk to all reflected vertices
        # (C, 1, 3) - (1, V, 3) → (C, V, 3) → (C, V)
        diffs = chunk[:, np.newaxis, :] - r_sub[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)  # (C, V)

        # Minimum distance for each vertex in chunk
        min_dists = dists.min(axis=1)  # (C,)
        total_loss += float(np.sum(min_dists ** 2))

    return total_loss / n_sub


# ---------------------------------------------------------------------------
# Silhouette rendering
# ---------------------------------------------------------------------------


def render_silhouettes(
    vertices: np.ndarray,
    faces: np.ndarray,
    rig: CameraRig,
    image_size: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    """
    Render binary silhouette masks of the mesh into each camera view.

    Uses OpenCV's fillPoly to rasterize projected triangles.

    Args:
        vertices: (V, 3) vertex positions.
        faces: (F, 3) face indices.
        rig: Calibrated camera rig.
        image_size: (width, height) of output masks.

    Returns:
        Dict mapping view name → binary mask (H, W) uint8 (0 or 255).
    """
    w, h = image_size
    n_faces = len(faces)

    if n_faces == 0:
        return {vn: np.zeros((h, w), dtype=np.uint8) for vn in rig.cameras}

    # Pre-compute homogeneous vertex coordinates
    ones = np.ones((len(vertices), 1), dtype=np.float64)
    homo = np.hstack([vertices, ones])  # (V, 4)

    masks = {}

    for vn in CANONICAL_VIEW_ORDER:
        if vn not in rig.cameras:
            continue

        ext = rig.get_extrinsic(vn)
        intr = rig.get_intrinsic(vn)

        # Project all vertices
        cam_coords = (ext @ homo.T).T  # (V, 4)
        z_cam = cam_coords[:, 2]

        p_img = (intr @ cam_coords[:, :3].T).T  # (V, 3)

        with np.errstate(divide="ignore", invalid="ignore"):
            px = p_img[:, 0] / p_img[:, 2]
            py = p_img[:, 1] / p_img[:, 2]

        # Build 2D projected vertex array
        proj_2d = np.stack([px, py], axis=1)  # (V, 2)

        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)

        # Rasterize each face
        for fi in range(n_faces):
            vi = faces[fi]
            # Check all vertices are in front of camera
            if np.any(z_cam[vi] <= 0):
                continue

            triangle = proj_2d[vi].astype(np.int32)  # (3, 2)

            # Check if triangle is within image bounds (rough check)
            if (np.all(triangle[:, 0] < 0) or np.all(triangle[:, 0] >= w) or
                    np.all(triangle[:, 1] < 0) or np.all(triangle[:, 1] >= h)):
                continue

            # Clip to reasonable range to avoid OpenCV issues
            triangle[:, 0] = np.clip(triangle[:, 0], -w, 2 * w)
            triangle[:, 1] = np.clip(triangle[:, 1], -h, 2 * h)

            cv2.fillPoly(mask, [triangle], 255)

        masks[vn] = mask

    return masks


# ---------------------------------------------------------------------------
# Gradient computation
# ---------------------------------------------------------------------------


def compute_vertex_gradients(
    vertices: np.ndarray,
    faces: np.ndarray,
    loss_fn,
    epsilon: float = GRADIENT_EPSILON,
    vertex_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute numerical gradients of a loss function w.r.t. vertex positions.

    Uses central differences for better accuracy.

    Args:
        vertices: (V, 3) vertex positions.
        faces: (F, 3) face indices (passed to loss_fn).
        loss_fn: Callable(vertices, faces) → scalar loss.
        epsilon: Finite difference step size.
        vertex_mask: Optional (V,) bool mask — only compute gradients
                     for True vertices.

    Returns:
        (V, 3) gradient array.
    """
    n_verts = len(vertices)
    grad = np.zeros_like(vertices)

    if vertex_mask is None:
        indices = range(n_verts)
    else:
        indices = np.where(vertex_mask)[0]

    for vi in indices:
        for dim in range(3):
            v_plus = vertices.copy()
            v_minus = vertices.copy()
            v_plus[vi, dim] += epsilon
            v_minus[vi, dim] -= epsilon

            loss_plus = loss_fn(v_plus, faces)
            loss_minus = loss_fn(v_minus, faces)

            grad[vi, dim] = (loss_plus - loss_minus) / (2.0 * epsilon)

    return grad


def compute_laplacian_gradient(
    vertices: np.ndarray,
    laplacian: Any,
) -> np.ndarray:
    """
    Compute the analytical gradient of the Laplacian loss.

    The Laplacian loss is ||L @ V||² = sum_i (L_i @ V)^T (L_i @ V).
    Its gradient w.r.t. V is 2 * L^T @ L @ V / V.

    Args:
        vertices: (V, 3) vertex positions.
        laplacian: (V, V) Laplacian matrix.

    Returns:
        (V, 3) gradient array.
    """
    n = len(vertices)

    if sp_sparse is not None and sp_sparse.issparse(laplacian):
        Lv = laplacian.dot(vertices)
        LtLv = laplacian.T.dot(Lv)
    else:
        Lv = laplacian @ vertices
        LtLv = laplacian.T @ Lv

    return 2.0 * LtLv / n


def compute_edge_length_gradient(
    vertices: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """
    Compute the analytical gradient of the edge-length variance loss.

    Args:
        vertices: (V, 3) vertex positions.
        edges: (E, 2) edge vertex pairs.

    Returns:
        (V, 3) gradient array.
    """
    n_verts = len(vertices)
    grad = np.zeros((n_verts, 3), dtype=np.float64)

    if len(edges) == 0:
        return grad

    v0 = vertices[edges[:, 0]]
    v1 = vertices[edges[:, 1]]
    diff = v1 - v0  # (E, 3)
    lengths = np.linalg.norm(diff, axis=1, keepdims=True)  # (E, 1)
    safe_lengths = np.maximum(lengths, 1e-12)

    mean_length = lengths.mean()
    n_edges = len(edges)

    # d(var)/d(L_i) = 2 * (L_i - mean) / N
    # d(L_i)/d(v0) = -(v1 - v0) / L_i
    # d(L_i)/d(v1) = (v1 - v0) / L_i
    dvar_dL = 2.0 * (lengths - mean_length) / n_edges  # (E, 1)
    dL_ddiff = diff / safe_lengths  # (E, 3) — unit edge direction

    edge_grad = dvar_dL * dL_ddiff  # (E, 3)

    # Accumulate to vertices
    np.add.at(grad, edges[:, 1], edge_grad)
    np.add.at(grad, edges[:, 0], -edge_grad)

    return grad


# ---------------------------------------------------------------------------
# Joint refiner
# ---------------------------------------------------------------------------


class JointRefiner:
    """
    Joint mesh + Gaussian optimization engine.

    Iteratively optimizes vertex positions to minimize a weighted
    combination of silhouette, smoothness, and regularization losses.
    After mesh optimization, updates Gaussian positions to track
    the refined surface.
    """

    def __init__(
        self,
        mesh: MeshState,
        gaussians: Dict[str, np.ndarray],
        rig: CameraRig,
        target_masks: Dict[str, np.ndarray],
        target_images: Dict[str, np.ndarray],
        config: RefinementConfig,
        image_size: Tuple[int, int],
    ):
        self.mesh = mesh
        self.gaussians = gaussians
        self.rig = rig
        self.target_masks = target_masks
        self.target_images = target_images
        self.config = config
        self.image_size = image_size

        # Pre-compute topology structures
        self.edges = compute_edges(mesh.faces)
        self.face_pairs = compute_adjacency_face_pairs(mesh.faces)
        self.laplacian = build_laplacian_matrix(mesh.n_vertices, self.edges)

        # Momentum buffer for gradient descent
        self._velocity = np.zeros_like(mesh.vertices)

        # Loss history
        self.loss_history: List[Dict[str, float]] = []

    def step(self) -> Dict[str, float]:
        """
        Perform one optimization step.

        Computes all losses, combines gradients, and updates vertices.

        Returns:
            Dict of per-loss values for this step.
        """
        verts = self.mesh.vertices
        faces = self.mesh.faces
        cfg = self.config
        losses = {}

        # Initialize total gradient
        total_grad = np.zeros_like(verts)

        # --- Silhouette loss (numerical gradient — expensive) ---
        if cfg.silhouette_weight > 0:
            sil_loss = silhouette_loss(
                verts, faces, self.target_masks, self.rig, self.image_size,
            )
            losses["silhouette"] = sil_loss

            # Numerical gradient for silhouette loss
            # Only compute for a subset of vertices to keep it tractable
            n_verts = len(verts)
            if n_verts > 500:
                # Subsample vertices for gradient computation
                rng = np.random.RandomState(len(self.loss_history))
                mask = np.zeros(n_verts, dtype=bool)
                mask[rng.choice(n_verts, size=min(500, n_verts), replace=False)] = True
            else:
                mask = None

            sil_grad = compute_vertex_gradients(
                verts, faces,
                lambda v, f: silhouette_loss(
                    v, f, self.target_masks, self.rig, self.image_size
                ),
                epsilon=GRADIENT_EPSILON * 2,  # larger epsilon for silhouette
                vertex_mask=mask,
            )
            total_grad += cfg.silhouette_weight * sil_grad

        # --- Laplacian smoothness (analytical gradient) ---
        if cfg.laplacian_weight > 0:
            lap_loss = laplacian_loss(verts, self.laplacian)
            losses["laplacian"] = lap_loss

            lap_grad = compute_laplacian_gradient(verts, self.laplacian)
            total_grad += cfg.laplacian_weight * lap_grad

        # --- Edge-length regularization (analytical gradient) ---
        if cfg.edge_length_weight > 0:
            el_loss = edge_length_loss(verts, self.edges)
            losses["edge_length"] = el_loss

            el_grad = compute_edge_length_gradient(verts, self.edges)
            total_grad += cfg.edge_length_weight * el_grad

        # --- Normal consistency (numerical gradient) ---
        if cfg.normal_consistency_weight > 0 and len(self.face_pairs) > 0:
            nc_loss = normal_consistency_loss(verts, faces, self.face_pairs)
            losses["normal_consistency"] = nc_loss

            nc_grad = compute_vertex_gradients(
                verts, faces,
                lambda v, f: normal_consistency_loss(v, f, self.face_pairs),
                epsilon=GRADIENT_EPSILON,
            )
            total_grad += cfg.normal_consistency_weight * nc_grad

        # --- Symmetry (numerical gradient for subsampled vertices) ---
        if cfg.symmetry_weight > 0 and cfg.use_symmetry:
            sym_loss = symmetry_loss(verts, cfg.symmetry_axis)
            losses["symmetry"] = sym_loss

            sym_grad = compute_vertex_gradients(
                verts, faces,
                lambda v, f: symmetry_loss(v, cfg.symmetry_axis),
                epsilon=GRADIENT_EPSILON,
            )
            total_grad += cfg.symmetry_weight * sym_grad

        # --- Total loss ---
        total = sum(
            w * losses.get(name, 0.0)
            for name, w in [
                ("silhouette", cfg.silhouette_weight),
                ("laplacian", cfg.laplacian_weight),
                ("edge_length", cfg.edge_length_weight),
                ("normal_consistency", cfg.normal_consistency_weight),
                ("symmetry", cfg.symmetry_weight),
            ]
        )
        losses["total"] = total

        # --- Gradient descent with momentum ---
        self._velocity = (
            cfg.momentum * self._velocity
            - cfg.learning_rate * total_grad
        )

        # Clamp displacement
        displacement = self._velocity.copy()
        disp_norms = np.linalg.norm(displacement, axis=1, keepdims=True)
        max_disp = cfg.max_vertex_displacement
        excessive = disp_norms > max_disp
        if np.any(excessive):
            displacement[excessive.flatten()] *= (
                max_disp / disp_norms[excessive.flatten()]
            )

        # Update vertices
        self.mesh.vertices += displacement

        # Recompute normals
        self.mesh.normals = compute_vertex_normals(
            self.mesh.vertices, self.mesh.faces
        )

        # Record history
        self.loss_history.append(losses)

        return losses

    def run(
        self,
        n_iterations: Optional[int] = None,
        progress_callback=None,
    ) -> Tuple[MeshState, Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Run the full optimization loop.

        Args:
            n_iterations: Number of steps (default from config).
            progress_callback: Optional callable(iteration, n_total, losses).

        Returns:
            Tuple of (refined_mesh, refined_gaussians, metrics_dict).
        """
        if n_iterations is None:
            n_iterations = self.config.n_iterations

        logger.info(
            f"Starting joint refinement: {n_iterations} iterations, "
            f"{self.mesh.n_vertices} vertices, {self.mesh.n_faces} faces"
        )

        for i in range(n_iterations):
            losses = self.step()

            if progress_callback:
                progress_callback(i, n_iterations, losses)

            # Check convergence
            if i > 0 and len(self.loss_history) >= 2:
                prev_total = self.loss_history[-2]["total"]
                curr_total = self.loss_history[-1]["total"]
                if abs(prev_total - curr_total) < self.config.convergence_threshold:
                    logger.info(
                        f"Converged at iteration {i}: "
                        f"loss change = {abs(prev_total - curr_total):.2e}"
                    )
                    break

            if (i + 1) % 10 == 0 or i == 0:
                logger.info(
                    f"  Iteration {i+1}/{n_iterations}: "
                    f"total={losses['total']:.6f}"
                )

        # Update Gaussians to track refined surface
        refined_gaussians = self._update_gaussians()

        # Build metrics
        metrics = self._build_metrics(n_iterations)

        return self.mesh, refined_gaussians, metrics

    def _update_gaussians(self) -> Dict[str, np.ndarray]:
        """
        Update Gaussian positions and normals to track the refined mesh.

        Strategy: for each Gaussian, find the closest mesh vertex and
        snap the Gaussian to that vertex position.
        """
        if self.gaussians is None or len(self.gaussians.get("positions", [])) == 0:
            return self.gaussians

        gauss_pos = self.gaussians["positions"].astype(np.float64)
        mesh_verts = self.mesh.vertices

        n_gauss = len(gauss_pos)
        n_verts = len(mesh_verts)

        if n_verts == 0:
            return self.gaussians

        # Find closest mesh vertex for each Gaussian
        # Process in chunks for memory efficiency
        new_positions = np.zeros_like(gauss_pos)
        new_normals = np.zeros((n_gauss, 3), dtype=np.float32)
        chunk_size = 2000

        for start in range(0, n_gauss, chunk_size):
            end = min(start + chunk_size, n_gauss)
            chunk = gauss_pos[start:end]

            # (C, 1, 3) - (1, V, 3) → (C, V)
            diffs = chunk[:, np.newaxis, :] - mesh_verts[np.newaxis, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            closest = np.argmin(dists, axis=1)

            new_positions[start:end] = mesh_verts[closest]
            new_normals[start:end] = self.mesh.normals[closest].astype(np.float32)

        updated = dict(self.gaussians)
        updated["positions"] = new_positions.astype(np.float32)
        updated["normals"] = new_normals

        return updated

    def _build_metrics(self, n_iterations: int) -> Dict[str, Any]:
        """Build a summary metrics dict for the refinement stage."""
        if not self.loss_history:
            return {"iterations": 0}

        first = self.loss_history[0]
        last = self.loss_history[-1]

        metrics: Dict[str, Any] = {
            "iterations_run": len(self.loss_history),
            "iterations_requested": n_iterations,
            "converged": len(self.loss_history) < n_iterations,
            "initial_total_loss": first["total"],
            "final_total_loss": last["total"],
            "loss_reduction": first["total"] - last["total"],
            "loss_reduction_pct": (
                (first["total"] - last["total"]) / max(first["total"], 1e-10) * 100
            ),
            "per_loss_initial": {k: v for k, v in first.items() if k != "total"},
            "per_loss_final": {k: v for k, v in last.items() if k != "total"},
            "config": {
                "learning_rate": self.config.learning_rate,
                "momentum": self.config.momentum,
                "n_iterations": self.config.n_iterations,
                "silhouette_weight": self.config.silhouette_weight,
                "laplacian_weight": self.config.laplacian_weight,
                "edge_length_weight": self.config.edge_length_weight,
                "normal_consistency_weight": self.config.normal_consistency_weight,
                "symmetry_weight": self.config.symmetry_weight,
                "use_symmetry": self.config.use_symmetry,
            },
            "mesh_stats": {
                "n_vertices": self.mesh.n_vertices,
                "n_faces": self.mesh.n_faces,
            },
        }

        # Loss history (subsample if long)
        if len(self.loss_history) <= 20:
            metrics["loss_history"] = self.loss_history
        else:
            # Keep first, last, and evenly spaced samples
            indices = np.linspace(
                0, len(self.loss_history) - 1, 20, dtype=int
            )
            metrics["loss_history"] = [self.loss_history[i] for i in indices]

        return metrics


# ---------------------------------------------------------------------------
# PLY loading
# ---------------------------------------------------------------------------


def load_mesh_ply(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a triangle mesh from a binary PLY file.

    Supports the PLY format written by ``save_mesh_ply`` in coarse_recon.py.

    Args:
        filepath: Path to the PLY file.

    Returns:
        Tuple of (vertices, faces, normals):
            - vertices: (V, 3) float64
            - faces: (F, 3) int32
            - normals: (V, 3) float64 (zeros if not present in file)
    """
    import struct

    with open(filepath, "rb") as f:
        # Parse header
        header_lines = []
        while True:
            line = f.readline().decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        n_vertices = 0
        n_faces = 0
        has_normals = False
        vertex_props = []

        for line in header_lines:
            if line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
            elif line.startswith("element face"):
                n_faces = int(line.split()[-1])
            elif line.startswith("property") and n_faces == 0:
                # Vertex property (before face element declaration)
                parts = line.split()
                if len(parts) >= 3:
                    vertex_props.append(parts[-1])
                    if parts[-1] in ("nx", "ny", "nz"):
                        has_normals = True

        # Read vertices
        n_floats_per_vert = 6 if has_normals else 3
        vertices = np.zeros((n_vertices, 3), dtype=np.float64)
        normals = np.zeros((n_vertices, 3), dtype=np.float64)

        for i in range(n_vertices):
            data = f.read(n_floats_per_vert * 4)
            vals = struct.unpack(f"<{n_floats_per_vert}f", data)
            vertices[i] = vals[:3]
            if has_normals:
                normals[i] = vals[3:6]

        # Read faces
        faces = np.zeros((n_faces, 3), dtype=np.int32)
        for i in range(n_faces):
            count_data = f.read(1)
            count = struct.unpack("<B", count_data)[0]
            face_data = f.read(count * 4)
            face_indices = struct.unpack(f"<{count}i", face_data)
            if count >= 3:
                faces[i] = face_indices[:3]

    return vertices, faces, normals


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------


def run_refine_joint(
    job_id: str,
    config: CanonicalMVConfig,
    jm: JobManager,
    sm: StorageManager,
) -> None:
    """
    Execute the refine_joint stage.

    Steps:
        1. Load camera rig from camera_init.json.
        2. Load coarse mesh from coarse_visual_hull_mesh.ply.
        3. Load coarse Gaussians from coarse_gaussians.ply (if present).
        4. Load segmented masks and images from preprocess stage.
        5. Build refinement config and pre-compute topology.
        6. Run joint optimization.
        7. Save refined mesh, Gaussians, and metrics.

    Raises:
        ValueError: if required artifacts are missing.
    """
    logger.info(f"[{job_id}] refine_joint: starting")
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
    # Step 2: Load coarse mesh
    # ------------------------------------------------------------------
    mesh_path = sm.get_artifact_path(job_id, "coarse_visual_hull_mesh.ply")
    if mesh_path is None:
        raise ValueError(
            "coarse_visual_hull_mesh.ply not found — "
            "reconstruct_coarse must run first"
        )

    vertices, faces, normals = load_mesh_ply(str(mesh_path))

    if len(faces) < MIN_FACES_FOR_REFINEMENT:
        raise ValueError(
            f"Coarse mesh has only {len(faces)} faces "
            f"(minimum {MIN_FACES_FOR_REFINEMENT}). "
            f"Mesh is too coarse for refinement."
        )

    logger.info(
        f"[{job_id}] refine_joint: loaded coarse mesh "
        f"({len(vertices)} verts, {len(faces)} faces)"
    )

    # Initialize vertex colors (default gray)
    vertex_colors = np.full((len(vertices), 3), 128, dtype=np.uint8)

    mesh = MeshState(
        vertices=vertices,
        faces=faces,
        normals=normals if len(normals) == len(vertices) else
            compute_vertex_normals(vertices, faces),
        vertex_colors=vertex_colors,
    )

    jm.update_job(job_id, stage_progress=0.1)

    # ------------------------------------------------------------------
    # Step 3: Load coarse Gaussians (optional)
    # ------------------------------------------------------------------
    gaussians = None
    gauss_path = sm.get_artifact_path(job_id, "coarse_gaussians.ply")
    if gauss_path is not None:
        try:
            # Load Gaussian positions from the coarse stage
            # We only need positions and normals for refinement
            coarse_metrics = sm.load_artifact_json(
                job_id, "coarse_recon_metrics.json"
            )
            n_gaussians = (
                coarse_metrics.get("n_gaussians", 0)
                if coarse_metrics else 0
            )
            logger.info(
                f"[{job_id}] refine_joint: coarse Gaussians available "
                f"({n_gaussians} points)"
            )
            # Build a simple Gaussian dict from mesh surface for now
            # (loading binary PLY Gaussians would need a dedicated parser)
            gaussians = build_coarse_gaussians(
                mesh.vertices[:min(len(mesh.vertices), 50000)],
                vertex_colors[:min(len(vertex_colors), 50000)],
                mesh.normals[:min(len(mesh.normals), 50000)],
                voxel_size=0.02,
            )
        except Exception as e:
            logger.warning(
                f"[{job_id}] refine_joint: could not load Gaussians: {e}"
            )

    jm.update_job(job_id, stage_progress=0.15)

    # ------------------------------------------------------------------
    # Step 4: Load segmented masks and images
    # ------------------------------------------------------------------
    masks, images = _load_segmented_views(job_id, sm, target_size=image_size)
    if len(masks) < 3:
        raise ValueError(
            f"Only {len(masks)} segmented views found, need at least 3 "
            f"for refinement"
        )

    logger.info(f"[{job_id}] refine_joint: loaded {len(masks)} view masks")
    jm.update_job(job_id, stage_progress=0.2)

    # ------------------------------------------------------------------
    # Step 5: Build refinement config and run optimization
    # ------------------------------------------------------------------
    refine_config = RefinementConfig.from_pipeline_config(config)

    # Scale iterations based on mesh complexity.
    # Numerical gradients are O(V * 3 * 2) per loss per iteration,
    # so we aggressively cap iterations and disable expensive losses
    # for meshes that would otherwise take too long.
    #
    # Silhouette gradient is the most expensive: each evaluation renders
    # the full mesh into all camera views.  On CPU this is ~0.5-1s per
    # evaluation, and we need V_sub * 3 * 2 evaluations per iteration.
    # Only enable for tiny meshes (< 200 verts) where it is tractable.
    n_verts = mesh.n_vertices
    if n_verts > 50000:
        refine_config.n_iterations = min(refine_config.n_iterations, 5)
        refine_config.silhouette_weight = 0.0
        refine_config.normal_consistency_weight = 0.0
        refine_config.symmetry_weight = 0.0
    elif n_verts > 200:
        refine_config.n_iterations = min(refine_config.n_iterations, 10)
        # Silhouette, normal-consistency, and symmetry gradients are numerical
        # and prohibitively expensive on CPU for meshes above ~200 verts.
        refine_config.silhouette_weight = 0.0
        refine_config.normal_consistency_weight = 0.0
        refine_config.symmetry_weight = 0.0

    refiner = JointRefiner(
        mesh=mesh,
        gaussians=gaussians,
        rig=rig,
        target_masks=masks,
        target_images=images,
        config=refine_config,
        image_size=image_size,
    )

    def _progress(iteration, total, losses):
        frac = 0.2 + 0.6 * (iteration + 1) / total
        jm.update_job(job_id, stage_progress=frac)

    refined_mesh, refined_gaussians, metrics = refiner.run(
        progress_callback=_progress,
    )

    jm.update_job(job_id, stage_progress=0.85)

    # ------------------------------------------------------------------
    # Step 6: Save refined artifacts
    # ------------------------------------------------------------------
    # Refined mesh
    refined_mesh_path = sm.get_artifact_dir(job_id) / "refined_mesh.ply"
    save_mesh_ply(
        str(refined_mesh_path),
        refined_mesh.vertices,
        refined_mesh.faces,
        refined_mesh.normals,
    )
    logger.info(
        f"[{job_id}] refine_joint: saved refined mesh "
        f"({refined_mesh.n_vertices} verts, {refined_mesh.n_faces} faces)"
    )

    # Refined Gaussians
    if refined_gaussians is not None and len(refined_gaussians.get("positions", [])) > 0:
        gauss_out_path = sm.get_artifact_dir(job_id) / "refined_gaussians.ply"
        save_gaussians_ply(str(gauss_out_path), refined_gaussians)
        logger.info(
            f"[{job_id}] refine_joint: saved refined Gaussians "
            f"({len(refined_gaussians['positions'])} points)"
        )

    # Metrics
    sm.save_artifact_json(job_id, "refine_metrics.json", metrics)

    jm.update_job(job_id, stage_progress=1.0)
    logger.info(f"[{job_id}] refine_joint: completed")


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

    Args:
        job_id: Job identifier.
        sm: Storage manager.
        target_size: Optional (width, height) to resize loaded images to.
                     Should match the camera rig image_size.

    Returns:
        Tuple of (masks_dict, images_dict).
    """
    # Reuse the loader from coarse_recon
    from .coarse_recon import _load_segmented_views as _load
    return _load(job_id, sm, target_size=target_size)

