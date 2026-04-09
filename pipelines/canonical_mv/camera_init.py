"""
Camera initialization stage for the canonical multi-view pipeline.

Responsibilities:
    - Build a canonical camera rig from fixed view angles (yaw/pitch)
    - Compute 4×4 world-to-camera (extrinsic) matrices for each view
    - Compute 3×3 intrinsic camera matrices (shared focal length)
    - Optionally refine camera distance / focal length from silhouette metrics
    - Provide projection and reprojection utilities for downstream stages
    - Save ``camera_init.json`` artifact with all camera parameters

Camera convention:
    - World space: Y-up, right-handed
    - Camera space: X-right, Y-down, Z-forward (OpenCV convention)
    - Cameras orbit the origin at a shared distance, looking at (0, 0, 0)
    - Yaw 0 = front (camera on +Z axis), 90 = right (+X axis),
      180 = back (-Z axis), 270 = left (-X axis)
    - Pitch 0 = horizontal, -90 = looking straight down from above

Refinement (v1):
    - Adjusts the shared camera distance so that the median observed
      bounding-box-to-image ratio matches the expected projection ratio
    - Adjusts focal length within a narrow band if silhouettes are
      consistently too large or too small
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from api.job_manager import JobManager
from api.storage import StorageManager

from .config import CanonicalMVConfig, CameraSpec, CANONICAL_CAMERAS, CANONICAL_VIEW_ORDER

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default sensor width (mm) — 35mm full-frame equivalent
DEFAULT_SENSOR_WIDTH_MM = 36.0

# Default near / far clip planes (in world units)
DEFAULT_NEAR_PLANE = 0.1
DEFAULT_FAR_PLANE = 10.0

# Maximum allowed scale adjustment during refinement (± 50%)
MAX_SCALE_ADJUSTMENT = 0.5

# Maximum allowed focal length adjustment during refinement (± 20%)
MAX_FOCAL_ADJUSTMENT = 0.2

# Expected bounding-box-to-image ratio for a well-framed canonical view
# This is the fraction of the image width/height that the subject's
# projected bounding box should occupy.  Derived from the preprocessing
# FFR target (~0.60 area → ~0.77 side ratio).
EXPECTED_BBOX_IMAGE_RATIO = 0.65


# ---------------------------------------------------------------------------
# Camera math utilities
# ---------------------------------------------------------------------------


def camera_position_from_angles(
    yaw_rad: float,
    pitch_rad: float,
    distance: float,
) -> np.ndarray:
    """
    Compute camera position on a sphere centered at the origin.

    Uses the convention:
        - yaw 0 → camera on +Z axis (front)
        - yaw π/2 → camera on +X axis (right)
        - pitch 0 → horizontal plane
        - pitch -π/2 → directly above (top)

    Args:
        yaw_rad: Azimuth angle in radians.
        pitch_rad: Elevation angle in radians (negative = above).
        distance: Distance from origin.

    Returns:
        3D position as (3,) numpy array.
    """
    cos_pitch = math.cos(pitch_rad)
    x = distance * math.sin(yaw_rad) * cos_pitch
    y = -distance * math.sin(pitch_rad)  # negative because pitch<0 → above
    z = distance * math.cos(yaw_rad) * cos_pitch
    return np.array([x, y, z], dtype=np.float64)


def look_at(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray = None,
) -> np.ndarray:
    """
    Compute a 4×4 world-to-camera (view) matrix using the look-at formula.

    The resulting matrix transforms world coordinates into camera space
    following the OpenCV convention:
        - X → right
        - Y → down
        - Z → forward (into the scene)

    Args:
        eye: Camera position in world space (3,).
        target: Point the camera looks at (3,).
        up: World up vector (3,). Defaults to (0, 1, 0).

    Returns:
        4×4 world-to-camera matrix (np.float64).
    """
    if up is None:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    # Forward direction (camera looks along +Z in camera space for OpenCV)
    forward = target - eye
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-10:
        raise ValueError("Camera eye and target are at the same position")
    forward = forward / forward_norm

    # Right direction
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-10:
        # Forward is parallel to up — use a fallback up vector
        fallback_up = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        right = np.cross(forward, fallback_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-10:
            fallback_up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            right = np.cross(forward, fallback_up)
            right_norm = np.linalg.norm(right)
    right = right / right_norm

    # True up (orthogonal to forward and right)
    true_up = np.cross(right, forward)
    true_up = true_up / np.linalg.norm(true_up)

    # OpenCV convention: X=right, Y=down, Z=forward
    # So camera_y = -true_up (world up → camera down)
    R = np.eye(3, dtype=np.float64)
    R[0, :] = right       # camera X = right
    R[1, :] = -true_up    # camera Y = down
    R[2, :] = forward     # camera Z = forward

    # Translation: t = -R @ eye
    t = -R @ eye

    # Build 4×4 matrix
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t

    return extrinsic


def build_intrinsic_matrix(
    focal_length_mm: float,
    sensor_width_mm: float,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """
    Build a 3×3 camera intrinsic matrix.

    Assumes zero skew and principal point at image center.

    Args:
        focal_length_mm: Focal length in millimeters.
        sensor_width_mm: Sensor width in millimeters.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        3×3 intrinsic matrix (np.float64).
    """
    # Focal length in pixels
    fx = focal_length_mm * image_width / sensor_width_mm
    fy = fx  # square pixels

    # Principal point at image center
    cx = image_width / 2.0
    cy = image_height / 2.0

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    return K


def focal_length_pixels(
    focal_length_mm: float,
    sensor_width_mm: float,
    image_width: int,
) -> float:
    """
    Convert focal length from mm to pixels.

    Args:
        focal_length_mm: Focal length in millimeters.
        sensor_width_mm: Sensor width in millimeters.
        image_width: Image width in pixels.

    Returns:
        Focal length in pixels.
    """
    return focal_length_mm * image_width / sensor_width_mm


def project_point(
    point_3d: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Project a 3D world point to 2D image coordinates.

    Args:
        point_3d: 3D point in world space (3,).
        extrinsic: 4×4 world-to-camera matrix.
        intrinsic: 3×3 intrinsic matrix.

    Returns:
        2D point (x, y) in pixel coordinates, or None if behind camera.
    """
    point_3d = np.asarray(point_3d, dtype=np.float64)

    # Transform to camera space
    p_homo = np.append(point_3d, 1.0)
    p_cam = extrinsic @ p_homo

    # Check if point is behind the camera (z <= 0 in camera space)
    if p_cam[2] <= 0:
        return None

    # Project to image plane
    p_img = intrinsic @ p_cam[:3]
    p_img = p_img / p_img[2]

    return p_img[:2]


def project_bbox_3d(
    bbox_3d: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Project a 3D axis-aligned bounding box to a 2D image bounding box.

    Args:
        bbox_3d: 3D bounding box as (2, 3) array: [[min_x, min_y, min_z],
                 [max_x, max_y, max_z]].
        extrinsic: 4×4 world-to-camera matrix.
        intrinsic: 3×3 intrinsic matrix.

    Returns:
        2D bounding box as (x_min, y_min, x_max, y_max) in pixel coords,
        or None if any corner is behind the camera.
    """
    bbox_3d = np.asarray(bbox_3d, dtype=np.float64)
    min_pt = bbox_3d[0]
    max_pt = bbox_3d[1]

    # Generate all 8 corners
    corners = np.array([
        [min_pt[0], min_pt[1], min_pt[2]],
        [max_pt[0], min_pt[1], min_pt[2]],
        [min_pt[0], max_pt[1], min_pt[2]],
        [max_pt[0], max_pt[1], min_pt[2]],
        [min_pt[0], min_pt[1], max_pt[2]],
        [max_pt[0], min_pt[1], max_pt[2]],
        [min_pt[0], max_pt[1], max_pt[2]],
        [max_pt[0], max_pt[1], max_pt[2]],
    ], dtype=np.float64)

    projected = []
    for corner in corners:
        p2d = project_point(corner, extrinsic, intrinsic)
        if p2d is None:
            return None
        projected.append(p2d)

    projected = np.array(projected)
    x_min = float(projected[:, 0].min())
    y_min = float(projected[:, 1].min())
    x_max = float(projected[:, 0].max())
    y_max = float(projected[:, 1].max())

    return (x_min, y_min, x_max, y_max)


def compute_reprojection_error(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
) -> float:
    """
    Compute mean reprojection error for a set of 3D-2D correspondences.

    Args:
        points_3d: (N, 3) array of 3D world points.
        points_2d: (N, 2) array of observed 2D image points.
        extrinsic: 4×4 world-to-camera matrix.
        intrinsic: 3×3 intrinsic matrix.

    Returns:
        Mean Euclidean reprojection error in pixels.
    """
    points_3d = np.asarray(points_3d, dtype=np.float64)
    points_2d = np.asarray(points_2d, dtype=np.float64)

    if len(points_3d) == 0:
        return 0.0

    errors = []
    for p3d, p2d_observed in zip(points_3d, points_2d):
        p2d_projected = project_point(p3d, extrinsic, intrinsic)
        if p2d_projected is not None:
            err = np.linalg.norm(p2d_projected - p2d_observed)
            errors.append(err)

    return float(np.mean(errors)) if errors else float("inf")


# ---------------------------------------------------------------------------
# Canonical camera rig builder
# ---------------------------------------------------------------------------


class CameraRig:
    """
    Represents a complete canonical camera rig with extrinsics,
    intrinsics, and metadata for all views.
    """

    def __init__(
        self,
        cameras: Dict[str, Dict[str, Any]],
        shared_params: Dict[str, Any],
        refinement: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            cameras: Per-view camera data. Each entry has:
                extrinsic (4x4), intrinsic (3x3), position (3,),
                yaw_deg, pitch_deg, distance, focal_length_mm,
                image_width, image_height.
            shared_params: Parameters shared across all cameras.
            refinement: Refinement metadata (method, adjustments made).
        """
        self.cameras = cameras
        self.shared_params = shared_params
        self.refinement = refinement or {
            "applied": False,
            "method": None,
            "scale_factor": 1.0,
            "focal_adjustment": 0.0,
        }

    def get_extrinsic(self, view_name: str) -> np.ndarray:
        """Get the 4×4 extrinsic matrix for a view."""
        return np.array(self.cameras[view_name]["extrinsic"], dtype=np.float64)

    def get_intrinsic(self, view_name: str) -> np.ndarray:
        """Get the 3×3 intrinsic matrix for a view."""
        return np.array(self.cameras[view_name]["intrinsic"], dtype=np.float64)

    def get_position(self, view_name: str) -> np.ndarray:
        """Get the camera position in world space."""
        return np.array(self.cameras[view_name]["position"], dtype=np.float64)

    def project(
        self, view_name: str, point_3d: np.ndarray
    ) -> Optional[np.ndarray]:
        """Project a 3D point through a specific camera."""
        return project_point(
            point_3d,
            self.get_extrinsic(view_name),
            self.get_intrinsic(view_name),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the rig to a JSON-compatible dictionary.

        Numpy arrays are converted to nested lists.
        """
        cameras_serialized = {}
        for vn, cam in self.cameras.items():
            cameras_serialized[vn] = {
                "extrinsic": _ndarray_to_list(cam["extrinsic"]),
                "intrinsic": _ndarray_to_list(cam["intrinsic"]),
                "position": _ndarray_to_list(cam["position"]),
                "yaw_deg": cam["yaw_deg"],
                "pitch_deg": cam["pitch_deg"],
                "distance": cam["distance"],
                "focal_length_mm": cam["focal_length_mm"],
                "image_width": cam["image_width"],
                "image_height": cam["image_height"],
            }

        return {
            "rig_type": "canonical_3view",
            "cameras": cameras_serialized,
            "shared_params": self.shared_params,
            "refinement": self.refinement,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraRig":
        """Deserialize a CameraRig from a dictionary (e.g. loaded from JSON)."""
        cameras = {}
        for vn, cam in data["cameras"].items():
            cameras[vn] = {
                "extrinsic": np.array(cam["extrinsic"], dtype=np.float64),
                "intrinsic": np.array(cam["intrinsic"], dtype=np.float64),
                "position": np.array(cam["position"], dtype=np.float64),
                "yaw_deg": cam["yaw_deg"],
                "pitch_deg": cam["pitch_deg"],
                "distance": cam["distance"],
                "focal_length_mm": cam["focal_length_mm"],
                "image_width": cam["image_width"],
                "image_height": cam["image_height"],
            }
        return cls(
            cameras=cameras,
            shared_params=data["shared_params"],
            refinement=data.get("refinement"),
        )


def build_canonical_rig(
    config: CanonicalMVConfig,
    image_size: Tuple[int, int],
    sensor_width_mm: float = DEFAULT_SENSOR_WIDTH_MM,
) -> CameraRig:
    """
    Build the canonical 3-view camera rig from configuration.

    Each camera is placed on a sphere at the configured distance,
    looking at the origin. All cameras share the same focal length
    and image dimensions.

    Args:
        config: Pipeline configuration (contains camera specs).
        image_size: (width, height) of the images in pixels.
        sensor_width_mm: Sensor width in mm for focal length conversion.

    Returns:
        CameraRig with all 3 cameras configured.
    """
    image_width, image_height = image_size

    cameras: Dict[str, Dict[str, Any]] = {}

    for view_name in CANONICAL_VIEW_ORDER:
        spec = config.cameras[view_name]

        # Compute camera position on the sphere
        position = camera_position_from_angles(
            spec.yaw_rad, spec.pitch_rad, spec.distance
        )

        # Compute world-to-camera matrix
        target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        extrinsic = look_at(position, target)

        # Build intrinsic matrix
        intrinsic = build_intrinsic_matrix(
            spec.focal_length, sensor_width_mm, image_width, image_height
        )

        cameras[view_name] = {
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
            "position": position,
            "yaw_deg": spec.yaw_deg,
            "pitch_deg": spec.pitch_deg,
            "distance": spec.distance,
            "focal_length_mm": spec.focal_length,
            "image_width": image_width,
            "image_height": image_height,
        }

    shared_params = {
        "focal_length_mm": config.cameras["front"].focal_length,
        "focal_length_px": focal_length_pixels(
            config.cameras["front"].focal_length, sensor_width_mm, image_width
        ),
        "sensor_width_mm": sensor_width_mm,
        "image_size": [image_width, image_height],
        "distance": config.cameras["front"].distance,
        "near_plane": DEFAULT_NEAR_PLANE,
        "far_plane": DEFAULT_FAR_PLANE,
    }

    return CameraRig(cameras=cameras, shared_params=shared_params)


# ---------------------------------------------------------------------------
# Scale / focal refinement from silhouettes
# ---------------------------------------------------------------------------


def refine_rig_from_silhouettes(
    rig: CameraRig,
    preprocess_metrics: Dict[str, Any],
    config: CanonicalMVConfig,
) -> CameraRig:
    """
    Refine camera distance and focal length using silhouette metrics
    from the preprocessing stage.

    Strategy:
        1. For each side view (front, side), compute the
           observed bounding-box-to-image ratio from preprocess metrics.
        2. Compare the median observed ratio to the expected ratio.
        3. Adjust the shared camera distance so the projected size
           better matches the observed silhouettes.
        4. Optionally adjust focal length within a narrow band.

    The top view is excluded from distance refinement because its
    foreshortening makes bbox ratio unreliable for scale estimation.

    Args:
        rig: Initial canonical camera rig.
        preprocess_metrics: Loaded from ``preprocess_metrics.json``.
        config: Pipeline configuration.

    Returns:
        New CameraRig with refined parameters (or the original if
        refinement is not needed / not possible).
    """
    per_view = preprocess_metrics.get("per_view", {})
    canvas_size = preprocess_metrics.get("canvas_size", 1024)

    if not per_view:
        logger.warning("No per-view metrics available for refinement")
        return rig

    # Collect observed bbox ratios for side views only
    side_views = ["front", "side"]
    observed_ratios = []

    for vn in side_views:
        vm = per_view.get(vn)
        if vm is None:
            continue

        bbox = vm.get("bbox")
        if bbox is None or len(bbox) < 4:
            continue

        # bbox is [x, y, w, h]
        bbox_w, bbox_h = bbox[2], bbox[3]
        if bbox_w <= 0 or bbox_h <= 0:
            continue

        # Ratio of max bbox dimension to canvas size
        max_dim = max(bbox_w, bbox_h)
        ratio = max_dim / canvas_size
        observed_ratios.append(ratio)

    if len(observed_ratios) < 2:
        logger.info("Not enough side-view metrics for refinement, skipping")
        return rig

    median_ratio = float(np.median(observed_ratios))

    if median_ratio < 0.01:
        logger.warning("Median bbox ratio is near zero, skipping refinement")
        return rig

    # Compute scale adjustment
    # If observed ratio > expected → subject appears larger → camera too close
    # If observed ratio < expected → subject appears smaller → camera too far
    # Scale factor adjusts distance: new_distance = old_distance * scale_factor
    # Larger distance → smaller projection → scale_factor > 1 means "move camera back"
    scale_factor = median_ratio / EXPECTED_BBOX_IMAGE_RATIO

    # Clamp to prevent extreme adjustments
    scale_factor = max(1.0 - MAX_SCALE_ADJUSTMENT, min(1.0 + MAX_SCALE_ADJUSTMENT, scale_factor))

    # Only apply if the adjustment is meaningful (> 5%)
    if abs(scale_factor - 1.0) < 0.05:
        logger.info(
            f"Scale adjustment too small ({scale_factor:.3f}), "
            f"keeping original distance"
        )
        return rig

    logger.info(
        f"Refining camera distance: median_ratio={median_ratio:.3f}, "
        f"expected={EXPECTED_BBOX_IMAGE_RATIO:.3f}, "
        f"scale_factor={scale_factor:.3f}"
    )

    # Rebuild rig with adjusted distance
    original_distance = rig.shared_params["distance"]
    new_distance = original_distance * scale_factor

    # Create a modified config with adjusted distance
    adjusted_cameras = {}
    for vn in CANONICAL_VIEW_ORDER:
        spec = config.cameras[vn]
        adjusted_cameras[vn] = CameraSpec(
            view_name=spec.view_name,
            yaw_deg=spec.yaw_deg,
            pitch_deg=spec.pitch_deg,
            distance=new_distance,
            focal_length=spec.focal_length,
        )

    adjusted_config = CanonicalMVConfig.from_params(
        {**config.__dict__, "cameras": adjusted_cameras}
    )
    # Manually set cameras since from_params won't handle CameraSpec objects
    adjusted_config.cameras = adjusted_cameras

    image_size = (
        rig.shared_params["image_size"][0],
        rig.shared_params["image_size"][1],
    )
    new_rig = build_canonical_rig(
        adjusted_config,
        image_size,
        rig.shared_params["sensor_width_mm"],
    )

    new_rig.refinement = {
        "applied": True,
        "method": "silhouette_scale",
        "scale_factor": float(scale_factor),
        "focal_adjustment": 0.0,
        "original_distance": float(original_distance),
        "new_distance": float(new_distance),
        "median_observed_ratio": float(median_ratio),
        "expected_ratio": float(EXPECTED_BBOX_IMAGE_RATIO),
        "side_views_used": len(observed_ratios),
    }

    return new_rig


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------


def run_initialize_cameras(
    job_id: str,
    config: CanonicalMVConfig,
    jm: JobManager,
    sm: StorageManager,
) -> None:
    """
    Execute the initialize_cameras stage.

    Steps:
        1. Load preprocess metrics to determine image size and bbox info.
        2. Build the canonical camera rig from config.
        3. Optionally refine scale from silhouette metrics.
        4. Validate camera rig (sanity checks).
        5. Save ``camera_init.json`` artifact.
        6. Update job metadata.

    Raises:
        ValueError: if preprocess metrics are missing or camera rig
                    fails validation.
    """
    logger.info(f"[{job_id}] initialize_cameras: starting")
    jm.update_job(job_id, stage_progress=0.0)

    # ------------------------------------------------------------------
    # Step 1: Load preprocess metrics
    # ------------------------------------------------------------------
    preprocess_metrics = sm.load_artifact_json(job_id, "preprocess_metrics.json")
    if preprocess_metrics is None:
        raise ValueError(
            "preprocess_metrics.json not found — preprocess_views must run first"
        )

    canvas_size = preprocess_metrics.get("canvas_size", config.shared_canvas_size)
    image_size = (canvas_size, canvas_size)  # normalized views are square

    jm.update_job(job_id, stage_progress=0.2)

    # ------------------------------------------------------------------
    # Step 2: Build canonical camera rig
    # ------------------------------------------------------------------
    logger.info(
        f"[{job_id}] initialize_cameras: building rig "
        f"(distance={config.cameras['front'].distance}, "
        f"focal={config.cameras['front'].focal_length}mm, "
        f"image_size={image_size})"
    )

    rig = build_canonical_rig(config, image_size)
    jm.update_job(job_id, stage_progress=0.4)

    # ------------------------------------------------------------------
    # Step 3: Optional scale refinement
    # ------------------------------------------------------------------
    rig = refine_rig_from_silhouettes(rig, preprocess_metrics, config)
    jm.update_job(job_id, stage_progress=0.6)

    if rig.refinement["applied"]:
        logger.info(
            f"[{job_id}] initialize_cameras: refinement applied — "
            f"scale_factor={rig.refinement['scale_factor']:.3f}, "
            f"new_distance={rig.refinement.get('new_distance', 'N/A')}"
        )
    else:
        logger.info(f"[{job_id}] initialize_cameras: no refinement needed")

    # ------------------------------------------------------------------
    # Step 4: Validate camera rig
    # ------------------------------------------------------------------
    _validate_rig(rig)
    jm.update_job(job_id, stage_progress=0.8)

    # ------------------------------------------------------------------
    # Step 5: Save artifact
    # ------------------------------------------------------------------
    rig_dict = rig.to_dict()
    sm.save_artifact_json(job_id, "camera_init.json", rig_dict)

    logger.info(f"[{job_id}] initialize_cameras: saved camera_init.json")

    # ------------------------------------------------------------------
    # Step 6: Update job
    # ------------------------------------------------------------------
    jm.update_job(job_id, stage_progress=1.0)
    logger.info(f"[{job_id}] initialize_cameras: completed")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_rig(rig: CameraRig) -> None:
    """
    Run sanity checks on the camera rig.

    Checks:
        - All 3 views are present
        - Extrinsic matrices are valid (orthogonal rotation, finite values)
        - Intrinsic matrices have positive focal lengths
        - Camera positions are at expected distances from origin
        - All cameras look roughly toward the origin

    Raises:
        ValueError: if any check fails.
    """
    # Check all views present
    expected_views = set(CANONICAL_VIEW_ORDER)
    actual_views = set(rig.cameras.keys())
    if expected_views != actual_views:
        raise ValueError(
            f"Camera rig missing views: {expected_views - actual_views}"
        )

    for vn, cam in rig.cameras.items():
        extrinsic = np.array(cam["extrinsic"], dtype=np.float64)
        intrinsic = np.array(cam["intrinsic"], dtype=np.float64)
        position = np.array(cam["position"], dtype=np.float64)

        # Check for NaN/Inf
        if not np.all(np.isfinite(extrinsic)):
            raise ValueError(f"Camera '{vn}' extrinsic contains NaN/Inf")
        if not np.all(np.isfinite(intrinsic)):
            raise ValueError(f"Camera '{vn}' intrinsic contains NaN/Inf")
        if not np.all(np.isfinite(position)):
            raise ValueError(f"Camera '{vn}' position contains NaN/Inf")

        # Check rotation is orthogonal (R^T @ R ≈ I)
        R = extrinsic[:3, :3]
        RtR = R.T @ R
        identity = np.eye(3, dtype=np.float64)
        if not np.allclose(RtR, identity, atol=1e-6):
            raise ValueError(
                f"Camera '{vn}' rotation matrix is not orthogonal "
                f"(max deviation={np.max(np.abs(RtR - identity)):.2e})"
            )

        # Check det(R) = 1 (proper rotation, not reflection)
        det = np.linalg.det(R)
        if not np.isclose(det, 1.0, atol=1e-6):
            raise ValueError(
                f"Camera '{vn}' rotation determinant is {det:.6f}, expected 1.0"
            )

        # Check positive focal lengths
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        if fx <= 0 or fy <= 0:
            raise ValueError(
                f"Camera '{vn}' has non-positive focal length: fx={fx}, fy={fy}"
            )

        # Check camera distance from origin is reasonable
        dist = np.linalg.norm(position)
        expected_dist = cam["distance"]
        if not np.isclose(dist, expected_dist, rtol=0.01):
            raise ValueError(
                f"Camera '{vn}' distance from origin ({dist:.3f}) "
                f"doesn't match expected ({expected_dist:.3f})"
            )

        # Check camera looks toward origin:
        # The origin should project to roughly the image center
        origin_proj = project_point(
            np.array([0.0, 0.0, 0.0]), extrinsic, intrinsic
        )
        if origin_proj is None:
            raise ValueError(
                f"Camera '{vn}' cannot see the origin (behind camera)"
            )

        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]
        proj_error = np.linalg.norm(origin_proj - np.array([cx, cy]))
        max_error = max(cam["image_width"], cam["image_height"]) * 0.1
        if proj_error > max_error:
            raise ValueError(
                f"Camera '{vn}' origin projection ({origin_proj[0]:.1f}, "
                f"{origin_proj[1]:.1f}) is far from image center "
                f"({cx:.1f}, {cy:.1f}), error={proj_error:.1f}px"
            )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _ndarray_to_list(arr) -> Any:
    """Convert a numpy array to a nested Python list for JSON serialization."""
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return arr

