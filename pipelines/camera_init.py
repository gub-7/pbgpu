"""
Camera initialisation for the canonical 3-view capture rig.

Defines the three canonical views (front, side, top) as spherical poses,
converts them to world-to-camera extrinsics in COLMAP convention, and
provides utilities for exporting to COLMAP-compatible text files.

World convention (object-centric):
  - Origin at the object centre
  - +Z = up
  - +Y = object forward (the face the "front" camera looks at)
  - +X = object right

COLMAP camera axes:
  - +X = right in image
  - +Y = down in image
  - +Z = forward (into the scene)

The three canonical views and their stitching geometry:
  - FRONT: looks at the object from -Y direction (azimuth=0, elevation=0)
  - SIDE:  looks at the object from -X direction (azimuth=90, elevation=0)
    → stitches to front's LEFT edge (side's RIGHT edge)
  - TOP:   looks at the object from +Z direction (azimuth=0, elevation=90)
    → stitches to front's TOP edge (top's BOTTOM edge)
    → top's LEFT edge meets side's TOP edge
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

from api.models import (
    CameraExtrinsics,
    CameraIntrinsics,
    ResolvedView,
    SphericalPose,
    ViewLabel,
    ViewSpec,
)
from pipelines.config import DEFAULT_RADIUS, get_default_intrinsics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical view definitions
# ---------------------------------------------------------------------------

# Front view: camera on -Y axis looking toward origin
CANONICAL_FRONT = SphericalPose(
    radius=DEFAULT_RADIUS,
    azimuth_deg=0.0,
    elevation_deg=0.0,
    roll_deg=0.0,
    target_world=[0.0, 0.0, 0.0],
)

# Side view: camera on -X axis looking toward origin
# Azimuth 90° places the camera at +X, but we want it looking from -X
# so azimuth=270° (or equivalently -90°), but per the stitching spec
# "front's left edge meets side's right edge" → the side camera is at
# azimuth = 90° (looking from +X toward origin).
#
# Actually, let's think about this carefully:
#   azimuth=0   → camera at (0, -r, 0) looking toward origin along +Y
#   azimuth=90  → camera at (r, 0, 0)  looking toward origin along -X
#   azimuth=270 → camera at (-r, 0, 0) looking toward origin along +X
#
# The front camera sees the object's front face (the -Y face).
# For stitching, front's LEFT edge = the object's +X side.
# So the side camera should see the +X face → camera at (+r, 0, 0)
# → azimuth = 90°.
CANONICAL_SIDE = SphericalPose(
    radius=DEFAULT_RADIUS,
    azimuth_deg=90.0,
    elevation_deg=0.0,
    roll_deg=0.0,
    target_world=[0.0, 0.0, 0.0],
)

# Top view: camera on +Z axis looking down toward origin
# elevation=90° places camera directly above.
CANONICAL_TOP = SphericalPose(
    radius=DEFAULT_RADIUS,
    azimuth_deg=0.0,
    elevation_deg=90.0,
    roll_deg=0.0,
    target_world=[0.0, 0.0, 0.0],
)


# ---------------------------------------------------------------------------
# Spherical → Cartesian → extrinsics
# ---------------------------------------------------------------------------


def spherical_to_camera_center(pose: SphericalPose) -> np.ndarray:
    """
    Convert a spherical pose to a camera centre in world coordinates.

    C_x = r * cos(elevation) * sin(azimuth)
    C_y = r * cos(elevation) * (-cos(azimuth))   [front at azimuth=0 → C_y = -r]
    C_z = r * sin(elevation)

    Note: We use azimuth=0 → camera on the -Y axis (looking at object front).
    """
    r = pose.radius
    az = math.radians(pose.azimuth_deg)
    el = math.radians(pose.elevation_deg)

    cx = r * math.cos(el) * math.sin(az)
    cy = r * math.cos(el) * (-math.cos(az))
    cz = r * math.sin(el)

    target = np.array(pose.target_world, dtype=np.float64)
    return target + np.array([cx, cy, cz], dtype=np.float64)


def build_lookat_rotation(
    camera_center: np.ndarray,
    target: np.ndarray,
    world_up: np.ndarray = np.array([0.0, 0.0, 1.0]),
    roll_deg: float = 0.0,
) -> np.ndarray:
    """
    Build a camera-to-world rotation matrix using look-at convention.

    Returns R_c2w where columns are [right, up, forward] in world coords.
    The camera looks along +Z in its own frame (COLMAP convention).

    Parameters
    ----------
    camera_center : world position of the camera
    target : world position the camera looks at
    world_up : world up direction (default +Z)
    roll_deg : roll angle around the forward axis
    """
    forward = target - camera_center
    forward = forward / (np.linalg.norm(forward) + 1e-12)

    # Handle degenerate case when forward is parallel to world_up (top view)
    if abs(np.dot(forward, world_up)) > 0.999:
        # For top-down view looking along -Z, use +Y as the "up" proxy
        # so that the image top aligns with -Y (back of object)
        # and image left aligns with -X.
        # This ensures top's bottom edge = front's top edge.
        if forward[2] < 0:
            # Looking down
            aux_up = np.array([0.0, -1.0, 0.0])
        else:
            # Looking up
            aux_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, aux_up)
        right = right / (np.linalg.norm(right) + 1e-12)
        up = np.cross(right, forward)
        up = up / (np.linalg.norm(up) + 1e-12)
    else:
        right = np.cross(forward, world_up)
        right = right / (np.linalg.norm(right) + 1e-12)
        up = np.cross(right, forward)
        up = up / (np.linalg.norm(up) + 1e-12)

    # Apply roll around the forward axis
    if abs(roll_deg) > 1e-6:
        roll_rad = math.radians(roll_deg)
        cos_r = math.cos(roll_rad)
        sin_r = math.sin(roll_rad)
        right_new = cos_r * right + sin_r * up
        up_new = -sin_r * right + cos_r * up
        right = right_new
        up = up_new

    # COLMAP convention: camera +X=right, +Y=down, +Z=forward
    # So camera-to-world maps:
    #   cam_x (right)   → world right
    #   cam_y (down)     → world (-up)
    #   cam_z (forward)  → world forward
    R_c2w = np.column_stack([right, -up, forward])

    return R_c2w


def rotation_to_quaternion_wxyz(R: np.ndarray) -> list[float]:
    """
    Convert a 3×3 rotation matrix to a unit quaternion [w, x, y, z].

    Uses the Shepperd method for numerical stability.
    """
    m = R
    trace = m[0, 0] + m[1, 1] + m[2, 2]

    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    # Ensure w > 0 for consistency
    quat = np.array([w, x, y, z])
    if w < 0:
        quat = -quat

    # Normalize
    quat = quat / (np.linalg.norm(quat) + 1e-12)
    return quat.tolist()


def pose_to_extrinsics(pose: SphericalPose) -> CameraExtrinsics:
    """
    Convert a SphericalPose to world-to-camera extrinsics (COLMAP convention).

    Returns R_w2c, t_w2c, and the COLMAP quaternion.
    """
    target = np.array(pose.target_world, dtype=np.float64)
    C = spherical_to_camera_center(pose)

    R_c2w = build_lookat_rotation(C, target, roll_deg=pose.roll_deg)

    # World-to-camera
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ C

    quat = rotation_to_quaternion_wxyz(R_w2c)

    return CameraExtrinsics(
        R_w2c=R_w2c.flatten().tolist(),
        t_w2c=t_w2c.tolist(),
        quaternion_wxyz=quat,
    )


# ---------------------------------------------------------------------------
# Build canonical views
# ---------------------------------------------------------------------------


def get_canonical_views(
    image_filenames: Optional[dict[ViewLabel, str]] = None,
    radius: Optional[float] = None,
) -> list[ViewSpec]:
    """
    Return the three canonical ViewSpecs for the capture rig.

    Parameters
    ----------
    image_filenames : optional mapping of ViewLabel → filename.
        Defaults to "front.png", "side.png", "top.png".
    radius : optional override for camera distance.
    """
    filenames = image_filenames or {
        ViewLabel.FRONT: "front.png",
        ViewLabel.SIDE: "side.png",
        ViewLabel.TOP: "top.png",
    }

    poses = {
        ViewLabel.FRONT: CANONICAL_FRONT.model_copy(),
        ViewLabel.SIDE: CANONICAL_SIDE.model_copy(),
        ViewLabel.TOP: CANONICAL_TOP.model_copy(),
    }

    if radius is not None:
        for p in poses.values():
            p.radius = radius

    views = []
    for label in (ViewLabel.FRONT, ViewLabel.SIDE, ViewLabel.TOP):
        views.append(
            ViewSpec(
                label=label,
                image_filename=filenames[label],
                pose=poses[label],
            )
        )

    return views


def resolve_views(
    view_specs: list[ViewSpec],
    intrinsics: Optional[CameraIntrinsics] = None,
) -> list[ResolvedView]:
    """
    Convert ViewSpecs into fully resolved views with computed extrinsics.

    Parameters
    ----------
    view_specs : list of ViewSpec with poses
    intrinsics : shared camera intrinsics (defaults to pipeline default)
    """
    intr = intrinsics or get_default_intrinsics()
    resolved = []

    for vs in view_specs:
        ext = pose_to_extrinsics(vs.pose)
        resolved.append(
            ResolvedView(
                label=vs.label,
                image_filename=vs.image_filename,
                intrinsics=intr,
                extrinsics=ext,
                pose=vs.pose,
            )
        )
        logger.info(
            "Resolved view %s: C=%s, quat=%s",
            vs.label.value,
            spherical_to_camera_center(vs.pose).tolist(),
            ext.quaternion_wxyz,
        )

    return resolved


# ---------------------------------------------------------------------------
# COLMAP export utilities
# ---------------------------------------------------------------------------


def export_colmap_cameras_txt(
    views: list[ResolvedView],
    output_path: Path,
) -> None:
    """
    Write a COLMAP-format cameras.txt file.

    All views share the same intrinsics (single camera model).
    Format: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    intr = views[0].intrinsics
    lines = [
        "# Camera list with one line of data per camera:",
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: 1",
        f"1 PINHOLE {intr.width} {intr.height} {intr.fx} {intr.fy} {intr.cx} {intr.cy}",
    ]

    output_path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote COLMAP cameras.txt to %s", output_path)


def export_colmap_images_txt(
    views: list[ResolvedView],
    output_path: Path,
) -> None:
    """
    Write a COLMAP-format images.txt file.

    Format per image (2 lines):
      IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
      <empty line for 2D points – we leave it blank>
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "# Image list with two lines of data per image:",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(views)}",
    ]

    lines = list(header)
    for idx, v in enumerate(views, start=1):
        q = v.extrinsics.quaternion_wxyz
        t = v.extrinsics.t_w2c
        lines.append(
            f"{idx} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} "
            f"{t[0]:.10f} {t[1]:.10f} {t[2]:.10f} 1 {v.image_filename}"
        )
        lines.append("")  # empty line for 2D points

    output_path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote COLMAP images.txt to %s", output_path)


def export_colmap_points3D_txt(output_path: Path) -> None:
    """Write an empty COLMAP points3D.txt (populated later by reconstruction)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# 3D point list with one line of data per point:",
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
        "# Number of points: 0",
    ]
    output_path.write_text("\n".join(header) + "\n")
    logger.info("Wrote empty COLMAP points3D.txt to %s", output_path)


def export_colmap_workspace(
    views: list[ResolvedView],
    workspace_dir: Path,
) -> Path:
    """
    Create a full COLMAP-compatible sparse model directory.

    Returns the path to the sparse/0/ directory.
    """
    sparse_dir = workspace_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    export_colmap_cameras_txt(views, sparse_dir / "cameras.txt")
    export_colmap_images_txt(views, sparse_dir / "images.txt")
    export_colmap_points3D_txt(sparse_dir / "points3D.txt")

    logger.info("Exported COLMAP workspace to %s", sparse_dir)
    return sparse_dir

