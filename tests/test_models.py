"""Tests for api/models.py – Pydantic data models."""

import pytest
from api.models import (
    CameraExtrinsics,
    CameraIntrinsics,
    CoarseReconResult,
    CreateJobRequest,
    IsolationResult,
    JobDetailResponse,
    JobStatus,
    JobStatusResponse,
    PipelineConfig,
    PointCloud,
    ReconBackend,
    ReconJob,
    ResolvedView,
    SphericalPose,
    TrellisResult,
    ViewLabel,
    ViewSpec,
    WorldConvention,
)


class TestEnums:
    def test_view_labels(self):
        assert ViewLabel.FRONT == "front"
        assert ViewLabel.SIDE == "side"
        assert ViewLabel.TOP == "top"

    def test_job_status_values(self):
        assert JobStatus.PENDING == "pending"
        assert JobStatus.COMPLETED == "completed"
        assert JobStatus.FAILED == "failed"

    def test_recon_backends(self):
        assert ReconBackend.DUST3R == "dust3r"
        assert ReconBackend.MAST3R == "mast3r"
        assert ReconBackend.VGGT == "vggt"


class TestWorldConvention:
    def test_defaults(self):
        wc = WorldConvention()
        assert wc.origin == "object_center"
        assert wc.up_axis == "Z"
        assert wc.forward_axis == "Y"


class TestCameraIntrinsics:
    def test_defaults(self):
        intr = CameraIntrinsics()
        assert intr.width == 2048
        assert intr.height == 2048
        assert intr.fx == 1700.0

    def test_custom(self):
        intr = CameraIntrinsics(width=1024, height=1024, fx=800, fy=800, cx=512, cy=512)
        assert intr.width == 1024
        assert intr.fx == 800


class TestSphericalPose:
    def test_defaults(self):
        pose = SphericalPose()
        assert pose.radius == 1.2
        assert pose.azimuth_deg == 0.0
        assert pose.target_world == [0.0, 0.0, 0.0]

    def test_custom(self):
        pose = SphericalPose(radius=2.0, azimuth_deg=90.0, elevation_deg=45.0)
        assert pose.radius == 2.0
        assert pose.azimuth_deg == 90.0


class TestViewSpec:
    def test_creation(self):
        vs = ViewSpec(
            label=ViewLabel.FRONT,
            image_filename="front.png",
            pose=SphericalPose(),
        )
        assert vs.label == ViewLabel.FRONT
        assert vs.image_filename == "front.png"


class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.recon_backend == ReconBackend.DUST3R
        assert cfg.use_background_for_pose is True
        assert cfg.trellis_enabled is True
        assert cfg.image_size == 2048

    def test_custom(self):
        cfg = PipelineConfig(
            recon_backend=ReconBackend.MAST3R,
            trellis_enabled=False,
            image_size=1024,
        )
        assert cfg.recon_backend == ReconBackend.MAST3R
        assert cfg.trellis_enabled is False


class TestReconJob:
    def test_defaults(self):
        job = ReconJob()
        assert job.status == JobStatus.PENDING
        assert len(job.job_id) == 32  # hex UUID
        assert job.views == []
        assert job.coarse_result is None

    def test_serialization_roundtrip(self):
        job = ReconJob()
        json_str = job.model_dump_json()
        restored = ReconJob.model_validate_json(json_str)
        assert restored.job_id == job.job_id
        assert restored.status == job.status


class TestPointCloud:
    def test_defaults(self):
        pc = PointCloud()
        assert pc.num_points == 0
        assert pc.ply_path is None


class TestCoarseReconResult:
    def test_defaults(self):
        result = CoarseReconResult()
        assert result.point_cloud.num_points == 0
        assert result.views == []


class TestIsolationResult:
    def test_defaults(self):
        result = IsolationResult()
        assert result.masked_image_paths == []
        assert result.num_points_retained == 0


class TestTrellisResult:
    def test_defaults(self):
        result = TrellisResult()
        assert result.mesh_path is None
        assert result.metadata == {}


class TestCreateJobRequest:
    def test_defaults(self):
        req = CreateJobRequest()
        assert req.config is None

    def test_with_config(self):
        cfg = PipelineConfig(recon_backend=ReconBackend.VGGT)
        req = CreateJobRequest(config=cfg)
        assert req.config.recon_backend == ReconBackend.VGGT

