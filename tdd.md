# Test-Driven Development Plan – GPU Cluster Multi-View Reconstruction

## Architecture Overview

The pipeline processes 3 canonical views (front, side, top) through:

1. **Preprocessing** – validate, resize, normalise images
2. **Camera Init** – resolve spherical poses to COLMAP extrinsics
3. **Coarse Recon** – dense geometry from full images (WITH background)
4. **Subject Isolation** – remove background, filter 3D points
5. **Trellis.2 Completion** – generative 3D asset production

## Test Files

| File | Module Under Test | Coverage |
|---|---|---|
| `tests/test_models.py` | `api/models.py` | Pydantic models, enums, serialization |
| `tests/test_pipeline_config.py` | `pipelines/config.py` | Configuration, defaults, env vars |
| `tests/test_pipeline_camera_init.py` | `pipelines/camera_init.py` | Spherical→cartesian, look-at, quaternions, COLMAP export, stitching geometry |
| `tests/test_preprocess.py` | `pipelines/preprocess.py` | Image validation, resize, normalise, batch processing |
| `tests/test_coarse_recon.py` | `pipelines/coarse_recon.py` | PLY I/O, backend factory (GPU tests require hardware) |
| `tests/test_storage.py` | `api/storage.py` | Job directory creation, uploads, artifacts, cleanup |

## Running Tests

```bash
# Run all tests (CPU only, no GPU required)
pytest tests/ -v

# Run specific test file
pytest tests/test_pipeline_camera_init.py -v

# Run with coverage
pytest tests/ --cov=api --cov=pipelines -v
```

## Test Categories

### Unit Tests (no external dependencies)
- Camera math (spherical coords, rotations, quaternions)
- Pydantic model validation
- Image preprocessing
- PLY file I/O
- Configuration defaults

### Integration Tests (require Redis)
- Job manager CRUD
- API endpoints
- Full pipeline orchestration

### GPU Tests (require CUDA + models)
- DUSt3R/MASt3R reconstruction
- Trellis.2 completion
- rembg background removal

