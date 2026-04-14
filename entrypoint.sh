#!/bin/bash
# Docker entrypoint for the GPU cluster service.
# Starts Redis, verifies GPU / model availability, then launches
# the API server and the canonical MV worker.

set -e

PORT="${PORT:-8001}"

echo "=========================================="
echo "GPU Cluster Multi-View Reconstruction"
echo "=========================================="

# ── GPU check ───────────────────────────────────────────────────
echo "Checking GPU..."
python3 -c "
import torch, sys
if not torch.cuda.is_available():
    print('WARNING: CUDA not available – will run on CPU (very slow)')
else:
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')
" || true

# ── Verify DUSt3R / MASt3R installation ─────────────────────────
echo ""
echo "Verifying reconstruction backends..."
python3 -c "
ok = True
try:
    from dust3r.model import AsymmetricCroCo3DStereo
    print('  DUSt3R .... OK')
except ImportError:
    print('  DUSt3R .... MISSING')
    ok = False
try:
    from mast3r.model import AsymmetricMASt3R
    print('  MASt3R .... OK')
except ImportError:
    print('  MASt3R .... MISSING')
    ok = False
if not ok:
    print('WARNING: Some backends are not installed. Coarse reconstruction may fail.')
" || true

# ── Start Redis if not already running ──────────────────────────
if ! pgrep -x "redis-server" > /dev/null 2>&1; then
    echo ""
    echo "Starting Redis..."
    redis-server --daemonize yes
    sleep 2
fi

# ── Ensure storage directories exist ────────────────────────────
python3 -c "from pipelines.config import ensure_directories; ensure_directories()"

# ── Start the canonical MV worker in the background ─────────────
echo "Starting canonical MV worker..."
python3 -m workers.canonical_mv_worker &
WORKER_PID=$!
echo "Worker PID: $WORKER_PID"

# ── Start the API server (foreground) ───────────────────────────
echo ""
echo "Starting API server on port ${PORT}..."
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT}"

