#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────
# setup_dust3r.sh – Install MASt3R + DUSt3R on a bare-metal GPU host
#
# MASt3R includes DUSt3R as a submodule, so installing MASt3R gives
# you both `import mast3r` and `import dust3r`.
#
# Neither MASt3R nor DUSt3R are pip-installable (no setup.py /
# pyproject.toml).  We clone the repo, install requirements, build
# CUDA extensions, and add the repo paths to PYTHONPATH.
#
# Prerequisites:
#   - NVIDIA GPU with driver installed (nvidia-smi works)
#   - CUDA toolkit 12.x installed (nvcc works)
#   - Python 3.10+ with pip
#
# Usage:
#   sudo bash setup/setup_dust3r.sh
#
# Environment overrides:
#   MAST3R_DIR     – where to clone MASt3R (default: /opt/mast3r)
#   CUDA_HOME      – CUDA toolkit path (default: /usr/local/cuda)
#   TORCH_CUDA_ARCH_LIST – GPU arch (default: 8.9 for Ada)
#   DOWNLOAD_CHECKPOINTS – set to "true" to download model weights
# ──────────────────────────────────────────────────────────────────

MAST3R_DIR="${MAST3R_DIR:-/opt/mast3r}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
DOWNLOAD_CHECKPOINTS="${DOWNLOAD_CHECKPOINTS:-false}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/app/model_cache}"

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST

log() { echo -e "\n\033[1;34m==>\033[0m $*\n"; }

# ── Preflight checks ────────────────────────────────────────────
log "Checking prerequisites"

if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not found at $CUDA_HOME/bin/nvcc"
    echo "Install CUDA toolkit or set CUDA_HOME correctly."
    exit 1
fi
nvcc --version

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found"
    exit 1
fi
python3 --version

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    log "PyTorch not found or CUDA not available – installing PyTorch"
    pip3 install --no-cache-dir \
        torch==2.6.0 torchvision==0.21.0 \
        --extra-index-url https://download.pytorch.org/whl/cu124
fi

log "Verifying PyTorch + CUDA"
python3 -c "
import torch
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
assert torch.cuda.is_available(), 'CUDA not available in torch'
"

# ── System dependencies ──────────────────────────────────────────
log "Installing system dependencies"
apt-get update
apt-get install -y --no-install-recommends \
    git build-essential pkg-config ninja-build cmake \
    ffmpeg libgl1 libglib2.0-0

# ── Clone MASt3R (includes DUSt3R + CroCo) ──────────────────────
if [ -d "$MAST3R_DIR/.git" ]; then
    log "MASt3R already cloned at $MAST3R_DIR – pulling latest"
    cd "$MAST3R_DIR"
    git pull
    git submodule update --init --recursive
else
    log "Cloning MASt3R (with DUSt3R + CroCo submodules)"
    git clone --recursive https://github.com/naver/mast3r "$MAST3R_DIR"
fi

cd "$MAST3R_DIR"

# ── Set PYTHONPATH ───────────────────────────────────────────────
# MASt3R and DUSt3R are NOT pip-installable (no setup.py / pyproject.toml).
# We add the repo paths to PYTHONPATH so `import mast3r` and `import dust3r`
# resolve globally.  This matches the approach used in Dockerfile.gpu.
export PYTHONPATH="$MAST3R_DIR:$MAST3R_DIR/dust3r${PYTHONPATH:+:$PYTHONPATH}"
log "PYTHONPATH set to: $PYTHONPATH"

# Persist PYTHONPATH for future shell sessions
PROFILE_SCRIPT="/etc/profile.d/mast3r.sh"
log "Writing PYTHONPATH to $PROFILE_SCRIPT"
cat > "$PROFILE_SCRIPT" <<EOF
export PYTHONPATH="$MAST3R_DIR:$MAST3R_DIR/dust3r\${PYTHONPATH:+:\$PYTHONPATH}"
EOF

# ── Install Python dependencies ──────────────────────────────────
log "Installing MASt3R Python requirements"
pip3 install --no-cache-dir -r requirements.txt

log "Installing DUSt3R Python requirements"
pip3 install --no-cache-dir -r dust3r/requirements.txt

log "Installing DUSt3R optional requirements (non-fatal)"
pip3 install --no-cache-dir -r dust3r/requirements_optional.txt || true

# ── ASMK (required by MASt3R for retrieval) ──────────────────────
log "Installing ASMK"
pip3 install --no-cache-dir cython

if [ -d "/opt/asmk/.git" ]; then
    log "ASMK already cloned at /opt/asmk"
else
    git clone https://github.com/jenicek/asmk /opt/asmk
fi

cd /opt/asmk/cython
cythonize *.pyx
cd /opt/asmk
pip3 install --no-cache-dir .

cd "$MAST3R_DIR"

# ── Build CroCo CUDA RoPE extension ─────────────────────────────
log "Building CroCo/curope CUDA extension"
cd dust3r/croco/models/curope
python3 setup.py build_ext --inplace
cd "$MAST3R_DIR"

# ── Download checkpoints (optional) ─────────────────────────────
if [ "$DOWNLOAD_CHECKPOINTS" = "true" ]; then
    log "Downloading model checkpoints to $CHECKPOINT_DIR"
    mkdir -p "$CHECKPOINT_DIR"

    DUST3R_URL="https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    MAST3R_URL="https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

    if [ ! -f "$CHECKPOINT_DIR/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" ]; then
        log "Downloading DUSt3R checkpoint (~1.1 GB)"
        wget -q --show-progress "$DUST3R_URL" -P "$CHECKPOINT_DIR/"
    else
        log "DUSt3R checkpoint already exists"
    fi

    if [ ! -f "$CHECKPOINT_DIR/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" ]; then
        log "Downloading MASt3R checkpoint (~1.1 GB)"
        wget -q --show-progress "$MAST3R_URL" -P "$CHECKPOINT_DIR/"
    else
        log "MASt3R checkpoint already exists"
    fi
fi

# ── Smoke test ───────────────────────────────────────────────────
log "Smoke test: importing dust3r and mast3r"
python3 -c "
import sys
print('PYTHONPATH:', sys.path)

import dust3r
print('dust3r imported OK')
from dust3r.model import AsymmetricCroCo3DStereo
print('DUSt3R model class imported OK')

import mast3r
print('mast3r imported OK')
from mast3r.model import AsymmetricMASt3R
print('MASt3R model class imported OK')
"

log "DONE – MASt3R + DUSt3R installed at $MAST3R_DIR"
echo ""
echo "PYTHONPATH has been persisted to $PROFILE_SCRIPT"
echo "For new shell sessions, run: source $PROFILE_SCRIPT"
echo ""
echo "Usage:"
echo "  # DUSt3R (auto-downloads checkpoint from HuggingFace on first use):"
echo "  python3 -c \"from dust3r.model import AsymmetricCroCo3DStereo; m = AsymmetricCroCo3DStereo.from_pretrained('naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt')\""
echo ""
echo "  # MASt3R:"
echo "  python3 -c \"from mast3r.model import AsymmetricMASt3R; m = AsymmetricMASt3R.from_pretrained('naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric')\""
echo ""

