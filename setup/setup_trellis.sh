#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# Config (override via env vars)
# ----------------------------
ENV_NAME="${ENV_NAME:-trellis2}"
REPO_DIR="${REPO_DIR:-/root/TRELLIS.2}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# PyTorch versions (the set you used successfully)
TORCH_VER="${TORCH_VER:-2.6.0}"
TV_VER="${TV_VER:-0.21.0}"
TRITON_VER="${TRITON_VER:-3.2.0}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"

# GPU arch: RTX 6000 Ada = 8.9
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

# Optional: set HF_TOKEN in env before running this script
HF_TOKEN="${HF_TOKEN:-}"

log() { echo -e "\n\033[1;34m==>\033[0m $*\n"; }

# ----------------------------
# System deps
# ----------------------------
log "Installing system dependencies"
sudo apt-get update
sudo apt-get install -y \
  git build-essential curl ca-certificates \
  ffmpeg libgl1 libglib2.0-0 \
  ninja-build

# ----------------------------
# micromamba install
# ----------------------------
if ! command -v micromamba >/dev/null 2>&1; then
  log "Installing micromamba"
  curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
  sudo mv bin/micromamba /usr/local/bin/micromamba
fi

# Enable micromamba activation in this shell
eval "$(micromamba shell hook --shell bash)"

# ----------------------------
# Create/activate Python 3.10 env
# ----------------------------
if ! micromamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  log "Creating micromamba env '$ENV_NAME' (python=3.10)"
  micromamba create -y -n "$ENV_NAME" -c conda-forge python=3.10 pip
fi

log "Activating env '$ENV_NAME'"
micromamba activate "$ENV_NAME"

log "Upgrading pip tooling"
python -m pip install -U pip setuptools wheel

# ----------------------------
# CUDA env vars (needed for extensions + flash-attn builds)
# ----------------------------
log "Configuring CUDA environment variables"
export CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found. Expected at: $CUDA_HOME/bin/nvcc"
  echo "Make sure CUDA toolkit is installed and CUDA_HOME is correct."
  exit 1
fi

# ----------------------------
# Install PyTorch cu124 stack
# ----------------------------
log "Installing PyTorch ${TORCH_VER}+cu124 stack"
python -m pip install --extra-index-url "$TORCH_CUDA_INDEX" \
  "torch==${TORCH_VER}" "torchvision==${TV_VER}" "triton==${TRITON_VER}"

log "Verifying torch + CUDA"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
assert torch.cuda.is_available(), "CUDA not available in torch"
PY

# ----------------------------
# Hugging Face auth (recommended)
# ----------------------------
log "Ensuring Hugging Face CLI is available (hf)"
python -m pip install -U huggingface_hub

if ! command -v hf >/dev/null 2>&1; then
  log "hf binary not found on PATH; you can still login via module:"
  echo "  python -m huggingface_hub.cli auth login"
fi

if [ -n "$HF_TOKEN" ]; then
  log "Logging into Hugging Face using HF_TOKEN (non-interactive)"
  # Prefer the hf binary if present; fall back to python -m
  if command -v hf >/dev/null 2>&1; then
    hf auth login --token "$HF_TOKEN" || true
  else
    python -m huggingface_hub.cli auth login --token "$HF_TOKEN" || true
  fi
else
  log "HF_TOKEN not set. If you haven't logged in on this machine, run:"
  echo "  hf auth login"
  echo "or:"
  echo "  python -m huggingface_hub.cli auth login"
  echo "Otherwise downloads may fail for gated models."
fi

# ----------------------------
# Clone TRELLIS.2 repo (recursive)
# ----------------------------
if [ ! -d "$REPO_DIR" ]; then
  log "Cloning TRELLIS.2 repo to $REPO_DIR"
  git clone --recursive https://github.com/microsoft/TRELLIS.2.git "$REPO_DIR"
else
  log "TRELLIS.2 repo already exists at $REPO_DIR (skipping clone)"
fi

cd "$REPO_DIR"

# ----------------------------
# Run repo setup.sh modules (must be sourced)
# ----------------------------
if [ ! -f "./setup.sh" ]; then
  echo "ERROR: setup.sh not found in $REPO_DIR"
  exit 1
fi

log "Installing TRELLIS.2 base deps via repo setup.sh"
# setup.sh uses 'return' so it MUST be sourced.
source ./setup.sh --basic

log "Installing core extensions via repo setup.sh"
source ./setup.sh --o-voxel
source ./setup.sh --nvdiffrast
source ./setup.sh --nvdiffrec

log "Installing flash-attn (no-build-isolation + deps)"
python -m pip install -U psutil
python -m pip install -v --no-build-isolation flash-attn==2.7.3

# Optional: if you want these (uncomment)
# log "Installing cumesh + flexgemm"
# source ./setup.sh --cumesh
 source ./setup.sh --flexgemm

# ----------------------------
# Smoke test: bundled example
# ----------------------------
log "Running bundled example.py smoke test"
python example.py

log "DONE."
echo "Next time:"
echo "  eval \"\$(micromamba shell hook --shell bash)\""
echo "  micromamba activate $ENV_NAME"
echo "  cd $REPO_DIR"
echo "  python example.py"
echo "  python app.py"
