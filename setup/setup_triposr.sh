#!/usr/bin/env bash
set -euo pipefail

### --- Config you may want to change ---
INSTALL_DIR="${INSTALL_DIR:-/root/TripoSR}"
VENV_DIR="${VENV_DIR:-/root/venvs/triposr}"
CUDA_PKG_VER="${CUDA_PKG_VER:-12-8}"            # CUDA 12.8 toolchain
PYTORCH_VER="${PYTORCH_VER:-2.10.0+cu128}"      # matches your working env
TORCHVISION_VER="${TORCHVISION_VER:-0.15.2+cu128}"
TORCHAUDIO_VER="${TORCHAUDIO_VER:-2.0.2+cu128}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVIDIA_DRIVER_PKG="${NVIDIA_DRIVER_PKG:-nvidia-driver-570-server}"  # good modern choice on 24.04
### ------------------------------------

log() { echo -e "\n\033[1;32m==>\033[0m $*\n"; }
warn() { echo -e "\n\033[1;33m[warn]\033[0m $*\n"; }

if [[ $EUID -ne 0 ]]; then
  echo "Please run as root (or with sudo)."
  exit 1
fi

log "1) Base OS packages (includes EGL/GLES/OSMesa for TripoSR texture baking)"
apt update
apt install -y \
  git build-essential pkg-config \
  python3 python3-venv python3-dev \
  ffmpeg libgl1 libglib2.0-0 \
  libegl1 libgles2 libosmesa6 \
  ca-certificates curl wget gnupg

log "2) NVIDIA driver (recommended)"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  warn "nvidia-smi not found; installing driver package: ${NVIDIA_DRIVER_PKG}"
  apt install -y "${NVIDIA_DRIVER_PKG}" || true
  warn "Driver install may require a reboot. If CUDA/GPU is not available after this step, reboot and rerun this script."
else
  log "nvidia-smi already present"
fi

# If driver is installed but kernel modules not loaded yet, a reboot may be needed.
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi >/dev/null 2>&1; then
    warn "nvidia-smi exists but failed to run. Reboot is likely required."
    echo "Reboot now: sudo reboot"
    exit 2
  fi
  log "GPU visible:"
  nvidia-smi || true
else
  warn "nvidia-smi still not present. You may be on a provider image that handles drivers differently."
  warn "Continuing anyway (PyTorch will not see GPU until driver is working)."
fi

log "3) Add NVIDIA CUDA apt repo (cuda-keyring) + install CUDA toolkit/nvcc ${CUDA_PKG_VER}"
# Clean up common duplicate repo files if present
rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list || true

wget -qO /tmp/cuda-keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
dpkg -i /tmp/cuda-keyring.deb
apt update

# Install toolkit + nvcc explicitly (matches what we ended with)
apt install -y "cuda-toolkit-${CUDA_PKG_VER}" "cuda-nvcc-${CUDA_PKG_VER}" "cuda-compiler-${CUDA_PKG_VER}"

log "4) Ensure CUDA env vars are persistent"
cat >/etc/profile.d/cuda.sh <<EOF
export CUDA_HOME=${CUDA_HOME}
export PATH="\$CUDA_HOME/bin:\$PATH"
export LD_LIBRARY_PATH="\$CUDA_HOME/lib64:\$CUDA_HOME/targets/x86_64-linux/lib:\$LD_LIBRARY_PATH"
EOF

# Apply to current shell too
# shellcheck disable=SC1091
source /etc/profile.d/cuda.sh

log "CUDA sanity checks"
ls -la "${CUDA_HOME}/bin/nvcc"
nvcc --version

log "5) Create Python venv (Python 3.12 on Ubuntu 24.04) and upgrade packaging tools"
mkdir -p "$(dirname "${VENV_DIR}")"
python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install -U pip setuptools wheel

log "6) Install PyTorch CUDA 12.8 (cu128) pinned to your working versions"
pip install \
  "torch==${PYTORCH_VER}" \
  "torchvision==${TORCHVISION_VER}" \
  "torchaudio==${TORCHAUDIO_VER}" \
  --index-url https://download.pytorch.org/whl/cu128

log "7) Clone (or update) TripoSR repo"
if [[ -d "${INSTALL_DIR}/.git" ]]; then
  git -C "${INSTALL_DIR}" fetch --all
  git -C "${INSTALL_DIR}" reset --hard origin/main || true
else
  git clone https://github.com/VAST-AI-Research/TripoSR.git "${INSTALL_DIR}"
fi
cd "${INSTALL_DIR}"

log "8) Install TripoSR python requirements"
pip install -r requirements.txt

log "9) Install moderngl + glcontext (required for TripoSR texture baking)"
pip install -U moderngl glcontext

log "10) Build/install torchmcubes (CUDA extension) from git"
# Ensure build sees nvcc
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=${CUDA_HOME}/bin/nvcc -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME}"

pip uninstall -y torchmcubes || true
pip install --no-cache-dir --verbose git+https://github.com/tatsy/torchmcubes.git

log "11) Install onnxruntime-gpu for rembg"
pip install -U onnxruntime-gpu

log "12) Verify imports + GPU availability"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

import rembg
print("rembg import: OK")

import onnxruntime as ort
print("onnxruntime providers:", ort.get_available_providers())

import torchmcubes
print("torchmcubes import: OK")

import moderngl
print("moderngl import: OK, version:", moderngl.__version__)

import glcontext
print("glcontext import: OK")
PY

log "13) Done. Test run (creates output/)"
echo "Run:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  cd ${INSTALL_DIR}"
echo "  python run.py examples/chair.png --output-dir output/"
