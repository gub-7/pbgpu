#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# Runtime entrypoint for BrickedUp GPU containers.
#
# TRELLIS.2's setup.sh requires nvidia-smi to detect the GPU platform,
# which is NOT available during `docker build`. Instead, we run the
# TRELLIS CUDA extension setup here at container start, when the
# NVIDIA runtime exposes the GPU to the container.
#
# A marker file at /app/.trellis_setup_done prevents re-running on
# every restart of the same container. On container recreation (e.g.
# docker-compose down + up), the marker is lost and setup re-runs,
# which is correct since the extensions live in the container layer.
# ─────────────────────────────────────────────────────────────────────
set -e

MARKER="/app/.trellis_setup_done"
TRELLIS_DIR="/opt/trellis2"

# ── TRELLIS.2 CUDA extension setup (runs once per container) ────────
if [ ! -f "$MARKER" ] && [ -d "$TRELLIS_DIR" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "══════════════════════════════════════════════════════════════"
        echo "  TRELLIS.2 first-run setup: building CUDA extensions..."
        echo "  (This only runs once per container lifecycle)"
        echo "══════════════════════════════════════════════════════════════"
        nvidia-smi || true

        cd "$TRELLIS_DIR"

        echo "[1/4] setup.sh --basic ..."
        bash -c 'source ./setup.sh --basic' || echo "WARN: --basic setup failed (non-fatal)"

        echo "[2/4] setup.sh --o-voxel ..."
        bash -c 'source ./setup.sh --o-voxel' || echo "WARN: --o-voxel setup failed (non-fatal)"

        echo "[3/4] setup.sh --nvdiffrast ..."
        bash -c 'source ./setup.sh --nvdiffrast' || echo "WARN: --nvdiffrast setup failed (non-fatal)"

        echo "[4/4] setup.sh --nvdiffrec ..."
        bash -c 'source ./setup.sh --nvdiffrec' || echo "WARN: --nvdiffrec setup failed (non-fatal)"

        # Write marker so we skip this on container restart
        touch "$MARKER"
        echo "══════════════════════════════════════════════════════════════"
        echo "  TRELLIS.2 CUDA extension setup complete."
        echo "══════════════════════════════════════════════════════════════"

        cd /app
    else
        echo "WARN: nvidia-smi not found at runtime; skipping TRELLIS CUDA setup."
        echo "      GPU-dependent features (mesh extraction) may not work."
    fi
else
    if [ -f "$MARKER" ]; then
        echo "TRELLIS.2 CUDA extensions already built (marker: $MARKER)."
    fi
fi

# ── Hand off to the original CMD ────────────────────────────────────
exec "$@"

