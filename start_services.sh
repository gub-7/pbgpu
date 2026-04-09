#!/bin/bash
# Start all GPU cluster services (non-Docker)

set -e

echo "=========================================="
echo "Starting GPU Cluster Services"
echo "=========================================="

# Check if Redis is running
if ! pgrep -x "redis-server" > /dev/null; then
    echo "Starting Redis..."
    redis-server --daemonize yes
    sleep 2
else
    echo "Redis already running"
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create log directory
mkdir -p logs

echo ""
echo "Starting services in background..."
echo "Logs will be written to logs/"
echo ""

# Define environment paths
TRIPOSR_VENV="/root/venvs/triposr"
TRELLIS_MICROMAMBA_ENV="trellis2"

# Start API server
echo "Starting API server (port 8000)..."
source "$TRIPOSR_VENV/bin/activate" && uvicorn api.main:app --host 0.0.0.0 --port 8000 > logs/api.log 2>&1 &
API_PID=$!
echo "API server started (PID: $API_PID)"

sleep 3

# Start TripoSR worker
echo "Starting TripoSR worker..."
source "$TRIPOSR_VENV/bin/activate" && MGL_BACKEND=egl python workers/triposr_worker.py > logs/triposr_worker.log 2>&1 &
TRIPOSR_PID=$!
echo "TripoSR worker started (PID: $TRIPOSR_PID)"

# Start Trellis.2 worker (uses micromamba, not stock venv)
if command -v micromamba &> /dev/null && micromamba env list 2>/dev/null | grep -q "$TRELLIS_MICROMAMBA_ENV"; then
    echo "Starting Trellis.2 worker (micromamba env: $TRELLIS_MICROMAMBA_ENV)..."
    eval "$(micromamba shell hook --shell bash)" && micromamba activate "$TRELLIS_MICROMAMBA_ENV" && python workers/trellis2_worker.py > logs/trellis2_worker.log 2>&1 &
    TRELLIS2_PID=$!
    echo "Trellis.2 worker started (PID: $TRELLIS2_PID)"
else
    echo "Skipping Trellis.2 worker (micromamba env '$TRELLIS_MICROMAMBA_ENV' not found)"
    echo "To set up Trellis.2, run the setup script for Trellis"
    TRELLIS2_PID=""
fi

# Save PIDs
echo $API_PID > logs/api.pid
echo $TRIPOSR_PID > logs/triposr_worker.pid
echo $TRELLIS2_PID > logs/trellis2_worker.pid

echo ""
echo "=========================================="
echo "All services started!"
echo "=========================================="
echo ""
echo "API:             http://localhost:8000"
echo "API docs:        http://localhost:8000/docs"
echo ""
echo "To view logs:"
echo "  tail -f logs/api.log"
echo "  tail -f logs/triposr_worker.log"
echo "  tail -f logs/trellis2_worker.log"
echo ""
echo "To stop services:"
echo "  ./stop_services.sh"
echo ""

