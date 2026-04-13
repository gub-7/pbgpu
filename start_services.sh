#!/bin/bash
# Start all GPU cluster services (non-Docker)
#
# Architecture:
#   1. Redis – job queue and state persistence
#   2. API server (FastAPI/uvicorn) – HTTP endpoints
#   3. Canonical MV worker – processes reconstruction jobs
#
# Pipeline stages run by the worker:
#   Preprocessing → Camera Init → Coarse Recon (DUSt3R/MASt3R) →
#   Subject Isolation → Trellis.2 Completion

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

# Create directories
mkdir -p logs storage model_cache

# Load environment if .env exists
if [ -f .env ]; then
    echo "Loading .env..."
    set -a
    source .env
    set +a
fi

# API port – matches docker-compose.gpu.yml and backend GPU_CLUSTER_URL
PORT="${PORT:-8001}"

echo ""
echo "Starting services in background..."
echo "Logs will be written to logs/"
echo ""

# Define environment paths
TRIPOSR_VENV="${TRIPOSR_VENV:-/root/venvs/triposr}"

# Start API server
echo "Starting API server (port ${PORT})..."
if [ -d "$TRIPOSR_VENV" ]; then
    source "$TRIPOSR_VENV/bin/activate"
fi
uvicorn api.main:app --host 0.0.0.0 --port "${PORT}" > logs/api.log 2>&1 &
API_PID=$!
echo "API server started (PID: $API_PID)"

sleep 3

# Start canonical MV worker
echo "Starting canonical MV worker..."
python -m workers.canonical_mv_worker > logs/canonical_mv_worker.log 2>&1 &
WORKER_PID=$!
echo "Canonical MV worker started (PID: $WORKER_PID)"

# Save PIDs
echo $API_PID > logs/api.pid
echo $WORKER_PID > logs/canonical_mv_worker.pid

echo ""
echo "=========================================="
echo "All services started!"
echo "=========================================="
echo ""
echo "API:             http://localhost:${PORT}"
echo "API docs:        http://localhost:${PORT}/docs"
echo ""
echo "Pipeline: preprocessing → camera init → coarse recon → isolation → trellis"
echo ""
echo "To view logs:"
echo "  tail -f logs/api.log"
echo "  tail -f logs/canonical_mv_worker.log"
echo ""
echo "To stop services:"
echo "  ./stop_services.sh"
echo ""

