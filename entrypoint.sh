#!/bin/bash
# Docker entrypoint for the GPU cluster service.
# Starts Redis, the API server, and the canonical MV worker.

set -e

echo "=========================================="
echo "GPU Cluster Multi-View Reconstruction"
echo "=========================================="

# Start Redis if not already running
if ! pgrep -x "redis-server" > /dev/null 2>&1; then
    echo "Starting Redis..."
    redis-server --daemonize yes
    sleep 2
fi

# Ensure storage directories exist
python3 -c "from pipelines.config import ensure_directories; ensure_directories()"

# Start the canonical MV worker in the background
echo "Starting canonical MV worker..."
python3 -m workers.canonical_mv_worker &
WORKER_PID=$!
echo "Worker PID: $WORKER_PID"

# Start the API server (foreground)
echo "Starting API server on port 8000..."
exec uvicorn api.main:app --host 0.0.0.0 --port 8000

