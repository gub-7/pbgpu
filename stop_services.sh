#!/bin/bash
# Stop all GPU cluster services

echo "=========================================="
echo "Stopping GPU Cluster Services"
echo "=========================================="

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if PID files exist
if [ ! -d "logs" ]; then
    echo "No logs directory found. Services may not be running."
    exit 0
fi

# Stop API server
if [ -f "logs/api.pid" ]; then
    API_PID=$(cat logs/api.pid)
    if ps -p $API_PID > /dev/null 2>&1; then
        echo "Stopping API server (PID: $API_PID)..."
        kill $API_PID
    fi
    rm -f logs/api.pid
fi

# Stop canonical MV worker
if [ -f "logs/canonical_mv_worker.pid" ]; then
    WORKER_PID=$(cat logs/canonical_mv_worker.pid)
    if ps -p $WORKER_PID > /dev/null 2>&1; then
        echo "Stopping canonical MV worker (PID: $WORKER_PID)..."
        kill $WORKER_PID
    fi
    rm -f logs/canonical_mv_worker.pid
fi

# Clean up any stray uvicorn processes
pkill -f "uvicorn api.main:app" 2>/dev/null || true

echo ""
echo "All services stopped."

