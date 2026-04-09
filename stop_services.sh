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
    if ps -p $API_PID > /dev/null; then
        echo "Stopping API server (PID: $API_PID)..."
        kill $API_PID
    fi
    pkill -9 uvicorn
    rm -f logs/api.pid
fi

# Stop TripoSR worker
if [ -f "logs/triposr_worker.pid" ]; then
    TRIPOSR_PID=$(cat logs/triposr_worker.pid)
    if ps -p $TRIPOSR_PID > /dev/null; then
        echo "Stopping TripoSR worker (PID: $TRIPOSR_PID)..."
        kill $TRIPOSR_PID
    fi
    rm -f logs/triposr_worker.pid
fi

# Stop Trellis.2 worker
if [ -f "logs/trellis2_worker.pid" ]; then
    TRELLIS2_PID=$(cat logs/trellis2_worker.pid)
    if ps -p $TRELLIS2_PID > /dev/null; then
        echo "Stopping Trellis.2 worker (PID: $TRELLIS2_PID)..."
        kill $TRELLIS2_PID
    fi
    rm -f logs/trellis2_worker.pid
fi

echo ""
echo "All services stopped."

