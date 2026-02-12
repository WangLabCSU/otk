#!/bin/bash

# OTK Prediction API Startup Script

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create necessary directories
mkdir -p uploads results logs models

# Check if virtual environment exists, if not use system python
PYTHON_CMD="python"
if [ -d "venv" ]; then
    # Use virtual environment Python directly
    PYTHON_CMD="${SCRIPT_DIR}/venv/bin/python"
    export PATH="${SCRIPT_DIR}/venv/bin:$PATH"
    export VIRTUAL_ENV="${SCRIPT_DIR}/venv"
elif [ -d ".venv" ]; then
    PYTHON_CMD="${SCRIPT_DIR}/.venv/bin/python"
    export PATH="${SCRIPT_DIR}/.venv/bin:$PATH"
    export VIRTUAL_ENV="${SCRIPT_DIR}/.venv"
fi

# Add parent src directory to Python path
export PYTHONPATH="${SCRIPT_DIR}/../src:${PYTHONPATH}"

# Set default host and port
HOST=${API_HOST:-0.0.0.0}
PORT=${API_PORT:-8000}

# Set base path for reverse proxy (e.g., /otk)
# This should match the path configured in the reverse proxy
export OTK_BASE_PATH="${OTK_BASE_PATH:-}"

echo "Starting OTK Prediction API..."
echo "Host: $HOST"
echo "Port: $PORT"
echo "Base Path: ${OTK_BASE_PATH:-/ (root)}"
echo "Python: $PYTHON_CMD"
echo "Python Path: ${PYTHONPATH}"
echo ""

# Start the API server
"$PYTHON_CMD" -m uvicorn api.main:app --host "$HOST" --port "$PORT" "$@"
