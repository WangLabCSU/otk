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

# Set base path for reverse proxy (default: /otk for deployment)
# Set OTK_BASE_PATH="" (empty) to serve at root path
if [ -z "${OTK_BASE_PATH+x}" ]; then
    # Variable not set at all, use default
    export OTK_BASE_PATH="/otk"
elif [ "$OTK_BASE_PATH" = "" ]; then
    # Variable set to empty string, keep it empty (serve at root)
    export OTK_BASE_PATH=""
fi

echo "Starting OTK Prediction API..."
echo "Host: $HOST"
echo "Port: $PORT"
if [ -n "$OTK_BASE_PATH" ]; then
    echo "Base Path: $OTK_BASE_PATH"
else
    echo "Base Path: / (root)"
fi
echo "Python: $PYTHON_CMD"
echo "Python Path: ${PYTHONPATH}"
echo ""

# Show URL hint
if [ -n "$OTK_BASE_PATH" ]; then
    echo "API will be available at: http://${HOST}:${PORT}${OTK_BASE_PATH}/"
    echo "API Documentation: http://${HOST}:${PORT}${OTK_BASE_PATH}/docs"
else
    echo "API will be available at: http://${HOST}:${PORT}/"
    echo "API Documentation: http://${HOST}:${PORT}/docs"
fi
echo ""

# Start the API server
"$PYTHON_CMD" -m uvicorn api.main:app --host "$HOST" --port "$PORT" "$@"
