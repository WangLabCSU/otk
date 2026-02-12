#!/bin/bash

# OTK Prediction API Startup Script

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create necessary directories
mkdir -p uploads results logs models

# Check if virtual environment exists, if not use system python
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Add parent src directory to Python path
export PYTHONPATH="${SCRIPT_DIR}/../src:${PYTHONPATH}"

# Set default host and port
HOST=${API_HOST:-0.0.0.0}
PORT=${API_PORT:-8000}

echo "Starting OTK Prediction API..."
echo "Host: $HOST"
echo "Port: $PORT"
echo "Python Path: ${PYTHONPATH}"
echo ""

# Start the API server
python -m uvicorn api.main:app --host "$HOST" --port "$PORT" "$@"
