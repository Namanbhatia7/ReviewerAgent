#!/usr/bin/env bash
set -e  # Exit on first error

# Set Python path so 'app' package is discoverable
export PYTHONPATH=./app

# Default host and port (can be overridden by env vars)
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8080}

# Start FastAPI app
echo "🚀 Starting FastAPI server on http://$HOST:$PORT"
LOG_LEVEL=DEBUG uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
