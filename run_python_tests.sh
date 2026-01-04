#!/bin/bash
# Run Python tests for pytensor4all
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Building tensor4all-capi ==="
cargo build --release -p tensor4all-capi

echo "=== Setting up Python environment ==="
cd pytensor4all

# Copy library
python scripts/build_capi.py

# Install dependencies and package
if command -v uv &> /dev/null; then
    echo "Using uv..."
    uv sync --all-extras
    uv pip install -e ".[dev]"
    echo "=== Running Python tests ==="
    uv run pytest -v
else
    echo "Using pip..."
    python -m pip install --upgrade pip
    pip install -e ".[dev]"
    echo "=== Running Python tests ==="
    pytest -v
fi

echo "=== All Python tests passed ==="
