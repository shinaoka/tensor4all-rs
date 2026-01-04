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

# Install dependencies and package
if command -v uv &> /dev/null; then
    uv run python scripts/build_capi.py
    echo "Using uv..."
    uv sync --all-extras
    uv pip install -e ".[dev]"
    echo "=== Running Python tests ==="
    uv run pytest -v
else
    python3 scripts/build_capi.py
    echo "Using pip..."
    python3 -m pip install --upgrade pip
    pip install -e ".[dev]"
    echo "=== Running Python tests ==="
    pytest -v
fi

echo "=== All Python tests passed ==="
