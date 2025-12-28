#!/bin/bash
# scripts/run_ci_local.sh

echo "=============================="
echo " Running Local CI Simulation "
echo "=============================="

# Get project root (one folder up from scripts/)
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"
echo "Project root: $PROJECT_ROOT"

# Set PYTHONPATH so Python can find 'src'
export PYTHONPATH="$PROJECT_ROOT"
echo "PYTHONPATH set to $PYTHONPATH"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install production dependencies
echo "Installing production dependencies..."
pip install -r "$PROJECT_ROOT/requirements.txt"

# Install dev dependencies
echo "Installing dev dependencies..."
pip install -r "$PROJECT_ROOT/requirements-dev.txt"

# Linting
echo "Running linting..."
flake8 src tests
black --check src tests

# Run tests
echo "Running tests..."
pytest tests/

# Optional: run training script to validate CI
echo "Running training script..."
python "$PROJECT_ROOT/src/train.py"

echo "=============================="
echo " Local CI Simulation Complete "
echo "=============================="
