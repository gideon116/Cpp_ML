#!/bin/bash
# build_pip.sh — Build and install the wefml pip package
set -e

echo "=== Building wefml Python package ==="
echo ""

# Check prerequisites
command -v python3 >/dev/null 2>&1 || { echo "python3 not found"; exit 1; }
command -v cmake >/dev/null 2>&1 || { echo "cmake not found"; exit 1; }

# Install build dependencies if needed
pip install --upgrade pip
pip install pybind11 scikit-build-core numpy

# Build and install in development mode
echo ""
echo "=== Installing wefml in development mode ==="
pip install -e .

echo ""
echo "=== Done! ==="
echo "Test with:"
echo "  python3 -c 'import wefml; print(\"backend:\", wefml.gpu_backend())'"
echo "  python3 -c 'import wefml; print(\"cuda:\", wefml.is_cuda_available(), \"vulkan:\", wefml.is_vulkan_available())'"
