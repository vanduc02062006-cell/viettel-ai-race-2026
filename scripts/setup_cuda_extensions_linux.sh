#!/bin/bash
set -e

echo "Installing build dependencies..."
python -m pip install --upgrade pip "setuptools<82" wheel ninja

# T4 on Colab is compute capability 7.5. This keeps builds smaller there,
# while still allowing other machines to override it before running the script.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5}"
export MAX_JOBS="${MAX_JOBS:-2}"

cd gaussian-splatting

echo "Installing diff-gaussian-rasterization..."
pip install --no-build-isolation --no-cache-dir --force-reinstall submodules/diff-gaussian-rasterization

echo "Installing simple-knn..."
pip install --no-build-isolation --no-cache-dir --force-reinstall submodules/simple-knn

cd ..

echo "Running environment check..."
python tools/check_env.py
