#!/bin/bash

echo "=== CUDA Diagnostic Script ==="
echo ""

# Check if nvcc is available
echo "1. Checking for nvcc..."
if command -v nvcc &> /dev/null; then
    echo "   ✓ nvcc found: $(which nvcc)"
    nvcc --version | head -1
else
    echo "   ✗ nvcc not found in PATH"
    echo "   Add CUDA to PATH: export PATH=/usr/local/cuda/bin:\$PATH"
fi
echo ""

# Check for nvidia-smi
echo "2. Checking for GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "   ✓ nvidia-smi found"
    nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader | head -1
else
    echo "   ✗ nvidia-smi not found"
    echo "   Install NVIDIA drivers"
fi
echo ""

# Check for CUDA libraries
echo "3. Checking for CUDA libraries..."
CUDA_LIB_PATHS=(
    "/usr/local/cuda/lib64"
    "/usr/local/cuda/lib"
    "/usr/lib/x86_64-linux-gnu"
    "/usr/lib64"
    "/opt/cuda/lib64"
    "/usr/lib/wsl/lib"
)

FOUND_CUDART=false
for path in "${CUDA_LIB_PATHS[@]}"; do
    if [ -d "$path" ]; then
        if ls "$path"/libcudart.so* 2>/dev/null | grep -q .; then
            echo "   ✓ Found libcudart in: $path"
            ls -lh "$path"/libcudart.so* | head -1
            FOUND_CUDART=true
            export LD_LIBRARY_PATH="$path:$LD_LIBRARY_PATH"
        fi
    fi
done

if [ "$FOUND_CUDART" = false ]; then
    echo "   ✗ libcudart.so not found in any common location"
    echo "   Install CUDA toolkit or set CUDA_PATH"
fi
echo ""

# Check CUDA_PATH
echo "4. Checking environment variables..."
if [ -n "$CUDA_PATH" ]; then
    echo "   ✓ CUDA_PATH is set: $CUDA_PATH"
else
    echo "   ⚠ CUDA_PATH not set (optional)"
fi

if [ -n "$LD_LIBRARY_PATH" ]; then
    echo "   ✓ LD_LIBRARY_PATH is set"
else
    echo "   ⚠ LD_LIBRARY_PATH not set"
fi
echo ""

# Recommended exports
echo "=== Recommended Setup ==="
echo ""
echo "Add these to your ~/.bashrc or ~/.zshrc:"
echo ""
echo "export PATH=/usr/local/cuda/bin:\$PATH"
if [ "$FOUND_CUDART" = true ]; then
    for path in "${CUDA_LIB_PATHS[@]}"; do
        if [ -d "$path" ] && ls "$path"/libcudart.so* 2>/dev/null | grep -q .; then
            echo "export LD_LIBRARY_PATH=$path:\$LD_LIBRARY_PATH"
            break
        fi
    done
else
    echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
fi
echo ""

# Try building
echo "=== Next Steps ==="
echo ""
echo "1. Run: source ~/.bashrc  # or restart terminal"
echo "2. Run: cargo clean"
echo "3. Run: cargo build --release"
echo ""
echo "If you still get linking errors, try:"
echo "  cargo build --release 2>&1 | tee build.log"
echo "  # Then share build.log"
