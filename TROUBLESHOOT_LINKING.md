# Fixing CUDA Linking Errors

If you're getting **"extern functions can't be found"** or **"native libraries may need to be installed"** errors, follow these steps:

## Quick Fix

Run the diagnostic script first:

```bash
./check_cuda.sh
```

This will tell you what's missing and what to export.

## Manual Steps

### Step 1: Verify CUDA is Installed

```bash
# Check nvcc
nvcc --version

# Check GPU
nvidia-smi

# Find CUDA libraries
find /usr -name "libcudart.so*" 2>/dev/null
# OR
find /opt -name "libcudart.so*" 2>/dev/null
```

### Step 2: Set Environment Variables

Based on where `libcudart.so` was found, export the path:

```bash
# If found in /usr/local/cuda/lib64:
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH

# If found in /usr/lib/x86_64-linux-gnu:
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# For WSL2:
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

### Step 3: Make it Permanent

Add to `~/.bashrc` or `~/.zshrc`:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Step 4: Clean Build

```bash
cargo clean
cargo build --release
```

## Common Issues

### Issue: "cannot find -lcudart"

**Cause:** CUDA runtime library not found

**Fix:**
```bash
# Find libcudart
sudo find / -name "libcudart.so*" 2>/dev/null

# Export the directory (not the file)
export LD_LIBRARY_PATH=/path/to/directory:$LD_LIBRARY_PATH
```

### Issue: "undefined reference to cuda functions"

**Cause:** Static library wasn't created or linked properly

**Fix:**
```bash
# Check if CUDA code compiled
ls -lh target/release/build/zoda-rs-*/out/

# Should see: libntt_cuda.a and ntt_kernel.o

# If missing, check build output:
cargo clean
cargo build --release 2>&1 | grep -i cuda
```

### Issue: CUDA compiles but still can't link

**Cause:** Wrong library search path

**Fix:**
```bash
# Find where libcudart actually is
ldconfig -p | grep cudart

# Use that path
export LD_LIBRARY_PATH=/actual/path:$LD_LIBRARY_PATH
cargo clean
cargo build --release
```

### Issue: "version GLIBCXX_X.X.X not found"

**Cause:** C++ standard library version mismatch

**Fix:**
```bash
# Check GCC version
gcc --version

# CUDA might need a specific GCC version
# Install compatible GCC if needed:
sudo apt-get install gcc-11 g++-11

# Tell CUDA to use it:
export CUDAHOSTCXX=/usr/bin/g++-11
cargo clean
cargo build --release
```

## Alternative: Link CUDA Statically

If dynamic linking keeps failing, edit `build.rs` line 106:

Change:
```rust
println!("cargo:rustc-link-lib=dylib=cudart");
```

To:
```rust
println!("cargo:rustc-link-lib=static=cudart_static");
println!("cargo:rustc-link-lib=dylib=dl");
println!("cargo:rustc-link-lib=dylib=rt");
println!("cargo:rustc-link-lib=dylib=pthread");
```

Then rebuild:
```bash
cargo clean
cargo build --release
```

## Still Not Working?

### Capture Full Error Output

```bash
cargo clean
cargo build --release 2>&1 | tee build_error.log
```

Share `build_error.log` and the output of:
```bash
./check_cuda.sh
```

### Test Without CUDA

To verify the rest of the code works:

```bash
# Temporarily hide nvcc
export PATH=$(echo $PATH | tr ':' '\n' | grep -v cuda | tr '\n' ':')

cargo clean
cargo build --release
cargo test --release

# This should build CPU-only version
```

## WSL2 Specific

If on WSL2, you may need:

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# And ensure Windows NVIDIA drivers are installed
# (CUDA toolkit should be installed inside WSL, not Windows)
```

## Docker Users

If building in Docker:

```Dockerfile
FROM nvidia/cuda:12.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    curl build-essential

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Build will automatically find CUDA
```
