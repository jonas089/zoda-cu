# Quick Start Guide for RTX 5090

## On Your RTX 5090 Machine

### 1. Copy the Code
Transfer the entire `/Users/chef/Desktop/joda` directory to your RTX 5090 machine.

### 2. Verify CUDA Installation
```bash
# Check CUDA is installed
nvcc --version

# Should show CUDA 12.x or later
# If not installed, download from: https://developer.nvidia.com/cuda-downloads
```

### 3. Set Environment Variables (if needed)
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda
```

### 4. Clean Build with CUDA
```bash
cd joda
cargo clean
cargo build --release

# You should see:
# "Compiling CUDA kernels..."
# NOT "nvcc not found"
```

### 5. Run Benchmarks
```bash
# This will show you CPU vs GPU performance
cargo test test_compare_implementations --release -- --nocapture

# Expected output:
# Testing 4x4 data square:
#   BigInt CPU:     ~8ms
#   BabyBear CPU:   ~140µs
#   BabyBear GPU:   ~10-20µs    <-- THIS IS THE MAGIC!
#   GPU Speedup vs BigInt: 400-800x
#   GPU Speedup vs CPU:    7-14x
```

### 6. Verify GPU Tests Pass
```bash
# Run all tests including GPU
cargo test --release -- --nocapture

# Specifically test CUDA functionality
cargo test test_cuda_ntt_vs_cpu --release -- --nocapture
cargo test test_cuda_intt_roundtrip --release -- --nocapture
```

## Expected Results

### What You Should See

✅ **Build succeeds with CUDA enabled**
✅ **All 16 tests pass**
✅ **GPU faster than CPU for NTT operations**
✅ **ZODA protocol 100-1000x faster than original BigInt implementation**

### Performance Targets (estimates for RTX 5090)

| Data Size | BigInt CPU | BabyBear CPU | BabyBear GPU | Total Speedup |
|-----------|------------|--------------|--------------|---------------|
| 4x4       | ~8 ms      | ~140 µs      | ~10-20 µs    | 400-800x      |
| 8x8       | ~16 ms     | ~320 µs      | ~20-40 µs    | 400-800x      |
| 16x16     | ~42 ms     | ~1.1 ms      | ~60-120 µs   | 350-700x      |
| 32x32     | ~138 ms    | ~8 ms        | ~400-800 µs  | 170-350x      |
| 64x64     | ~500 ms    | ~30 ms       | ~1-2 ms      | 250-500x      |
| 128x128   | ~2 s       | ~120 ms      | ~3-6 ms      | 330-660x      |

For larger data sizes, GPU advantage increases significantly!

## Troubleshooting

### Problem: "nvcc not found"
```bash
# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH

# Verify
which nvcc
```

### Problem: "cannot find -lcudart"
```bash
# Add CUDA libraries to library path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Or add to ~/.bashrc for permanence
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

### Problem: Build succeeds but tests fail
```bash
# Check GPU is visible
nvidia-smi

# Ensure CUDA drivers are installed
# Reboot if you just installed CUDA
```

### Problem: GPU slower than CPU (small sizes)
This is expected! GPU kernel launch has overhead (~10µs). For tiny NTTs (<256 elements), CPU might be faster. GPU wins for larger sizes (1024+).

## Using the New Implementation

### In Your Code

```rust
use zoda_rs::babybear::BabyBear;
use zoda_rs::zoda_babybear::run_zoda_test_babybear;

// Run ZODA with GPU acceleration (automatic fallback to CPU if needed)
let duration = run_zoda_test_babybear(64, true);  // true = use GPU if available
println!("ZODA 64x64: {:?}", duration);

// For manual NTT operations
#[cfg(feature = "cuda")]
{
    use zoda_rs::cuda_ntt::{ntt_cuda, intt_cuda};
    let mut values: Vec<BabyBear> = (0..1024).map(|i| BabyBear::new(i)).collect();
    ntt_cuda(&mut values).unwrap();  // GPU NTT
    intt_cuda(&mut values).unwrap(); // GPU INTT
}
```

## What to Report Back

Please share:
1. Build output (confirm CUDA is detected)
2. Test output from `test_compare_implementations`
3. Actual speedup numbers (GPU vs CPU vs BigInt)
4. Any errors or unexpected behavior

## Next Steps After Verification

1. **Scale up** - Try larger data sizes (128x128, 256x256)
2. **Integrate** - Use the new API in your production code
3. **Optimize** - Profile to find any remaining bottlenecks
4. **Batch** - Process multiple NTTs in parallel for even better GPU utilization

---

**Expected outcome:** ZODA protocol running **100-1000x faster** than your original implementation! 🚀
