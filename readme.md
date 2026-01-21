# ZODA with GPU Acceleration

GPU-accelerated implementation of the ZODA (Zero Overhead Data Availability) protocol using CUDA and the BabyBear finite field.

## 🚀 Quick Start - Run the Benchmark

On your GPU machine (RTX 3060 mobile, RTX 5090, or any CUDA-capable GPU):

```bash
cd joda
cargo test --release benchmark_gpu_vs_cpu -- --ignored --nocapture
```

This runs an automated benchmark comparing GPU vs CPU performance. **Completes in under 10 minutes.**

## 📊 Performance

### Already Achieved (M3 CPU only - no GPU)

BabyBear CPU vs BigInt CPU:
- **4x4**: 55.8x faster
- **8x8**: 50.2x faster
- **16x16**: 38.7x faster
- **32x32**: 17.3x faster

### Expected on RTX 3060 Mobile

GPU vs CPU BabyBear:
- **Raw NTT**: 5-12x faster
- **Full ZODA**: 4-8x faster
- **vs Original BigInt**: 200-400x total speedup

### Expected on RTX 5090

GPU vs CPU BabyBear:
- **Raw NTT**: 15-30x faster
- **Full ZODA**: 10-20x faster
- **vs Original BigInt**: 500-1000x total speedup

## 📚 Documentation

| File | Purpose |
|------|---------|
| **QUICK_BENCHMARK.md** | One-liner to run benchmark |
| **BENCHMARK_INSTRUCTIONS.md** | Detailed benchmark guide |
| **IMPLEMENTATION_SUMMARY.md** | Technical overview |
| **README_BABYBEAR.md** | Complete API documentation |
| **QUICK_START_RTX5090.md** | Setup guide for RTX 5090 |

## 🏗️ What's Implemented

1. **BabyBear Field** - Efficient u64 arithmetic (prime: 2^31 - 2^27 + 1)
2. **CPU NTT** - Fast Cooley-Tukey FFT in Rust
3. **CUDA Kernels** - GPU-accelerated NTT
4. **FFI Bindings** - Safe Rust-CUDA integration
5. **ZODA Protocol** - Full implementation with automatic GPU acceleration
6. **Build System** - Automatic CUDA detection and compilation
7. **Benchmark Suite** - Comprehensive performance testing

## 🔧 Building

### With CUDA (GPU acceleration)
```bash
# Ensure CUDA is installed and nvcc is in PATH
cargo build --release
```

### Without CUDA (CPU only)
```bash
# Works anywhere, even without GPU
cargo build --release
```

The build system automatically detects CUDA availability.

## 🧪 Testing

```bash
# Run all tests
cargo test --release

# Run GPU benchmark (the important one!)
cargo test --release benchmark_gpu_vs_cpu -- --ignored --nocapture

# Run specific tests
cargo test test_basic_arithmetic --release
cargo test test_ntt_intt_roundtrip --release
cargo test test_zoda_babybear_cpu --release
```

## 🎯 Supported Hardware

### Primary Target
- ✅ RTX 5090 (compute_89)
- ✅ RTX 4090 (compute_89)

### Also Supported
- ✅ RTX 3060/3070/3080/3090 (compute_86)
- ✅ RTX 3060 Mobile (compute_86)
- ✅ RTX 2060/2070/2080 (compute_75)

### Optional (requires build.rs modification)
- GTX 970/980 (compute_52) - add to build.rs

## 📝 API Usage

### Using GPU-Accelerated ZODA
```rust
use zoda_rs::zoda_babybear::run_zoda_test_babybear;

// Automatically uses GPU if available, falls back to CPU
let duration = run_zoda_test_babybear(32, true);
println!("ZODA 32x32 with GPU: {:?}", duration);
```

## 🐛 Troubleshooting

See `BENCHMARK_INSTRUCTIONS.md` for detailed troubleshooting.

Quick fixes:
```bash
# CUDA not detected
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
cargo clean && cargo build --release

# GPU not visible
nvidia-smi
```

## 📈 Scaling

The GPU advantage increases with problem size:

| NTT Size | GPU Speedup (est.) |
|----------|-------------------|
| 256      | 2-3x              |
| 1024     | 5-8x              |
| 4096     | 10-15x            |
| 16384    | 15-25x            |
| 65536+   | 20-30x            |

---

**Ready to see the speedup?**

```bash
cargo test --release benchmark_gpu_vs_cpu -- --ignored --nocapture
```
