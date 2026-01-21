# ZODA with BabyBear Field and CUDA Acceleration

This implementation adds GPU-accelerated NTT over the BabyBear field to the ZODA protocol.

## What's New

### BabyBear Field
- **Prime**: `p = 2^31 - 2^27 + 1 = 2013265921`
- **Arithmetic**: Efficient u64-based operations (fits in 32 bits, but uses 64 for intermediate calculations)
- **NTT Support**: Supports NTT up to size 2^27
- **GPU-Friendly**: Optimized for NVIDIA GPU computation

### CUDA Implementation
- **NTT Kernels**: Optimized Cooley-Tukey butterfly operations
- **Bit Reversal**: Parallel bit-reversal permutation
- **Memory Management**: RAII wrappers for safe device memory handling
- **Compute Capabilities**: Supports RTX 30xx, 40xx, and 50xx series (sm_86, sm_89)

## Architecture

```
src/
├── babybear.rs           # BabyBear field arithmetic
├── ntt_babybear.rs       # CPU NTT implementation
├── cuda_ntt.rs           # CUDA FFI bindings and wrappers
├── zoda_babybear.rs      # ZODA protocol using BabyBear
├── ff.rs                 # Old BigInt field (for comparison)
├── ntt.rs                # Old BigInt NTT (for comparison)
└── lib.rs                # Main library and tests

cuda/
└── ntt_kernel.cu         # CUDA NTT kernels

build.rs                   # Build script for CUDA compilation
```

## Building

### Prerequisites

**On your RTX 5090 system:**
- NVIDIA CUDA Toolkit (12.x or later)
- nvcc compiler
- Rust toolchain

**On M3 (for CPU testing):**
- Rust toolchain only
- CUDA will be disabled automatically

### Build Commands

```bash
# On M3 (CPU only, no CUDA)
cargo build --release

# On RTX 5090 (with CUDA)
cargo build --release

# The build script automatically detects nvcc availability
# and enables CUDA support if found
```

### Environment Variables

If CUDA is installed in a non-standard location, set:
```bash
export CUDA_PATH=/path/to/cuda
```

## Running Tests

### CPU Tests (works on M3)
```bash
# Test BabyBear field arithmetic
cargo test test_basic_arithmetic

# Test CPU NTT
cargo test test_ntt_intt_roundtrip

# Test ZODA with BabyBear CPU
cargo test test_zoda_babybear_cpu
```

### GPU Tests (requires CUDA device)
```bash
# Test CUDA availability
cargo test test_cuda_available

# Test CUDA NTT vs CPU
cargo test test_cuda_ntt_vs_cpu

# Test ZODA with GPU acceleration
cargo test test_zoda_babybear_gpu

# Run comprehensive performance comparison
cargo test test_compare_implementations -- --nocapture
```

## Performance Benchmarks

The `test_compare_implementations` test compares three implementations:

1. **BigInt CPU** (original): Uses BigInt arithmetic over BN254 field
2. **BabyBear CPU**: Uses u64 arithmetic over BabyBear field
3. **BabyBear GPU**: Uses CUDA-accelerated NTT

Expected speedups (will vary based on hardware):
- BabyBear CPU vs BigInt: **10-50x faster**
- BabyBear GPU vs BigInt: **100-500x faster** (on RTX 5090)
- BabyBear GPU vs CPU: **10-20x faster**

## Implementation Details

### BabyBear Field Optimizations

1. **Modular Reduction**: Fast reduction using properties of the prime
2. **Montgomery Form**: Could be added for further GPU optimization
3. **Root of Unity**: Precomputed primitive 2^27-th root (value: 440564289 = 31^15 mod p)

### CUDA Kernel Optimizations

1. **Bit Reversal**: Parallel permutation kernel
2. **Butterfly Operations**: Staged FFT with optimized memory access
3. **Shared Memory**: Optional shared memory version for larger transforms
4. **Twiddle Factors**: Computed on-the-fly to reduce memory bandwidth

### Compute Capability Support

The kernels are compiled for:
- **sm_75**: RTX 2060/2070/2080
- **sm_86**: RTX 3060/3070/3080/3090
- **sm_89**: RTX 4090/5090

## API Usage

### Using BabyBear Field
```rust
use zoda_rs::babybear::BabyBear;

let a = BabyBear::new(100);
let b = BabyBear::new(200);
let c = a + b;  // Fast u64 arithmetic
```

### Using CPU NTT
```rust
use zoda_rs::babybear::BabyBear;
use zoda_rs::ntt_babybear::{ntt, intt};

let n = 256;
let omega = BabyBear::get_root_of_unity(n.trailing_zeros());
let mut values: Vec<BabyBear> = (0..n).map(|i| BabyBear::new(i)).collect();

ntt(&mut values, omega);  // Forward NTT
intt(&mut values, omega); // Inverse NTT
```

### Using CUDA NTT
```rust
#[cfg(feature = "cuda")]
use zoda_rs::cuda_ntt::{cuda_available, ntt_cuda, intt_cuda};

#[cfg(feature = "cuda")]
if cuda_available() {
    let mut values: Vec<BabyBear> = (0..256).map(|i| BabyBear::new(i)).collect();
    ntt_cuda(&mut values).unwrap();  // GPU-accelerated NTT
    intt_cuda(&mut values).unwrap(); // GPU-accelerated INTT
}
```

### Running ZODA Protocol
```rust
use zoda_rs::zoda_babybear::run_zoda_test_babybear;

// CPU version
let duration = run_zoda_test_babybear(32, false);
println!("CPU: {:?}", duration);

// GPU version (if available)
let duration = run_zoda_test_babybear(32, true);
println!("GPU: {:?}", duration);
```

## Testing on Different Platforms

### On M3 (Development)
The code will compile and run CPU tests. All BabyBear arithmetic and CPU NTT tests should pass. The build script will detect the absence of CUDA and disable GPU features automatically.

### On RTX 5090 (Production)
When you move to the RTX 5090:

1. Ensure CUDA toolkit is installed
2. Run `cargo clean` to rebuild with CUDA support
3. Run `cargo build --release`
4. Run `cargo test test_compare_implementations -- --nocapture` to see performance comparison

## Troubleshooting

### CUDA not detected
```bash
# Check if nvcc is in PATH
nvcc --version

# If not, add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Compute capability mismatch
If you have an older GPU (e.g., GTX 970), you may need to edit `build.rs` to add `compute_52`:

```rust
let compute_capabilities = vec![
    "compute_52",  // GTX 970, 980
    "compute_75",  // RTX 2060, 2070, 2080
    "compute_86",  // RTX 3060, 3070, 3080, 3090
    "compute_89",  // RTX 4090, RTX 5090
];
```

### Link errors
Ensure CUDA libraries are in the library path:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Future Optimizations

1. **Montgomery Multiplication**: Further speedup for GPU operations
2. **Batch Processing**: Process multiple NTTs simultaneously
3. **Unified Memory**: Reduce copy overhead
4. **cuFFT Integration**: Consider using NVIDIA's optimized FFT library
5. **Larger NTTs**: Support sizes up to 2^27 for production use

## License

Same as the original ZODA implementation.
