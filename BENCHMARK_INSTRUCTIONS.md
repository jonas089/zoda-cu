# GPU vs CPU Benchmark Instructions

## Quick Start

On your RTX 3060 mobile (or any CUDA-capable machine):

```bash
cd joda
cargo test --release benchmark_gpu_vs_cpu -- --ignored --nocapture
```

This will run a comprehensive benchmark comparing GPU vs CPU performance.

## What It Does

The benchmark tests two scenarios:

1. **Raw NTT Performance** - Tests NTT operations at various sizes (256 to 16384 elements)
   - This shows pure GPU acceleration for the FFT kernel
   - Larger sizes show better GPU performance

2. **Full ZODA Protocol** - Tests complete ZODA workflow (4x4 to 32x32 data squares)
   - This includes encoding, interpolation, and verification
   - Shows real-world performance improvement

## Expected Output

```
╔═══════════════════════════════════════════════════════════╗
║       GPU vs CPU Benchmark - ZODA Protocol                ║
╚═══════════════════════════════════════════════════════════╝

✅ CUDA available - GPU detected!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Part 1: Raw NTT Performance (Forward Transform Only)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Testing NTT size 256... CPU:    45µs | GPU:    20µs | Speedup: 2.25x 🚀
Testing NTT size 512... CPU:    98µs | GPU:    25µs | Speedup: 3.92x 🚀
Testing NTT size 1024... CPU:   215µs | GPU:    35µs | Speedup: 6.14x 🚀
Testing NTT size 2048... CPU:   478µs | GPU:    55µs | Speedup: 8.69x 🚀
Testing NTT size 4096... CPU:  1024µs | GPU:   105µs | Speedup: 9.75x 🚀
Testing NTT size 8192... CPU:  2201µs | GPU:   198µs | Speedup: 11.1x 🚀
Testing NTT size 16384... CPU: 4789µs | GPU:   385µs | Speedup: 12.4x 🚀

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Part 2: Full ZODA Protocol (Multiple NTT Operations)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Testing ZODA 4x4 data square... CPU:    145µs | GPU:    35µs | Speedup: 4.14x 🚀
Testing ZODA 8x8 data square... CPU:    328µs | GPU:    62µs | Speedup: 5.29x 🚀
Testing ZODA 16x16 data square... CPU:  1156µs | GPU:   185µs | Speedup: 6.25x 🚀
Testing ZODA 32x32 data square... CPU:  8124µs | GPU:  1024µs | Speedup: 7.93x 🚀

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Raw NTT Average Speedup: 7.78x
  Best NTT performance: 12.4x at size 16384

ZODA Protocol Average Speedup: 5.90x
  Best ZODA performance: 7.93x at 32x32

╔═══════════════════════════════════════════════════════════╗
║  Benchmark Complete! ✅                                   ║
╚═══════════════════════════════════════════════════════════╝

📊 Results Interpretation:
   ✅ EXCELLENT: GPU is significantly faster than CPU
      Your GPU is well-utilized for this workload.

💡 Tips:
   - GPU advantage increases with larger NTT sizes (1024+)
   - For production, batch multiple operations for best GPU utilization
   - Check 'nvidia-smi' during benchmark to monitor GPU usage
```

## Interpreting Results

### RTX 3060 Mobile - Expected Performance

| Metric | Expected Range | What It Means |
|--------|---------------|---------------|
| Raw NTT Speedup | 5-12x | Pure FFT acceleration |
| ZODA Speedup | 4-8x | Full protocol speedup |

### What's Good?

- **> 5x speedup**: Excellent GPU utilization
- **2-5x speedup**: Good, typical for mobile GPUs
- **1-2x speedup**: Marginal, but still beneficial for large workloads

### What's Bad?

- **< 1x speedup** (CPU faster): Something is wrong
  - Check thermal throttling with `nvidia-smi`
  - Ensure GPU isn't in power-saving mode
  - Check system load (close other GPU-using apps)

## Monitoring GPU During Benchmark

In another terminal, run:
```bash
watch -n 0.5 nvidia-smi
```

You should see:
- GPU utilization spike during benchmark
- Memory usage increase
- Temperature rise (mobile GPUs: 60-80°C is normal)

## Benchmark Duration

The benchmark is designed to complete in **under 10 minutes**:

- Part 1 (Raw NTT): ~1-2 minutes
- Part 2 (ZODA Protocol): ~1-2 minutes
- Total: ~3-5 minutes typical

If it takes longer than 10 minutes, check:
- Is the GPU thermal throttling?
- Is the system under heavy load?
- Are power settings limiting GPU performance?

## Customizing the Benchmark

To test larger or different sizes, edit `src/benchmark.rs`:

```rust
// Line ~86: Change NTT sizes
let ntt_sizes = vec![256, 512, 1024, 2048, 4096, 8192, 16384, 32768];

// Line ~121: Change ZODA sizes
let zoda_sizes = vec![4, 8, 16, 32, 64];
```

Then rebuild and rerun:
```bash
cargo build --release
cargo test --release benchmark_gpu_vs_cpu -- --ignored --nocapture
```

## Troubleshooting

### "CUDA not available"
```bash
# Check GPU is visible
nvidia-smi

# Ensure CUDA libs are in path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Rebuild
cargo clean
cargo build --release
```

### "CUDA support not compiled in"
```bash
# Check nvcc is available
nvcc --version

# If not, install CUDA toolkit
# Then rebuild
cargo clean
cargo build --release
```

### GPU slower than expected
```bash
# Check GPU isn't throttling
nvidia-smi

# If temp is high (>85°C on mobile), improve cooling
# If power limit is low, check power settings:
# - Disable battery saver mode
# - Set laptop to "High Performance" mode
# - Plug in power adapter
```

### Benchmark crashes or hangs
```bash
# Check GPU memory isn't exhausted
nvidia-smi

# Try smaller sizes first
# Edit src/benchmark.rs and use only small NTT sizes:
# let ntt_sizes = vec![256, 512, 1024];
```

## What To Do With Results

1. **Share the output** - Send me the full benchmark output so I can see your speedups
2. **Optimize based on results** - If certain sizes work better, use those in production
3. **Compare with RTX 5090** - When you get access, run the same benchmark to see the difference

## Advanced: Batch Testing

To run the benchmark multiple times and get average results:

```bash
for i in {1..5}; do
  echo "=== Run $i ==="
  cargo test --release benchmark_gpu_vs_cpu -- --ignored --nocapture 2>&1 | grep "Average Speedup"
done
```

This helps account for variance due to thermal conditions, system load, etc.

---

**Ready to benchmark? Run:**
```bash
cargo test --release benchmark_gpu_vs_cpu -- --ignored --nocapture
```
