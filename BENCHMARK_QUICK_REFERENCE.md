# Benchmark Quick Reference

## Two Benchmarks Available

### 1. General GPU vs CPU Benchmark
**What**: Tests raw NTT performance at various sizes
**When**: Quick check if GPU is working, general performance overview
**Time**: ~3-5 minutes
**Command**:
```bash
cargo test --release benchmark_gpu_vs_cpu -- --ignored --nocapture
```

### 2. ZODA Configuration Benchmark
**What**: Tests specific Reed-Solomon encoding configurations
**When**: Production planning, specific use case optimization
**Time**: ~5-10 minutes
**Command**:
```bash
cargo test --release benchmark_zoda_configurations -- --ignored --nocapture
```

## Quick Comparison

| Feature | General Benchmark | ZODA Config Benchmark |
|---------|------------------|----------------------|
| **Tests** | NTT sizes (256-16384) | Data sizes (128KB-8MB) |
| **Metrics** | Time, Speedup | MB/s, ns/op, Speedup |
| **Use Case** | Quick validation | Production planning |
| **Duration** | 3-5 minutes | 5-10 minutes |
| **Detail Level** | Basic | Detailed |
| **Configurations** | 7 NTT sizes, 4 ZODA sizes | 16 specific configs |

## Which Should You Run?

### Run General Benchmark If:
- ✓ First time setup, want to verify GPU works
- ✓ Quick sanity check
- ✓ Want to know general speedup
- ✓ Short on time

### Run ZODA Config Benchmark If:
- ✓ Planning production deployment
- ✓ Need throughput (MB/s) numbers
- ✓ Need latency (ns/op) numbers
- ✓ Testing specific k/n configurations
- ✓ Want detailed performance data

### Run Both If:
- ✓ Complete performance characterization
- ✓ Comparing different GPUs
- ✓ Preparing performance report
- ✓ Optimizing for specific workload

## Output Comparison

### General Benchmark Output
```
Testing NTT size 1024... CPU: 215µs | GPU: 35µs | Speedup: 6.14x 🚀
Testing ZODA 16x16 data square... CPU: 1156µs | GPU: 185µs | Speedup: 6.25x 🚀

Raw NTT Average Speedup: 7.78x
ZODA Protocol Average Speedup: 5.90x
```

### ZODA Config Benchmark Output
```
Size     k      n      │ CPU MB/s     GPU MB/s     │ CPU ns/op    GPU ns/op    │ Speedup
128KB    1024   1024   │ 103.45       815.23       │ 1205.3       152.8        │ 7.91x
1MB      4096   12288  │ 156.78       1234.56      │ 6387.2       810.5        │ 7.88x

Average GPU Speedup:     7.85x
Average GPU Throughput:  623.45 MB/s
Peak GPU Throughput:     1024.78 MB/s
```

## Common Commands

```bash
# First time - validate GPU works
cargo test --release benchmark_gpu_vs_cpu -- --ignored --nocapture

# Detailed analysis
cargo test --release benchmark_zoda_configurations -- --ignored --nocapture

# Run both back-to-back
cargo test --release benchmark_gpu_vs_cpu -- --ignored --nocapture && \
cargo test --release benchmark_zoda_configurations -- --ignored --nocapture

# Save output to file
cargo test --release benchmark_zoda_configurations -- --ignored --nocapture 2>&1 | tee benchmark_results.txt
```

## Understanding Metrics

### From General Benchmark
- **Time (µs)**: Microseconds to complete operation
- **Speedup**: How many times faster GPU is than CPU
- **Simple interpretation**: Higher speedup = better GPU utilization

### From ZODA Config Benchmark
- **MB/s**: Megabytes processed per second (throughput)
  - Higher = faster bulk processing
- **ns/op**: Nanoseconds per operation (latency)
  - Lower = faster individual operations
- **Speedup**: GPU time / CPU time ratio
  - Higher = more efficient GPU usage

## Recommended Workflow

1. **Initial Setup**
   ```bash
   ./check_cuda.sh  # Verify CUDA is available
   cargo build --release
   ```

2. **Quick Validation**
   ```bash
   cargo test --release benchmark_gpu_vs_cpu -- --ignored --nocapture
   ```
   *Should take 3-5 minutes. Verify GPU speedup > 1.0x*

3. **Detailed Analysis** (if step 2 passes)
   ```bash
   cargo test --release benchmark_zoda_configurations -- --ignored --nocapture
   ```
   *Takes 5-10 minutes. Get production metrics*

4. **Save Results**
   ```bash
   cargo test --release benchmark_zoda_configurations -- --ignored --nocapture 2>&1 | tee results_$(date +%Y%m%d).txt
   ```

## Troubleshooting

If either benchmark fails:
1. Check `TROUBLESHOOT_LINKING.md` for CUDA linking issues
2. Run `./check_cuda.sh` to diagnose
3. Verify GPU visibility: `nvidia-smi`
4. Check exports:
   ```bash
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   export PATH=/usr/local/cuda/bin:$PATH
   ```

## Getting Help

Include this information when asking for help:
```bash
# System info
nvidia-smi
nvcc --version
rustc --version

# Build output
cargo clean
cargo build --release 2>&1 | tee build.log

# Benchmark output
cargo test --release benchmark_gpu_vs_cpu -- --ignored --nocapture 2>&1 | tee benchmark.log
```

---

**Most common first command:**
```bash
cargo test --release benchmark_gpu_vs_cpu -- --ignored --nocapture
```

**For production planning:**
```bash
cargo test --release benchmark_zoda_configurations -- --ignored --nocapture
```
