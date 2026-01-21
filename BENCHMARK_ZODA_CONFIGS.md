# ZODA Configuration Benchmark

This benchmark tests specific Reed-Solomon encoding configurations with detailed performance metrics.

## Quick Start

```bash
cargo test --release benchmark_zoda_configurations -- --ignored --nocapture
```

## What It Tests

The benchmark tests 16 specific configurations across different data sizes:

### 128KB Data
- k=1024, n=1024
- k=1024, n=3072
- k=4096, n=4096
- k=4096, n=12288

### 1MB Data
- k=1024, n=1024
- k=1024, n=3072
- k=4096, n=4096
- k=4096, n=12288

### 4MB Data
- k=1024, n=1024
- k=1024, n=3072
- k=4096, n=4096
- k=4096, n=12288

### 8MB Data
- k=1024, n=1024
- k=1024, n=3072
- k=4096, n=4096
- k=4096, n=12288

## Parameters Explained

- **Data Size**: Total amount of data to encode (128KB, 1MB, 4MB, 8MB)
- **k**: Number of data chunks (1024 or 4096)
- **n**: Total chunks including parity (1024, 3072, 4096, 12288)
- **Redundancy**: (n-k)/k ratio
  - n=1024, k=1024: No redundancy (1:1)
  - n=3072, k=1024: 2x redundancy (3:1)
  - n=4096, k=4096: No redundancy (1:1)
  - n=12288, k=4096: 2x redundancy (3:1)

## Metrics Reported

### Throughput (MB/s)
- **CPU MB/s**: Data processed per second on CPU
- **GPU MB/s**: Data processed per second on GPU
- Higher is better

### Latency (ns/operation)
- **CPU ns/op**: Nanoseconds per encoding operation on CPU
- **GPU ns/op**: Nanoseconds per encoding operation on GPU
- Lower is better

### Speedup
- GPU speedup over CPU (ratio of CPU time / GPU time)
- Higher means GPU is more efficient

## Expected Output

```
╔════════════════════════════════════════════════════════════╗
║     ZODA Benchmark - Specific Configurations              ║
╚════════════════════════════════════════════════════════════╝

✅ CUDA available - GPU detected!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Configuration: 128KB, k=1024, n=1024
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    CPU... 1234µs
    GPU... 156µs

Configuration: 128KB, k=1024, n=3072
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    CPU... 2456µs
    GPU... 298µs

... (continues for all configurations)

╔════════════════════════════════════════════════════════════╗
║                    Results Summary                         ║
╚════════════════════════════════════════════════════════════╝

Size     k      n      │ CPU MB/s     GPU MB/s     │ CPU ns/op    GPU ns/op    │ Speedup
────────────────────────────────────────────────────────────────────────────────────────────────
128KB    1024   1024   │ 103.45       815.23       │ 1205.3       152.8        │ 7.91x
128KB    1024   3072   │ 52.03        412.56       │ 2397.1       302.5        │ 8.24x
128KB    4096   4096   │ 98.12        756.89       │ 1283.5       165.7        │ 7.75x
...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Summary Statistics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Average GPU Speedup:     7.85x
Best GPU Speedup:        12.34x
Worst GPU Speedup:       4.56x

Average GPU Throughput:  623.45 MB/s
Peak GPU Throughput:     1024.78 MB/s

╔════════════════════════════════════════════════════════════╗
║  Benchmark Complete! ✅                                    ║
╚════════════════════════════════════════════════════════════╝
```

## Understanding Results

### Good Performance Indicators

**High Throughput (MB/s)**
- 100+ MB/s: Good for CPU
- 500+ MB/s: Good for GPU
- 1000+ MB/s: Excellent for GPU

**Low Latency (ns/op)**
- <10,000 ns/op: Good for CPU
- <1,000 ns/op: Good for GPU
- <500 ns/op: Excellent for GPU

**High Speedup**
- 5-10x: Good GPU acceleration
- 10-15x: Excellent GPU acceleration
- 15x+: Outstanding GPU acceleration

### Performance Patterns

**Small k, Large n**
- More redundancy, more computation
- Better GPU advantage due to more parallel work

**Large k, Small n**
- Less redundancy, less computation per chunk
- Still benefits from GPU but smaller speedup

**Larger Data Sizes**
- Better GPU utilization
- Higher throughput
- Amortizes kernel launch overhead

## Benchmark Duration

Expected runtime: **5-10 minutes** for all 16 configurations

Each configuration includes:
- Warm-up run
- CPU benchmark
- GPU benchmark

## Interpreting for Your Use Case

### If you prioritize throughput (batch processing)
Look at **MB/s** metrics. Higher is better. GPU should excel here.

### If you prioritize latency (real-time)
Look at **ns/op** metrics. Lower is better. For small operations, CPU might compete.

### If you prioritize efficiency (cost per operation)
Look at **Speedup**. Higher speedup means GPU is more cost-effective.

## Customizing Configurations

To test different configurations, edit `src/benchmark_zoda.rs`:

```rust
let configs = vec![
    BenchmarkConfig { data_size_kb: 256, k: 2048, n: 4096 },
    // Add your custom configs here
];
```

Then rebuild and rerun:
```bash
cargo build --release
cargo test --release benchmark_zoda_configurations -- --ignored --nocapture
```

## Comparing with General Benchmark

The general benchmark (`benchmark_gpu_vs_cpu`) tests raw NTT performance.

This benchmark (`benchmark_zoda_configurations`) tests realistic ZODA encoding scenarios.

Run both to get a complete picture:
```bash
# General NTT performance
cargo test --release benchmark_gpu_vs_cpu -- --ignored --nocapture

# ZODA-specific configurations
cargo test --release benchmark_zoda_configurations -- --ignored --nocapture
```

## Troubleshooting

### "CUDA not available"
See `TROUBLESHOOT_LINKING.md` for linking fixes.

### Benchmark runs slowly
- Ensure GPU isn't thermal throttling: `watch nvidia-smi`
- Close other GPU-using applications
- Check system isn't under heavy load

### Out of memory errors
Try smaller configurations first:
```rust
let configs = vec![
    BenchmarkConfig { data_size_kb: 128, k: 1024, n: 1024 },
];
```

### Results seem wrong
Verify correctness with general benchmark first:
```bash
cargo test --release benchmark_gpu_vs_cpu -- --ignored --nocapture
```

## Next Steps

After running the benchmark:
1. Identify which configurations work best for your use case
2. Note the GPU speedup for your specific parameters
3. Use these metrics to estimate production performance
4. Share results if you'd like optimization suggestions!

---

**Run the benchmark:**
```bash
cargo test --release benchmark_zoda_configurations -- --ignored --nocapture
```
