# Quick Benchmark Cheat Sheet

## On RTX 3060 Mobile or RTX 5090

### One Command to Rule Them All

```bash
cargo test --release benchmark_gpu_vs_cpu -- --ignored --nocapture
```

That's it! This will:
- ✅ Test GPU vs CPU performance
- ✅ Run multiple NTT sizes (256 to 16384)
- ✅ Test full ZODA protocol (4x4 to 32x32)
- ✅ Show speedup numbers
- ✅ Verify GPU results match CPU (correctness check)
- ✅ Complete in under 10 minutes

### What You'll See

```
✅ CUDA available - GPU detected!
Testing NTT size 1024... CPU: 215µs | GPU: 35µs | Speedup: 6.14x 🚀
...
Raw NTT Average Speedup: 7.78x
ZODA Protocol Average Speedup: 5.90x
```

### Expected Speedups

| Hardware | NTT Speedup | ZODA Speedup |
|----------|-------------|--------------|
| RTX 3060 Mobile | 5-12x | 4-8x |
| RTX 5090 | 15-30x | 10-20x |

### Troubleshooting

**If you see "CUDA not available":**
```bash
nvidia-smi  # Check GPU is detected
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
cargo clean && cargo build --release
```

**If you see "CUDA support not compiled in":**
```bash
nvcc --version  # Check CUDA is installed
export PATH=/usr/local/cuda/bin:$PATH
cargo clean && cargo build --release
```

### Monitor GPU During Test

In another terminal:
```bash
watch -n 0.5 nvidia-smi
```

You should see GPU usage spike and memory increase.

---

**For detailed instructions, see:** `BENCHMARK_INSTRUCTIONS.md`
