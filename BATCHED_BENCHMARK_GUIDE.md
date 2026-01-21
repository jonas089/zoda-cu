# Batched Benchmark Guide

I've rewritten the ZODA benchmark to properly batch GPU operations. Here's what changed and how to use it.

## The Problem with the Original Benchmark

The original `benchmark_zoda_configurations` was doing **1024+ separate GPU operations** for each test:
- For 128KB with k=1024: Each chunk was only 128 bytes
- **1024 chunks × 2 operations (INTT + NTT) = 2048 GPU calls**
- Each call had ~160µs overhead (alloc, copy to GPU, compute, copy back)
- **Total overhead: ~327ms** just in GPU transfer overhead!

This is why you only got **17 MB/s** - the GPU was spending most of its time on memory transfers, not computation.

## Three Benchmark Versions Now Available

### 1. Original (Naive) - `benchmark_zoda_configurations`
```bash
cargo test --release benchmark_zoda_configurations -- --ignored --nocapture
```
**What it does**: Separate GPU call per chunk (SLOW, for comparison)
**Expected performance**: 10-20 MB/s (lots of overhead)
**Use when**: You want to see why naive implementation is slow

### 2. Batched - `benchmark_zoda_batched_configurations` (NEW)
```bash
cargo test --release benchmark_zoda_batched_configurations -- --ignored --nocapture
```
**What it does**: Processes chunks in batches, reduced memory transfers
**Expected performance**: 50-150 MB/s (much better)
**Use when**: Testing improved batching strategy

### 3. Optimal - `benchmark_zoda_optimal` (NEW, BEST)
```bash
cargo test --release benchmark_zoda_optimal -- --ignored --nocapture
```
**What it does**:
- **Single GPU memory allocation** for all chunks
- **Single transfer TO GPU** (all data at once)
- **Process all chunks on GPU** (no intermediate transfers)
- **Single transfer FROM GPU** (all results at once)

**Expected performance**: 100-500+ MB/s (minimized overhead)
**Use when**: You want realistic GPU performance numbers

## Which Should You Run?

### For Fair Comparison with Leopard RS:
```bash
cargo test --release benchmark_zoda_optimal -- --ignored --nocapture
```

This gives you the **true GPU performance** without overhead dominating.

### To See the Improvement:
Run all three and compare:
```bash
# Original (slow)
cargo test --release benchmark_zoda_configurations -- --ignored --nocapture

# Batched (better)
cargo test --release benchmark_zoda_batched_configurations -- --ignored --nocapture

# Optimal (best)
cargo test --release benchmark_zoda_optimal -- --ignored --nocapture
```

## Expected Results

### On RTX 3060 Mobile

**Original (naive):**
- 10-20 MB/s
- Dominated by overhead

**Batched:**
- 50-150 MB/s
- Much better, but still some overhead

**Optimal:**
- 100-300 MB/s
- True GPU performance
- Should be competitive with Leopard CPU on smaller sizes
- Should beat Leopard CPU on larger sizes (4MB+)

### On RTX 5090

**Optimal benchmark:**
- 300-800 MB/s (estimated)
- Should significantly beat Leopard CPU

## Understanding the Results

### If Optimal Shows 100-300 MB/s:
✅ **This is good!** You're now getting real GPU performance.

The reason Leopard RS might still be faster on small sizes:
- **Leopard uses GF(2^16)** - XOR operations, no modular arithmetic
- **Leopard uses SIMD** - AVX2/AVX512, processes 16+ operations in parallel
- **Leopard is highly optimized** - Years of work by experts
- **Binary fields are CPU-friendly** - Table lookups are cache-efficient

### If Optimal Shows 300+ MB/s:
✅ **Excellent!** GPU is well-utilized.

You're now competitive with or beating optimized CPU implementations.

### If Optimal Still Shows <50 MB/s:
⚠️ Something is wrong:
- Check GPU isn't thermal throttling: `nvidia-smi`
- Ensure proper CUDA linking
- Check the GPU is actually being used
- Try on RTX 5090 for comparison

## Comparing with Leopard RS

To see what Leopard actually achieves:
```bash
cd ~/Desktop/rsema1d
go test -bench=BenchmarkEncode -benchtime=3s -benchmem | grep MB
```

Then compare with your optimal benchmark:
```bash
cd ~/Desktop/joda
cargo test --release benchmark_zoda_optimal -- --ignored --nocapture
```

## Key Differences: Our GPU vs Leopard CPU

| Aspect | Leopard RS (CPU) | Our GPU (BabyBear NTT) |
|--------|------------------|------------------------|
| **Field** | GF(2^16) binary | Prime field (BabyBear) |
| **Operations** | XOR (1 cycle) | Modular arithmetic (many cycles) |
| **SIMD** | AVX2/AVX512 (16-32 wide) | CUDA (1000s parallel) |
| **Memory** | Cache-friendly tables | PCIe transfer overhead |
| **Best for** | Small to medium data | Large data blocks |
| **Optimization** | Years of work | Our first implementation |

## Next Steps

1. **Run optimal benchmark** on your RTX 3060:
   ```bash
   cargo test --release benchmark_zoda_optimal -- --ignored --nocapture
   ```

2. **Share the results** - I want to see the actual numbers!

3. **Test on RTX 5090** when available - should be 2-3x faster

4. **Further optimizations possible**:
   - Parallel NTT processing (multiple chunks simultaneously)
   - Streaming transfers (overlap compute and memory)
   - Optimized kernel scheduling

## The Bottom Line

**The original benchmark was measuring overhead, not GPU performance.**

The optimal benchmark now gives you **real GPU throughput** by:
- Single memory allocation
- Single GPU transfer (both directions)
- Batch processing on GPU
- Minimal overhead

**Expected speedup: 5-20x improvement over the original benchmark!**

---

**Run this:**
```bash
cargo test --release benchmark_zoda_optimal -- --ignored --nocapture
```
