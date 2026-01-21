## CUDA + ZODA = SPEED at SCALE
This is a CUDA accelerated implementation of ZODA that leverages prime field arithmetic over u64 BABYBEAR fields.

## Benchmark Results

Encoding performance [see benchmark](src/benchmark_zoda_optimal.rs) on an RTX 5090 32 GB:

| Data Size | CUDA (GPU) | CPU | Speedup |
|-----------|------------|-----|---------|
| 2 GB      | 1,890 ms   | 65,423 ms | 34.6x |
| 1 GB      | 961 ms     | 28,450 ms | 29.6x |
| 512 MB    | 478 ms     | 1,395 ms  | 2.9x  |
| 256 MB    | 239 ms     | 7,403 ms  | 30.9x |

## Validated Benchmark

A validated benchmark is available that verifies the correctness of the GPU encoding using ZODA protocol verification: [benchmark_zoda_validated.rs](src/benchmark_zoda_validated.rs)

This test:
1. Encodes data using the GPU-accelerated Reed-Solomon implementation
2. Validates the encoding using ZODA's random linear combination check
3. Verifies that parity rows satisfy the polynomial structure

Run with:
```bash
cargo test --features cuda --release benchmark_zoda_validated -- --ignored --nocapture
```

The validation ensures that the encoded data forms a mathematically correct Reed-Solomon codeword by:
- Extracting each column's first k values (original data)
- Using INTT to interpolate and get polynomial coefficients
- Zero-padding and using NTT to evaluate at k+n points
- Verifying that the result matches the GPU-encoded column values