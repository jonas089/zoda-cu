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