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

A validated benchmark is available that verifies the correctness of the GPU encoding using the full ZODA protocol: [benchmark_zoda_validated.rs](src/benchmark_zoda_validated.rs)

This test performs **two-phase verification**:

### Phase 1: Column Encoding Verification
- Encodes data using the GPU-accelerated vertical Reed-Solomon implementation
- Verifies each column forms a valid Reed-Solomon codeword
- Compares GPU output against CPU reference implementation

### Phase 2: RLC Soundness Check (ZODA/RSEMA1D)
- Derives random linear combination coefficients from commitment
- Computes RLC for each row: `∑(row[col] × coeff[col])`
- Extends original k RLC values to k+n via Reed-Solomon
- Verifies extended rows satisfy the RLC consistency property

Run with:
```bash
cargo test --features cuda --release benchmark_zoda_validated -- --ignored --nocapture
```

This provides **full ZODA soundness** for data availability sampling:
- **Column-wise**: Each column is a valid Reed-Solomon codeword
- **Row-wise**: Each row is a consistent linear combination across columns
- **Commitment binding**: Random coefficients prevent forgery attacks