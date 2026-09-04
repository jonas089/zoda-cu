## Cuda accelerated ZODA experiment
This is a CUDA accelerated implementation of ZODA that leverages prime field arithmetic over the u32 BabyBear field.

Full ZODA encoding test with y-column verification:

`cargo test test_zoda_babybear_gpu --features cuda --release -- --nocapture`

> **⚠️ Security Notice**: This is a proof-of-concept implementation using the BabyBear field (p = 2^31 - 2^27 + 1), which provides ~31 bits of security. This is **NOT cryptographically secure** for production use. For production data availability systems, a larger field is required (e.g., Goldilocks p = 2^64 - 2^32 + 1 for 64-bit security, or a 128-bit field for full cryptographic security). The encoding is mathematically correct and performance characteristics would scale similarly to larger fields.

## Kernel status

`cuda/ntt_kernel.cu` transforms every column of a row-major `[n rows][cols]` square in one call.
It mirrors the radix-2 CPU reference in `src/ntt/cpu.rs`: bit reversal, then one kernel launch per
stage, with one thread per butterfly per column. Field elements are `u32` BabyBear in Montgomery
form on the device and canonical on the Rust side. Columns are processed in chunks that rotate over
three streams so uploads and downloads overlap the kernels.

## Benchmark Results

> The numbers below were measured with an earlier `u64` kernel and have not yet been re-run with the
> current one.

cargo test benchmark_zoda_eigenda_comparison --features cuda --release -- --ignored --nocapture

Direct comparison against [rsema1d](https://github.com/celestiaorg/eigenda-kzg-bench):

On a single RTX 5090 (32 GB):

| Configuration | K | N | GPU 1D ZODA (ns/op) | GPU 1D ZODA (MB/s) |  EMA (ns/op) |  EMA (MB/s) | Speedup | Status |
|---------------|---|---|--------------|-------------|---------------------|--------------------|---------|---------| 
| 128KB | 1024 | 1024 | 446807 | 279.76 | 885254 | 148.06 | 1.98x | 
| 128KB | 1024 | 3072 | 301284 | 414.89 | 1537837 | 85.23 | 5.10x | 
| 128KB | 4096 | 4096 | 241281 | 518.07 | - | - | - | 
| 128KB | 4096 | 12288 | 1225914 | 101.96 | - | - | - | 
| 1MB | 1024 | 1024 | 1160890 | 861.41 | 3532739 | 296.82 | 3.04x | 
| 1MB | 1024 | 3072 | 1738083 | 575.35 | 4732775 | 221.56 | 2.72x | 
| 1MB | 4096 | 4096 | 546308 | 1830.47 | 4724641 | 221.94 | 8.65x | 
| 1MB | 4096 | 12288 | 822874 | 1215.25 | 7131521 | 147.03 | 8.67x | 
| 4MB | 1024 | 1024 | 3452894 | 1158.45 | 12449984 | 336.89 | 3.61x | 
| 4MB | 1024 | 3072 | 7415193 | 539.43 | 15774507 | 265.89 | 2.13x | 
| 4MB | 4096 | 4096 | 1809383 | 2210.70 | 13541566 | 309.74 | 7.48x | 
| 4MB | 4096 | 12288 | 6689245 | 597.97 | 22033068 | 190.36 | 3.29x | 
| 8MB | 1024 | 1024 | 7042468 | 1135.97 | 22388042 | 374.69 | 3.18x | 
| 8MB | 1024 | 3072 | 12491441 | 640.44 | 35500902 | 236.29 | 2.84x | 
| 8MB | 4096 | 4096 | 7299058 | 1096.03 | 26230505 | 319.80 | 3.59x | 
| 8MB | 4096 | 12288 | 13355007 | 599.03 | 40912656 | 205.04 | 3.06x | 
| 16MB | 1024 | 1024 | 13958160 | 1146.28 | - | - | - | 
| 16MB | 1024 | 3072 | 27054283 | 591.40 | - | - | - | 
| 16MB | 4096 | 4096 | 14496499 | 1103.71 | - | - | - | 
| 16MB | 4096 | 12288 | 27454539 | 582.78 | - | - | - | 
| 32MB | 1024 | 1024 | 28673439 | 1116.02 | - | - | - | 
| 32MB | 1024 | 3072 | 53376836 | 599.51 | - | - | - | 
| 32MB | 4096 | 4096 | 29829277 | 1072.77 | - | - | - | 
| 32MB | 4096 | 12288 | 55523676 | 576.33 | - | - | - | 
| 64MB | 1024 | 1024 | 57057025 | 1121.68 | - | - | - | 
| 64MB | 1024 | 3072 | 103952057 | 615.67 | - | - | - | 
| 64MB | 4096 | 4096 | 59123501 | 1082.48 | - | - | - | 
| 64MB | 4096 | 12288 | 111542138 | 573.77 | - | - | - | 
| 128MB | 1024 | 1024 | 113273023 | 1130.01 | - | - | - | 
| 128MB | 1024 | 3072 | 211676550 | 604.70 | - | - | - | 
| 128MB | 4096 | 4096 | 118497436 | 1080.19 | - | - | - | 
| 128MB | 4096 | 12288 | 214147433 | 597.72 | - | - | - | 
| 256MB | 1024 | 1024 | 228780269 | 1118.98 | - | - | - | 
| 256MB | 1024 | 3072 | 420637690 | 608.60 | - | - | - | 
| 256MB | 4096 | 4096 | 230378623 | 1111.21 | - | - | - | 
| 256MB | 4096 | 12288 | 429416899 | 596.16 | - | - | - | 
| 512MB | 1024 | 1024 | 447773663 | 1143.43 | - | - | - | 
| 512MB | 1024 | 3072 | 834373660 | 613.63 | - | - | - | 
| 512MB | 4096 | 4096 | 467116253 | 1096.09 | - | - | - | 
| 512MB | 4096 | 12288 | 853806266 | 599.67 | - | - | - | 
| 1024MB | 1024 | 1024 | 898437256 | 1139.76 | - | - | - | 
| 1024MB | 1024 | 3072 | 1676912693 | 610.65 | - | - | - | 
| 1024MB | 4096 | 4096 | 924001725 | 1108.22 | - | - | - | 
| 1024MB | 4096 | 12288 | 1703623025 | 601.07 | - | - | - | 


## Statistics by Data Size

| Data Size | Avg MB/s | Min MB/s | Max MB/s | Configs |
|-----------|----------|----------|----------|---------|
| 128KB | 328.67 | 101.96 | 518.07 | 4 |
| 1MB | 1120.62 | 575.35 | 1830.47 | 4 |
| 4MB | 1126.64 | 539.43 | 2210.70 | 4 |
| 8MB | 867.87 | 599.03 | 1135.97 | 4 |
| 16MB | 856.05 | 582.78 | 1146.28 | 4 |
| 32MB | 841.16 | 576.33 | 1116.02 | 4 |
| 64MB | 848.40 | 573.77 | 1121.68 | 4 |
| 128MB | 853.16 | 597.72 | 1130.01 | 4 |
| 256MB | 858.74 | 596.16 | 1118.98 | 4 |
| 512MB | 863.21 | 599.67 | 1143.43 | 4 |
| 1GB | 864.92 | 601.07 | 1139.76 | 4 |



## Validated Benchmark

A validated benchmark is available that verifies the correctness of the GPU encoding using the full ZODA protocol: [zoda_validated.rs](src/benchmarks/zoda_validated.rs)

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
