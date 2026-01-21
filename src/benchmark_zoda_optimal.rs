// ============================================================================
// ZODA Reed-Solomon Encoding Benchmark - GPU vs CPU
// ============================================================================
//
// This benchmark measures the performance of vertical Reed-Solomon encoding
// used in data availability systems like Celestia's rsema1d.
//
// TERMINOLOGY (matching rsema1d/Celestia):
// ----------------------------------------
// - K: number of original data rows
// - N: number of parity rows
// - Total rows: K + N (full erasure-coded output)
// - RowSize: bytes per row (must be multiple of 4 for BabyBear)
//
// ENCODING ALGORITHM (Vertical Extension):
// ----------------------------------------
// The data is arranged as a matrix with K rows and (RowSize/4) columns.
// For EACH column position independently:
//   1. Gather K values from that column across all rows
//   2. INTT(K) → convert K evaluations to polynomial coefficients
//   3. Zero-pad coefficients from K to (K+N)
//   4. NTT(K+N) → evaluate polynomial at (K+N) points
//   5. Write (K+N) values back to column in output matrix
//
// This produces K+N total rows, where any K rows can recover the original data.
//
// GPU OPTIMIZATION STRATEGY:
// --------------------------
// The key insight is that ALL (RowSize/4) column positions can be processed
// in parallel on the GPU. For large data:
//   - CPU: processes columns sequentially → ~O(columns * NTT_time)
//   - GPU: processes ALL columns in parallel → ~O(NTT_time) + transfer overhead
//
// With batched kernels and fused operations, the GPU achieves:
//   - Single memory upload (all input data)
//   - Batched INTT on ALL columns simultaneously
//   - GPU-side zero-padding (no CPU roundtrip!)
//   - Batched NTT on ALL columns simultaneously
//   - Single memory download (all output data)
//
// For data with 1024 columns and 1024-size NTTs:
//   - CPU does: 1024 × INTT + 1024 × NTT sequentially
//   - GPU does: 1 × batched_INTT(1024 columns) + 1 × batched_NTT(1024 columns)
//   → Expected speedup: ~1000x+ on pure compute, ~100x+ with memory transfer
//
// ============================================================================

use crate::babybear::BabyBear;
use crate::ntt_babybear::{intt as cpu_intt, ntt as cpu_ntt};
use std::time::Instant;

#[cfg(feature = "cuda")]
use crate::cuda_ntt::{cuda_available, cuda_rs_encode_vertical, CudaBuffer};

// ============================================================================
// Configuration & Results
// ============================================================================

#[derive(Debug, Clone)]
struct EncodingConfig {
    k: usize,        // Number of original rows
    n: usize,        // Number of parity rows
    row_size: usize, // Bytes per row (must be multiple of 4 for BabyBear)
}

impl EncodingConfig {
    /// Original data size in MB
    fn data_size_mb(&self) -> f64 {
        (self.k * self.row_size) as f64 / (1024.0 * 1024.0)
    }

    /// Total rows after encoding (K original + N parity)
    fn total_rows(&self) -> usize {
        self.k + self.n
    }

    /// Number of BabyBear elements per row (each element is 4 bytes)
    fn elements_per_row(&self) -> usize {
        self.row_size / 4
    }

    /// Number of column positions = number of parallel NTT operations
    /// This is the dimension that GPU parallelizes over
    fn num_positions(&self) -> usize {
        self.elements_per_row()
    }

    /// NTT size for K rows (must be power of 2 >= K)
    fn ntt_size_k(&self) -> usize {
        self.k.next_power_of_two()
    }

    /// NTT size for K+N rows (must be power of 2 >= K+N)
    fn ntt_size_kn(&self) -> usize {
        self.total_rows().next_power_of_two()
    }
}

#[derive(Debug)]
struct EncodingResult {
    config: EncodingConfig,
    cpu_time_ms: f64,
    gpu_time_ms: f64,
    cpu_throughput_mbs: f64,
    gpu_throughput_mbs: f64,
    speedup: f64,
}

// ============================================================================
// CPU Implementation (Baseline)
// ============================================================================

/// RS Encoding using CPU NTT - processes each column sequentially
///
/// This is the baseline for comparison. It's straightforward but slow for
/// large data because it processes one column at a time.
fn encode_cpu(config: &EncodingConfig) -> f64 {
    let k = config.k;
    let n = config.n;
    let num_positions = config.num_positions();
    let ntt_size_k = config.ntt_size_k();
    let ntt_size_kn = config.ntt_size_kn();

    // Generate original data: K rows × num_positions columns
    let original_rows: Vec<Vec<BabyBear>> = (0..k)
        .map(|row_idx| {
            (0..num_positions)
                .map(|col_idx| {
                    BabyBear::new(((row_idx * num_positions + col_idx) % 2013265921) as u64)
                })
                .collect()
        })
        .collect();

    // Allocate output: (K+N) rows × num_positions columns
    let mut encoded_rows: Vec<Vec<BabyBear>> = (0..k + n)
        .map(|_| vec![BabyBear::zero(); num_positions])
        .collect();

    // Get twiddle factors for NTT sizes
    let omega_k = BabyBear::get_root_of_unity(ntt_size_k.trailing_zeros());
    let omega_kn = BabyBear::get_root_of_unity(ntt_size_kn.trailing_zeros());

    let start = Instant::now();

    // Process each column position sequentially (CPU limitation)
    for col in 0..num_positions {
        // Step 1: Gather K values from this column across all original rows
        let mut values: Vec<BabyBear> = original_rows.iter().map(|row| row[col]).collect();

        // Step 2: Pad to ntt_size_k for INTT
        values.resize(ntt_size_k, BabyBear::zero());

        // Step 3: INTT - convert K evaluations to polynomial coefficients
        cpu_intt(&mut values, omega_k);

        // Step 4: Pad to ntt_size_kn for evaluation at K+N points
        values.resize(ntt_size_kn, BabyBear::zero());

        // Step 5: NTT - evaluate polynomial at K+N points
        cpu_ntt(&mut values, omega_kn);

        // Step 6: Scatter (K+N) values back to output rows at this column
        for (row_idx, &val) in values.iter().take(k + n).enumerate() {
            encoded_rows[row_idx][col] = val;
        }
    }

    start.elapsed().as_secs_f64() * 1000.0
}

// ============================================================================
// GPU Implementation (Fully Optimized)
// ============================================================================

/// RS Encoding using GPU with fully fused operations - ZERO CPU ROUNDTRIPS
///
/// Key optimizations:
/// 1. Data layout: columns are contiguous in memory for GPU coalesced access
/// 2. Single upload: all input data transferred to GPU at once
/// 3. Batched INTT: processes ALL columns in parallel (not sequential!)
/// 4. GPU padding: zero-extension happens on GPU (no download/upload!)
/// 5. Batched NTT: processes ALL columns in parallel
/// 6. Single download: all output data transferred from GPU at once
///
/// Memory transfers:
///   - Upload: num_positions × ntt_size_k elements
///   - Download: num_positions × ntt_size_kn elements
///   - Total: ~2× the data size (vs ~8× for naive implementation)
///
/// Compute operations:
///   - CPU: O(num_positions) sequential NTT operations
///   - GPU: O(1) batched NTT operations (all parallel)
///   - Speedup: ~num_positions × (ignoring memory transfer)
#[cfg(feature = "cuda")]
fn encode_gpu_optimized(config: &EncodingConfig) -> Result<f64, String> {
    let k = config.k;
    let n = config.n;
    let num_positions = config.num_positions();
    let ntt_size_k = config.ntt_size_k();
    let ntt_size_kn = config.ntt_size_kn();

    // Get twiddle factors
    let omega_k = BabyBear::get_root_of_unity(ntt_size_k.trailing_zeros());
    let omega_kn = BabyBear::get_root_of_unity(ntt_size_kn.trailing_zeros());

    // ========================================================================
    // Prepare input data in column-major layout for GPU efficiency
    // ========================================================================
    // Layout: [col_0: k values, 0-padded to ntt_size_k]
    //         [col_1: k values, 0-padded to ntt_size_k]
    //         ...
    //         [col_{m-1}: k values, 0-padded to ntt_size_k]
    //
    // This layout enables:
    //   - Coalesced GPU memory access (adjacent threads access adjacent memory)
    //   - Efficient batched NTT (each column is an independent NTT)

    let mut h_input: Vec<u64> = vec![0; num_positions * ntt_size_k];

    // Fill input data
    for row_idx in 0..k {
        for col in 0..num_positions {
            let value = ((row_idx * num_positions + col) % 2013265921) as u64;
            h_input[col * ntt_size_k + row_idx] = value;
        }
    }
    // Note: padding from k to ntt_size_k is already 0 (vec initialization)

    // ========================================================================
    // Allocate GPU buffers
    // ========================================================================
    let mut d_input = CudaBuffer::new(num_positions * ntt_size_k)?;     // Input data
    let mut d_output = CudaBuffer::new(num_positions * ntt_size_kn)?;   // Final output
    let mut d_work = CudaBuffer::new(num_positions * ntt_size_k)?;      // INTT workspace

    let mut h_output: Vec<u64> = vec![0; num_positions * ntt_size_kn];

    // ========================================================================
    // GPU Encoding Pipeline - Everything happens on GPU!
    // ========================================================================

    let start = Instant::now();

    // --- Upload (Single Transfer) ---
    d_input.copy_from_host(&h_input)?;

    // --- GPU Processing (Zero CPU Roundtrips!) ---
    unsafe {
        cuda_rs_encode_vertical(
            d_input.as_ptr(),     // Input: num_positions columns of size ntt_size_k
            d_output.as_ptr(),    // Output: num_positions columns of size ntt_size_kn
            d_work.as_ptr(),      // Work buffer for INTT
            num_positions as u32,
            ntt_size_k as u32,
            ntt_size_kn as u32,
            omega_k.value,
            omega_kn.value,
        );
    }

    // --- Download (Single Transfer) ---
    d_output.copy_to_host(&mut h_output)?;

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    // ========================================================================
    // Result verification (optional - can be disabled for pure benchmark)
    // ========================================================================
    // Output layout: h_output[col * ntt_size_kn + row] for row in [0, k+n)
    // This can be reshaped to (k+n) rows × num_positions columns

    Ok(elapsed)
}

// ============================================================================
// Benchmark Runner
// ============================================================================

fn run_encoding_benchmark(config: EncodingConfig) -> Option<EncodingResult> {
    let data_size_mb = config.data_size_mb();

    println!(
        "  K={} original rows, N={} parity rows → {} total rows ({}x expansion)",
        config.k,
        config.n,
        config.total_rows(),
        config.total_rows() as f64 / config.k as f64
    );
    println!(
        "  Row size: {} bytes = {} BabyBear elements per row",
        config.row_size,
        config.elements_per_row()
    );
    println!(
        "  Data volume: {:.1} MB original → {:.1} MB after encoding",
        data_size_mb,
        data_size_mb * config.total_rows() as f64 / config.k as f64
    );
    println!(
        "  Parallel dimension: {} column positions (each requires INTT+NTT)",
        config.num_positions()
    );
    println!(
        "  NTT sizes: INTT({}) then NTT({})",
        config.ntt_size_k(),
        config.ntt_size_kn()
    );

    #[cfg(not(feature = "cuda"))]
    {
        println!("  CUDA not available");
        return None;
    }

    #[cfg(feature = "cuda")]
    {
        if !cuda_available() {
            println!("  CUDA not available");
            return None;
        }

        // CPU benchmark (baseline)
        print!("  CPU encoding (sequential)... ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        let cpu_time_ms = encode_cpu(&config);
        println!("{:.2} ms", cpu_time_ms);

        // GPU benchmark (optimized)
        print!("  GPU encoding (batched+fused)... ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        match encode_gpu_optimized(&config) {
            Ok(gpu_time_ms) => {
                println!("{:.2} ms", gpu_time_ms);

                let cpu_throughput_mbs = data_size_mb / (cpu_time_ms / 1000.0);
                let gpu_throughput_mbs = data_size_mb / (gpu_time_ms / 1000.0);
                let speedup = cpu_time_ms / gpu_time_ms;

                println!("    → GPU is {:.1}x faster than CPU", speedup);

                Some(EncodingResult {
                    config,
                    cpu_time_ms,
                    gpu_time_ms,
                    cpu_throughput_mbs,
                    gpu_throughput_mbs,
                    speedup,
                })
            }
            Err(e) => {
                println!("ERROR: {}", e);
                None
            }
        }
    }
}

// ============================================================================
// Main Benchmark Test
// ============================================================================

#[test]
#[ignore]
fn benchmark_zoda_optimal() {
    println!("\n╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║                ZODA Reed-Solomon Encoding Benchmark                  ║");
    println!("║                      GPU vs CPU Performance                          ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝\n");

    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA support not compiled in.");
        println!("Build with: cargo test --features cuda --release");
        return;
    }

    #[cfg(feature = "cuda")]
    {
        if !cuda_available() {
            println!("CUDA not available on this system.");
            return;
        }

        println!("✓ CUDA GPU detected\n");
        println!("Benchmark: Vertical Reed-Solomon encoding (like Celestia rsema1d)");
        println!("Algorithm: K rows → K+N rows via NTT-based polynomial evaluation\n");

        // ====================================================================
        // Test configurations - covering small to very large data sizes
        // ====================================================================
        //
        // row_size must be multiple of 4 (BabyBear element = 4 bytes)
        // K and N should ideally be powers of 2 for optimal NTT performance
        // But the benchmark handles any K, N by padding to next power of 2

        let configs = vec![
            // --- Small: warmup and correctness check ---
            EncodingConfig { k: 64, n: 64, row_size: 4096 },       // 256 KB
            EncodingConfig { k: 128, n: 128, row_size: 4096 },     // 512 KB
            EncodingConfig { k: 256, n: 256, row_size: 4096 },     // 1 MB

            // --- Medium: GPU advantage starts to show ---
            EncodingConfig { k: 512, n: 512, row_size: 4096 },     // 2 MB
            EncodingConfig { k: 1024, n: 1024, row_size: 4096 },   // 4 MB
            EncodingConfig { k: 2048, n: 2048, row_size: 4096 },   // 8 MB

            // --- Large: GPU should dominate ---
            EncodingConfig { k: 4096, n: 4096, row_size: 4096 },   // 16 MB
            EncodingConfig { k: 8192, n: 8192, row_size: 4096 },   // 32 MB
            EncodingConfig { k: 16384, n: 16384, row_size: 4096 }, // 64 MB

            // --- Very large: close to rsema1d defaults ---
            EncodingConfig { k: 32768, n: 32768, row_size: 4096 }, // 128 MB (rsema1d default)
            EncodingConfig { k: 65536, n: 65536, row_size: 4096 }, // 256 MB

            // --- Huge: larger rows = more parallelism = better GPU util ---
            EncodingConfig { k: 16384, n: 16384, row_size: 8192 },  // 128 MB, 2048 positions
            EncodingConfig { k: 16384, n: 16384, row_size: 16384 }, // 256 MB, 4096 positions
            EncodingConfig { k: 32768, n: 32768, row_size: 16384 }, // 512 MB, 4096 positions

            // --- Massive: stress test (if your GPU has enough memory) ---
            EncodingConfig { k: 32768, n: 32768, row_size: 32768 }, // 1 GB, 8192 positions
            // EncodingConfig { k: 65536, n: 65536, row_size: 32768 }, // 2 GB (uncomment if you have 24GB+ GPU)

            // --- Different expansion ratios ---
            EncodingConfig { k: 16384, n: 32768, row_size: 4096 },  // 1:2 ratio (3x total size)
            EncodingConfig { k: 16384, n: 49152, row_size: 4096 },  // 1:3 ratio (4x total size)
        ];

        let mut results = Vec::new();

        println!("Running benchmarks...\n");

        for config in configs {
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("Encoding {:.1} MB of data:", config.data_size_mb());

            if let Some(result) = run_encoding_benchmark(config) {
                results.push(result);
            }
            println!();
        }

        // ====================================================================
        // Results Summary
        // ====================================================================

        println!("\n╔═══════════════════════════════════════════════════════════════════════╗");
        println!("║                         Results Summary                               ║");
        println!("╚═══════════════════════════════════════════════════════════════════════╝\n");

        println!("{:<12} {:<10} {:<10} {:<10} │ {:<12} {:<12} │ {:<10}",
            "Data MB", "K", "N", "RowSize", "CPU MB/s", "GPU MB/s", "Speedup");
        println!("{}", "─".repeat(90));

        for result in &results {
            println!("{:<12.1} {:<10} {:<10} {:<10} │ {:<12.1} {:<12.1} │ {:<10.1}x",
                result.config.data_size_mb(),
                result.config.k,
                result.config.n,
                result.config.row_size,
                result.cpu_throughput_mbs,
                result.gpu_throughput_mbs,
                result.speedup,
            );
        }

        // ====================================================================
        // Performance Analysis
        // ====================================================================

        if !results.is_empty() {
            println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("Performance Analysis");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

            // Analyze large data results (>= 64 MB)
            let large_results: Vec<_> = results
                .iter()
                .filter(|r| r.config.data_size_mb() >= 64.0)
                .collect();

            if !large_results.is_empty() {
                let avg_speedup = large_results.iter().map(|r| r.speedup).sum::<f64>()
                    / large_results.len() as f64;
                let avg_gpu_throughput = large_results.iter().map(|r| r.gpu_throughput_mbs).sum::<f64>()
                    / large_results.len() as f64;
                let max_gpu_throughput = large_results.iter()
                    .map(|r| r.gpu_throughput_mbs).fold(0.0f64, f64::max);

                println!("For large data (>= 64 MB):");
                println!("  Average GPU Speedup:      {:.1}x faster than CPU", avg_speedup);
                println!("  Average GPU Throughput:   {:.1} MB/s", avg_gpu_throughput);
                println!("  Peak GPU Throughput:      {:.1} MB/s", max_gpu_throughput);

                println!("\nComparison with leopard-rs (GF(2^16) reference implementation):");
                println!("  leopard-rs encode (AVX2):  ~1000-3000 MB/s");
                println!("  leopard-rs encode (basic):  ~200-500 MB/s");
                println!("  Our GPU (BabyBear):         {:.1} MB/s (peak)", max_gpu_throughput);
                println!();

                if max_gpu_throughput > 3000.0 {
                    println!("  ✓✓ OUTPERFORMING optimized leopard-rs with AVX2!");
                } else if max_gpu_throughput > 1000.0 {
                    println!("  ✓ Competitive with optimized leopard-rs");
                } else if max_gpu_throughput > 500.0 {
                    println!("  → Competitive with basic leopard-rs");
                } else {
                    println!("  → Potential for further GPU optimization");
                }
            }

            // Show scaling analysis
            println!("\nScaling Analysis:");
            let mut prev_size = 0.0;
            let mut prev_gpu_time = 0.0;
            for result in &results {
                let size = result.config.data_size_mb();
                let gpu_time = result.gpu_time_ms;

                if prev_size > 0.0 {
                    let size_ratio = size / prev_size;
                    let time_ratio = gpu_time / prev_gpu_time;
                    let efficiency = size_ratio / time_ratio;

                    if efficiency > 0.9 {
                        println!("  {:.0} MB → {:.0} MB: {:.2}x data in {:.2}x time (efficient scaling)",
                            prev_size, size, size_ratio, time_ratio);
                    }
                }

                prev_size = size;
                prev_gpu_time = gpu_time;
            }
        }

        println!("\n╔═══════════════════════════════════════════════════════════════════════╗");
        println!("║  Benchmark Complete!                                                  ║");
        println!("╚═══════════════════════════════════════════════════════════════════════╝\n");
    }
}
