// ZODA Reed-Solomon Encoding Benchmark
// Compares GPU vs CPU encoding performance for erasure coding
//
// This measures actual RS encoding: given k data shards, produce n total shards
// using NTT-based polynomial interpolation/evaluation.
//
// GPU optimization strategy:
// - Single memory transfer to GPU
// - Batched NTT kernels process ALL positions in parallel
// - Single memory transfer back
// - This gives GPU massive parallelism advantage on large data

use crate::babybear::BabyBear;
use crate::ntt_babybear::{intt as cpu_intt, ntt as cpu_ntt};
use std::time::Instant;

#[cfg(feature = "cuda")]
use crate::cuda_ntt::{cuda_available, cuda_intt_batched, cuda_ntt_batched, CudaBuffer};

#[derive(Debug, Clone)]
struct EncodingConfig {
    data_size_mb: usize, // Original data size in MB
    k: usize,            // Number of original data shards (must be power of 2)
    expansion: usize,    // Expansion factor (n = k * expansion, n must be power of 2)
}

impl EncodingConfig {
    fn n(&self) -> usize {
        self.k * self.expansion
    }

    fn shard_elements(&self) -> usize {
        // Each shard has this many field elements
        // data_size_bytes / k / sizeof(element)
        // We treat each u32 as one BabyBear element
        (self.data_size_mb * 1024 * 1024) / self.k / 4
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

// CUDA kernel interface for single NTT (used by CPU comparison, not benchmark)
#[cfg(feature = "cuda")]
extern "C" {
    fn cuda_ntt(d_values: *mut u64, n: u32, omega: u64);
    fn cuda_intt(d_values: *mut u64, n: u32, omega: u64);
}

/// RS Encoding using CPU NTT
///
/// For each position across shards, we:
/// 1. Gather k values (one from each original shard)
/// 2. INTT to get polynomial coefficients
/// 3. Pad coefficients to n
/// 4. NTT to evaluate at n points (producing n shard values)
fn encode_cpu(config: &EncodingConfig) -> f64 {
    let k = config.k;
    let n = config.n();
    let shard_elements = config.shard_elements();

    // Generate original data: k shards, each with shard_elements
    let original_shards: Vec<Vec<BabyBear>> = (0..k)
        .map(|shard_idx| {
            (0..shard_elements)
                .map(|elem_idx| {
                    BabyBear::new(((shard_idx * shard_elements + elem_idx) % 2013265921) as u64)
                })
                .collect()
        })
        .collect();

    // Output: n shards (k original + n-k parity)
    let mut encoded_shards: Vec<Vec<BabyBear>> = (0..n)
        .map(|_| vec![BabyBear::zero(); shard_elements])
        .collect();

    let omega_k = BabyBear::get_root_of_unity(k.trailing_zeros());
    let omega_n = BabyBear::get_root_of_unity(n.trailing_zeros());

    let start = Instant::now();

    // For each position across all shards
    for pos in 0..shard_elements {
        // Gather k values from original shards at this position
        let mut coeffs: Vec<BabyBear> = original_shards.iter().map(|shard| shard[pos]).collect();

        // INTT: convert k evaluations to k coefficients
        cpu_intt(&mut coeffs, omega_k);

        // Pad to n for evaluation at n points
        coeffs.resize(n, BabyBear::zero());

        // NTT: evaluate polynomial at n points
        cpu_ntt(&mut coeffs, omega_n);

        // Scatter results to encoded shards
        for (shard_idx, &val) in coeffs.iter().enumerate() {
            encoded_shards[shard_idx][pos] = val;
        }
    }

    start.elapsed().as_secs_f64() * 1000.0
}

/// RS Encoding using GPU with BATCHED NTT operations
///
/// Key optimizations:
/// 1. Data layout optimized for GPU: positions are contiguous
/// 2. Single memory transfer to GPU
/// 3. Batched INTT processes ALL positions in parallel
/// 4. Batched NTT processes ALL positions in parallel
/// 5. Single memory transfer back
#[cfg(feature = "cuda")]
fn encode_gpu_batched(config: &EncodingConfig) -> Result<f64, String> {
    let k = config.k;
    let n = config.n();
    let shard_elements = config.shard_elements();

    // Verify k and n are powers of 2 (required for NTT)
    if !k.is_power_of_two() || !n.is_power_of_two() {
        return Err(format!("k={} and n={} must be powers of 2", k, n));
    }

    let omega_k = BabyBear::get_root_of_unity(k.trailing_zeros());
    let omega_n = BabyBear::get_root_of_unity(n.trailing_zeros());

    // Memory layout for GPU efficiency:
    // We process all positions in parallel, so arrange data as:
    // [pos_0: k values][pos_1: k values]...[pos_{m-1}: k values]
    // Then after INTT, we need to expand each k-block to n-block for NTT

    // Step 1: Prepare input data in position-major order
    // d_intt_buffer[pos * k + shard_idx] = original_shards[shard_idx][pos]
    let mut h_intt_buffer: Vec<u64> = vec![0; shard_elements * k];
    for shard_idx in 0..k {
        for pos in 0..shard_elements {
            let value = ((shard_idx * shard_elements + pos) % 2013265921) as u64;
            h_intt_buffer[pos * k + shard_idx] = value;
        }
    }

    // Allocate GPU buffers
    // INTT buffer: shard_elements positions, k elements each
    let mut d_intt_buffer = CudaBuffer::new(shard_elements * k)?;

    // NTT buffer: shard_elements positions, n elements each (larger for expansion)
    let mut d_ntt_buffer = CudaBuffer::new(shard_elements * n)?;

    // Host buffer for NTT (for padding operation)
    let mut h_ntt_buffer: Vec<u64> = vec![0; shard_elements * n];

    let start = Instant::now();

    // Step 2: Upload data to GPU (single transfer)
    d_intt_buffer.copy_from_host(&h_intt_buffer)?;

    // Step 3: Batched INTT - process ALL positions in parallel!
    // Each position has k elements, stride = k
    unsafe {
        cuda_intt_batched(
            d_intt_buffer.as_ptr(),
            shard_elements as u32,
            k as u32,
            k as u32,
            omega_k.value,
        );
    }

    // Step 4: Download INTT results
    d_intt_buffer.copy_to_host(&mut h_intt_buffer)?;

    // Step 5: Pad from k to n (CPU for now - could be GPU kernel)
    // Copy k coefficients, pad with zeros to n
    for pos in 0..shard_elements {
        for i in 0..k {
            h_ntt_buffer[pos * n + i] = h_intt_buffer[pos * k + i];
        }
        for i in k..n {
            h_ntt_buffer[pos * n + i] = 0;
        }
    }

    // Step 6: Upload padded data for NTT
    d_ntt_buffer.copy_from_host(&h_ntt_buffer)?;

    // Step 7: Batched NTT - process ALL positions in parallel!
    unsafe {
        cuda_ntt_batched(
            d_ntt_buffer.as_ptr(),
            shard_elements as u32,
            n as u32,
            n as u32,
            omega_n.value,
        );
    }

    // Step 8: Download final results (single transfer)
    d_ntt_buffer.copy_to_host(&mut h_ntt_buffer)?;

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    // Verify we got results (optional - comment out for pure benchmark)
    // Results are in h_ntt_buffer[pos * n + shard_idx]

    Ok(elapsed)
}

fn run_encoding_benchmark(config: EncodingConfig) -> Option<EncodingResult> {
    let data_size_mb = config.data_size_mb;

    println!(
        "  k={} shards, {}x expansion (n={})",
        config.k,
        config.expansion,
        config.n()
    );
    println!(
        "  Shard size: {} elements ({:.2} MB per shard)",
        config.shard_elements(),
        config.shard_elements() as f64 * 4.0 / (1024.0 * 1024.0)
    );
    println!(
        "  Total NTT operations: {} (INTT) + {} (NTT) = {} parallel ops",
        config.shard_elements(),
        config.shard_elements(),
        config.shard_elements() * 2
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

        // CPU benchmark
        print!("  CPU encoding... ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        let cpu_time_ms = encode_cpu(&config);
        println!("{:.2} ms", cpu_time_ms);

        // GPU benchmark (batched)
        print!("  GPU encoding (batched)... ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        match encode_gpu_batched(&config) {
            Ok(gpu_time_ms) => {
                println!("{:.2} ms", gpu_time_ms);

                let cpu_throughput_mbs = data_size_mb as f64 / (cpu_time_ms / 1000.0);
                let gpu_throughput_mbs = data_size_mb as f64 / (gpu_time_ms / 1000.0);
                let speedup = cpu_time_ms / gpu_time_ms;

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

#[test]
#[ignore]
fn benchmark_zoda_optimal() {
    println!(
        "\n╔══════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║   ZODA Reed-Solomon Encoding Benchmark (BATCHED GPU)            ║"
    );
    println!(
        "║   GPU NTT vs CPU NTT - Competing with leopard-rs                ║"
    );
    println!(
        "╚══════════════════════════════════════════════════════════════════╝\n"
    );

    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA support not compiled in.");
        return;
    }

    #[cfg(feature = "cuda")]
    {
        if !cuda_available() {
            println!("CUDA not available on this system.");
            return;
        }

        println!("CUDA available - GPU detected!\n");
        println!("Strategy: Batched NTT kernels process ALL positions in parallel\n");

        // All k and n values must be powers of 2 for NTT
        let configs = vec![
            // Small warmup
            EncodingConfig { data_size_mb: 16, k: 64, expansion: 2 },
            EncodingConfig { data_size_mb: 32, k: 128, expansion: 2 },
            EncodingConfig { data_size_mb: 64, k: 256, expansion: 2 },

            // Medium - GPU advantage emerges
            EncodingConfig { data_size_mb: 128, k: 256, expansion: 2 },
            EncodingConfig { data_size_mb: 256, k: 512, expansion: 2 },
            EncodingConfig { data_size_mb: 512, k: 1024, expansion: 2 },

            // Large - GPU should dominate
            EncodingConfig { data_size_mb: 1024, k: 1024, expansion: 2 },  // 1 GB, 256K positions
            EncodingConfig { data_size_mb: 1024, k: 2048, expansion: 2 },  // 1 GB, 128K positions
            EncodingConfig { data_size_mb: 2048, k: 2048, expansion: 2 },  // 2 GB
            EncodingConfig { data_size_mb: 4096, k: 4096, expansion: 2 },  // 4 GB

            // Extra large - maximum GPU advantage
            EncodingConfig { data_size_mb: 8192, k: 4096, expansion: 2 },  // 8 GB
            EncodingConfig { data_size_mb: 8192, k: 8192, expansion: 2 },  // 8 GB, more shards

            // Different expansion factors (4x redundancy)
            EncodingConfig { data_size_mb: 1024, k: 1024, expansion: 4 },  // 1 GB, 4x
            EncodingConfig { data_size_mb: 2048, k: 2048, expansion: 4 },  // 2 GB, 4x
        ];

        let mut results = Vec::new();

        for config in configs {
            println!(
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            );
            println!("Encoding {} MB of data:", config.data_size_mb);

            if let Some(result) = run_encoding_benchmark(config) {
                results.push(result);
            }
            println!();
        }

        // Print summary table
        println!(
            "\n╔══════════════════════════════════════════════════════════════════╗"
        );
        println!(
            "║                      Results Summary                             ║"
        );
        println!(
            "╚══════════════════════════════════════════════════════════════════╝\n"
        );

        println!(
            "{:<10} {:<8} {:<6} │ {:<14} {:<14} │ {:<10}",
            "Data", "k", "n", "CPU MB/s", "GPU MB/s", "Speedup"
        );
        println!("{}", "─".repeat(75));

        for result in &results {
            println!(
                "{:<10} {:<8} {:<6} │ {:<14.1} {:<14.1} │ {:<10.2}x",
                format!("{} MB", result.config.data_size_mb),
                result.config.k,
                result.config.n(),
                result.cpu_throughput_mbs,
                result.gpu_throughput_mbs,
                result.speedup,
            );
        }

        if !results.is_empty() {
            println!(
                "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            );
            println!("Performance Summary");
            println!(
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            );

            let large_results: Vec<_> = results
                .iter()
                .filter(|r| r.config.data_size_mb >= 1024)
                .collect();

            if !large_results.is_empty() {
                let avg_speedup: f64 =
                    large_results.iter().map(|r| r.speedup).sum::<f64>() / large_results.len() as f64;
                let avg_gpu_throughput: f64 = large_results
                    .iter()
                    .map(|r| r.gpu_throughput_mbs)
                    .sum::<f64>()
                    / large_results.len() as f64;
                let max_gpu_throughput = large_results
                    .iter()
                    .map(|r| r.gpu_throughput_mbs)
                    .fold(0.0f64, f64::max);

                println!("For large data (>= 1GB):");
                println!("  Average GPU Speedup:     {:.2}x over CPU", avg_speedup);
                println!("  Average GPU Throughput:  {:.1} MB/s", avg_gpu_throughput);
                println!("  Peak GPU Throughput:     {:.1} MB/s", max_gpu_throughput);

                println!("\nComparison with leopard-rs (reference):");
                println!("  leopard-rs encode (AVX2): ~1000-3000 MB/s");
                println!("  leopard-rs encode (basic): ~200-500 MB/s");
                println!(
                    "  Our GPU implementation:    {:.1} MB/s (peak)",
                    max_gpu_throughput
                );
                println!();

                if max_gpu_throughput > 3000.0 {
                    println!("  >> Outperforming optimized leopard-rs!");
                } else if max_gpu_throughput > 1000.0 {
                    println!("  >> Competitive with optimized leopard-rs");
                } else if max_gpu_throughput > 500.0 {
                    println!("  >> Competitive with basic leopard-rs");
                } else {
                    println!("  >> Room for GPU optimization");
                }
            }
        }

        println!(
            "\n╔══════════════════════════════════════════════════════════════════╗"
        );
        println!(
            "║  Benchmark Complete                                              ║"
        );
        println!(
            "╚══════════════════════════════════════════════════════════════════╝\n"
        );
    }
}
