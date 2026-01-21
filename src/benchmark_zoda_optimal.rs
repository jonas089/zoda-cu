// ZODA Reed-Solomon Encoding Benchmark
// Compares GPU vs CPU encoding performance for erasure coding
//
// This measures actual RS encoding: given k data shards, produce n total shards
// using NTT-based polynomial interpolation/evaluation.

use crate::babybear::BabyBear;
use crate::ntt_babybear::{intt as cpu_intt, ntt as cpu_ntt};
use std::time::Instant;

#[cfg(feature = "cuda")]
use crate::cuda_ntt::{cuda_available, CudaBuffer};

#[derive(Debug, Clone)]
struct EncodingConfig {
    data_size_mb: usize,    // Original data size in MB
    k: usize,               // Number of original data shards
    expansion: usize,       // Expansion factor (n = k * expansion)
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

    fn ntt_size(&self) -> usize {
        // NTT size must be >= n and a power of 2
        self.n().next_power_of_two()
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

// CUDA kernel interface
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
    let ntt_size = config.ntt_size();
    let shard_elements = config.shard_elements();

    // Generate original data: k shards, each with shard_elements
    let original_shards: Vec<Vec<BabyBear>> = (0..k)
        .map(|shard_idx| {
            (0..shard_elements)
                .map(|elem_idx| BabyBear::new(((shard_idx * shard_elements + elem_idx) % 2013265921) as u64))
                .collect()
        })
        .collect();

    // Output: n shards (k original + n-k parity)
    let mut encoded_shards: Vec<Vec<BabyBear>> = (0..n)
        .map(|_| vec![BabyBear::zero(); shard_elements])
        .collect();

    let omega_k = BabyBear::get_root_of_unity(k.trailing_zeros());
    let omega_n = BabyBear::get_root_of_unity(ntt_size.trailing_zeros());

    let start = Instant::now();

    // For each position across all shards
    for pos in 0..shard_elements {
        // Gather k values from original shards at this position
        let mut coeffs: Vec<BabyBear> = original_shards.iter()
            .map(|shard| shard[pos])
            .collect();

        // INTT: convert k evaluations to k coefficients
        cpu_intt(&mut coeffs, omega_k);

        // Pad to ntt_size for evaluation at n points
        coeffs.resize(ntt_size, BabyBear::zero());

        // NTT: evaluate polynomial at n points
        cpu_ntt(&mut coeffs, omega_n);

        // Scatter results to encoded shards
        for (shard_idx, &val) in coeffs.iter().take(n).enumerate() {
            encoded_shards[shard_idx][pos] = val;
        }
    }

    start.elapsed().as_secs_f64() * 1000.0
}

/// RS Encoding using GPU NTT with batched transfers
#[cfg(feature = "cuda")]
fn encode_gpu(config: &EncodingConfig) -> Result<f64, String> {
    let k = config.k;
    let n = config.n();
    let ntt_size = config.ntt_size();
    let shard_elements = config.shard_elements();

    // For GPU, we batch all position NTTs together
    // We'll process in batches to manage memory

    let omega_k = BabyBear::get_root_of_unity(k.trailing_zeros());
    let omega_n = BabyBear::get_root_of_unity(ntt_size.trailing_zeros());

    // Generate original data
    let original_data: Vec<u64> = (0..k)
        .flat_map(|shard_idx| {
            (0..shard_elements)
                .map(move |elem_idx| ((shard_idx * shard_elements + elem_idx) % 2013265921) as u64)
        })
        .collect();

    // Batch size for GPU processing (number of positions to process at once)
    // Each position needs ntt_size elements
    let gpu_batch_size = (64 * 1024 * 1024 / (ntt_size * 8)).max(1024); // ~64MB GPU buffer per batch
    let num_batches = (shard_elements + gpu_batch_size - 1) / gpu_batch_size;

    // Allocate GPU buffers
    // For INTT: k elements per position, batch_size positions
    let intt_buffer_size = k.next_power_of_two() * gpu_batch_size;
    // For NTT: ntt_size elements per position, batch_size positions
    let ntt_buffer_size = ntt_size * gpu_batch_size;

    let mut d_intt_buffer = CudaBuffer::new(intt_buffer_size)?;
    let mut d_ntt_buffer = CudaBuffer::new(ntt_buffer_size)?;

    let mut intt_host: Vec<u64> = vec![0; intt_buffer_size];
    let mut ntt_host: Vec<u64> = vec![0; ntt_buffer_size];

    // Output storage
    let mut encoded_shards: Vec<Vec<u64>> = (0..n)
        .map(|_| vec![0u64; shard_elements])
        .collect();

    let start = Instant::now();

    for batch_idx in 0..num_batches {
        let batch_start = batch_idx * gpu_batch_size;
        let batch_end = (batch_start + gpu_batch_size).min(shard_elements);
        let actual_batch_size = batch_end - batch_start;

        // Gather data for this batch: for each position, collect k values
        for (local_pos, global_pos) in (batch_start..batch_end).enumerate() {
            for shard_idx in 0..k {
                intt_host[local_pos * k.next_power_of_two() + shard_idx] =
                    original_data[shard_idx * shard_elements + global_pos];
            }
            // Zero-pad to power of 2 for INTT
            for pad_idx in k..k.next_power_of_two() {
                intt_host[local_pos * k.next_power_of_two() + pad_idx] = 0;
            }
        }

        // Copy to GPU and run batched INTTs
        d_intt_buffer.copy_from_host(&intt_host)?;

        unsafe {
            for local_pos in 0..actual_batch_size {
                let ptr = d_intt_buffer.as_ptr().add(local_pos * k.next_power_of_two());
                cuda_intt(ptr, k.next_power_of_two() as u32, omega_k.value);
            }
        }

        d_intt_buffer.copy_to_host(&mut intt_host)?;

        // Prepare NTT buffer: copy coefficients and zero-pad to ntt_size
        for local_pos in 0..actual_batch_size {
            for coeff_idx in 0..k {
                ntt_host[local_pos * ntt_size + coeff_idx] =
                    intt_host[local_pos * k.next_power_of_two() + coeff_idx];
            }
            for pad_idx in k..ntt_size {
                ntt_host[local_pos * ntt_size + pad_idx] = 0;
            }
        }

        // Copy to GPU and run batched NTTs
        d_ntt_buffer.copy_from_host(&ntt_host)?;

        unsafe {
            for local_pos in 0..actual_batch_size {
                let ptr = d_ntt_buffer.as_ptr().add(local_pos * ntt_size);
                cuda_ntt(ptr, ntt_size as u32, omega_n.value);
            }
        }

        d_ntt_buffer.copy_to_host(&mut ntt_host)?;

        // Scatter results to encoded shards
        for (local_pos, global_pos) in (batch_start..batch_end).enumerate() {
            for shard_idx in 0..n {
                encoded_shards[shard_idx][global_pos] = ntt_host[local_pos * ntt_size + shard_idx];
            }
        }
    }

    Ok(start.elapsed().as_secs_f64() * 1000.0)
}

fn run_encoding_benchmark(config: EncodingConfig) -> Option<EncodingResult> {
    let data_size_mb = config.data_size_mb;

    println!("  k={} shards, {}x expansion (n={}), NTT size={}",
        config.k, config.expansion, config.n(), config.ntt_size());
    println!("  Shard size: {} elements ({:.2} MB per shard)",
        config.shard_elements(),
        config.shard_elements() as f64 * 4.0 / (1024.0 * 1024.0));

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

        // GPU benchmark
        print!("  GPU encoding... ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        match encode_gpu(&config) {
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
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║   ZODA Reed-Solomon Encoding Benchmark                           ║");
    println!("║   GPU NTT vs CPU NTT - Competing with leopard-rs                 ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

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
        println!("Encoding: k original shards -> n total shards (n-k parity)\n");

        let configs = vec![
            // Small sizes for warmup/baseline
            EncodingConfig { data_size_mb: 16, k: 64, expansion: 2 },
            EncodingConfig { data_size_mb: 32, k: 128, expansion: 2 },
            EncodingConfig { data_size_mb: 64, k: 256, expansion: 2 },

            // Medium sizes - GPU starts showing advantage
            EncodingConfig { data_size_mb: 128, k: 256, expansion: 2 },
            EncodingConfig { data_size_mb: 256, k: 512, expansion: 2 },
            EncodingConfig { data_size_mb: 512, k: 1024, expansion: 2 },

            // Large sizes - GPU should dominate
            EncodingConfig { data_size_mb: 1024, k: 1024, expansion: 2 },    // 1 GB
            EncodingConfig { data_size_mb: 1024, k: 2048, expansion: 2 },    // 1 GB, more shards
            EncodingConfig { data_size_mb: 2048, k: 2048, expansion: 2 },    // 2 GB
            EncodingConfig { data_size_mb: 4096, k: 4096, expansion: 2 },    // 4 GB

            // Different expansion factors
            EncodingConfig { data_size_mb: 1024, k: 1024, expansion: 4 },    // 1 GB, 4x expansion
            EncodingConfig { data_size_mb: 2048, k: 2048, expansion: 4 },    // 2 GB, 4x expansion
        ];

        let mut results = Vec::new();

        for config in configs {
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("Encoding {} MB of data:", config.data_size_mb);

            if let Some(result) = run_encoding_benchmark(config) {
                results.push(result);
            }
            println!();
        }

        // Print summary table
        println!("\n╔══════════════════════════════════════════════════════════════════╗");
        println!("║                      Results Summary                             ║");
        println!("╚══════════════════════════════════════════════════════════════════╝\n");

        println!("{:<10} {:<8} {:<6} │ {:<14} {:<14} │ {:<10}",
            "Data", "k", "n", "CPU MB/s", "GPU MB/s", "Speedup");
        println!("{}", "─".repeat(75));

        for result in &results {
            println!("{:<10} {:<8} {:<6} │ {:<14.1} {:<14.1} │ {:<10.2}x",
                format!("{} MB", result.config.data_size_mb),
                result.config.k,
                result.config.n(),
                result.cpu_throughput_mbs,
                result.gpu_throughput_mbs,
                result.speedup,
            );
        }

        if !results.is_empty() {
            println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("Performance Summary");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

            let large_results: Vec<_> = results.iter()
                .filter(|r| r.config.data_size_mb >= 1024)
                .collect();

            if !large_results.is_empty() {
                let avg_speedup: f64 = large_results.iter().map(|r| r.speedup).sum::<f64>()
                    / large_results.len() as f64;
                let avg_gpu_throughput: f64 = large_results.iter().map(|r| r.gpu_throughput_mbs).sum::<f64>()
                    / large_results.len() as f64;
                let max_gpu_throughput = large_results.iter()
                    .map(|r| r.gpu_throughput_mbs).fold(0.0f64, f64::max);

                println!("For large data (>= 1GB):");
                println!("  Average GPU Speedup:     {:.2}x over CPU", avg_speedup);
                println!("  Average GPU Throughput:  {:.1} MB/s", avg_gpu_throughput);
                println!("  Peak GPU Throughput:     {:.1} MB/s", max_gpu_throughput);

                println!("\nComparison with leopard-rs (reference):");
                println!("  leopard-rs encode (AVX2): ~1000-3000 MB/s");
                println!("  leopard-rs encode (basic): ~200-500 MB/s");
                println!("  Our GPU implementation:    {:.1} MB/s (peak)", max_gpu_throughput);
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

        println!("\n╔══════════════════════════════════════════════════════════════════╗");
        println!("║  Benchmark Complete                                              ║");
        println!("╚══════════════════════════════════════════════════════════════════╝\n");
    }
}
