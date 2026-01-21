// Fixed ZODA benchmark - correct memory calculations
// Designed for mobile GPUs with proper chunk sizing

use crate::babybear::BabyBear;
use crate::ntt_babybear::{intt as cpu_intt, ntt as cpu_ntt};
use std::time::Instant;

#[cfg(feature = "cuda")]
use crate::cuda_ntt::{cuda_available, CudaBuffer};

#[derive(Debug, Clone)]
struct BenchmarkConfig {
    data_size_kb: usize,
    k: usize,
    n: usize,
}

#[derive(Debug)]
struct BenchmarkResult {
    config: BenchmarkConfig,
    cpu_time_us: u128,
    gpu_time_us: u128,
    cpu_throughput_mbs: f64,
    gpu_throughput_mbs: f64,
    speedup: f64,
    total_vram_mb: f64,
    ntt_size: usize,
    elements_per_chunk: usize,
}

#[cfg(feature = "cuda")]
extern "C" {
    fn cuda_ntt(d_values: *mut u64, n: u32, omega: u64);
    fn cuda_intt(d_values: *mut u64, n: u32, omega: u64);
}

fn calculate_params(config: &BenchmarkConfig) -> (usize, usize, f64) {
    // Each BabyBear element encodes ~4 bytes of data
    let bytes_per_element = 4usize;
    let total_elements = (config.data_size_kb * 1024) / bytes_per_element;
    let elements_per_chunk = total_elements / config.k;

    // NTT size must be power of 2 and >= n for RS encoding
    // For RS, we need ntt_size >= n to evaluate at n points
    let ntt_size = config.n.next_power_of_two();

    // Total GPU memory: k chunks × ntt_size elements × 8 bytes per u64
    let total_gpu_bytes = config.k * ntt_size * 8;
    let total_vram_mb = total_gpu_bytes as f64 / (1024.0 * 1024.0);

    (elements_per_chunk, ntt_size, total_vram_mb)
}

fn benchmark_cpu(config: &BenchmarkConfig) -> (u128, usize, usize) {
    let (elements_per_chunk, ntt_size, _) = calculate_params(config);

    // Create data: k chunks, each with elements_per_chunk elements, padded to ntt_size
    let mut all_data: Vec<BabyBear> = Vec::with_capacity(config.k * ntt_size);

    for chunk_idx in 0..config.k {
        // Data elements
        for elem_idx in 0..elements_per_chunk {
            let val = ((chunk_idx * elements_per_chunk + elem_idx) % 1000) as u64;
            all_data.push(BabyBear::new(val));
        }
        // Padding to ntt_size
        for _ in elements_per_chunk..ntt_size {
            all_data.push(BabyBear::zero());
        }
    }

    let omega = BabyBear::get_root_of_unity(ntt_size.trailing_zeros());

    let start = Instant::now();

    // Process all chunks
    for chunk_idx in 0..config.k {
        let chunk_start = chunk_idx * ntt_size;
        let chunk_end = chunk_start + ntt_size;
        let mut chunk = all_data[chunk_start..chunk_end].to_vec();

        // RS encoding: INTT to get coefficients, then NTT to evaluate at n points
        cpu_intt(&mut chunk, omega);
        cpu_ntt(&mut chunk, omega);

        all_data[chunk_start..chunk_end].copy_from_slice(&chunk);
    }

    (start.elapsed().as_micros(), elements_per_chunk, ntt_size)
}

#[cfg(feature = "cuda")]
fn benchmark_gpu(config: &BenchmarkConfig) -> Result<(u128, usize, usize), String> {
    let (elements_per_chunk, ntt_size, _) = calculate_params(config);

    // Create data as raw u64
    let total_elements = config.k * ntt_size;
    let mut all_data: Vec<u64> = Vec::with_capacity(total_elements);

    for chunk_idx in 0..config.k {
        for elem_idx in 0..elements_per_chunk {
            let val = ((chunk_idx * elements_per_chunk + elem_idx) % 1000) as u64;
            all_data.push(val);
        }
        for _ in elements_per_chunk..ntt_size {
            all_data.push(0);
        }
    }

    let omega = BabyBear::get_root_of_unity(ntt_size.trailing_zeros());

    // Single GPU allocation
    let mut d_buffer = CudaBuffer::new(total_elements)?;

    let start = Instant::now();

    // Single transfer to GPU
    d_buffer.copy_from_host(&all_data)?;

    // Process all chunks on GPU
    unsafe {
        for chunk_idx in 0..config.k {
            let chunk_ptr = d_buffer.as_ptr().add(chunk_idx * ntt_size);
            cuda_intt(chunk_ptr, ntt_size as u32, omega.value);
            cuda_ntt(chunk_ptr, ntt_size as u32, omega.value);
        }
    }

    // Single transfer from GPU
    d_buffer.copy_to_host(&mut all_data)?;

    Ok((start.elapsed().as_micros(), elements_per_chunk, ntt_size))
}

fn run_config(config: BenchmarkConfig) -> Option<BenchmarkResult> {
    let (elements_per_chunk, ntt_size, total_vram_mb) = calculate_params(&config);

    println!("  Elements/chunk: {}, NTT size: {}, VRAM: {:.1} MB",
             elements_per_chunk, ntt_size, total_vram_mb);

    // Skip if too much VRAM
    if total_vram_mb > 256.0 {
        println!("  ⚠️  Skipping - needs {:.1} MB VRAM", total_vram_mb);
        return None;
    }

    // Skip if elements_per_chunk is 0 or too small
    if elements_per_chunk < 16 {
        println!("  ⚠️  Skipping - chunk too small ({} elements)", elements_per_chunk);
        return None;
    }

    // Skip if elements_per_chunk > ntt_size (can't fit in NTT)
    if elements_per_chunk > ntt_size {
        println!("  ⚠️  Skipping - chunk ({}) > NTT size ({})", elements_per_chunk, ntt_size);
        return None;
    }

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

        let data_size_bytes = config.data_size_kb * 1024;

        // Warm-up
        let _ = benchmark_cpu(&config);

        // CPU
        print!("    CPU... ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        let (cpu_time_us, _, _) = benchmark_cpu(&config);
        println!("{}µs", cpu_time_us);

        // GPU
        print!("    GPU... ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        match benchmark_gpu(&config) {
            Ok((gpu_time_us, _, _)) => {
                println!("{}µs", gpu_time_us);

                let cpu_time_s = cpu_time_us as f64 / 1_000_000.0;
                let gpu_time_s = gpu_time_us as f64 / 1_000_000.0;
                let data_size_mb = data_size_bytes as f64 / (1024.0 * 1024.0);

                Some(BenchmarkResult {
                    config,
                    cpu_time_us,
                    gpu_time_us,
                    cpu_throughput_mbs: data_size_mb / cpu_time_s,
                    gpu_throughput_mbs: data_size_mb / gpu_time_s,
                    speedup: cpu_time_us as f64 / gpu_time_us as f64,
                    total_vram_mb,
                    ntt_size,
                    elements_per_chunk,
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
fn benchmark_zoda_fixed() {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║   ZODA Fixed Benchmark - Correct Memory Calculations      ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    #[cfg(not(feature = "cuda"))]
    {
        println!("❌ CUDA not compiled in.");
        return;
    }

    #[cfg(feature = "cuda")]
    {
        if !cuda_available() {
            println!("❌ CUDA not available.");
            return;
        }

        println!("✅ CUDA available!\n");
        println!("Memory calculation: data_size / 4 bytes per element / k = elements_per_chunk");
        println!("NTT size = next_power_of_two(n)\n");

        // Configs that should work on mobile GPUs
        let configs = vec![
            // 128KB
            BenchmarkConfig { data_size_kb: 128, k: 1024, n: 1024 },
            BenchmarkConfig { data_size_kb: 128, k: 1024, n: 3072 },
            BenchmarkConfig { data_size_kb: 128, k: 4096, n: 4096 },

            // 1MB
            BenchmarkConfig { data_size_kb: 1024, k: 1024, n: 1024 },
            BenchmarkConfig { data_size_kb: 1024, k: 1024, n: 3072 },
            BenchmarkConfig { data_size_kb: 1024, k: 4096, n: 4096 },

            // 4MB
            BenchmarkConfig { data_size_kb: 4096, k: 1024, n: 1024 },
            BenchmarkConfig { data_size_kb: 4096, k: 1024, n: 3072 },

            // 8MB
            BenchmarkConfig { data_size_kb: 8192, k: 1024, n: 1024 },
            BenchmarkConfig { data_size_kb: 8192, k: 1024, n: 3072 },
        ];

        let mut results = Vec::new();

        for config in configs {
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("Config: {}KB, k={}, n={}", config.data_size_kb, config.k, config.n);
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

            if let Some(result) = run_config(config) {
                results.push(result);
            }
            println!();
        }

        // Summary
        println!("\n╔════════════════════════════════════════════════════════════╗");
        println!("║                    Results Summary                         ║");
        println!("╚════════════════════════════════════════════════════════════╝\n");

        println!("{:<8} {:<6} {:<6} {:<6} {:<8} │ {:<10} {:<10} │ {:<8}",
            "Size", "k", "n", "NTT", "VRAM", "CPU MB/s", "GPU MB/s", "Speedup");
        println!("{}", "─".repeat(90));

        for r in &results {
            println!("{:<8} {:<6} {:<6} {:<6} {:<8.1} │ {:<10.2} {:<10.2} │ {:<8.2}x",
                format!("{}KB", r.config.data_size_kb),
                r.config.k,
                r.config.n,
                r.ntt_size,
                r.total_vram_mb,
                r.cpu_throughput_mbs,
                r.gpu_throughput_mbs,
                r.speedup,
            );
        }

        if !results.is_empty() {
            let avg_speedup: f64 = results.iter().map(|r| r.speedup).sum::<f64>() / results.len() as f64;
            let avg_gpu: f64 = results.iter().map(|r| r.gpu_throughput_mbs).sum::<f64>() / results.len() as f64;
            let max_gpu = results.iter().map(|r| r.gpu_throughput_mbs).fold(0.0f64, f64::max);

            println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("Average Speedup:    {:.2}x", avg_speedup);
            println!("Average GPU:        {:.2} MB/s", avg_gpu);
            println!("Peak GPU:           {:.2} MB/s", max_gpu);

            if avg_speedup < 1.0 {
                println!("\n⚠️  GPU is SLOWER than CPU!");
                println!("   This is likely due to:");
                println!("   1. Small NTT sizes (kernel launch overhead dominates)");
                println!("   2. Too many small chunks (k is too large for the data size)");
                println!("   3. Memory transfer overhead for small operations");
                println!("\n   Recommendations:");
                println!("   - Use larger data sizes (4MB+)");
                println!("   - Use smaller k values for small data");
                println!("   - Consider CPU for small workloads");
            }
        }

        println!("\n╔════════════════════════════════════════════════════════════╗");
        println!("║  Benchmark Complete! ✅                                    ║");
        println!("╚════════════════════════════════════════════════════════════╝\n");
    }
}
