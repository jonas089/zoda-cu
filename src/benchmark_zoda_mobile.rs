// Mobile GPU friendly benchmark - reduced memory usage for GPUs with limited VRAM
// Designed for RTX 3060 mobile and similar GPUs

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
    vram_mb: f64,
}

#[cfg(feature = "cuda")]
extern "C" {
    fn cuda_ntt(d_values: *mut u64, n: u32, omega: u64);
    fn cuda_intt(d_values: *mut u64, n: u32, omega: u64);
}

fn estimate_vram_mb(config: &BenchmarkConfig) -> f64 {
    let ntt_size = config.n.next_power_of_two();
    let total_elements = config.k * ntt_size;
    let bytes = total_elements * 8; // 8 bytes per u64
    bytes as f64 / (1024.0 * 1024.0)
}

fn benchmark_cpu(config: &BenchmarkConfig) -> u128 {
    let data_size_bytes = config.data_size_kb * 1024;
    let chunk_size = data_size_bytes / config.k;
    let ntt_size = config.n.next_power_of_two();

    let total_elements = config.k * ntt_size;
    let mut all_data: Vec<BabyBear> = Vec::with_capacity(total_elements);

    for chunk_idx in 0..config.k {
        for elem_idx in 0..chunk_size {
            all_data.push(BabyBear::new(((chunk_idx * chunk_size + elem_idx) % 1000) as u64));
        }
        for _ in 0..(ntt_size - chunk_size) {
            all_data.push(BabyBear::zero());
        }
    }

    let omega = BabyBear::get_root_of_unity(ntt_size.trailing_zeros());

    let start = Instant::now();

    for chunk_idx in 0..config.k {
        let chunk_start = chunk_idx * ntt_size;
        let chunk_end = chunk_start + ntt_size;
        let mut chunk = all_data[chunk_start..chunk_end].to_vec();

        cpu_intt(&mut chunk, omega);
        cpu_ntt(&mut chunk, omega);

        all_data[chunk_start..chunk_end].copy_from_slice(&chunk);
    }

    start.elapsed().as_micros()
}

#[cfg(feature = "cuda")]
fn benchmark_gpu_mobile(config: &BenchmarkConfig) -> Result<u128, String> {
    let data_size_bytes = config.data_size_kb * 1024;
    let chunk_size = data_size_bytes / config.k;
    let ntt_size = config.n.next_power_of_two();

    let total_elements = config.k * ntt_size;
    let mut all_data_raw: Vec<u64> = Vec::with_capacity(total_elements);

    for chunk_idx in 0..config.k {
        for elem_idx in 0..chunk_size {
            all_data_raw.push(((chunk_idx * chunk_size + elem_idx) % 1000) as u64);
        }
        for _ in 0..(ntt_size - chunk_size) {
            all_data_raw.push(0);
        }
    }

    let omega = BabyBear::get_root_of_unity(ntt_size.trailing_zeros());

    // Allocate GPU memory
    let mut d_buffer = CudaBuffer::new(total_elements)?;

    let start = Instant::now();

    // Single transfer TO GPU
    d_buffer.copy_from_host(&all_data_raw)?;

    // Process all chunks on GPU
    unsafe {
        for chunk_idx in 0..config.k {
            let chunk_offset = chunk_idx * ntt_size;
            let chunk_ptr = d_buffer.as_ptr().add(chunk_offset);

            cuda_intt(chunk_ptr, ntt_size as u32, omega.value);
            cuda_ntt(chunk_ptr, ntt_size as u32, omega.value);
        }
    }

    // Single transfer FROM GPU
    d_buffer.copy_to_host(&mut all_data_raw)?;

    let elapsed = start.elapsed().as_micros();

    Ok(elapsed)
}

fn run_benchmark_config(config: BenchmarkConfig) -> Option<BenchmarkResult> {
    #[cfg(not(feature = "cuda"))]
    {
        println!("  CUDA not available, skipping GPU test");
        return None;
    }

    #[cfg(feature = "cuda")]
    {
        if !cuda_available() {
            println!("  CUDA not available, skipping GPU test");
            return None;
        }

        let data_size_bytes = config.data_size_kb * 1024;
        let vram_mb = estimate_vram_mb(&config);

        // Skip if too much VRAM needed
        if vram_mb > 500.0 {
            println!("  ⚠️  Skipping (needs {:.1} MB VRAM - too large for mobile GPU)", vram_mb);
            return None;
        }

        println!("  Estimated VRAM: {:.1} MB", vram_mb);

        // Warm-up
        let _ = benchmark_cpu(&config);

        // CPU benchmark
        print!("    CPU... ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        let cpu_time_us = benchmark_cpu(&config);
        println!("{}µs", cpu_time_us);

        // GPU benchmark
        print!("    GPU... ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        let gpu_result = benchmark_gpu_mobile(&config);

        match gpu_result {
            Ok(gpu_time_us) => {
                println!("{}µs", gpu_time_us);

                let cpu_time_s = cpu_time_us as f64 / 1_000_000.0;
                let gpu_time_s = gpu_time_us as f64 / 1_000_000.0;
                let data_size_mb = data_size_bytes as f64 / (1024.0 * 1024.0);

                let cpu_throughput_mbs = data_size_mb / cpu_time_s;
                let gpu_throughput_mbs = data_size_mb / gpu_time_s;

                let speedup = cpu_time_us as f64 / gpu_time_us as f64;

                Some(BenchmarkResult {
                    config,
                    cpu_time_us,
                    gpu_time_us,
                    cpu_throughput_mbs,
                    gpu_throughput_mbs,
                    speedup,
                    vram_mb,
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
fn benchmark_zoda_mobile() {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║   ZODA Mobile GPU Benchmark - RTX 3060 Friendly           ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    #[cfg(not(feature = "cuda"))]
    {
        println!("❌ CUDA support not compiled in.");
        return;
    }

    #[cfg(feature = "cuda")]
    {
        if !cuda_available() {
            println!("❌ CUDA not available on this system.");
            return;
        }

        println!("✅ CUDA available - GPU detected!\n");
        println!("NOTE: Configurations limited to <500 MB VRAM for mobile GPUs.\n");

        // Mobile-friendly configs (smaller VRAM usage)
        let configs = vec![
            // 128KB - all configs should work
            BenchmarkConfig { data_size_kb: 128, k: 1024, n: 1024 },
            BenchmarkConfig { data_size_kb: 128, k: 1024, n: 3072 },
            BenchmarkConfig { data_size_kb: 128, k: 4096, n: 4096 },
            BenchmarkConfig { data_size_kb: 128, k: 4096, n: 12288 },

            // 1MB - should all work
            BenchmarkConfig { data_size_kb: 1024, k: 1024, n: 1024 },
            BenchmarkConfig { data_size_kb: 1024, k: 1024, n: 3072 },
            BenchmarkConfig { data_size_kb: 1024, k: 4096, n: 4096 },
            BenchmarkConfig { data_size_kb: 1024, k: 4096, n: 12288 },

            // 2MB - testing middle ground
            BenchmarkConfig { data_size_kb: 2048, k: 1024, n: 1024 },
            BenchmarkConfig { data_size_kb: 2048, k: 1024, n: 3072 },
            BenchmarkConfig { data_size_kb: 2048, k: 4096, n: 4096 },

            // 4MB - smaller n values only
            BenchmarkConfig { data_size_kb: 4096, k: 1024, n: 1024 },
            BenchmarkConfig { data_size_kb: 4096, k: 1024, n: 3072 },
            // Skip 4MB with n=12288 (too much VRAM)
        ];

        let mut results = Vec::new();

        for config in configs {
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("Config: {}KB, k={}, n={} (NTT size={})",
                config.data_size_kb, config.k, config.n, config.n.next_power_of_two());
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

            if let Some(result) = run_benchmark_config(config) {
                results.push(result);
            }
            println!();
        }

        // Print summary
        println!("\n╔════════════════════════════════════════════════════════════╗");
        println!("║                    Results Summary                         ║");
        println!("╚════════════════════════════════════════════════════════════╝\n");

        println!("{:<8} {:<6} {:<6} │ {:<10} │ {:<12} {:<12} │ {:<8}",
            "Size", "k", "n", "VRAM MB", "CPU MB/s", "GPU MB/s", "Speedup");
        println!("{}", "─".repeat(90));

        for result in &results {
            println!("{:<8} {:<6} {:<6} │ {:<10.1} │ {:<12.2} {:<12.2} │ {:<8.2}x",
                format!("{}KB", result.config.data_size_kb),
                result.config.k,
                result.config.n,
                result.vram_mb,
                result.cpu_throughput_mbs,
                result.gpu_throughput_mbs,
                result.speedup,
            );
        }

        if !results.is_empty() {
            println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("Summary Statistics");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

            let avg_speedup: f64 = results.iter().map(|r| r.speedup).sum::<f64>() / results.len() as f64;
            let avg_gpu_throughput: f64 = results.iter().map(|r| r.gpu_throughput_mbs).sum::<f64>() / results.len() as f64;
            let max_gpu_throughput = results.iter().map(|r| r.gpu_throughput_mbs).fold(0.0f64, f64::max);

            println!("Configurations tested:   {}", results.len());
            println!("Average GPU Speedup:     {:.2}x", avg_speedup);
            println!("Average GPU Throughput:  {:.2} MB/s", avg_gpu_throughput);
            println!("Peak GPU Throughput:     {:.2} MB/s", max_gpu_throughput);

            println!("\n📊 RTX 3060 Mobile Performance:");
            if avg_gpu_throughput > 150.0 {
                println!("   ✅ Excellent performance for a mobile GPU!");
                println!("   Your 3060 mobile is delivering great results.");
            } else if avg_gpu_throughput > 80.0 {
                println!("   ✓ Good mobile GPU performance");
                println!("   Competitive with optimized CPU implementations.");
            } else {
                println!("   Results look reasonable for a mobile GPU.");
                println!("   Larger configs would show better GPU advantage.");
            }

            println!("\n💡 Note: Desktop RTX 5090 would be 2-4x faster and handle");
            println!("   larger configs (4MB+, 8MB+) that exceed mobile VRAM.");
        }

        println!("\n╔════════════════════════════════════════════════════════════╗");
        println!("║  Benchmark Complete! ✅                                    ║");
        println!("╚════════════════════════════════════════════════════════════╝\n");
    }
}
