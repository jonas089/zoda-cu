// ZODA-specific benchmark with Reed-Solomon parameters
// Tests specific (data_size, k, n) configurations

use crate::babybear::BabyBear;
use crate::ntt_babybear::{intt as cpu_intt, ntt as cpu_ntt};
use std::time::Instant;

#[cfg(feature = "cuda")]
use crate::cuda_ntt::{cuda_available, intt_cuda, ntt_cuda};

#[derive(Debug)]
struct BenchmarkConfig {
    data_size_kb: usize,  // Data size in KB
    k: usize,             // Number of data chunks
    n: usize,             // Total chunks (data + parity)
}

#[derive(Debug)]
struct BenchmarkResult {
    config: BenchmarkConfig,
    cpu_time_us: u128,
    gpu_time_us: u128,
    cpu_throughput_mbs: f64,
    gpu_throughput_mbs: f64,
    cpu_ns_per_op: f64,
    gpu_ns_per_op: f64,
    speedup: f64,
}

fn benchmark_zoda_config(config: BenchmarkConfig, use_gpu: bool) -> u128 {
    let data_size_bytes = config.data_size_kb * 1024;
    let chunk_size = data_size_bytes / config.k;

    // Calculate NTT size (must be power of 2, >= n)
    let ntt_size = config.n.next_power_of_two();

    // Generate random data chunks
    let mut data_chunks: Vec<Vec<BabyBear>> = Vec::new();
    for i in 0..config.k {
        let mut chunk = Vec::new();
        for j in 0..chunk_size {
            chunk.push(BabyBear::new(((i * chunk_size + j) % 1000) as u64));
        }
        data_chunks.push(chunk);
    }

    let omega = BabyBear::get_root_of_unity(ntt_size.trailing_zeros());

    let start = Instant::now();

    // Encode each chunk using NTT (RS encoding)
    for chunk_data in data_chunks.iter() {
        // Pad to ntt_size with zeros
        let mut padded: Vec<BabyBear> = chunk_data.clone();
        padded.resize(ntt_size, BabyBear::zero());

        // INTT to get coefficients
        #[cfg(feature = "cuda")]
        if use_gpu && cuda_available() {
            let mut coeffs = padded.clone();
            intt_cuda(&mut coeffs).unwrap();

            // NTT to encode
            let mut encoded = coeffs;
            ntt_cuda(&mut encoded).unwrap();
        } else {
            let mut coeffs = padded.clone();
            cpu_intt(&mut coeffs, omega);

            let mut encoded = coeffs;
            cpu_ntt(&mut encoded, omega);
        }

        #[cfg(not(feature = "cuda"))]
        {
            let mut coeffs = padded.clone();
            cpu_intt(&mut coeffs, omega);

            let mut encoded = coeffs;
            cpu_ntt(&mut encoded, omega);
        }
    }

    start.elapsed().as_micros()
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
        let num_operations = config.k; // One encode per chunk

        // Warm-up
        benchmark_zoda_config(
            BenchmarkConfig {
                data_size_kb: config.data_size_kb,
                k: config.k,
                n: config.n,
            },
            false,
        );

        // CPU benchmark
        print!("    CPU... ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        let cpu_time_us = benchmark_zoda_config(
            BenchmarkConfig {
                data_size_kb: config.data_size_kb,
                k: config.k,
                n: config.n,
            },
            false,
        );
        println!("{}µs", cpu_time_us);

        // GPU benchmark
        print!("    GPU... ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        let gpu_time_us = benchmark_zoda_config(
            BenchmarkConfig {
                data_size_kb: config.data_size_kb,
                k: config.k,
                n: config.n,
            },
            true,
        );
        println!("{}µs", gpu_time_us);

        // Calculate metrics
        let cpu_time_s = cpu_time_us as f64 / 1_000_000.0;
        let gpu_time_s = gpu_time_us as f64 / 1_000_000.0;
        let data_size_mb = data_size_bytes as f64 / (1024.0 * 1024.0);

        let cpu_throughput_mbs = data_size_mb / cpu_time_s;
        let gpu_throughput_mbs = data_size_mb / gpu_time_s;

        let cpu_ns_per_op = (cpu_time_us as f64 * 1000.0) / num_operations as f64;
        let gpu_ns_per_op = (gpu_time_us as f64 * 1000.0) / num_operations as f64;

        let speedup = cpu_time_us as f64 / gpu_time_us as f64;

        Some(BenchmarkResult {
            config,
            cpu_time_us,
            gpu_time_us,
            cpu_throughput_mbs,
            gpu_throughput_mbs,
            cpu_ns_per_op,
            gpu_ns_per_op,
            speedup,
        })
    }
}

#[test]
#[ignore]
fn benchmark_zoda_configurations() {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║     ZODA Benchmark - Specific Configurations              ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    #[cfg(not(feature = "cuda"))]
    {
        println!("❌ CUDA support not compiled in.");
        println!("   Build with: cargo build --release");
        return;
    }

    #[cfg(feature = "cuda")]
    {
        if !cuda_available() {
            println!("❌ CUDA not available on this system.");
            return;
        }

        println!("✅ CUDA available - GPU detected!\n");

        // Define all test configurations
        let configs = vec![
            // 128KB
            BenchmarkConfig { data_size_kb: 128, k: 1024, n: 1024 },
            BenchmarkConfig { data_size_kb: 128, k: 1024, n: 3072 },
            BenchmarkConfig { data_size_kb: 128, k: 4096, n: 4096 },
            BenchmarkConfig { data_size_kb: 128, k: 4096, n: 12288 },

            // 1MB
            BenchmarkConfig { data_size_kb: 1024, k: 1024, n: 1024 },
            BenchmarkConfig { data_size_kb: 1024, k: 1024, n: 3072 },
            BenchmarkConfig { data_size_kb: 1024, k: 4096, n: 4096 },
            BenchmarkConfig { data_size_kb: 1024, k: 4096, n: 12288 },

            // 4MB
            BenchmarkConfig { data_size_kb: 4096, k: 1024, n: 1024 },
            BenchmarkConfig { data_size_kb: 4096, k: 1024, n: 3072 },
            BenchmarkConfig { data_size_kb: 4096, k: 4096, n: 4096 },
            BenchmarkConfig { data_size_kb: 4096, k: 4096, n: 12288 },

            // 8MB
            BenchmarkConfig { data_size_kb: 8192, k: 1024, n: 1024 },
            BenchmarkConfig { data_size_kb: 8192, k: 1024, n: 3072 },
            BenchmarkConfig { data_size_kb: 8192, k: 4096, n: 4096 },
            BenchmarkConfig { data_size_kb: 8192, k: 4096, n: 12288 },
        ];

        let mut results = Vec::new();

        for config in configs {
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("Configuration: {}KB, k={}, n={}",
                config.data_size_kb, config.k, config.n);
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

            if let Some(result) = run_benchmark_config(config) {
                results.push(result);
            }
            println!();
        }

        // Print summary table
        println!("\n╔════════════════════════════════════════════════════════════╗");
        println!("║                    Results Summary                         ║");
        println!("╚════════════════════════════════════════════════════════════╝\n");

        println!("{:<8} {:<6} {:<6} │ {:<12} {:<12} │ {:<12} {:<12} │ {:<8}",
            "Size", "k", "n", "CPU MB/s", "GPU MB/s", "CPU ns/op", "GPU ns/op", "Speedup");
        println!("{}", "─".repeat(120));

        for result in &results {
            println!("{:<8} {:<6} {:<6} │ {:<12.2} {:<12.2} │ {:<12.1} {:<12.1} │ {:<8.2}x",
                format!("{}KB", result.config.data_size_kb),
                result.config.k,
                result.config.n,
                result.cpu_throughput_mbs,
                result.gpu_throughput_mbs,
                result.cpu_ns_per_op,
                result.gpu_ns_per_op,
                result.speedup,
            );
        }

        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Summary Statistics");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        if !results.is_empty() {
            let avg_speedup: f64 = results.iter().map(|r| r.speedup).sum::<f64>() / results.len() as f64;
            let max_speedup = results.iter().map(|r| r.speedup).fold(0.0f64, f64::max);
            let min_speedup = results.iter().map(|r| r.speedup).fold(f64::INFINITY, f64::min);

            let avg_gpu_throughput: f64 = results.iter().map(|r| r.gpu_throughput_mbs).sum::<f64>() / results.len() as f64;
            let max_gpu_throughput = results.iter().map(|r| r.gpu_throughput_mbs).fold(0.0f64, f64::max);

            println!("Average GPU Speedup:     {:.2}x", avg_speedup);
            println!("Best GPU Speedup:        {:.2}x", max_speedup);
            println!("Worst GPU Speedup:       {:.2}x", min_speedup);
            println!();
            println!("Average GPU Throughput:  {:.2} MB/s", avg_gpu_throughput);
            println!("Peak GPU Throughput:     {:.2} MB/s", max_gpu_throughput);
        }

        println!("\n╔════════════════════════════════════════════════════════════╗");
        println!("║  Benchmark Complete! ✅                                    ║");
        println!("╚════════════════════════════════════════════════════════════╝\n");
    }
}
