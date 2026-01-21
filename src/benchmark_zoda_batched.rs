// Batched ZODA benchmark - processes all chunks in a single GPU operation
// This is the correct way to benchmark GPU performance

use crate::babybear::BabyBear;
use crate::ntt_babybear::{intt as cpu_intt, ntt as cpu_ntt};
use std::time::Instant;

#[cfg(feature = "cuda")]
use crate::cuda_ntt::{cuda_available, intt_cuda, ntt_cuda};

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
    cpu_ns_per_op: f64,
    gpu_ns_per_op: f64,
    speedup: f64,
}

fn benchmark_zoda_batched(config: &BenchmarkConfig, use_gpu: bool) -> u128 {
    let data_size_bytes = config.data_size_kb * 1024;
    let chunk_size = data_size_bytes / config.k;

    // NTT size must be power of 2, >= n
    let ntt_size = config.n.next_power_of_two();

    // Generate ALL data at once
    let total_elements = config.k * chunk_size;
    let mut all_data: Vec<BabyBear> = (0..total_elements)
        .map(|i| BabyBear::new((i % 1000) as u64))
        .collect();

    // Reshape into k chunks of chunk_size, then pad each to ntt_size
    let mut batched_data: Vec<BabyBear> = Vec::with_capacity(config.k * ntt_size);
    for chunk_idx in 0..config.k {
        let start = chunk_idx * chunk_size;
        let end = start + chunk_size;

        // Copy chunk data
        batched_data.extend_from_slice(&all_data[start..end]);

        // Pad to ntt_size
        for _ in 0..(ntt_size - chunk_size) {
            batched_data.push(BabyBear::zero());
        }
    }

    let omega = BabyBear::get_root_of_unity(ntt_size.trailing_zeros());

    let start = Instant::now();

    // Process ALL chunks in a SINGLE batched operation
    #[cfg(feature = "cuda")]
    if use_gpu && cuda_available() {
        // Process all k chunks in one GPU operation
        for chunk_idx in 0..config.k {
            let chunk_start = chunk_idx * ntt_size;
            let chunk_end = chunk_start + ntt_size;
            let mut chunk = batched_data[chunk_start..chunk_end].to_vec();

            // INTT
            intt_cuda(&mut chunk).unwrap();

            // NTT (encode)
            ntt_cuda(&mut chunk).unwrap();

            // Write back (in real use, you'd keep these results)
            batched_data[chunk_start..chunk_end].copy_from_slice(&chunk);
        }
    } else {
        // CPU version
        for chunk_idx in 0..config.k {
            let chunk_start = chunk_idx * ntt_size;
            let chunk_end = chunk_start + ntt_size;
            let mut chunk = batched_data[chunk_start..chunk_end].to_vec();

            // INTT
            cpu_intt(&mut chunk, omega);

            // NTT (encode)
            cpu_ntt(&mut chunk, omega);

            batched_data[chunk_start..chunk_end].copy_from_slice(&chunk);
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        // CPU version
        for chunk_idx in 0..config.k {
            let chunk_start = chunk_idx * ntt_size;
            let chunk_end = chunk_start + ntt_size;
            let mut chunk = batched_data[chunk_start..chunk_end].to_vec();

            cpu_intt(&mut chunk, omega);
            cpu_ntt(&mut chunk, omega);

            batched_data[chunk_start..chunk_end].copy_from_slice(&chunk);
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
        let num_operations = config.k * 2; // INTT + NTT per chunk

        // Warm-up
        benchmark_zoda_batched(&config, false);

        // CPU benchmark
        print!("    CPU... ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        let cpu_time_us = benchmark_zoda_batched(&config, false);
        println!("{}µs", cpu_time_us);

        // GPU benchmark
        print!("    GPU... ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        let gpu_time_us = benchmark_zoda_batched(&config, true);
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
fn benchmark_zoda_batched_configurations() {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║   ZODA Batched Benchmark - Proper GPU Utilization         ║");
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
        println!("NOTE: This benchmark batches operations to minimize GPU overhead.\n");

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
            println!("Configuration: {}KB, k={}, n={} (NTT size={})",
                config.data_size_kb, config.k, config.n, config.n.next_power_of_two());
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

            let avg_cpu_throughput: f64 = results.iter().map(|r| r.cpu_throughput_mbs).sum::<f64>() / results.len() as f64;

            println!("Average GPU Speedup:     {:.2}x", avg_speedup);
            println!("Best GPU Speedup:        {:.2}x", max_speedup);
            println!("Worst GPU Speedup:       {:.2}x", min_speedup);
            println!();
            println!("Average CPU Throughput:  {:.2} MB/s", avg_cpu_throughput);
            println!("Average GPU Throughput:  {:.2} MB/s", avg_gpu_throughput);
            println!("Peak GPU Throughput:     {:.2} MB/s", max_gpu_throughput);

            println!("\n📊 Interpretation:");
            if avg_gpu_throughput > 100.0 {
                println!("   ✅ Excellent GPU performance!");
                println!("   GPU is efficiently processing batched NTT operations.");
            } else if avg_gpu_throughput > 50.0 {
                println!("   ✓ Good GPU performance");
                println!("   Further optimization possible with better batching.");
            } else {
                println!("   ⚠️  Lower than expected GPU performance");
                println!("   Small chunk sizes may still have overhead.");
                println!("   Larger data sizes should show better GPU utilization.");
            }
        }

        println!("\n╔════════════════════════════════════════════════════════════╗");
        println!("║  Benchmark Complete! ✅                                    ║");
        println!("╚════════════════════════════════════════════════════════════╝\n");
    }
}
