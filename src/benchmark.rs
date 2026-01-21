// Automated GPU vs CPU benchmark
// Run with: cargo test --release benchmark_gpu_vs_cpu -- --nocapture --ignored

use crate::babybear::BabyBear;
use crate::ntt_babybear::{intt as cpu_intt, ntt as cpu_ntt};
use std::time::Instant;

#[cfg(feature = "cuda")]
use crate::cuda_ntt::{cuda_available, intt_cuda, ntt_cuda};

#[derive(Debug)]
struct BenchmarkResult {
    size: usize,
    cpu_ntt_us: u64,
    gpu_ntt_us: u64,
    speedup: f64,
}

fn benchmark_ntt_size(size: usize) -> Option<BenchmarkResult> {
    #[cfg(not(feature = "cuda"))]
    {
        println!("  Size {}: CUDA not compiled in", size);
        return None;
    }

    #[cfg(feature = "cuda")]
    {
        if !cuda_available() {
            println!("  Size {}: CUDA not available", size);
            return None;
        }

        // Generate test data
        let data: Vec<BabyBear> = (0..size)
            .map(|i| BabyBear::new((i as u64 * 7 + 3) % 1000000))
            .collect();

        let omega = BabyBear::get_root_of_unity(size.trailing_zeros());

        // Benchmark CPU NTT
        let mut cpu_values = data.clone();
        let cpu_start = Instant::now();
        cpu_ntt(&mut cpu_values, omega);
        let cpu_time = cpu_start.elapsed();

        // Benchmark GPU NTT
        let mut gpu_values = data.clone();
        let gpu_start = Instant::now();
        match ntt_cuda(&mut gpu_values) {
            Ok(_) => {
                let gpu_time = gpu_start.elapsed();

                // Verify results match
                for (i, (cpu_val, gpu_val)) in
                    cpu_values.iter().zip(gpu_values.iter()).enumerate()
                {
                    if cpu_val.value != gpu_val.value {
                        println!(
                            "  Size {}: MISMATCH at index {}: CPU={}, GPU={}",
                            size, i, cpu_val.value, gpu_val.value
                        );
                        return None;
                    }
                }

                let cpu_us = cpu_time.as_micros() as u64;
                let gpu_us = gpu_time.as_micros() as u64;
                let speedup = cpu_us as f64 / gpu_us as f64;

                Some(BenchmarkResult {
                    size,
                    cpu_ntt_us: cpu_us,
                    gpu_ntt_us: gpu_us,
                    speedup,
                })
            }
            Err(e) => {
                println!("  Size {}: GPU error: {}", size, e);
                None
            }
        }
    }
}

fn benchmark_zoda_size(size: usize) -> Option<(u128, u128, f64)> {
    #[cfg(not(feature = "cuda"))]
    {
        return None;
    }

    #[cfg(feature = "cuda")]
    {
        use crate::zoda_babybear::run_zoda_test_babybear;

        if !cuda_available() {
            return None;
        }

        // Run CPU version
        let cpu_duration = run_zoda_test_babybear(size, false);
        let cpu_us = cpu_duration.as_micros();

        // Run GPU version
        let gpu_duration = run_zoda_test_babybear(size, true);
        let gpu_us = gpu_duration.as_micros();

        let speedup = cpu_us as f64 / gpu_us as f64;

        Some((cpu_us, gpu_us, speedup))
    }
}

#[test]
#[ignore] // Use --ignored to run this benchmark
fn benchmark_gpu_vs_cpu() {
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║       GPU vs CPU Benchmark - ZODA Protocol                ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    #[cfg(not(feature = "cuda"))]
    {
        println!("❌ CUDA support not compiled in.");
        println!("   Build with: cargo build --release");
        println!("   Make sure nvcc is in your PATH.");
        return;
    }

    #[cfg(feature = "cuda")]
    {
        if !cuda_available() {
            println!("❌ CUDA not available on this system.");
            println!("   - Check nvidia-smi shows your GPU");
            println!("   - Verify CUDA drivers are installed");
            println!("   - Try: export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH");
            return;
        }

        println!("✅ CUDA available - GPU detected!\n");

        // Part 1: Raw NTT Performance
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Part 1: Raw NTT Performance (Forward Transform Only)");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        let ntt_sizes = vec![256, 512, 1024, 2048, 4096, 8192, 16384];
        let mut ntt_results = Vec::new();

        for size in ntt_sizes {
            print!("Testing NTT size {}... ", size);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();

            if let Some(result) = benchmark_ntt_size(size) {
                println!(
                    "CPU: {:>6}µs | GPU: {:>6}µs | Speedup: {:.2}x {}",
                    result.cpu_ntt_us,
                    result.gpu_ntt_us,
                    result.speedup,
                    if result.speedup > 1.0 { "🚀" } else { "⚠️ " }
                );
                ntt_results.push(result);
            } else {
                println!("FAILED");
            }
        }

        // Part 2: Full ZODA Protocol
        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Part 2: Full ZODA Protocol (Multiple NTT Operations)");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        let zoda_sizes = vec![4, 8, 16, 32];
        let mut zoda_results = Vec::new();

        for size in zoda_sizes {
            print!("Testing ZODA {}x{} data square... ", size, size);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();

            if let Some((cpu_us, gpu_us, speedup)) = benchmark_zoda_size(size) {
                println!(
                    "CPU: {:>6}µs | GPU: {:>6}µs | Speedup: {:.2}x {}",
                    cpu_us,
                    gpu_us,
                    speedup,
                    if speedup > 1.0 { "🚀" } else { "⚠️ " }
                );
                zoda_results.push((size, cpu_us, gpu_us, speedup));
            } else {
                println!("FAILED");
            }
        }

        // Summary
        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Summary");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        if !ntt_results.is_empty() {
            let avg_ntt_speedup: f64 =
                ntt_results.iter().map(|r| r.speedup).sum::<f64>() / ntt_results.len() as f64;
            println!("Raw NTT Average Speedup: {:.2}x", avg_ntt_speedup);

            // Find best case
            let best_ntt = ntt_results
                .iter()
                .max_by(|a, b| a.speedup.partial_cmp(&b.speedup).unwrap())
                .unwrap();
            println!(
                "  Best NTT performance: {:.2}x at size {}",
                best_ntt.speedup, best_ntt.size
            );
        }

        if !zoda_results.is_empty() {
            let avg_zoda_speedup: f64 =
                zoda_results.iter().map(|(_, _, _, s)| s).sum::<f64>() / zoda_results.len() as f64;
            println!("\nZODA Protocol Average Speedup: {:.2}x", avg_zoda_speedup);

            let best_zoda = zoda_results
                .iter()
                .max_by(|a, b| a.3.partial_cmp(&b.3).unwrap())
                .unwrap();
            println!(
                "  Best ZODA performance: {:.2}x at {}x{}",
                best_zoda.3, best_zoda.0, best_zoda.0
            );
        }

        println!("\n╔═══════════════════════════════════════════════════════════╗");
        println!("║  Benchmark Complete! ✅                                   ║");
        println!("╚═══════════════════════════════════════════════════════════╝\n");

        // Interpretation
        println!("📊 Results Interpretation:");
        if !ntt_results.is_empty() {
            let avg_speedup = ntt_results.iter().map(|r| r.speedup).sum::<f64>()
                / ntt_results.len() as f64;

            if avg_speedup > 5.0 {
                println!("   ✅ EXCELLENT: GPU is significantly faster than CPU");
                println!("      Your GPU is well-utilized for this workload.");
            } else if avg_speedup > 2.0 {
                println!("   ✓  GOOD: GPU provides solid acceleration");
                println!("      Larger data sizes will show better GPU performance.");
            } else if avg_speedup > 1.0 {
                println!("   ⚠️  MARGINAL: GPU is slightly faster");
                println!("      Small sizes have kernel launch overhead.");
                println!("      Try larger NTT sizes for better GPU utilization.");
            } else {
                println!("   ⚠️  WARNING: CPU is faster than GPU");
                println!("      This is unexpected. Possible causes:");
                println!("      - GPU thermal throttling (check temps)");
                println!("      - GPU memory bandwidth limitations");
                println!("      - System under heavy load");
            }
        }

        println!("\n💡 Tips:");
        println!("   - GPU advantage increases with larger NTT sizes (1024+)");
        println!("   - For production, batch multiple operations for best GPU utilization");
        println!("   - Check 'nvidia-smi' during benchmark to monitor GPU usage");
    }
}
