use crate::benchmark_utils::{encode_gpu_with_output, validate_zoda_encoding, EncodingConfig};
use std::fs::File;
use std::io::Write;

#[cfg(feature = "cuda")]
use crate::cuda_ntt::cuda_available;


struct BenchmarkResult {
    config: EncodingConfig,
    encode_time_ns: u64,
    throughput_mbs: f64,
    validation_passed: bool,
}

// EigenDA EMA benchmark reference results from celestiaorg/eigenda-kzg-bench
fn get_eigenda_reference(data_mb: f64, k: usize, n: usize) -> Option<(u64, f64)> {
    let key = (data_mb, k, n);

    // (encode_time_ns, throughput_mbs)
    match key {
        // 128KB configurations
        (0.125, 1024, 1024) => Some((885_254, 148.06)),
        (0.125, 1024, 3072) => Some((1_537_837, 85.23)),

        // 1MB configurations
        (1.0, 1024, 1024) => Some((3_532_739, 296.82)),
        (1.0, 1024, 3072) => Some((4_732_775, 221.56)),
        (1.0, 4096, 4096) => Some((4_724_641, 221.94)),
        (1.0, 4096, 12288) => Some((7_131_521, 147.03)),

        // 4MB configurations
        (4.0, 1024, 1024) => Some((12_449_984, 336.89)),
        (4.0, 1024, 3072) => Some((15_774_507, 265.89)),
        (4.0, 4096, 4096) => Some((13_541_566, 309.74)),
        (4.0, 4096, 12288) => Some((22_033_068, 190.36)),

        // 8MB configurations
        (8.0, 1024, 1024) => Some((22_388_042, 374.69)),
        (8.0, 1024, 3072) => Some((35_500_902, 236.29)),
        (8.0, 4096, 4096) => Some((26_230_505, 319.80)),
        (8.0, 4096, 12288) => Some((40_912_656, 205.04)),

        _ => None,
    }
}


#[cfg(feature = "cuda")]
fn run_benchmark(config: EncodingConfig) -> Option<BenchmarkResult> {
    // Encode on GPU and measure time
    let (encoded_rows, encode_time_ns) = match encode_gpu_with_output(&config) {
        Ok(result) => result,
        Err(_) => return None,
    };

    // Validate correctness (not timed in benchmark)
    let validation_passed = match validate_zoda_encoding(&config, &encoded_rows) {
        Ok((result, _elapsed, _checks)) => result,
        Err(_) => false,
    };

    // Calculate throughput
    let data_size_mb = config.data_size_mb();
    let throughput_mbs = data_size_mb / (encode_time_ns as f64 / 1_000_000_000.0);

    Some(BenchmarkResult {
        config,
        encode_time_ns,
        throughput_mbs,
        validation_passed,
    })
}

#[test]
#[ignore]
fn benchmark_zoda_eigenda_comparison() {
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

        println!("Performance Comparison: ZODA Encoding vs EigenDA KZG\n");

        // EigenDA-style configurations from the benchmark table
        // Extended to 1GB data size
        // Data size = k * row_size
        let configs = vec![
            // 128KB configurations
            EncodingConfig {
                k: 1024,
                n: 1024,
                row_size: 128,
            }, // 128KB, k=1024, n=1024
            EncodingConfig {
                k: 1024,
                n: 3072,
                row_size: 128,
            }, // 128KB, k=1024, n=3072
            EncodingConfig {
                k: 4096,
                n: 4096,
                row_size: 32,
            }, // 128KB, k=4096, n=4096
            EncodingConfig {
                k: 4096,
                n: 12288,
                row_size: 32,
            }, // 128KB, k=4096, n=12288
            // 1MB configurations
            EncodingConfig {
                k: 1024,
                n: 1024,
                row_size: 1024,
            }, // 1MB, k=1024, n=1024
            EncodingConfig {
                k: 1024,
                n: 3072,
                row_size: 1024,
            }, // 1MB, k=1024, n=3072
            EncodingConfig {
                k: 4096,
                n: 4096,
                row_size: 256,
            }, // 1MB, k=4096, n=4096
            EncodingConfig {
                k: 4096,
                n: 12288,
                row_size: 256,
            }, // 1MB, k=4096, n=12288
            // 4MB configurations
            EncodingConfig {
                k: 1024,
                n: 1024,
                row_size: 4096,
            }, // 4MB, k=1024, n=1024
            EncodingConfig {
                k: 1024,
                n: 3072,
                row_size: 4096,
            }, // 4MB, k=1024, n=3072
            EncodingConfig {
                k: 4096,
                n: 4096,
                row_size: 1024,
            }, // 4MB, k=4096, n=4096
            EncodingConfig {
                k: 4096,
                n: 12288,
                row_size: 1024,
            }, // 4MB, k=4096, n=12288
            // 8MB configurations
            EncodingConfig {
                k: 1024,
                n: 1024,
                row_size: 8192,
            }, // 8MB, k=1024, n=1024
            EncodingConfig {
                k: 1024,
                n: 3072,
                row_size: 8192,
            }, // 8MB, k=1024, n=3072
            EncodingConfig {
                k: 4096,
                n: 4096,
                row_size: 2048,
            }, // 8MB, k=4096, n=4096
            EncodingConfig {
                k: 4096,
                n: 12288,
                row_size: 2048,
            }, // 8MB, k=4096, n=12288
            // 16MB configurations
            EncodingConfig {
                k: 1024,
                n: 1024,
                row_size: 16384,
            }, // 16MB, k=1024, n=1024
            EncodingConfig {
                k: 1024,
                n: 3072,
                row_size: 16384,
            }, // 16MB, k=1024, n=3072
            EncodingConfig {
                k: 4096,
                n: 4096,
                row_size: 4096,
            }, // 16MB, k=4096, n=4096
            EncodingConfig {
                k: 4096,
                n: 12288,
                row_size: 4096,
            }, // 16MB, k=4096, n=12288
            // 32MB configurations
            EncodingConfig {
                k: 1024,
                n: 1024,
                row_size: 32768,
            }, // 32MB, k=1024, n=1024
            EncodingConfig {
                k: 1024,
                n: 3072,
                row_size: 32768,
            }, // 32MB, k=1024, n=3072
            EncodingConfig {
                k: 4096,
                n: 4096,
                row_size: 8192,
            }, // 32MB, k=4096, n=4096
            EncodingConfig {
                k: 4096,
                n: 12288,
                row_size: 8192,
            }, // 32MB, k=4096, n=12288
            // 64MB configurations
            EncodingConfig {
                k: 1024,
                n: 1024,
                row_size: 65536,
            }, // 64MB, k=1024, n=1024
            EncodingConfig {
                k: 1024,
                n: 3072,
                row_size: 65536,
            }, // 64MB, k=1024, n=3072
            EncodingConfig {
                k: 4096,
                n: 4096,
                row_size: 16384,
            }, // 64MB, k=4096, n=4096
            EncodingConfig {
                k: 4096,
                n: 12288,
                row_size: 16384,
            }, // 64MB, k=4096, n=12288
            // 128MB configurations
            EncodingConfig {
                k: 1024,
                n: 1024,
                row_size: 131072,
            }, // 128MB, k=1024, n=1024
            EncodingConfig {
                k: 1024,
                n: 3072,
                row_size: 131072,
            }, // 128MB, k=1024, n=3072
            EncodingConfig {
                k: 4096,
                n: 4096,
                row_size: 32768,
            }, // 128MB, k=4096, n=4096
            EncodingConfig {
                k: 4096,
                n: 12288,
                row_size: 32768,
            }, // 128MB, k=4096, n=12288
            // 256MB configurations
            EncodingConfig {
                k: 1024,
                n: 1024,
                row_size: 262144,
            }, // 256MB, k=1024, n=1024
            EncodingConfig {
                k: 1024,
                n: 3072,
                row_size: 262144,
            }, // 256MB, k=1024, n=3072
            EncodingConfig {
                k: 4096,
                n: 4096,
                row_size: 65536,
            }, // 256MB, k=4096, n=4096
            EncodingConfig {
                k: 4096,
                n: 12288,
                row_size: 65536,
            }, // 256MB, k=4096, n=12288
            // 512MB configurations
            EncodingConfig {
                k: 1024,
                n: 1024,
                row_size: 524288,
            }, // 512MB, k=1024, n=1024
            EncodingConfig {
                k: 1024,
                n: 3072,
                row_size: 524288,
            }, // 512MB, k=1024, n=3072
            EncodingConfig {
                k: 4096,
                n: 4096,
                row_size: 131072,
            }, // 512MB, k=4096, n=4096
            EncodingConfig {
                k: 4096,
                n: 12288,
                row_size: 131072,
            }, // 512MB, k=4096, n=12288
            // 1GB configurations
            EncodingConfig {
                k: 1024,
                n: 1024,
                row_size: 1048576,
            }, // 1GB, k=1024, n=1024
            EncodingConfig {
                k: 1024,
                n: 3072,
                row_size: 1048576,
            }, // 1GB, k=1024, n=3072
            EncodingConfig {
                k: 4096,
                n: 4096,
                row_size: 262144,
            }, // 1GB, k=4096, n=4096
            EncodingConfig {
                k: 4096,
                n: 12288,
                row_size: 262144,
            }, // 1GB, k=4096, n=12288
        ];

        let mut results = Vec::new();

        println!("Running {} benchmark configurations...\n", configs.len());

        for (idx, config) in configs.iter().enumerate() {
            print!("Progress: {}/{} - ", idx + 1, configs.len());
            if config.data_size_mb() >= 1.0 {
                print!("{:.0}MB ", config.data_size_mb());
            } else {
                print!("{:.0}KB ", config.data_size_kb());
            }
            print!("(k={}, n={})... ", config.k, config.n);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();

            if let Some(result) = run_benchmark(config.clone()) {
                println!("✓ {:.2} MB/s", result.throughput_mbs);
                results.push(result);
            } else {
                println!("✗ FAILED");
            }
        }

        // Write results to files
        println!("\nWriting results to files...");

        // Write Markdown table
        if let Ok(mut md_file) = File::create("zoda_eigenda_benchmark.md") {
            writeln!(md_file, "# ZODA Encoding Benchmark Results\n").ok();
            writeln!(
                md_file,
                "## Performance Comparison: ZODA vs EigenDA EMA Encoding\n"
            )
            .ok();
            writeln!(md_file, "| Configuration | K | N | ZODA (ns/op) | ZODA (MB/s) | EigenDA EMA (ns/op) | EigenDA EMA (MB/s) | Speedup | Status |").ok();
            writeln!(md_file, "|---------------|---|---|--------------|-------------|---------------------|--------------------|---------|---------| ").ok();

            for result in &results {
                let data_label = if result.config.data_size_kb() < 1024.0 {
                    format!("{:.0}KB", result.config.data_size_kb())
                } else {
                    format!("{:.0}MB", result.config.data_size_mb())
                };
                let status = if result.validation_passed {
                    "✓ PASS"
                } else {
                    "✗ FAIL"
                };

                let (eigenda_time, eigenda_throughput, speedup) =
                    if let Some((ref_time, ref_throughput)) = get_eigenda_reference(
                        result.config.data_size_mb(),
                        result.config.k,
                        result.config.n,
                    ) {
                        let speedup_val = ref_time as f64 / result.encode_time_ns as f64;
                        (
                            format!("{}", ref_time),
                            format!("{:.2}", ref_throughput),
                            format!("{:.2}x", speedup_val),
                        )
                    } else {
                        ("-".to_string(), "-".to_string(), "-".to_string())
                    };

                writeln!(
                    md_file,
                    "| {} | {} | {} | {} | {:.2} | {} | {} | {} | {} |",
                    data_label,
                    result.config.k,
                    result.config.n,
                    result.encode_time_ns,
                    result.throughput_mbs,
                    eigenda_time,
                    eigenda_throughput,
                    speedup,
                    status
                )
                .ok();
            }

            // Add statistics section
            writeln!(md_file, "\n## Statistics by Data Size\n").ok();
            writeln!(
                md_file,
                "| Data Size | Avg MB/s | Min MB/s | Max MB/s | Configs |"
            )
            .ok();
            writeln!(
                md_file,
                "|-----------|----------|----------|----------|---------|"
            )
            .ok();

            let data_sizes = vec![
                (128.0, "128KB"),
                (1024.0, "1MB"),
                (4096.0, "4MB"),
                (8192.0, "8MB"),
                (16384.0, "16MB"),
                (32768.0, "32MB"),
                (65536.0, "64MB"),
                (131072.0, "128MB"),
                (262144.0, "256MB"),
                (524288.0, "512MB"),
                (1048576.0, "1GB"),
            ];

            for (size_kb, label) in data_sizes {
                let size_results: Vec<_> = results
                    .iter()
                    .filter(|r| (r.config.data_size_kb() - size_kb).abs() < 1.0)
                    .collect();

                if !size_results.is_empty() {
                    let avg_throughput = size_results.iter().map(|r| r.throughput_mbs).sum::<f64>()
                        / size_results.len() as f64;
                    let max_throughput = size_results
                        .iter()
                        .map(|r| r.throughput_mbs)
                        .fold(0.0f64, f64::max);
                    let min_throughput = size_results
                        .iter()
                        .map(|r| r.throughput_mbs)
                        .fold(f64::INFINITY, f64::min);

                    writeln!(
                        md_file,
                        "| {} | {:.2} | {:.2} | {:.2} | {} |",
                        label,
                        avg_throughput,
                        min_throughput,
                        max_throughput,
                        size_results.len()
                    )
                    .ok();
                }
            }

            // Overall statistics
            if !results.is_empty() {
                let avg_throughput =
                    results.iter().map(|r| r.throughput_mbs).sum::<f64>() / results.len() as f64;
                let max_throughput = results
                    .iter()
                    .map(|r| r.throughput_mbs)
                    .fold(0.0f64, f64::max);
                let passed = results.iter().filter(|r| r.validation_passed).count();

                writeln!(md_file, "\n## Overall Statistics\n").ok();
                writeln!(
                    md_file,
                    "- **Average Throughput**: {:.2} MB/s",
                    avg_throughput
                )
                .ok();
                writeln!(md_file, "- **Peak Throughput**: {:.2} MB/s", max_throughput).ok();
                writeln!(
                    md_file,
                    "- **Validation Success**: {}/{}",
                    passed,
                    results.len()
                )
                .ok();
                writeln!(md_file, "- **Total Configurations**: {}", results.len()).ok();

                // Add speedup comparison
                let mut speedups = Vec::new();
                for result in &results {
                    if let Some((ref_time, _)) = get_eigenda_reference(
                        result.config.data_size_mb(),
                        result.config.k,
                        result.config.n,
                    ) {
                        let speedup = ref_time as f64 / result.encode_time_ns as f64;
                        speedups.push(speedup);
                    }
                }

                if !speedups.is_empty() {
                    let avg_speedup = speedups.iter().sum::<f64>() / speedups.len() as f64;
                    let min_speedup = speedups.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    let max_speedup = speedups.iter().fold(0.0f64, |a, &b| a.max(b));

                    writeln!(md_file, "\n## Speedup vs EigenDA EMA\n").ok();
                    writeln!(
                        md_file,
                        "Based on {} comparable configurations:\n",
                        speedups.len()
                    )
                    .ok();
                    writeln!(md_file, "- **Average Speedup**: {:.2}x", avg_speedup).ok();
                    writeln!(md_file, "- **Minimum Speedup**: {:.2}x", min_speedup).ok();
                    writeln!(md_file, "- **Maximum Speedup**: {:.2}x", max_speedup).ok();
                }
            }

            println!("  ✓ Markdown results written to: zoda_eigenda_benchmark.md");
        }

        // Print results table matching EigenDA format
        println!("\n{}", "═".repeat(130));
        println!("Performance Comparison: ZODA vs EigenDA EMA Encoding");
        println!("{}", "═".repeat(130));
        println!(
            "{:<15} {:<6} {:<6} │ {:<12} {:<10} │ {:<12} {:<10} │ {:<10} {:<10}",
            "Config",
            "k",
            "n",
            "ZODA (ns)",
            "ZODA MB/s",
            "EMA (ns)",
            "EMA MB/s",
            "Speedup",
            "Status"
        );
        println!("{}", "─".repeat(130));

        let mut all_passed = true;
        for result in &results {
            let status = if result.validation_passed {
                "✓ PASS"
            } else {
                all_passed = false;
                "✗ FAIL"
            };

            let data_label = if result.config.data_size_kb() < 1024.0 {
                format!("{:.0}KB", result.config.data_size_kb())
            } else {
                format!("{:.0}MB", result.config.data_size_mb())
            };

            let (eigenda_time_str, eigenda_throughput_str, speedup_str) =
                if let Some((ref_time, ref_throughput)) = get_eigenda_reference(
                    result.config.data_size_mb(),
                    result.config.k,
                    result.config.n,
                ) {
                    let speedup_val = ref_time as f64 / result.encode_time_ns as f64;
                    (
                        format!("{}", ref_time),
                        format!("{:.2}", ref_throughput),
                        format!("{:.2}x", speedup_val),
                    )
                } else {
                    ("-".to_string(), "-".to_string(), "-".to_string())
                };

            println!(
                "{:<15} {:<6} {:<6} │ {:<12} {:<10.2} │ {:<12} {:<10} │ {:<10} {}",
                data_label,
                result.config.k,
                result.config.n,
                result.encode_time_ns,
                result.throughput_mbs,
                eigenda_time_str,
                eigenda_throughput_str,
                speedup_str,
                status,
            );
        }

        println!("{}", "═".repeat(130));

        if all_passed {
            println!("\n✓ All validations PASSED");
        } else {
            println!("\n✗ Some validations FAILED");
        }

        // Calculate speedup statistics for configs with EigenDA reference data
        let mut speedups = Vec::new();
        for result in &results {
            if let Some((ref_time, _)) = get_eigenda_reference(
                result.config.data_size_mb(),
                result.config.k,
                result.config.n,
            ) {
                let speedup = ref_time as f64 / result.encode_time_ns as f64;
                speedups.push(speedup);
            }
        }

        if !speedups.is_empty() {
            let avg_speedup = speedups.iter().sum::<f64>() / speedups.len() as f64;
            let min_speedup = speedups.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_speedup = speedups.iter().fold(0.0f64, |a, &b| a.max(b));

            println!(
                "\nSpeedup vs EigenDA EMA ({} comparable configs):",
                speedups.len()
            );
            println!("  Average: {:.2}x", avg_speedup);
            println!("  Min:     {:.2}x", min_speedup);
            println!("  Max:     {:.2}x", max_speedup);
        }

        // Statistics by data size
        println!("\nPerformance by Data Size:");
        println!("{}", "─".repeat(80));

        let data_sizes = vec![
            (128.0, "128KB"),
            (1024.0, "1MB"),
            (4096.0, "4MB"),
            (8192.0, "8MB"),
            (16384.0, "16MB"),
            (32768.0, "32MB"),
            (65536.0, "64MB"),
            (131072.0, "128MB"),
            (262144.0, "256MB"),
            (524288.0, "512MB"),
            (1048576.0, "1GB"),
        ];

        for (size_kb, label) in data_sizes {
            let size_results: Vec<_> = results
                .iter()
                .filter(|r| (r.config.data_size_kb() - size_kb).abs() < 1.0)
                .collect();

            if !size_results.is_empty() {
                let avg_throughput = size_results.iter().map(|r| r.throughput_mbs).sum::<f64>()
                    / size_results.len() as f64;
                let max_throughput = size_results
                    .iter()
                    .map(|r| r.throughput_mbs)
                    .fold(0.0f64, f64::max);
                let min_throughput = size_results
                    .iter()
                    .map(|r| r.throughput_mbs)
                    .fold(f64::INFINITY, f64::min);

                println!(
                    "  {:<8} Avg: {:>8.2} MB/s  Min: {:>8.2} MB/s  Max: {:>8.2} MB/s  ({} configs)",
                    label,
                    avg_throughput,
                    min_throughput,
                    max_throughput,
                    size_results.len()
                );
            }
        }

        // Overall statistics
        if !results.is_empty() {
            println!("\nOverall Statistics:");
            println!("{}", "─".repeat(80));

            let avg_throughput =
                results.iter().map(|r| r.throughput_mbs).sum::<f64>() / results.len() as f64;
            let max_throughput = results
                .iter()
                .map(|r| r.throughput_mbs)
                .fold(0.0f64, f64::max);

            println!("  Average Throughput:     {:.2} MB/s", avg_throughput);
            println!("  Peak Throughput:        {:.2} MB/s", max_throughput);
            println!(
                "  Validation Success:     {}/{}",
                results.iter().filter(|r| r.validation_passed).count(),
                results.len()
            );

            println!("\nKey Observations:");
            println!("  - EMA (ZODA) encoding uses Reed-Solomon with NTT");
            println!("  - All results validated against CPU reference implementation");
            println!("  - Benchmark times GPU encoding only (validation separate)");
            println!("  - Higher throughput indicates faster encoding");
        }
    }
}
