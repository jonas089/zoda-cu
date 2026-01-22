use crate::babybear::BabyBear;
use crate::ntt_babybear::{intt as cpu_intt, ntt as cpu_ntt};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Write;
use std::time::Instant;

#[cfg(feature = "cuda")]
use crate::cuda_ntt::{cuda_available, cuda_rs_encode_vertical, CudaBuffer};

#[derive(Debug, Clone)]
struct EncodingConfig {
    k: usize,
    n: usize,
    row_size: usize,
}

impl EncodingConfig {
    fn data_size_bytes(&self) -> usize {
        self.k * self.row_size
    }

    fn data_size_kb(&self) -> f64 {
        self.data_size_bytes() as f64 / 1024.0
    }

    fn data_size_mb(&self) -> f64 {
        self.data_size_bytes() as f64 / (1024.0 * 1024.0)
    }

    fn total_rows(&self) -> usize {
        self.k + self.n
    }

    fn elements_per_row(&self) -> usize {
        self.row_size / 4
    }

    fn num_positions(&self) -> usize {
        self.elements_per_row()
    }

    fn ntt_size_k(&self) -> usize {
        self.k.next_power_of_two()
    }

    fn ntt_size_kn(&self) -> usize {
        (self.k + self.n).next_power_of_two()
    }
}

struct BenchmarkResult {
    config: EncodingConfig,
    encode_time_ns: u64,
    throughput_mbs: f64,
    validation_passed: bool,
}

#[cfg(feature = "cuda")]
fn encode_gpu_with_output(config: &EncodingConfig) -> Result<(Vec<Vec<BabyBear>>, u64), String> {
    let k = config.k;
    let n = config.n;
    let num_positions = config.num_positions();
    let ntt_size_k = config.ntt_size_k();
    let ntt_size_kn = config.ntt_size_kn();

    let omega_k = BabyBear::get_root_of_unity(ntt_size_k.trailing_zeros());
    let omega_kn = BabyBear::get_root_of_unity(ntt_size_kn.trailing_zeros());

    let total_input_size = num_positions * ntt_size_k;
    let total_output_size = num_positions * ntt_size_kn;
    let work_size = num_positions * ntt_size_k;

    let mut h_input: Vec<u64> = Vec::with_capacity(total_input_size);
    for col in 0..num_positions {
        for row in 0..k {
            let value = ((row * num_positions + col) % 2013265921) as u64;
            h_input.push(value);
        }
        for _ in k..ntt_size_k {
            h_input.push(0);
        }
    }

    let mut h_output = vec![0u64; total_output_size];
    let mut d_input = CudaBuffer::new(total_input_size)?;
    let mut d_output = CudaBuffer::new(total_output_size)?;
    let d_work = CudaBuffer::new(work_size)?;

    let start = Instant::now();

    d_input.copy_from_host(&h_input)?;

    unsafe {
        cuda_rs_encode_vertical(
            d_input.as_ptr(),
            d_output.as_ptr(),
            d_work.as_ptr(),
            num_positions as u32,
            ntt_size_k as u32,
            ntt_size_kn as u32,
            omega_k.value,
            omega_kn.value,
        );
    }

    d_output.copy_to_host(&mut h_output)?;

    let elapsed_ns = start.elapsed().as_nanos() as u64;

    // Convert output to row-major format
    let mut encoded_rows: Vec<Vec<BabyBear>> = vec![vec![BabyBear::zero(); num_positions]; k + n];
    for col in 0..num_positions {
        for row in 0..(k + n) {
            encoded_rows[row][col] = BabyBear::new(h_output[col * ntt_size_kn + row]);
        }
    }

    Ok((encoded_rows, elapsed_ns))
}

fn compute_data_root(encoded_rows: &[Vec<BabyBear>]) -> String {
    let mut hasher = Sha256::new();
    for row in encoded_rows {
        for &value in row {
            hasher.update(value.value.to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

fn generate_deterministic_coefficients(data_root: &str, num_columns: usize) -> Vec<BabyBear> {
    (0..num_columns)
        .map(|i| {
            let mut hasher = Sha256::new();
            hasher.update(data_root.as_bytes());
            hasher.update(i.to_le_bytes());
            let digest = hasher.finalize();
            let val = u64::from_be_bytes([
                digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6],
                digest[7],
            ]);
            BabyBear::new(val) + BabyBear::new(i as u64)
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn validate_zoda_encoding(
    config: &EncodingConfig,
    encoded_rows: &[Vec<BabyBear>],
) -> Result<bool, String> {
    let k = config.k;
    let n = config.n;
    let num_positions = config.num_positions();
    let ntt_size_k = config.ntt_size_k();
    let ntt_size_kn = config.ntt_size_kn();

    let omega_k = BabyBear::get_root_of_unity(ntt_size_k.trailing_zeros());
    let omega_kn = BabyBear::get_root_of_unity(ntt_size_kn.trailing_zeros());

    // Phase 1: Column Encoding Verification
    let num_column_checks = 64.min(num_positions);

    for check_idx in 0..num_column_checks {
        let col_idx = (check_idx * num_positions) / num_column_checks;

        // Extract column from encoded data
        let column: Vec<BabyBear> = encoded_rows
            .iter()
            .take(k + n)
            .map(|row| row[col_idx])
            .collect();

        // Recreate original input for this column
        let mut original_input: Vec<BabyBear> = Vec::with_capacity(k);
        for row_idx in 0..k {
            let value = ((row_idx * num_positions + col_idx) % 2013265921) as u64;
            original_input.push(BabyBear::new(value));
        }

        // Encode column on CPU: INTT → pad → NTT
        original_input.resize(ntt_size_k, BabyBear::zero());
        cpu_intt(&mut original_input, omega_k);
        original_input.resize(ntt_size_kn, BabyBear::zero());
        cpu_ntt(&mut original_input, omega_kn);

        // Verify GPU matches CPU
        for row_idx in 0..(k + n) {
            if original_input[row_idx].value != column[row_idx].value {
                return Ok(false);
            }
        }
    }

    // Phase 2: RLC Soundness Check
    let data_root = compute_data_root(encoded_rows);
    let coefficients = generate_deterministic_coefficients(&data_root, num_positions);

    // Compute RLC for ALL k+n rows from GPU output
    let all_gpu_rlc: Vec<BabyBear> = encoded_rows
        .iter()
        .take(k + n)
        .map(|row| {
            row.iter()
                .zip(coefficients.iter())
                .fold(BabyBear::zero(), |acc, (&val, &coeff)| acc + (val * coeff))
        })
        .collect();

    // Compute what RLC values SHOULD be by encoding on CPU
    let mut encoded_rlc_columns: Vec<Vec<BabyBear>> = Vec::new();

    for col_idx in 0..num_positions {
        // Encode this column on CPU
        let mut column_data: Vec<BabyBear> = Vec::with_capacity(k);
        for row_idx in 0..k {
            let value = ((row_idx * num_positions + col_idx) % 2013265921) as u64;
            column_data.push(BabyBear::new(value));
        }

        column_data.resize(ntt_size_k, BabyBear::zero());
        cpu_intt(&mut column_data, omega_k);
        column_data.resize(ntt_size_kn, BabyBear::zero());
        cpu_ntt(&mut column_data, omega_kn);

        encoded_rlc_columns.push(column_data);
    }

    // Now compute RLC for each row from the encoded columns
    let mut all_cpu_rlc: Vec<BabyBear> = Vec::with_capacity(k + n);
    for row_idx in 0..(k + n) {
        let mut rlc_sum = BabyBear::zero();
        for col_idx in 0..num_positions {
            rlc_sum = rlc_sum + (encoded_rlc_columns[col_idx][row_idx] * coefficients[col_idx]);
        }
        all_cpu_rlc.push(rlc_sum);
    }

    // Verify GPU RLC matches CPU RLC for all k+n rows
    let num_rlc_checks = 64.min(k + n);

    for check_idx in 0..num_rlc_checks {
        let row_idx = (check_idx * (k + n)) / num_rlc_checks;

        if all_gpu_rlc[row_idx].value != all_cpu_rlc[row_idx].value {
            return Ok(false);
        }
    }

    Ok(true)
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
        Ok(result) => result,
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
            EncodingConfig { k: 1024, n: 1024, row_size: 128 },   // 128KB, k=1024, n=1024
            EncodingConfig { k: 1024, n: 3072, row_size: 128 },   // 128KB, k=1024, n=3072
            EncodingConfig { k: 4096, n: 4096, row_size: 32 },    // 128KB, k=4096, n=4096
            EncodingConfig { k: 4096, n: 12288, row_size: 32 },   // 128KB, k=4096, n=12288

            // 1MB configurations
            EncodingConfig { k: 1024, n: 1024, row_size: 1024 },  // 1MB, k=1024, n=1024
            EncodingConfig { k: 1024, n: 3072, row_size: 1024 },  // 1MB, k=1024, n=3072
            EncodingConfig { k: 4096, n: 4096, row_size: 256 },   // 1MB, k=4096, n=4096
            EncodingConfig { k: 4096, n: 12288, row_size: 256 },  // 1MB, k=4096, n=12288

            // 4MB configurations
            EncodingConfig { k: 1024, n: 1024, row_size: 4096 },  // 4MB, k=1024, n=1024
            EncodingConfig { k: 1024, n: 3072, row_size: 4096 },  // 4MB, k=1024, n=3072
            EncodingConfig { k: 4096, n: 4096, row_size: 1024 },  // 4MB, k=4096, n=4096
            EncodingConfig { k: 4096, n: 12288, row_size: 1024 }, // 4MB, k=4096, n=12288

            // 8MB configurations
            EncodingConfig { k: 1024, n: 1024, row_size: 8192 },  // 8MB, k=1024, n=1024
            EncodingConfig { k: 1024, n: 3072, row_size: 8192 },  // 8MB, k=1024, n=3072
            EncodingConfig { k: 4096, n: 4096, row_size: 2048 },  // 8MB, k=4096, n=4096
            EncodingConfig { k: 4096, n: 12288, row_size: 2048 }, // 8MB, k=4096, n=12288

            // 16MB configurations
            EncodingConfig { k: 1024, n: 1024, row_size: 16384 }, // 16MB, k=1024, n=1024
            EncodingConfig { k: 1024, n: 3072, row_size: 16384 }, // 16MB, k=1024, n=3072
            EncodingConfig { k: 4096, n: 4096, row_size: 4096 },  // 16MB, k=4096, n=4096
            EncodingConfig { k: 4096, n: 12288, row_size: 4096 }, // 16MB, k=4096, n=12288

            // 32MB configurations
            EncodingConfig { k: 1024, n: 1024, row_size: 32768 }, // 32MB, k=1024, n=1024
            EncodingConfig { k: 1024, n: 3072, row_size: 32768 }, // 32MB, k=1024, n=3072
            EncodingConfig { k: 4096, n: 4096, row_size: 8192 },  // 32MB, k=4096, n=4096
            EncodingConfig { k: 4096, n: 12288, row_size: 8192 }, // 32MB, k=4096, n=12288

            // 64MB configurations
            EncodingConfig { k: 1024, n: 1024, row_size: 65536 }, // 64MB, k=1024, n=1024
            EncodingConfig { k: 1024, n: 3072, row_size: 65536 }, // 64MB, k=1024, n=3072
            EncodingConfig { k: 4096, n: 4096, row_size: 16384 }, // 64MB, k=4096, n=4096
            EncodingConfig { k: 4096, n: 12288, row_size: 16384 }, // 64MB, k=4096, n=12288

            // 128MB configurations
            EncodingConfig { k: 1024, n: 1024, row_size: 131072 }, // 128MB, k=1024, n=1024
            EncodingConfig { k: 1024, n: 3072, row_size: 131072 }, // 128MB, k=1024, n=3072
            EncodingConfig { k: 4096, n: 4096, row_size: 32768 },  // 128MB, k=4096, n=4096
            EncodingConfig { k: 4096, n: 12288, row_size: 32768 }, // 128MB, k=4096, n=12288

            // 256MB configurations
            EncodingConfig { k: 1024, n: 1024, row_size: 262144 }, // 256MB, k=1024, n=1024
            EncodingConfig { k: 1024, n: 3072, row_size: 262144 }, // 256MB, k=1024, n=3072
            EncodingConfig { k: 4096, n: 4096, row_size: 65536 },  // 256MB, k=4096, n=4096
            EncodingConfig { k: 4096, n: 12288, row_size: 65536 }, // 256MB, k=4096, n=12288

            // 512MB configurations
            EncodingConfig { k: 1024, n: 1024, row_size: 524288 }, // 512MB, k=1024, n=1024
            EncodingConfig { k: 1024, n: 3072, row_size: 524288 }, // 512MB, k=1024, n=3072
            EncodingConfig { k: 4096, n: 4096, row_size: 131072 }, // 512MB, k=4096, n=4096
            EncodingConfig { k: 4096, n: 12288, row_size: 131072 }, // 512MB, k=4096, n=12288

            // 1GB configurations
            EncodingConfig { k: 1024, n: 1024, row_size: 1048576 }, // 1GB, k=1024, n=1024
            EncodingConfig { k: 1024, n: 3072, row_size: 1048576 }, // 1GB, k=1024, n=3072
            EncodingConfig { k: 4096, n: 4096, row_size: 262144 },  // 1GB, k=4096, n=4096
            EncodingConfig { k: 4096, n: 12288, row_size: 262144 }, // 1GB, k=4096, n=12288
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

        // Write CSV file
        if let Ok(mut csv_file) = File::create("zoda_eigenda_benchmark.csv") {
            writeln!(csv_file, "Data Size,Data MB,K,N,Encode Time (ns),Throughput (MB/s),Validation Status").ok();
            for result in &results {
                let data_label = if result.config.data_size_kb() < 1024.0 {
                    format!("{:.0}KB", result.config.data_size_kb())
                } else {
                    format!("{:.0}MB", result.config.data_size_mb())
                };
                let status = if result.validation_passed { "PASS" } else { "FAIL" };
                writeln!(
                    csv_file,
                    "{},{:.6},{},{},{},{:.2},{}",
                    data_label,
                    result.config.data_size_mb(),
                    result.config.k,
                    result.config.n,
                    result.encode_time_ns,
                    result.throughput_mbs,
                    status
                ).ok();
            }
            println!("  ✓ CSV results written to: zoda_eigenda_benchmark.csv");
        }

        // Write Markdown table
        if let Ok(mut md_file) = File::create("zoda_eigenda_benchmark.md") {
            writeln!(md_file, "# ZODA Encoding Benchmark Results\n").ok();
            writeln!(md_file, "## Performance Comparison: ZODA (EMA-style) vs KZG Encoding\n").ok();
            writeln!(md_file, "| Configuration | K | N | EMA (ns/op) | EMA (MB/s) | Status |").ok();
            writeln!(md_file, "|---------------|---|---|-------------|------------|--------|").ok();

            for result in &results {
                let data_label = if result.config.data_size_kb() < 1024.0 {
                    format!("{:.0}KB", result.config.data_size_kb())
                } else {
                    format!("{:.0}MB", result.config.data_size_mb())
                };
                let status = if result.validation_passed { "✓ PASS" } else { "✗ FAIL" };
                writeln!(
                    md_file,
                    "| {} | {} | {} | {:,} | {:.2} | {} |",
                    data_label,
                    result.config.k,
                    result.config.n,
                    result.encode_time_ns,
                    result.throughput_mbs,
                    status
                ).ok();
            }

            // Add statistics section
            writeln!(md_file, "\n## Statistics by Data Size\n").ok();
            writeln!(md_file, "| Data Size | Avg MB/s | Min MB/s | Max MB/s | Configs |").ok();
            writeln!(md_file, "|-----------|----------|----------|----------|---------|").ok();

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
                        label, avg_throughput, min_throughput, max_throughput, size_results.len()
                    ).ok();
                }
            }

            // Overall statistics
            if !results.is_empty() {
                let avg_throughput = results.iter().map(|r| r.throughput_mbs).sum::<f64>()
                    / results.len() as f64;
                let max_throughput = results
                    .iter()
                    .map(|r| r.throughput_mbs)
                    .fold(0.0f64, f64::max);
                let passed = results.iter().filter(|r| r.validation_passed).count();

                writeln!(md_file, "\n## Overall Statistics\n").ok();
                writeln!(md_file, "- **Average Throughput**: {:.2} MB/s", avg_throughput).ok();
                writeln!(md_file, "- **Peak Throughput**: {:.2} MB/s", max_throughput).ok();
                writeln!(md_file, "- **Validation Success**: {}/{}", passed, results.len()).ok();
                writeln!(md_file, "- **Total Configurations**: {}", results.len()).ok();
            }

            println!("  ✓ Markdown results written to: zoda_eigenda_benchmark.md");
        }

        // Print results table matching EigenDA format
        println!("\n{}", "═".repeat(100));
        println!("Performance Comparison: ZODA (EMA-style) vs KZG Encoding");
        println!("{}", "═".repeat(100));
        println!(
            "{:<20} {:<8} {:<8} │ {:<15} {:<15} │ {:<10}",
            "Configuration", "k", "n", "EMA (ns/op)", "EMA (MB/s)", "Status"
        );
        println!("{}", "─".repeat(100));

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

            println!(
                "{:<20} {:<8} {:<8} │ {:<15,} {:<15.2} │ {}",
                data_label,
                result.config.k,
                result.config.n,
                result.encode_time_ns,
                result.throughput_mbs,
                status,
            );
        }

        println!("{}", "═".repeat(100));

        if all_passed {
            println!("\n✓ All validations PASSED");
        } else {
            println!("\n✗ Some validations FAILED");
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
                    label, avg_throughput, min_throughput, max_throughput, size_results.len()
                );
            }
        }

        // Overall statistics
        if !results.is_empty() {
            println!("\nOverall Statistics:");
            println!("{}", "─".repeat(80));

            let avg_throughput = results.iter().map(|r| r.throughput_mbs).sum::<f64>()
                / results.len() as f64;
            let max_throughput = results
                .iter()
                .map(|r| r.throughput_mbs)
                .fold(0.0f64, f64::max);

            println!("  Average Throughput:     {:.2} MB/s", avg_throughput);
            println!("  Peak Throughput:        {:.2} MB/s", max_throughput);
            println!("  Validation Success:     {}/{}",
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
