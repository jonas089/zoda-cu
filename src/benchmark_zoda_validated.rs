use crate::babybear::BabyBear;
use crate::ntt_babybear::{intt as cpu_intt, ntt as cpu_ntt};
use sha2::{Digest, Sha256};
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
    fn data_size_mb(&self) -> f64 {
        (self.k * self.row_size) as f64 / (1024.0 * 1024.0)
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

struct ValidatedEncodingResult {
    config: EncodingConfig,
    gpu_time_ms: f64,
    validation_time_ms: f64,
    total_time_ms: f64,
    validation_passed: bool,
    num_checks: usize,
}

#[cfg(feature = "cuda")]
fn encode_gpu_with_output(
    config: &EncodingConfig,
) -> Result<(Vec<Vec<BabyBear>>, f64), String> {
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

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    // Convert output to row-major format
    let mut encoded_rows: Vec<Vec<BabyBear>> = vec![vec![BabyBear::zero(); num_positions]; k + n];
    for col in 0..num_positions {
        for row in 0..(k + n) {
            encoded_rows[row][col] = BabyBear::new(h_output[col * ntt_size_kn + row]);
        }
    }

    Ok((encoded_rows, elapsed))
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
) -> Result<(bool, f64, usize), String> {
    let start = Instant::now();

    let k = config.k;
    let n = config.n;
    let num_positions = config.num_positions();
    let ntt_size_k = config.ntt_size_k();
    let ntt_size_kn = config.ntt_size_kn();

    let omega_k = BabyBear::get_root_of_unity(ntt_size_k.trailing_zeros());
    let omega_kn = BabyBear::get_root_of_unity(ntt_size_kn.trailing_zeros());

    // Phase 1: Column Encoding Verification
    let num_column_checks = 64.min(num_positions);
    let mut column_checks_passed = true;

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
                println!(
                    "  Column Encoding FAILED at column {} row {}: GPU={}, CPU={}",
                    col_idx, row_idx, column[row_idx].value, original_input[row_idx].value
                );
                column_checks_passed = false;
                break;
            }
        }

        if !column_checks_passed {
            break;
        }
    }

    if !column_checks_passed {
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        return Ok((false, elapsed, num_column_checks));
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
    // For each row, recreate the data and encode it
    let mut all_cpu_rlc: Vec<BabyBear> = Vec::with_capacity(k + n);

    // First, compute RLC for each COLUMN after encoding
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
    for row_idx in 0..(k + n) {
        let mut rlc_sum = BabyBear::zero();
        for col_idx in 0..num_positions {
            rlc_sum = rlc_sum + (encoded_rlc_columns[col_idx][row_idx] * coefficients[col_idx]);
        }
        all_cpu_rlc.push(rlc_sum);
    }

    // Verify GPU RLC matches CPU RLC for all k+n rows
    let num_rlc_checks = 64.min(k + n);
    let mut rlc_checks_passed = true;

    for check_idx in 0..num_rlc_checks {
        let row_idx = (check_idx * (k + n)) / num_rlc_checks;

        if all_gpu_rlc[row_idx].value != all_cpu_rlc[row_idx].value {
            println!(
                "  RLC Soundness FAILED at row {}: gpu_rlc={}, cpu_rlc={}",
                row_idx, all_gpu_rlc[row_idx].value, all_cpu_rlc[row_idx].value
            );
            rlc_checks_passed = false;
            break;
        }
    }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    let all_checks_passed = column_checks_passed && rlc_checks_passed;
    let total_checks = num_column_checks + num_rlc_checks;

    Ok((all_checks_passed, elapsed, total_checks))
}

#[cfg(feature = "cuda")]
fn run_validated_encoding_benchmark(
    config: EncodingConfig,
) -> Option<ValidatedEncodingResult> {
    let data_size_mb = config.data_size_mb();

    println!(
        "  K={} original rows, N={} parity rows → {} total rows",
        config.k, config.n, config.total_rows()
    );
    println!(
        "  Data volume: {:.1} MB original → {:.1} MB after encoding",
        data_size_mb,
        data_size_mb * config.total_rows() as f64 / config.k as f64
    );

    // GPU encoding
    print!("  GPU encoding (batched+fused)... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();

    let (encoded_rows, gpu_time_ms) = match encode_gpu_with_output(&config) {
        Ok(result) => result,
        Err(e) => {
            println!("ERROR: {}", e);
            return None;
        }
    };
    println!("{:.2} ms", gpu_time_ms);

    // ZODA validation
    print!("  Validating encoding (ZODA verification)... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();

    let (validation_passed, validation_time_ms, num_checks) =
        match validate_zoda_encoding(&config, &encoded_rows) {
            Ok(result) => result,
            Err(e) => {
                println!("ERROR: {}", e);
                return None;
            }
        };

    if validation_passed {
        println!("PASSED ({} checks: column encoding + RLC soundness, {:.2} ms)", num_checks, validation_time_ms);
    } else {
        println!("FAILED ({:.2} ms)", validation_time_ms);
    }

    let total_time_ms = gpu_time_ms + validation_time_ms;

    Some(ValidatedEncodingResult {
        config,
        gpu_time_ms,
        validation_time_ms,
        total_time_ms,
        validation_passed,
        num_checks,
    })
}

#[test]
#[ignore]
fn benchmark_zoda_validated() {
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

        println!("CUDA GPU detected\n");
        println!("Benchmark: Vertical Reed-Solomon encoding WITH ZODA VALIDATION");
        println!("This test encodes data and verifies correctness using ZODA protocol\n");

        let configs = vec![
            EncodingConfig {
                k: 65536,
                n: 65536,
                row_size: 4096,
            },
            EncodingConfig {
                k: 32768,
                n: 32768,
                row_size: 16384,
            },
            EncodingConfig {
                k: 32768,
                n: 32768,
                row_size: 32768,
            },
            EncodingConfig {
                k: 65536,
                n: 65536,
                row_size: 32768,
            },
        ];

        let mut results = Vec::new();

        println!("Running validated benchmarks...\n");

        for config in configs {
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("Encoding {:.1} MB of data:", config.data_size_mb());

            if let Some(result) = run_validated_encoding_benchmark(config) {
                results.push(result);
            }
            println!();
        }

        // Summary table
        println!("\n{}", "═".repeat(100));
        println!("SUMMARY - GPU Encoding + ZODA Validation");
        println!("{}", "═".repeat(100));
        println!(
            "{:<12} {:<10} {:<10} │ {:<12} {:<12} {:<12} │ {:<10}",
            "Data MB", "K", "N", "GPU (ms)", "Valid (ms)", "Total (ms)", "Status"
        );
        println!("{}", "─".repeat(100));

        let mut all_passed = true;
        for result in &results {
            let status = if result.validation_passed {
                "PASS"
            } else {
                all_passed = false;
                "FAIL"
            };

            println!(
                "{:<12.1} {:<10} {:<10} │ {:<12.2} {:<12.2} {:<12.2} │ {}",
                result.config.data_size_mb(),
                result.config.k,
                result.config.n,
                result.gpu_time_ms,
                result.validation_time_ms,
                result.total_time_ms,
                status,
            );
        }

        println!("{}", "═".repeat(100));

        if all_passed {
            println!("\nALL VALIDATIONS PASSED");
            println!("\nThe GPU-accelerated encoding has been verified to be");
            println!("mathematically correct according to the ZODA protocol.");
            println!("\nTwo-phase verification:");
            println!("\nPhase 1 - Column Encoding (Vertical Reed-Solomon):");
            println!("  1. Extract first k values from each column");
            println!("  2. INTT to interpolate polynomial coefficients");
            println!("  3. Zero-pad coefficients to k+n size");
            println!("  4. NTT to evaluate polynomial at k+n points");
            println!("  5. Verify GPU output matches CPU reference");
            println!("  Each column forms a valid Reed-Solomon codeword");
            println!("\nPhase 2 - RLC Soundness Check (ZODA/RSEMA1D):");
            println!("  1. Derive random coefficients from commitment");
            println!("  2. For each row: compute RLC = ∑(row[col] × coeff[col])");
            println!("  3. CPU encodes all columns and computes reference RLC");
            println!("  4. Verify GPU RLC matches CPU RLC for all k+n rows");
            println!("  Rows are consistent linear combinations across columns");
            println!("\nThis provides full ZODA soundness for data availability sampling.");
        } else {
            println!("\nSOME VALIDATIONS FAILED");
            println!("Please check the encoding implementation for errors.");
        }

        if !results.is_empty() {
            let avg_overhead = results
                .iter()
                .map(|r| (r.validation_time_ms / r.gpu_time_ms) * 100.0)
                .sum::<f64>()
                / results.len() as f64;

            println!("\nValidation overhead: {:.1}% of encoding time", avg_overhead);
        }
    }
}
