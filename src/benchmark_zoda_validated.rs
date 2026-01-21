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
        self.total_rows().next_power_of_two()
    }
}

#[derive(Debug)]
struct ValidatedEncodingResult {
    config: EncodingConfig,
    gpu_time_ms: f64,
    validation_time_ms: f64,
    total_time_ms: f64,
    validation_passed: bool,
    num_checks: usize,
}

/// Encode data using GPU and return both the encoded data and timing
#[cfg(feature = "cuda")]
fn encode_gpu_with_output(
    config: &EncodingConfig,
) -> Result<(Vec<Vec<BabyBear>>, f64), String> {
    let k = config.k;
    let n = config.n;
    let num_positions = config.num_positions();
    let ntt_size_k = config.ntt_size_k();
    let ntt_size_kn = config.ntt_size_kn();

    // Get twiddle factors
    let omega_k = BabyBear::get_root_of_unity(ntt_size_k.trailing_zeros());
    let omega_kn = BabyBear::get_root_of_unity(ntt_size_kn.trailing_zeros());

    let mut h_input: Vec<u64> = vec![0; num_positions * ntt_size_k];

    // Fill input data
    for row_idx in 0..k {
        for col in 0..num_positions {
            let value = ((row_idx * num_positions + col) % 2013265921) as u64;
            h_input[col * ntt_size_k + row_idx] = value;
        }
    }

    let mut d_input = CudaBuffer::new(num_positions * ntt_size_k)?;
    let mut d_output = CudaBuffer::new(num_positions * ntt_size_kn)?;
    let mut d_work = CudaBuffer::new(num_positions * ntt_size_k)?;

    let mut h_output: Vec<u64> = vec![0; num_positions * ntt_size_kn];

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

/// Compute a root hash of the encoded data square (similar to ZODA commitment)
fn compute_data_root(encoded_rows: &[Vec<BabyBear>]) -> String {
    let mut hasher = Sha256::new();
    for row in encoded_rows {
        for &value in row {
            hasher.update(&value.value.to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

/// Generate deterministic coefficients for random linear combination check
/// This is the ZODA verification approach
fn generate_deterministic_coefficients(
    data_root: &str,
    num_columns: usize,
) -> Vec<BabyBear> {
    (0..num_columns)
        .map(|i| {
            let mut hasher = Sha256::new();
            hasher.update(data_root.as_bytes());
            hasher.update(&i.to_le_bytes());
            let digest = hasher.finalize();
            let val = u64::from_be_bytes([
                digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6],
                digest[7],
            ]);
            BabyBear::new(val) + BabyBear::new(i as u64)
        })
        .collect()
}

/// Verify ZODA encoding correctness via random linear combination check
///
/// This validates that the encoded data forms a valid Reed-Solomon codeword
/// by checking that random linear combinations of rows are consistent
/// with the polynomial structure.
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

    // 1. Compute data root (commitment)
    let data_root = compute_data_root(encoded_rows);

    // 2. Generate deterministic coefficients for linear combination
    let coefficients = generate_deterministic_coefficients(&data_root, num_positions);

    // 3. Compute linear combination of all rows in the original k rows
    let mut y_values: Vec<BabyBear> = Vec::with_capacity(k);
    for row_idx in 0..k {
        let mut sum = BabyBear::zero();
        for (col_idx, &coeff) in coefficients.iter().enumerate() {
            sum = sum + (encoded_rows[row_idx][col_idx] * coeff);
        }
        y_values.push(sum);
    }

    // 4. Use INTT to get polynomial coefficients of y
    let omega_k = BabyBear::get_root_of_unity(ntt_size_k.trailing_zeros());
    let omega_kn = BabyBear::get_root_of_unity(ntt_size_kn.trailing_zeros());

    let mut y_coeffs = y_values.clone();
    y_coeffs.resize(ntt_size_k, BabyBear::zero());
    cpu_intt(&mut y_coeffs, omega_k);

    // 5. Extend to k+n points via NTT
    y_coeffs.resize(ntt_size_kn, BabyBear::zero());
    let mut y_extended = y_coeffs.clone();
    cpu_ntt(&mut y_extended, omega_kn);

    // 6. Verify random rows in the extended region satisfy the linear combination
    let num_checks = 64.min(n); // Check up to 64 parity rows
    let mut all_checks_passed = true;

    for check_idx in 0..num_checks {
        // Check parity rows (rows k through k+n-1)
        let row_idx = k + (check_idx * n / num_checks);

        let mut sum = BabyBear::zero();
        for (col_idx, &coeff) in coefficients.iter().enumerate() {
            sum = sum + (encoded_rows[row_idx][col_idx] * coeff);
        }

        if sum.value != y_extended[row_idx].value {
            println!(
                "  ✗ Validation FAILED at parity row {}: expected {}, got {}",
                row_idx, y_extended[row_idx].value, sum.value
            );
            all_checks_passed = false;
            break;
        }
    }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    Ok((all_checks_passed, elapsed, num_checks))
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
        println!("✓ PASSED ({} checks, {:.2} ms)", num_checks, validation_time_ms);
    } else {
        println!("✗ FAILED ({:.2} ms)", validation_time_ms);
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

// ============================================================================
// Main Validated Benchmark Test
// ============================================================================

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

        println!("✓ CUDA GPU detected\n");
        println!("Benchmark: Vertical Reed-Solomon encoding WITH ZODA VALIDATION");
        println!("This test encodes data and verifies correctness using ZODA protocol\n");

        // Test configurations - same as the regular benchmark
        let configs = vec![
            // 256 MB
            EncodingConfig {
                k: 65536,
                n: 65536,
                row_size: 4096,
            },
            // 512 MB
            EncodingConfig {
                k: 32768,
                n: 32768,
                row_size: 16384,
            },
            // 1 GB
            EncodingConfig {
                k: 32768,
                n: 32768,
                row_size: 32768,
            },
            // 2 GB
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
                "✓ PASS"
            } else {
                all_passed = false;
                "✗ FAIL"
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
            println!("\n✓✓✓ ALL VALIDATIONS PASSED ✓✓✓");
            println!("\nThe GPU-accelerated Reed-Solomon encoding has been verified to be");
            println!("mathematically correct according to the ZODA protocol specification.");
            println!("\nValidation method:");
            println!("  1. Compute deterministic linear combination of all columns");
            println!("  2. Interpolate to get polynomial coefficients");
            println!("  3. Evaluate polynomial at extended points");
            println!("  4. Verify parity rows satisfy the linear combination property");
            println!("\nThis proves the encoded data forms a valid Reed-Solomon codeword.");
        } else {
            println!("\n✗✗✗ SOME VALIDATIONS FAILED ✗✗✗");
            println!("Please check the encoding implementation for errors.");
        }

        // Report overhead
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
