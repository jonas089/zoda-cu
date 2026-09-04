use crate::benchmarks::utils::{encode_gpu_with_output, validate_zoda_encoding, EncodingConfig};

#[cfg(feature = "cuda")]
use crate::ntt::cuda::cuda_available;


struct ValidatedEncodingResult {
    config: EncodingConfig,
    /// End-to-end encode (host layout work + GPU), median of the timed runs.
    encode_time_ms: f64,
    /// The GPU transform calls alone.
    gpu_time_ms: f64,
    validation_time_ms: f64,
    total_time_ms: f64,
    validation_passed: bool,
}


#[cfg(feature = "cuda")]
fn run_validated_encoding_benchmark(config: EncodingConfig) -> Option<ValidatedEncodingResult> {
    let data_size_mb = config.data_size_mb();

    println!(
        "  K={} original rows, N={} parity rows → {} total rows",
        config.k,
        config.n,
        config.total_rows()
    );
    println!(
        "  Data volume: {:.1} MB original → {:.1} MB after encoding",
        data_size_mb,
        data_size_mb * config.total_rows() as f64 / config.k as f64
    );

    // GPU encoding: median end-to-end encode after a warm-up, see utils::EncodeTiming.
    print!("  GPU encoding (batched)... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();

    let (encoded_rows, timing) = match encode_gpu_with_output(&config) {
        Ok(result) => result,
        Err(e) => {
            println!("ERROR: {}", e);
            return None;
        }
    };
    let encode_time_ms = timing.encode_ns as f64 / 1_000_000.0;
    let gpu_time_ms = timing.transform_ns as f64 / 1_000_000.0;
    println!("{:.2} ms end-to-end, of which {:.2} ms GPU transforms", encode_time_ms, gpu_time_ms);

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
        println!(
            "PASSED ({} checks: column encoding + RLC soundness, {:.2} ms)",
            num_checks, validation_time_ms
        );
    } else {
        println!("FAILED ({:.2} ms)", validation_time_ms);
    }

    let total_time_ms = encode_time_ms + validation_time_ms;

    Some(ValidatedEncodingResult {
        config,
        encode_time_ms,
        gpu_time_ms,
        validation_time_ms,
        total_time_ms,
        validation_passed,
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
            "{:<12} {:<10} {:<10} │ {:<12} {:<12} {:<12} {:<12} │ {:<10}",
            "Data MB", "K", "N", "Encode (ms)", "GPU (ms)", "Valid (ms)", "Total (ms)", "Status"
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
                "{:<12.1} {:<10} {:<10} │ {:<12.2} {:<12.2} {:<12.2} {:<12.2} │ {}",
                result.config.data_size_mb(),
                result.config.k,
                result.config.n,
                result.encode_time_ms,
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
                .map(|r| (r.validation_time_ms / r.encode_time_ms) * 100.0)
                .sum::<f64>()
                / results.len() as f64;

            println!(
                "\nValidation overhead: {:.1}% of encoding time",
                avg_overhead
            );
        }
    }
}
