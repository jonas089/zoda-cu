//! Encoding throughput on Celestia-shaped data squares.
//!
//! A Celestia square of side `side` holds `side * side` shares of `SHARE_BYTES`
//! bytes. ZODA extends it column-wise: the `side` original rows become
//! `side + side` rows (the 2x Reed-Solomon extension), so `k = n = side` and
//! every row carries `side * SHARE_BYTES` bytes of payload.
//!
//! Two times are reported per square, both from `utils::encode_gpu`:
//!
//!   * `encode`    - end-to-end, host memory in to host memory out.
//!   * `GPU xform` - the two transform calls inside that window.
//!
//! `encode - GPU xform` is the single-threaded host layout work, so the two
//! numbers bracket what the kernel can do against what the pipeline delivers.
//!
//! Throughput is also reported in Gb/s, because Celestia's Fibre figures are
//! quoted that way and 1 Gb/s is not 1 GiB/s.

use crate::benchmarks::utils::{
    encode_gpu, encode_gpu_with_output, generate_input, validate_zoda_encoding, EncodeTiming,
    EncodingConfig,
};
use crate::ntt::cuda::PinnedSquare;

#[cfg(feature = "cuda")]
use crate::ntt::cuda::cuda_available;

/// Celestia share size.
const SHARE_BYTES: usize = 512;

/// Square sides to measure. Payload is `side * side * SHARE_BYTES`.
const SIDES: [usize; 7] = [64, 128, 256, 512, 1024, 2048, 4096];

/// Above this side the CPU-reference validation is skipped, because it needs
/// more host memory than this box has. Since the RLC check now extends the RLC
/// vector instead of re-encoding every column, it no longer allocates a second
/// copy of the square, so 4096 is in reach where it previously was not.
const VALIDATE_MAX_SIDE: usize = 4096;

fn config_for(side: usize) -> EncodingConfig {
    EncodingConfig {
        k: side,
        n: side,
        row_size: side * SHARE_BYTES,
    }
}

/// Timed iterations per square; the median is reported. `ZODA_BENCH_ITERS`, same
/// variable the shared harness reads.
fn bench_iters() -> usize {
    std::env::var("ZODA_BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(3)
}

struct SquareResult {
    side: usize,
    payload_mib: f64,
    /// Extended square, `2 * side` rows, in MiB.
    extended_mib: f64,
    encode_ns: u64,
    transform_ns: u64,
    encode_gibs: f64,
    transform_gibs: f64,
    /// `None` when the square was too large to validate on this host.
    validated: Option<bool>,
}

fn gibs_to_gbits(gibs: f64) -> f64 {
    gibs * 1024.0 * 1024.0 * 1024.0 * 8.0 / 1e9
}

fn result_from(side: usize, timing: EncodeTiming, validated: Option<bool>) -> SquareResult {
    let payload_mib = config_for(side).data_size_mb();
    let secs = |ns: u64| ns as f64 / 1_000_000_000.0;
    SquareResult {
        side,
        payload_mib,
        extended_mib: payload_mib * 2.0,
        encode_ns: timing.encode_ns,
        transform_ns: timing.transform_ns,
        encode_gibs: payload_mib / 1024.0 / secs(timing.encode_ns),
        transform_gibs: payload_mib / 1024.0 / secs(timing.transform_ns),
        validated,
    }
}

/// Small enough to validate: encode, keep the rows, check them against the CPU.
#[cfg(feature = "cuda")]
fn run_validated(side: usize) -> Option<SquareResult> {
    let config = config_for(side);
    let (encoded_rows, timing) = encode_gpu_with_output(&config).ok()?;
    let ok = matches!(validate_zoda_encoding(&config, &encoded_rows), Ok((true, _, _)));
    Some(result_from(side, timing, Some(ok)))
}

/// Too large to validate: same warm-up and median as the shared harness, but the
/// encoded rows are never gathered into a second host buffer, so the only large
/// allocations are the pinned square and the input.
#[cfg(feature = "cuda")]
fn run_unvalidated(side: usize) -> Option<SquareResult> {
    let config = config_for(side);
    let input = generate_input(&config);
    let mut square = PinnedSquare::new(config.ntt_size_kn(), config.num_positions()).ok()?;

    encode_gpu(&config, &input, &mut square).ok()?; // untimed warm-up
    let mut runs = Vec::with_capacity(bench_iters());
    for _ in 0..bench_iters() {
        runs.push(encode_gpu(&config, &input, &mut square).ok()?);
    }
    runs.sort_by_key(|t| t.encode_ns);
    let median = runs[runs.len() / 2];
    Some(result_from(side, median, None))
}

/// Above this side the host runs short of memory during an encode. At side 4096
/// the pinned square alone is 16 GiB, so the `encode` column there is degraded
/// by memory pressure and understates the pipeline. `GPU xform` is unaffected,
/// because the device only ever sees streamed chunks.
///
/// Note that this is a separate question from whether we validated the square.
/// Validation used to be the binding memory constraint, so "largest validated"
/// used to be a usable proxy for "largest unpressured". It is not any more: the
/// RLC check no longer allocates a second copy of the square, so we can now
/// validate sides whose `encode` figure we still should not extrapolate from.
const ENCODE_UNPRESSURED_MAX_SIDE: usize = 2048;

/// The square the extrapolations are based on: the largest one whose `encode`
/// figure is not degraded by host memory pressure.
#[cfg(feature = "cuda")]
fn projection_basis(results: &[SquareResult]) -> &SquareResult {
    results
        .iter()
        .rev()
        .find(|r| r.side <= ENCODE_UNPRESSURED_MAX_SIDE)
        .unwrap_or_else(|| results.last().unwrap())
}

/// Square side, in shares, whose payload fits in `budget_secs` at `gibs`.
/// Payload is `side^2 * SHARE_BYTES`, so the side scales as the square root of
/// the byte budget. Pure extrapolation from a measured rate.
fn side_within(gibs: f64, budget_secs: f64) -> f64 {
    let bytes = gibs * 1024.0 * 1024.0 * 1024.0 * budget_secs;
    (bytes / SHARE_BYTES as f64).sqrt()
}

#[test]
#[ignore]
fn benchmark_celestia_squares() {
    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA support not compiled in. Build with: cargo test --features cuda --release");
    }

    #[cfg(feature = "cuda")]
    {
        if !cuda_available() {
            println!("CUDA not available on this system.");
            return;
        }

        println!("ZODA encoding throughput on Celestia data squares");
        println!("{SHARE_BYTES} byte shares, k = n = side (2x row extension)\n");

        let mut results = Vec::new();
        for (idx, &side) in SIDES.iter().enumerate() {
            let config = config_for(side);
            print!(
                "  [{}/{}] {side}x{side} square, {:.0} MiB payload{} ... ",
                idx + 1,
                SIDES.len(),
                config.data_size_mb(),
                if side > VALIDATE_MAX_SIDE { ", unvalidated" } else { "" }
            );
            std::io::Write::flush(&mut std::io::stdout()).unwrap();

            let run = if side > VALIDATE_MAX_SIDE {
                run_unvalidated(side)
            } else {
                run_validated(side)
            };

            match run {
                Some(r) => {
                    println!(
                        "encode {:.1} ms ({:.2} GiB/s), xform {:.1} ms ({:.2} GiB/s){}",
                        r.encode_ns as f64 / 1e6,
                        r.encode_gibs,
                        r.transform_ns as f64 / 1e6,
                        r.transform_gibs,
                        match r.validated {
                            Some(false) => "  [VALIDATION FAILED]",
                            _ => "",
                        }
                    );
                    results.push(r);
                }
                None => println!("FAILED (likely out of host memory)"),
            }
        }

        if results.is_empty() {
            println!("\nNo squares completed.");
            return;
        }

        print_table(&results);
    }
}

#[cfg(feature = "cuda")]
fn valid_label(v: Option<bool>) -> &'static str {
    match v {
        Some(true) => "yes",
        Some(false) => "NO",
        None => "skipped",
    }
}

#[cfg(feature = "cuda")]
fn print_table(results: &[SquareResult]) {
    println!("\n{}", "=".repeat(118));
    println!(
        "{:<12} {:>9} {:>10} {:>11} {:>10} {:>13} {:>12} {:>11} {:>10} {:>8}",
        "Square",
        "payload",
        "extended",
        "encode ms",
        "xform ms",
        "encode GiB/s",
        "xform GiB/s",
        "encode Gb/s",
        "xform Gb/s",
        "valid"
    );
    println!("{}", "-".repeat(118));
    for r in results {
        println!(
            "{:<12} {:>8.0}M {:>9.0}M {:>11.1} {:>10.1} {:>13.2} {:>12.2} {:>11.1} {:>10.1} {:>8}",
            format!("{}x{}", r.side, r.side),
            r.payload_mib,
            r.extended_mib,
            r.encode_ns as f64 / 1e6,
            r.transform_ns as f64 / 1e6,
            r.encode_gibs,
            r.transform_gibs,
            gibs_to_gbits(r.encode_gibs),
            gibs_to_gbits(r.transform_gibs),
            valid_label(r.validated)
        );
    }
    println!("{}", "=".repeat(118));

    let big = results.last().unwrap();
    println!(
        "\nLargest square: {}x{} ({:.0} MiB payload, {:.0} MiB extended). Transforms in {:.0} ms \
         of a {:.0} ms encode, so {:.0}% of the wall clock is host layout work.",
        big.side,
        big.side,
        big.payload_mib,
        big.extended_mib,
        big.transform_ns as f64 / 1e6,
        big.encode_ns as f64 / 1e6,
        100.0 * (big.encode_ns - big.transform_ns) as f64 / big.encode_ns as f64
    );
    println!(
        "At that size: {:.1} Gb/s transform, {:.1} Gb/s end-to-end.",
        gibs_to_gbits(big.transform_gibs),
        gibs_to_gbits(big.encode_gibs)
    );

    let basis = projection_basis(results);
    if basis.side != big.side {
        println!(
            "\nNote: at side > {ENCODE_UNPRESSURED_MAX_SIDE} the pinned square alone is {:.0} GiB and \
             the host runs short of memory, so the `encode` column there is degraded by memory \
             pressure and understates the pipeline. `GPU xform` is unaffected - the device only \
             sees streamed chunks. Extrapolations below therefore use side {}.",
            big.extended_mib / 1024.0,
            basis.side
        );
    }
    println!(
        "\nSquare side sustainable per time budget, extrapolated from the {}x{} rates ({:.2} GiB/s \
         transform, {:.2} GiB/s end-to-end). The end-to-end rate falls as squares grow, so its peak \
         (at the smallest square) would overstate the ceiling.",
        basis.side, basis.side, basis.transform_gibs, basis.encode_gibs
    );
    for budget in [1.0f64, 6.0] {
        println!(
            "  {:>4.0} s : {:>7.0} shares/side at the transform rate, {:>7.0} at the end-to-end rate",
            budget,
            side_within(basis.transform_gibs, budget),
            side_within(basis.encode_gibs, budget)
        );
    }
}
