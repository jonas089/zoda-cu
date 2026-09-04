use crate::field::babybear::BabyBear;
use crate::ntt::{intt_babybear as cpu_intt, ntt_babybear as cpu_ntt};
use sha2::{Digest, Sha256};
use std::time::Instant;

#[cfg(feature = "cuda")]
use crate::ntt::cuda::{intt_cuda, ntt_cuda, PinnedSquare};

#[derive(Debug, Clone)]
pub struct EncodingConfig {
    pub k: usize,
    pub n: usize,
    pub row_size: usize,
}

impl EncodingConfig {
    pub fn data_size_bytes(&self) -> usize {
        self.k * self.row_size
    }

    pub fn data_size_kb(&self) -> f64 {
        self.data_size_bytes() as f64 / 1024.0
    }

    pub fn data_size_mb(&self) -> f64 {
        self.data_size_bytes() as f64 / (1024.0 * 1024.0)
    }

    pub fn total_rows(&self) -> usize {
        self.k + self.n
    }

    pub fn elements_per_row(&self) -> usize {
        self.row_size / 4
    }

    pub fn num_positions(&self) -> usize {
        self.elements_per_row()
    }

    pub fn ntt_size_k(&self) -> usize {
        self.k.next_power_of_two()
    }

    pub fn ntt_size_kn(&self) -> usize {
        (self.k + self.n).next_power_of_two()
    }
}

pub fn compute_data_root(encoded_rows: &[Vec<BabyBear>]) -> String {
    let mut hasher = Sha256::new();
    for row in encoded_rows {
        for &value in row {
            hasher.update(value.value.to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

pub fn generate_deterministic_coefficients(data_root: &str, num_columns: usize) -> Vec<BabyBear> {
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

/// How long one encode took.
///
/// `encode_ns` is the end-to-end number: row-major input already in host memory
/// -> encoded square in host memory. It covers the transpose into the column-major
/// pinned buffer, the zero padding, both transforms with their uploads and
/// downloads, and the per-call device setup inside the kernel wrapper. It does
/// not cover generating the synthetic input or gathering rows back out for
/// validation, neither of which a real encoder does.
///
/// `transform_ns` is only the two GPU transform calls inside that window, so
/// `encode_ns - transform_ns` is the host-side layout work.
#[derive(Debug, Clone, Copy, Default)]
pub struct EncodeTiming {
    pub encode_ns: u64,
    pub transform_ns: u64,
}

/// The synthetic input: `k` rows of `cols` field elements, row-major, with
/// element (row, col) = (row * cols + col) mod p. This is the data being encoded;
/// producing it is not part of the encode and is never timed.
pub fn generate_input(config: &EncodingConfig) -> Vec<u32> {
    let cols = config.num_positions();
    (0..config.k * cols).map(|i| BabyBear::new(i as u64).value).collect()
}

/// Timed iterations per configuration; the median is reported. Override with
/// `ZODA_BENCH_ITERS=<n>`.
fn bench_iters() -> usize {
    std::env::var("ZODA_BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(3)
}

/// One encode of `input` into `square`, timed. `square` must have `ntt_size_kn`
/// rows and `cols` columns and is fully overwritten, so it can be reused across
/// iterations (allocating it is the encoder's setup, not part of an encode).
#[cfg(feature = "cuda")]
pub fn encode_gpu(
    config: &EncodingConfig,
    input: &[u32],
    square: &mut PinnedSquare,
) -> Result<EncodeTiming, String> {
    let k = config.k;
    let cols = config.num_positions();
    let ntt_size_k = config.ntt_size_k();
    let ntt_size_kn = config.ntt_size_kn();
    assert_eq!(input.len(), k * cols, "input must be k rows of cols elements");
    assert_eq!((square.rows(), square.cols()), (ntt_size_kn, cols), "square has the wrong shape");

    let start = Instant::now();

    // Transpose the row-major input into one contiguous polynomial per column.
    // Any column-wise encoder has to do this before it can transform, so it is
    // inside the stopwatch. Tiled so neither side is walked with a huge stride.
    const TILE: usize = 64;
    for r0 in (0..k).step_by(TILE) {
        let r1 = (r0 + TILE).min(k);
        for c0 in (0..cols).step_by(TILE) {
            let c1 = (c0 + TILE).min(cols);
            for row in r0..r1 {
                let src = &input[row * cols + c0..row * cols + c1];
                for (col, &v) in (c0..c1).zip(src) {
                    square[col * ntt_size_kn + row] = v;
                }
            }
        }
    }
    // Rows k.. of every column are the zero padding for both transforms.
    for col in 0..cols {
        square[col * ntt_size_kn + k..(col + 1) * ntt_size_kn].fill(0);
    }

    // INTT over the first ntt_size_k rows of every column (stride ntt_size_kn
    // leaves the zero rows below untouched), then NTT over all ntt_size_kn rows.
    let transform_start = Instant::now();
    intt_cuda(&mut square[..], ntt_size_k, ntt_size_kn)?;
    ntt_cuda(&mut square[..], ntt_size_kn, ntt_size_kn)?;
    let transform_ns = transform_start.elapsed().as_nanos() as u64;

    Ok(EncodeTiming {
        encode_ns: start.elapsed().as_nanos() as u64,
        transform_ns,
    })
}

/// Benchmark one configuration: one untimed warm-up encode (CUDA context,
/// first-touch allocations, module load), then `bench_iters()` timed encodes of
/// the same input into the same buffer. Returns the median run and the encoded
/// rows of the last run for validation.
#[cfg(feature = "cuda")]
pub fn encode_gpu_with_output(
    config: &EncodingConfig,
) -> Result<(Vec<Vec<BabyBear>>, EncodeTiming), String> {
    let input = generate_input(config);
    let mut square = PinnedSquare::new(config.ntt_size_kn(), config.num_positions())?;

    encode_gpu(config, &input, &mut square)?;
    let mut runs = Vec::with_capacity(bench_iters());
    for _ in 0..bench_iters() {
        runs.push(encode_gpu(config, &input, &mut square)?);
    }
    runs.sort_by_key(|t| t.encode_ns);
    let median = runs[runs.len() / 2];

    let encoded_rows: Vec<Vec<BabyBear>> =
        (0..config.k + config.n).map(|row| square.row(row)).collect();

    Ok((encoded_rows, median))
}

#[cfg(feature = "cuda")]
pub fn validate_zoda_encoding(
    config: &EncodingConfig,
    encoded_rows: &[Vec<BabyBear>],
) -> Result<(bool, f64, usize), String> {
    let start = Instant::now();

    let k = config.k;
    let n = config.n;
    let num_positions = config.num_positions();
    let ntt_size_k = config.ntt_size_k();
    let ntt_size_kn = config.ntt_size_kn();

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
        cpu_intt(&mut original_input);
        original_input.resize(ntt_size_kn, BabyBear::zero());
        cpu_ntt(&mut original_input);

        // Verify GPU matches CPU
        for row_idx in 0..(k + n) {
            if original_input[row_idx].value != column[row_idx].value {
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                return Ok((false, elapsed, num_column_checks));
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
        cpu_intt(&mut column_data);
        column_data.resize(ntt_size_kn, BabyBear::zero());
        cpu_ntt(&mut column_data);

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
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            let total_checks = num_column_checks + check_idx;
            return Ok((false, elapsed, total_checks));
        }
    }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    let total_checks = num_column_checks + num_rlc_checks;

    Ok((true, elapsed, total_checks))
}
