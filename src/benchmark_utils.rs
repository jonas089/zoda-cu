use crate::babybear::BabyBear;
use crate::ntt_babybear::{intt as cpu_intt, ntt as cpu_ntt};
use sha2::{Digest, Sha256};
use std::time::Instant;

#[cfg(feature = "cuda")]
use crate::cuda_ntt::{cuda_rs_encode_vertical, CudaBuffer};

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

#[cfg(feature = "cuda")]
pub fn encode_gpu_with_output(
    config: &EncodingConfig,
) -> Result<(Vec<Vec<BabyBear>>, u64), String> {
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
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            let total_checks = num_column_checks + check_idx;
            return Ok((false, elapsed, total_checks));
        }
    }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    let total_checks = num_column_checks + num_rlc_checks;

    Ok((true, elapsed, total_checks))
}
