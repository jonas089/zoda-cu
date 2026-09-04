// ZODA implementation using BabyBear field with GPU acceleration

use crate::field::babybear::BabyBear;
use crate::ntt::{intt_babybear as intt, ntt_babybear as ntt};
use rand::Rng;
use sha2::{Digest, Sha256};
use std::cmp::max;

#[cfg(feature = "cuda")]
use crate::ntt::cuda::{cuda_available, intt_cuda, ntt_cuda, PinnedSquare};

/// GPU transform of polynomials laid end to end, each `n` long, in place.
#[cfg(feature = "cuda")]
fn gpu_transform(values: &mut [BabyBear], n: usize, inverse: bool) {
    let mut square = PinnedSquare::new(n, values.len() / n).expect("pinned alloc failed");
    for (i, v) in values.iter().enumerate() {
        square[i] = v.value as u32;
    }
    if inverse {
        intt_cuda(&mut square, n, n).expect("CUDA INTT failed");
    } else {
        ntt_cuda(&mut square, n, n).expect("CUDA NTT failed");
    }
    for (v, r) in values.iter_mut().zip(square.iter()) {
        *v = BabyBear::new(*r as u64);
    }
}

/// Data cell in the square
#[derive(Clone)]
pub struct BabyBearCell {
    pub value: BabyBear,
    pub column: usize,
    pub row: usize,
}

/// Data square for ZODA protocol
pub struct BabyBearDataSquare {
    pub cells: Vec<BabyBearCell>,
    pub columns: usize,
    pub rows: usize,
}

impl BabyBearDataSquare {
    pub fn new(cells: Vec<BabyBearCell>, columns: usize, rows: usize) -> Self {
        Self {
            cells,
            columns,
            rows,
        }
    }

    pub fn set_cell(&mut self, column: usize, row: usize, value: BabyBear) {
        if let Some(cell) = self
            .cells
            .iter_mut()
            .find(|c| c.column == column && c.row == row)
        {
            cell.value = value;
        } else {
            self.cells.push(BabyBearCell { value, column, row });
        }
        self.rows = max(self.rows, row + 1);
        self.columns = max(self.columns, column + 1);
    }

    pub fn get_row(&self, row: usize) -> Vec<BabyBear> {
        let mut row_cells: Vec<_> = self.cells.iter().filter(|cell| cell.row == row).collect();
        row_cells.sort_by_key(|c| c.column);
        row_cells.into_iter().map(|c| c.value).collect()
    }

    pub fn get_column(&self, column: usize) -> Vec<BabyBear> {
        let mut col_cells: Vec<_> = self
            .cells
            .iter()
            .filter(|cell| cell.column == column)
            .collect();
        col_cells.sort_by_key(|c| c.row);
        col_cells.into_iter().map(|c| c.value).collect()
    }

    pub fn hash_root(&self) -> String {
        let mut hasher = Sha256::new();
        let all_bytes: Vec<u8> = self
            .cells
            .iter()
            .flat_map(|cell| cell.value.to_bytes())
            .collect();
        hasher.update(&all_bytes);
        format!("{:x}", hasher.finalize())
    }
}

/// Run ZODA test with BabyBear field
/// Use GPU acceleration if available
pub fn run_zoda_test_babybear(data_size: usize, use_gpu: bool) -> std::time::Duration {
    use std::time::Instant;
    let start_time = Instant::now();

    #[cfg(feature = "cuda")]
    let gpu_available = use_gpu && cuda_available();
    #[cfg(not(feature = "cuda"))]
    let gpu_available = false;

    if use_gpu && !gpu_available {
        println!("Warning: GPU requested but not available, falling back to CPU");
    }

    let mut data_square = BabyBearDataSquare::new(vec![], 0, 0);
    for col in 0..data_size {
        for row in 0..data_size {
            let value = rand::rng().random_range(1..256);
            data_square.set_cell(col, row, BabyBear::new(value));
        }
    }

    // NTT domain (power-of-two)
    let ntt_n = data_square.rows.next_power_of_two();
    let cols = data_square.columns;

    // All columns one after another, each zero-padded to ntt_n values:
    // column c is square[c * ntt_n .. (c + 1) * ntt_n].
    let mut square = vec![BabyBear::zero(); ntt_n * cols];
    for col in 0..cols {
        for (row, value) in data_square.get_column(col).into_iter().enumerate() {
            square[col * ntt_n + row] = value;
        }
    }

    // INTT to get coefficients, NTT to get evaluations: every column in one call each.
    if gpu_available {
        #[cfg(feature = "cuda")]
        gpu_transform(&mut square, ntt_n, true);
        #[cfg(feature = "cuda")]
        gpu_transform(&mut square, ntt_n, false);
    } else {
        for column in square.chunks_mut(ntt_n) {
            intt(column);
            ntt(column);
        }
    }

    let mut extended_data_square = BabyBearDataSquare::new(vec![], 0, 0);
    for col in 0..cols {
        for row in 0..ntt_n {
            extended_data_square.set_cell(col, row, square[col * ntt_n + row]);
        }
    }

    let encoded_data_square_root = extended_data_square.hash_root();

    // Generate deterministic coefficients
    let mut deterministic_coefficients: Vec<BabyBear> = (0..extended_data_square.rows)
        .map(|i| {
            let mut hasher = Sha256::new();
            hasher.update(encoded_data_square_root.as_bytes());
            hasher.update(&i.to_le_bytes());
            let digest = hasher.finalize();
            let val = u64::from_be_bytes([
                digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6],
                digest[7],
            ]);
            BabyBear::new(val)
        })
        .collect();

    for i in 0..deterministic_coefficients.len() {
        deterministic_coefficients[i] = deterministic_coefficients[i] + BabyBear::new(i as u64);
    }

    let mut y: Vec<BabyBear> = Vec::new();
    for row_idx in 0..data_square.rows {
        let row_data = extended_data_square.get_row(row_idx);
        let mut running_sum = BabyBear::zero();
        for (x, coeff) in row_data.iter().zip(deterministic_coefficients.iter()) {
            running_sum = running_sum + (*x * *coeff);
        }
        y.push(running_sum);
    }

    let mut y_coeffs = y.clone();
    y_coeffs.resize(ntt_n, BabyBear::zero());
    if gpu_available {
        #[cfg(feature = "cuda")]
        gpu_transform(&mut y_coeffs, ntt_n, true);
    } else {
        intt(&mut y_coeffs);
    }

    // Evaluate y over extended domain
    let mut y_encoded = y_coeffs.clone();
    if gpu_available {
        #[cfg(feature = "cuda")]
        gpu_transform(&mut y_encoded, ntt_n, false);
    } else {
        ntt(&mut y_encoded);
    }

    for _ in 0..64 {
        let random_row = rand::rng().random_range(0..extended_data_square.rows);
        let row_data = extended_data_square.get_row(random_row);
        let mut running_sum = BabyBear::zero();
        for (x, coeff) in row_data.iter().zip(deterministic_coefficients.iter()) {
            running_sum = running_sum + (*x * *coeff);
        }

        assert_eq!(running_sum.value, y_encoded[random_row].value);
    }

    start_time.elapsed()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zoda_babybear_cpu() {
        let duration = run_zoda_test_babybear(4, false);
        println!("[BabyBear CPU 4x4]: {:?}", duration);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_zoda_babybear_gpu() {
        if !cuda_available() {
            println!("CUDA not available, skipping GPU test");
            return;
        }
        let duration = run_zoda_test_babybear(4, true);
        println!("[BabyBear GPU 4x4]: {:?}", duration);
    }

    #[test]
    fn test_compare_cpu_gpu() {
        for size in [4, 8, 16, 32] {
            println!("Testing {}x{} data square:", size, size);

            let duration_cpu = run_zoda_test_babybear(size, false);
            println!("  BabyBear CPU:   {:?}", duration_cpu);

            #[cfg(feature = "cuda")]
            if cuda_available() {
                let duration_gpu = run_zoda_test_babybear(size, true);
                println!("  BabyBear GPU:   {:?}", duration_gpu);
                println!(
                    "  GPU Speedup vs CPU: {:.2}x",
                    duration_cpu.as_secs_f64() / duration_gpu.as_secs_f64()
                );
            } else {
                println!("  BabyBear GPU:   Not available");
            }

            println!();
        }
    }
}
