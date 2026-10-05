// ZODA implementation using BabyBear field with GPU acceleration

use crate::field::babybear::BabyBear;
use crate::ntt::{intt_babybear as intt, ntt_babybear as ntt};
use crate::zoda::rlc::{generate_deterministic_coefficients, rlc_row, RLC_LIMBS};
use rand::Rng;
use sha2::{Digest, Sha256};
use std::cmp::max;

#[cfg(feature = "cuda")]
use crate::ntt::cuda::{cuda_available, intt_cuda, ntt_cuda, PinnedSquare};

/// GPU transform of polynomials laid end to end, each `stride` long, in place.
/// Only the first `n` values of every polynomial are transformed.
#[cfg(feature = "cuda")]
fn gpu_transform(values: &mut [BabyBear], n: usize, stride: usize, inverse: bool) {
    let mut square =
        PinnedSquare::new(stride, values.len() / stride).expect("pinned alloc failed");
    for (i, v) in values.iter().enumerate() {
        square[i] = v.value as u32;
    }
    if inverse {
        intt_cuda(&mut square, n, stride).expect("CUDA INTT failed");
    } else {
        ntt_cuda(&mut square, n, stride).expect("CUDA NTT failed");
    }
    for (v, r) in values.iter_mut().zip(square.iter()) {
        *v = BabyBear::new(*r as u64);
    }
}

/// Reed-Solomon extend polynomials laid end to end, each `n_kn` long with its
/// data in the first `n_k` values: INTT those, then NTT all `n_kn`.
fn extend(values: &mut [BabyBear], n_k: usize, n_kn: usize, gpu_available: bool) {
    if gpu_available {
        #[cfg(feature = "cuda")]
        gpu_transform(values, n_k, n_kn, true);
        #[cfg(feature = "cuda")]
        gpu_transform(values, n_kn, n_kn, false);
    } else {
        for poly in values.chunks_mut(n_kn) {
            intt(&mut poly[..n_k]);
            ntt(poly);
        }
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

    // Interpolate the k data rows at size n_k, then evaluate at n_kn = 2 * n_k
    // points. The NTT has to be larger than the INTT: at the same size the two
    // cancel and the "encoding" hands back its input with no parity rows.
    let k = data_square.rows;
    let n_k = k.next_power_of_two();
    let n_kn = (2 * k).next_power_of_two();
    let cols = data_square.columns;

    // All columns one after another, each zero-padded to n_kn values:
    // column c is square[c * n_kn .. (c + 1) * n_kn].
    let mut square = vec![BabyBear::zero(); n_kn * cols];
    for col in 0..cols {
        for (row, value) in data_square.get_column(col).into_iter().enumerate() {
            square[col * n_kn + row] = value;
        }
    }

    // INTT over the first n_k rows of every column, NTT over all n_kn of them.
    extend(&mut square, n_k, n_kn, gpu_available);

    let mut extended_data_square = BabyBearDataSquare::new(vec![], 0, 0);
    for col in 0..cols {
        for row in 0..n_kn {
            extended_data_square.set_cell(col, row, square[col * n_kn + row]);
        }
    }

    let encoded_data_square_root = extended_data_square.hash_root();

    // One coefficient per column, each `RLC_LIMBS` base-field elements, so a
    // forged row passes the check with probability `p^-RLC_LIMBS` rather than
    // `1/p`. See `zoda::rlc`.
    let coefficients = generate_deterministic_coefficients(&encoded_data_square_root, cols);

    // The RLC of each original row, one sequence per limb, laid end to end so
    // every limb is extended in a single call each way.
    // The encoding is not systematic, so these are the rows of `data_square`,
    // not the first k rows of the extended square.
    let mut y = vec![BabyBear::zero(); RLC_LIMBS * n_kn];
    for row_idx in 0..k {
        let row_rlc = rlc_row(&data_square.get_row(row_idx), &coefficients);
        for limb in 0..RLC_LIMBS {
            y[limb * n_kn + row_idx] = row_rlc[limb];
        }
    }

    // Evaluate y over extended domain, limb by limb
    extend(&mut y, n_k, n_kn, gpu_available);

    for _ in 0..64 {
        let random_row = rand::rng().random_range(0..extended_data_square.rows);
        let row_rlc = rlc_row(&extended_data_square.get_row(random_row), &coefficients);
        for limb in 0..RLC_LIMBS {
            assert_eq!(row_rlc[limb].value, y[limb * n_kn + random_row].value);
        }
    }

    // At least one parity row per data row; with n_kn == n_k the check below
    // would pass vacuously on an identity "extension".
    assert!(n_kn >= 2 * k, "no parity rows: {n_kn} rows for {k} data rows");

    // Every column must be a codeword: a polynomial of degree < n_k, so its
    // coefficients above n_k are zero. It must also be this data's codeword:
    // INTT then NTT puts the data at every (n_kn / n_k)-th row.
    let data_stride = n_kn / n_k;
    for col in 0..cols {
        for (row, value) in data_square.get_column(col).into_iter().enumerate() {
            assert_eq!(square[col * n_kn + row * data_stride].value, value.value);
        }
        let mut coeffs = square[col * n_kn..(col + 1) * n_kn].to_vec();
        intt(&mut coeffs);
        assert!(
            coeffs[n_k..].iter().all(|c| c.value == 0),
            "column {col} is not a Reed-Solomon codeword"
        );
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
