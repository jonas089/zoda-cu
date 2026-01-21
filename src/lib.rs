#![allow(unused)]
pub mod babybear;
mod ntt;
pub mod ntt_babybear;
pub mod zoda_babybear;

#[cfg(feature = "cuda")]
pub mod cuda_ntt;

#[cfg(test)]
mod benchmark_zoda_optimal;

#[repr(C)]
#[derive(Clone, Copy)]
struct FieldElem {
    x: u64,
    y: u64,
    z: u64,
    w: u64,
}

#[cfg(target_os = "macos")]
use ::metal::{Device, MTLResourceOptions, MTLSize};

use num_bigint::BigUint;
use rand::Rng;
use std::{cmp::max, sync::Arc, time::Instant};

pub mod ff;
pub mod polynomial;
mod types;
use num_traits::ToPrimitive;
use sha2::{Digest, Sha256};

use crate::{ff::F, ntt::ntt, polynomial::Polynomial};

pub struct DataSquare {
    pub cells: Vec<Cell>,
    pub columns: usize,
    pub rows: usize,
}
pub struct Cell {
    pub value: F,
    pub column: usize,
    pub row: usize,
}

impl DataSquare {
    pub fn new(cells: Vec<Cell>, columns: usize, rows: usize) -> Self {
        Self {
            cells,
            columns,
            rows,
        }
    }

    pub fn get_cell(&self, column: usize, row: usize) -> Option<&Cell> {
        self.cells
            .iter()
            .find(|cell| cell.column == column && cell.row == row)
    }

    pub fn set_cell(&mut self, column: usize, row: usize, value: F) {
        if let Some(cell) = self
            .cells
            .iter_mut()
            .find(|c| c.column == column && c.row == row)
        {
            cell.value = value;
        } else {
            self.cells.push(Cell { value, column, row });
        }
        self.rows = max(self.rows, row + 1); // store dimensions as counts
        self.columns = max(self.columns, column + 1);
    }

    pub fn get_row(&self, row: usize) -> Vec<F> {
        let mut row_cells: Vec<_> = self.cells.iter().filter(|cell| cell.row == row).collect();
        row_cells.sort_by_key(|c| c.column);
        row_cells.into_iter().map(|c| c.value.clone()).collect()
    }

    pub fn get_column(&self, column: usize) -> Vec<F> {
        let mut col_cells: Vec<_> = self
            .cells
            .iter()
            .filter(|cell| cell.column == column)
            .collect();
        col_cells.sort_by_key(|c| c.row);
        col_cells.into_iter().map(|c| c.value.clone()).collect()
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

fn run_zoda_test_cpu(data_size: usize) -> std::time::Duration {
    use crate::ntt::{intt, roots_of_unity_domain};
    let start_time = Instant::now();

    // BN254 modulus
    let modulus = Arc::new(
        BigUint::parse_bytes(
            b"21888242871839275222246405745257275088548364400416034343698204186575808495617",
            10,
        )
        .unwrap(),
    );

    // Build data square
    let mut data_square = DataSquare::new(vec![], 0, 0);
    for col in 0..data_size {
        for row in 0..data_size {
            let value = rand::rng().random_range(1..256);
            data_square.set_cell(col, row, F::new(value, modulus.clone()));
        }
    }

    // NTT domain (power-of-two), primitive root ω = domain[1]
    let ntt_n = data_square.rows.next_power_of_two();
    let domain = roots_of_unity_domain(ntt_n, modulus.clone());
    let omega = domain[1].clone();

    // We'll also use a roots-of-unity "extended" domain (same ω) for parity/evals,
    // mirroring your GPU code: just take the first columns*5 powers.
    let extended_domain: Vec<F> = (0..(data_square.columns * 5))
        .map(|i| {
            let mut x = F::new(1, modulus.clone());
            // fast pow by repeated multiply (avoids BigUint pow in tight loop)
            for _ in 0..i {
                x = &x * &omega;
            }
            x
        })
        .collect();

    let mut column_polys = Vec::new();
    for column_idx in 0..data_square.columns {
        // column values are evaluations at [1, ω, ω^2, ...]; pad to ntt_n
        let mut evals = data_square.get_column(column_idx);
        evals.resize(ntt_n, F::zero(modulus.clone()));

        let mut coeffs = evals;
        intt(&mut coeffs, &omega);

        column_polys.push(Polynomial::from_coeffs(coeffs));
    }

    let mut extended_data_square = DataSquare::new(vec![], 0, 0);
    for (col_idx, column_poly) in column_polys.into_iter().enumerate() {
        let mut evals = column_poly.coeffs.clone();
        ntt(&mut evals, &omega);
        for (i, y) in evals.into_iter().enumerate() {
            extended_data_square.set_cell(col_idx, i, y);
        }
    }

    let encoded_data_square_root = extended_data_square.hash_root();

    let mut deterministic_coefficients: Vec<F> = (0..extended_data_square.rows)
        .map(|i| {
            let mut hasher = Sha256::new();
            hasher.update(encoded_data_square_root.as_bytes());
            hasher.update(&i.to_le_bytes());
            let digest = hasher.finalize();
            let big = BigUint::from_bytes_be(&digest);
            let val = (big % u64::MAX).to_u64().unwrap();
            F::new(val, modulus.clone())
        })
        .collect();

    for i in 0..deterministic_coefficients.len() {
        deterministic_coefficients[i] =
            deterministic_coefficients[i].clone() + F::new(i as u64, modulus.clone());
    }

    let mut y: Vec<F> = Vec::new();
    for row_idx in 0..data_square.rows {
        let row_data = extended_data_square.get_row(row_idx);
        let running_sum = row_data
            .iter()
            .zip(deterministic_coefficients.iter())
            .map(|(x, y)| x * y)
            .fold(F::zero(modulus.clone()), |acc, x| acc + x);
        y.push(running_sum);
    }

    // Interpolate y over the same roots-of-unity domain using Intt as well
    let mut y_coeffs = y.clone();
    y_coeffs.resize(ntt_n, F::zero(modulus.clone()));
    intt(&mut y_coeffs, &omega);
    let y_poly = Polynomial::from_coeffs(y_coeffs);

    // Evaluate y over the extended domain
    let mut y_encoded: Vec<F> = Vec::with_capacity(extended_domain.len());
    let mut evals = y_poly.coeffs.clone();
    ntt(&mut evals, &omega);
    for (i, y) in evals.into_iter().enumerate() {
        y_encoded.push(y);
    }

    // Simulate sampling
    for _ in 0..64 {
        let random_row = rand::rng().random_range(0..extended_data_square.rows);
        let row_data = extended_data_square.get_row(random_row);
        let running_sum = row_data
            .iter()
            .zip(deterministic_coefficients.iter())
            .map(|(x, y)| x * y)
            .fold(F::zero(modulus.clone()), |acc, x| acc + x);

        assert_eq!(running_sum, y_encoded[random_row]);
    }

    start_time.elapsed()
}

#[test]
fn test_zoda_impl_cpu() {
    let duration = run_zoda_test_cpu(4);
    println!("[CPU 4x4]: {:?}", duration);
}

#[test]
fn test_compare_implementations() {
    use crate::zoda_babybear::run_zoda_test_babybear;

    let sizes = vec![4, 8, 16, 32];

    for size in sizes {
        println!("Testing {}x{} data square:", size, size);

        // Old BigInt implementation
        let duration_bigint = run_zoda_test_cpu(size);
        println!("  BigInt CPU:     {:?}", duration_bigint);

        // BabyBear CPU implementation
        let duration_bb_cpu = run_zoda_test_babybear(size, false);
        println!("  BabyBear CPU:   {:?}", duration_bb_cpu);

        let speedup = duration_bigint.as_secs_f64() / duration_bb_cpu.as_secs_f64();
        println!("  CPU Speedup:    {:.2}x", speedup);

        // BabyBear GPU implementation (if available)
        #[cfg(feature = "cuda")]
        {
            use crate::cuda_ntt::cuda_available;
            if cuda_available() {
                let duration_bb_gpu = run_zoda_test_babybear(size, true);
                println!("  BabyBear GPU:   {:?}", duration_bb_gpu);

                let gpu_speedup_vs_bigint =
                    duration_bigint.as_secs_f64() / duration_bb_gpu.as_secs_f64();
                let gpu_speedup_vs_cpu =
                    duration_bb_cpu.as_secs_f64() / duration_bb_gpu.as_secs_f64();
                println!("  GPU Speedup vs BigInt: {:.2}x", gpu_speedup_vs_bigint);
                println!("  GPU Speedup vs CPU:    {:.2}x", gpu_speedup_vs_cpu);
            } else {
                println!("  BabyBear GPU:   Not available");
            }
        }

        println!();
    }
}
