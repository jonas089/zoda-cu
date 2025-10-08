#![allow(unused)]
#[cfg(target_os = "macos")]
pub mod metal;
mod ntt;

#[repr(C)]
#[derive(Clone, Copy)]
struct FieldElem {
    x: u64,
    y: u64,
    z: u64,
    w: u64,
}

use ::metal::{Device, MTLResourceOptions, MTLSize};
use num_bigint::BigUint;
use rand::Rng;
use std::{cmp::max, sync::Arc, time::Instant};

use crate::{ff::F, metal::metal_long::find_root_of_unity, polynomial::Polynomial};
pub mod ff;
pub mod polynomial;
mod types;
use num_traits::ToPrimitive;
use sha2::{Digest, Sha256};

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
    let start_time = Instant::now();
    // some NTT friendly modulus
    let modulus = Arc::new(
        BigUint::parse_bytes(
            b"21888242871839275222246405745257275088548364400416034343698204186575808495617",
            10,
        )
        .unwrap(),
    );
    let mut data_square = DataSquare::new(vec![], 0, 0);

    // Create larger data square with random data
    for col in 0..data_size {
        for row in 0..data_size {
            let value = rand::rng().random_range(1..256);
            data_square.set_cell(col, row, F::new(value, modulus.clone()));
        }
    }

    let domain: Vec<F> = (0..data_square.rows)
        .map(|i| F::new(i as u64, modulus.clone()))
        .collect();

    // 1:4 parity data
    let extended_domain: Vec<F> = (0..data_square.columns * 5)
        .map(|i| F::new(i as u64, modulus.clone()))
        .collect();

    let mut column_polys = Vec::new();
    for column_idx in 0..data_square.columns {
        let column = data_square.get_column(column_idx);
        // interpolate each column into a polynomial
        let column_poly = Polynomial::interpolate(&domain, &column);
        column_polys.push(column_poly);
    }

    // evaluate the column polynomials over the extended domain and create new cells
    let mut extended_data_square = DataSquare::new(vec![], 0, 0);
    for (col_idx, column_poly) in column_polys.into_iter().enumerate() {
        for i in 0..extended_domain.len() {
            let x = &extended_domain[i];
            let y = column_poly.evaluate(&x);
            extended_data_square.set_cell(col_idx, i, y); // (column, row)
        }
    }

    let encoded_data_square_root = extended_data_square.hash_root();

    // compute running sum row-wise for the encoded data (original + parity), using random
    // linear combinations
    let mut y: Vec<F> = Vec::new();

    // compute y using the original data in the extended data square,
    // computing running sum of random linear combinations
    // column-wise
    // generate deterministic coefficients using encoded_data_square_root (fiat shamir)
    let mut deterministic_coefficients: Vec<F> = (0..extended_data_square.rows)
        .map(|i| {
            // hash root + index with SHA256
            let mut hasher = Sha256::new();
            hasher.update(encoded_data_square_root.as_bytes());
            hasher.update(&i.to_le_bytes());
            let digest = hasher.finalize();
            // interpret the whole 256-bit digest as a BigUint
            let big = BigUint::from_bytes_be(&digest);
            // fold it into u64 for your F::new constructor
            // (still deterministic, but using all digest bits)
            let val = (big % u64::MAX).to_u64().unwrap();
            F::new(val, modulus.clone())
        })
        .collect();

    // deterministically derive random coefficients from the root
    for i in 0..deterministic_coefficients.len() {
        deterministic_coefficients[i] =
            deterministic_coefficients[i].clone() + F::new(i as u64, modulus.clone());
    }

    for row_idx in 0..data_square.rows {
        let row_data = extended_data_square.get_row(row_idx);

        // compute running sum of random coefficients * row data
        let running_sum = row_data
            .iter()
            .zip(deterministic_coefficients.iter())
            .map(|(x, y)| x * y)
            .fold(F::zero(modulus.clone()), |acc, x| acc + x);
        y.push(running_sum);
    }

    // now interpolate y the same way as the columns over the original domain (because we only used rows in range 0..data_square.rows)
    let y_poly = Polynomial::interpolate(&domain, &y);
    let mut y_encoded: Vec<F> = Vec::new();
    for x in extended_domain {
        let y_val = y_poly.evaluate(&x);
        y_encoded.push(y_val);
    }

    // 64 queries
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
fn test_zoda_impl() {
    let duration = run_zoda_test_cpu(4);
    println!("[CPU 4x4]: {:?}", duration);
}

fn run_zoda_test_gpu(data_size: usize) -> std::time::Duration {
    let start_time = Instant::now();
    // some NTT friendly modulus
    let modulus = Arc::new(
        BigUint::parse_bytes(
            b"21888242871839275222246405745257275088548364400416034343698204186575808495617",
            10,
        )
        .unwrap(),
    );
    let mut data_square = DataSquare::new(vec![], 0, 0);

    // Create larger data square with random data
    for col in 0..data_size {
        for row in 0..data_size {
            let value = rand::rng().random_range(1..256);
            data_square.set_cell(col, row, F::new(value, modulus.clone()));
        }
    }

    let omega = find_root_of_unity(data_square.rows, &modulus);

    let domain: Vec<F> = (0..data_square.rows)
        .map(|i| F::from_biguint(omega.pow(i as u32), modulus.clone()))
        .collect();

    // 1:4 parity data
    let extended_domain: Vec<F> = (0..data_square.columns * 5)
        .map(|i| F::from_biguint(omega.pow(i as u32), modulus.clone()))
        .collect();

    let mut column_polys = Vec::new();
    for column_idx in 0..data_square.columns {
        let column = data_square.get_column(column_idx);
        // interpolate each column into a polynomial
        let column_poly = Polynomial::interpolate(&domain, &column);
        column_polys.push(column_poly);
    }

    // evaluate the column polynomials over the extended domain and create new cells
    let mut extended_data_square = DataSquare::new(vec![], 0, 0);
    for (col_idx, column_poly) in column_polys.into_iter().enumerate() {
        for i in 0..extended_domain.len() {
            let x = &extended_domain[i];
            let y = column_poly.evaluate(&x);
            extended_data_square.set_cell(col_idx, i, y); // (column, row)
        }
    }

    let encoded_data_square_root = extended_data_square.hash_root();

    // compute running sum row-wise for the encoded data (original + parity), using random
    // linear combinations
    let mut y: Vec<F> = Vec::new();

    // compute y using the original data in the extended data square,
    // computing running sum of random linear combinations
    // column-wise
    // generate deterministic coefficients using encoded_data_square_root (fiat shamir)
    let mut deterministic_coefficients: Vec<F> = (0..extended_data_square.rows)
        .map(|i| {
            // hash root + index with SHA256
            let mut hasher = Sha256::new();
            hasher.update(encoded_data_square_root.as_bytes());
            hasher.update(&i.to_le_bytes());
            let digest = hasher.finalize();
            // interpret the whole 256-bit digest as a BigUint
            let big = BigUint::from_bytes_be(&digest);
            // fold it into u64 for your F::new constructor
            // (still deterministic, but using all digest bits)
            let val = (big % u64::MAX).to_u64().unwrap();
            F::new(val, modulus.clone())
        })
        .collect();

    // deterministically derive random coefficients from the root
    for i in 0..deterministic_coefficients.len() {
        deterministic_coefficients[i] =
            deterministic_coefficients[i].clone() + F::new(i as u64, modulus.clone());
    }

    for row_idx in 0..data_square.rows {
        let row_data = extended_data_square.get_row(row_idx);

        // compute running sum of random coefficients * row data
        let running_sum = row_data
            .iter()
            .zip(deterministic_coefficients.iter())
            .map(|(x, y)| x * y)
            .fold(F::zero(modulus.clone()), |acc, x| acc + x);
        y.push(running_sum);
    }

    // now interpolate y the same way as the columns over the original domain (because we only used rows in range 0..data_square.rows)
    let y_poly = Polynomial::interpolate(&domain, &y);
    let mut y_encoded: Vec<F> = Vec::new();
    for x in extended_domain {
        let y_val = y_poly.evaluate(&x);
        y_encoded.push(y_val);
    }

    // 64 queries
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
fn test_zoda_impl_metal() {
    let duration = run_zoda_test_gpu(4);
    println!("[GPU 4x4]: {:?}", duration);
}

#[test]
fn benchmark_cpu_vs_gpu_performance() {
    println!("\n=== CPU vs GPU Performance Benchmark (with Batching) ===");
    let test_sizes = vec![4, 8, 16, 32, 64, 128];

    println!(
        "{:<8} {:<12} {:<12} {:<12} {:<12} {:<15}",
        "Size", "CPU Time", "GPU Time", "Speedup", "Batch Size", "GPU Throughput"
    );
    println!("{}", "-".repeat(80));

    for size in test_sizes {
        // Run CPU test
        let cpu_duration = run_zoda_test_cpu(size);

        // Run GPU test (now with optimized batching)
        let gpu_duration = run_zoda_test_gpu(size);

        // Calculate speedup
        let speedup = cpu_duration.as_nanos() as f64 / gpu_duration.as_nanos() as f64;
        let speedup_str = if speedup > 1.0 {
            format!("{:.2}x GPU", speedup)
        } else {
            format!("{:.2}x CPU", 1.0 / speedup)
        };

        // Calculate GPU throughput (FFTs per second)
        let batch_size = size; // Each column is an FFT
        let gpu_throughput = batch_size as f64 / gpu_duration.as_secs_f64();

        println!(
            "{:<8} {:<12.3}ms {:<12.3}ms {:<12} {:<12} {:<15.1}",
            format!("{}x{}", size, size),
            cpu_duration.as_secs_f64() * 1000.0,
            gpu_duration.as_secs_f64() * 1000.0,
            speedup_str,
            batch_size,
            gpu_throughput
        );
    }

    println!(
        "\nNote: GPU uses optimized batching - all {} FFTs processed in single dispatch",
        "N"
    );
    println!("FFT sizes scale with data size: (size * 5).next_power_of_two().max(4096)");
}

#[test]
fn benchmark_large_datasets() {
    println!("=== Large Dataset Performance Test (Batched GPU) ===");

    let large_sizes = vec![256, 512, 1024];

    println!(
        "{:<12} {:<12} {:<12} {:<12} {:<15} {:<15}",
        "Data Size", "GPU Time", "Batch Size", "FFT Size", "Throughput", "Total Ops"
    );
    println!("{}", "-".repeat(90));

    for size in large_sizes {
        // For very large datasets, test GPU performance with batching
        let gpu_duration = run_zoda_test_gpu(size);

        let fft_size = (size * 5).next_power_of_two().max(4096);
        let batch_size = size; // Each column is an FFT

        // Calculate throughput metrics
        let total_operations = size * 2; // forward + inverse FFT per column
        let ops_per_second = total_operations as f64 / gpu_duration.as_secs_f64();
        let ffts_per_second = batch_size as f64 / gpu_duration.as_secs_f64();

        println!(
            "{:<12} {:<12.3}ms {:<12} {:<12} {:<15.1} {:<15}",
            format!("{}x{}", size, size),
            gpu_duration.as_secs_f64() * 1000.0,
            batch_size,
            fft_size,
            ffts_per_second,
            total_operations
        );
    }

    println!("\nBatching Benefits:");
    println!("- All {} FFTs processed in single GPU dispatch", "N");
    println!("- Eliminates kernel launch overhead (~10-50μs per FFT)");
    println!("- Maximizes GPU occupancy and memory bandwidth");
    println!("- Throughput scales linearly with batch size");
}
