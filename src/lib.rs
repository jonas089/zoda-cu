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
    use crate::ntt::{ifft, roots_of_unity_domain};
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
    let fft_n = data_square.rows.next_power_of_two();
    let domain = roots_of_unity_domain(fft_n, modulus.clone());
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

    // ----- Interpolate each column via IFFT (values -> coeffs) -----
    let mut column_polys = Vec::new();
    for column_idx in 0..data_square.columns {
        // column values are evaluations at [1, ω, ω^2, ...]; pad to fft_n
        let mut evals = data_square.get_column(column_idx);
        evals.resize(fft_n, F::zero(modulus.clone()));

        // IFFT over ω gives coefficients (scaled correctly; ifft() divides by n)
        let mut coeffs = evals;
        ifft(&mut coeffs, &omega);

        // Wrap as Polynomial so your downstream evaluate() stays unchanged
        column_polys.push(Polynomial::from_coeffs(coeffs));
    }

    // ----- Evaluate the column polys over the (roots-based) extended domain -----
    let mut extended_data_square = DataSquare::new(vec![], 0, 0);
    for (col_idx, column_poly) in column_polys.into_iter().enumerate() {
        for i in 0..extended_domain.len() {
            let x = &extended_domain[i];
            let y = column_poly.evaluate(&x);
            extended_data_square.set_cell(col_idx, i, y);
        }
    }

    let encoded_data_square_root = extended_data_square.hash_root();

    // ----- Fiat–Shamir coefficients (unchanged) -----
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

    // ----- Row-wise running sums -----
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

    // Interpolate y over the same roots-of-unity domain using IFFT as well
    let mut y_coeffs = y.clone();
    y_coeffs.resize(fft_n, F::zero(modulus.clone()));
    ifft(&mut y_coeffs, &omega);
    let y_poly = Polynomial::from_coeffs(y_coeffs);

    // Evaluate y over the extended domain
    let mut y_encoded: Vec<F> = Vec::with_capacity(extended_domain.len());
    for x in extended_domain {
        let y_val = y_poly.evaluate(&x);
        y_encoded.push(y_val);
    }

    // ----- Queries / checks (unchanged) -----
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
    use crate::metal::metal_long::*;
    use metal::*;
    use num_bigint::BigUint;
    use num_traits::{One, ToPrimitive};
    use rand::Rng;
    use sha2::{Digest, Sha256};
    use std::{ops::Deref, sync::Arc, time::Instant};

    let start_time = Instant::now();

    // ----------------------------
    // 1. Setup field + Metal device
    // ----------------------------
    let modulus = Arc::new(
        BigUint::parse_bytes(
            b"21888242871839275222246405745257275088548364400416034343698204186575808495617",
            10,
        )
        .unwrap(),
    );
    let (nprime, r2) = compute_montgomery_params(&modulus);
    let device = Device::system_default().unwrap();
    let library = device
        .new_library_with_file("./metal/fft-big.metallib")
        .unwrap();
    let shared_fft_kernel = library.get_function("fft_shared_memory", None).unwrap();
    let shared_fft_pipeline = device
        .new_compute_pipeline_state_with_function(&shared_fft_kernel)
        .unwrap();
    let bitrev_kernel = library.get_function("bitrev_permute", None).unwrap();
    let bitrev_pipeline = device
        .new_compute_pipeline_state_with_function(&bitrev_kernel)
        .unwrap();
    let command_queue = device.new_command_queue();

    // ----------------------------
    // 2. Create data square
    // ----------------------------
    let mut data_square = DataSquare::new(vec![], 0, 0);
    for col in 0..data_size {
        for row in 0..data_size {
            let value = rand::rng().random_range(1..256);
            data_square.set_cell(col, row, F::new(value, modulus.clone()));
        }
    }

    // ----------------------------
    // 3. Setup domain and root of unity
    // ----------------------------
    let fft_n = data_square.rows.next_power_of_two();
    let omega = find_root_of_unity(fft_n, &modulus);
    let inv_omega = omega.modpow(&(modulus.deref() - BigUint::from(2u32)), &modulus);

    let domain: Vec<F> = (0..data_square.rows)
        .map(|i| {
            F::from_biguint(
                omega.modpow(&BigUint::from(i as u64), &modulus),
                modulus.clone(),
            )
        })
        .collect();

    let extended_domain: Vec<F> = (0..data_square.columns * 5)
        .map(|i| {
            F::from_biguint(
                omega.modpow(&BigUint::from(i as u64), &modulus),
                modulus.clone(),
            )
        })
        .collect();

    // ----------------------------
    // 4. GPU-based interpolation of all columns
    // ----------------------------
    let batch_size = data_square.columns;

    // --- Pack all columns into one GPU buffer ---
    let mut host_columns: Vec<FieldElem> = Vec::with_capacity(batch_size * fft_n);
    for col_idx in 0..batch_size {
        let mut column = data_square.get_column(col_idx);
        column.resize(fft_n, F::zero(modulus.clone()));
        for f in &column {
            let mont = to_montgomery(&f.value, &modulus, &r2);
            let f_mont = F {
                value: mont,
                modulus: modulus.clone(),
            };
            let limbs = f_to_limbs(&f_mont);
            host_columns.push(FieldElem {
                x: limbs[0],
                y: limbs[1],
                z: limbs[2],
                w: limbs[3],
            });
        }
    }

    let data_buf = device.new_buffer_with_data(
        host_columns.as_ptr() as *const _,
        (host_columns.len() * std::mem::size_of::<FieldElem>()) as u64,
        MTLResourceOptions::StorageModeManaged,
    );

    // --- Precompute inverse FFT twiddles for interpolation ---
    let inv_twiddles = precompute_all_twiddles_flat(fft_n, modulus.clone(), &inv_omega, &r2);
    let inv_tw_data: Vec<FieldElem> = inv_twiddles
        .iter()
        .map(|w| FieldElem {
            x: w[0],
            y: w[1],
            z: w[2],
            w: w[3],
        })
        .collect();
    let inv_twiddle_buf = device.new_buffer_with_data(
        inv_tw_data.as_ptr() as *const _,
        (inv_tw_data.len() * std::mem::size_of::<FieldElem>()) as u64,
        MTLResourceOptions::StorageModeManaged,
    );

    let modulus_limbs = biguint_to_limbs(&modulus);
    let modulus_buf = device.new_buffer_with_data(
        unsafe { std::mem::transmute(modulus_limbs.as_ptr()) },
        (modulus_limbs.len() * std::mem::size_of::<u64>()) as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    let nprime_buf = device.new_buffer_with_data(
        unsafe { std::mem::transmute(&nprime) },
        std::mem::size_of::<u64>() as u64,
        MTLResourceOptions::StorageModeManaged,
    );

    // --- Run batched inverse FFT (interpolation) ---
    run_batched_fft_operations(
        &device,
        &shared_fft_pipeline,
        &bitrev_pipeline,
        &command_queue,
        &data_buf,
        &inv_twiddle_buf,
        &inv_twiddle_buf,
        &modulus_buf,
        &nprime_buf,
        fft_n,
        batch_size,
    );

    // --- Read back interpolated coefficients ---
    let ptr = data_buf.contents() as *const FieldElem;
    let raw = unsafe { std::slice::from_raw_parts(ptr, batch_size * fft_n) };
    let ninv =
        BigUint::from(fft_n as u64).modpow(&(modulus.deref() - BigUint::from(2u32)), &modulus);

    let mut column_polys: Vec<Polynomial> = Vec::new();
    for col_idx in 0..batch_size {
        let mut coeffs = Vec::with_capacity(fft_n);
        for j in 0..fft_n {
            let fe = &raw[col_idx * fft_n + j];
            let limbs = [fe.x, fe.y, fe.z, fe.w];
            let f_mont = limbs_to_f(&limbs, modulus.clone());
            let mut val = from_montgomery(&f_mont.value, &modulus, nprime);
            val = (&val * &ninv) % &*modulus;
            coeffs.push(F {
                value: val,
                modulus: modulus.clone(),
            });
        }
        column_polys.push(Polynomial::from_coeffs(coeffs));
    }

    // ----------------------------
    // 5. Evaluate columns on extended domain
    // ----------------------------
    let mut extended_data_square = DataSquare::new(vec![], 0, 0);
    for (col_idx, column_poly) in column_polys.into_iter().enumerate() {
        for i in 0..extended_domain.len() {
            let x = &extended_domain[i];
            let y = column_poly.evaluate(&x);
            extended_data_square.set_cell(col_idx, i, y);
        }
    }

    // ----------------------------
    // 6. Fiat–Shamir random coefficients
    // ----------------------------
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

    // ----------------------------
    // 7. Compute row-wise running sums
    // ----------------------------
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

    // ----------------------------
    // 8. Interpolate y (CPU or GPU)
    // ----------------------------
    let y_poly = Polynomial::interpolate(&domain, &y);
    let mut y_encoded: Vec<F> = Vec::new();
    for x in extended_domain {
        let y_val = y_poly.evaluate(&x);
        y_encoded.push(y_val);
    }

    // ----------------------------
    // 9. Verify correctness
    // ----------------------------
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
