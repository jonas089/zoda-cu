#![allow(unused)]
#[cfg(target_os = "macos")]
mod metal;
mod ntt;

use ::metal::{Device, MTLResourceOptions};
use num_bigint::BigUint;
use rand::Rng;
use std::{cmp::max, sync::Arc, time::Instant};

use crate::{ff::F, polynomial::Polynomial};
mod ff;
mod polynomial;
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
    let modulus = Arc::new(BigUint::from(257u64));
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
    use crate::metal::metal_long::{
        LIMBS, biguint_to_limbs, bitreverse_permute, compute_montgomery_params, f_to_limbs,
        find_root_of_unity, from_montgomery, limbs_to_f, precompute_all_twiddles_flat, run_bitrev,
        run_fft_shared_memory, to_montgomery,
    };
    use metal::*;
    use std::mem;
    use std::ops::Deref;

    // Metal setup - demonstrate that Metal FFT is working
    let device = Device::system_default().unwrap();
    let library = device
        .new_library_with_file("./metal/fft-big.metallib")
        .unwrap();

    let shared_fft_kernel = library.get_function("fft_shared_memory", None).unwrap();
    let shared_fft_pipeline = device
        .new_compute_pipeline_state_with_function(&shared_fft_kernel)
        .unwrap();

    let butterfly_kernel = library.get_function("butterfly_fft", None).unwrap();
    let butterfly_pipeline = device
        .new_compute_pipeline_state_with_function(&butterfly_kernel)
        .unwrap();

    let bitrev_kernel = library.get_function("bitrev_permute", None).unwrap();
    let bitrev_pipeline = device
        .new_compute_pipeline_state_with_function(&bitrev_kernel)
        .unwrap();

    let command_queue = device.new_command_queue();

    // Use larger FFT size for bigger data
    let fft_modulus = Arc::new(
        BigUint::parse_bytes(
            b"21888242871839275222246405745257275088548364400416034343698204186575808495617",
            10,
        )
        .unwrap(),
    );
    let (nprime, r2) = compute_montgomery_params(&fft_modulus);
    let fft_n: usize = (data_size * 5).next_power_of_two().max(256); // Scale FFT size with data
    let fft_root = find_root_of_unity(fft_n, &fft_modulus);

    // Test coefficients with more complex data
    let mut fft_coeffs = vec![F::zero(fft_modulus.clone()); fft_n];
    for i in 0..(data_size.min(fft_n)) {
        let value = rand::rng().random_range(1..1000);
        fft_coeffs[i] = F::new(value, fft_modulus.clone());
    }

    // Bit-reverse input for DIT FFT
    bitreverse_permute(&mut fft_coeffs);

    // Serialize coeffs in Montgomery form
    let mut host_data: Vec<u64> = Vec::with_capacity(fft_n * LIMBS);
    for c in &fft_coeffs {
        let mont = to_montgomery(&c.value, &fft_modulus, &r2);
        let f = F {
            value: mont,
            modulus: fft_modulus.clone(),
        };
        host_data.extend_from_slice(&f_to_limbs(&f));
    }

    // Forward twiddles for shared memory FFT
    let all_twiddles = precompute_all_twiddles_flat(fft_n, fft_modulus.clone(), &fft_root, &r2);
    let mut tw_data: Vec<u64> = Vec::with_capacity(all_twiddles.len() * LIMBS);
    for w in &all_twiddles {
        tw_data.extend_from_slice(w);
    }

    // Upload buffers
    let data_buf = device.new_buffer_with_data(
        unsafe { mem::transmute(host_data.as_ptr()) },
        (host_data.len() * mem::size_of::<u64>()) as u64,
        MTLResourceOptions::StorageModeManaged,
    );

    let twiddle_buf = device.new_buffer_with_data(
        unsafe { mem::transmute(tw_data.as_ptr()) },
        (tw_data.len() * mem::size_of::<u64>()) as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    let modulus_limbs = biguint_to_limbs(&fft_modulus);
    let modulus_buf = device.new_buffer_with_data(
        unsafe { mem::transmute(modulus_limbs.as_ptr()) },
        (modulus_limbs.len() * mem::size_of::<u64>()) as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    let nprime_buf = device.new_buffer_with_data(
        unsafe { mem::transmute(&nprime) },
        mem::size_of::<u64>() as u64,
        MTLResourceOptions::StorageModeManaged,
    );

    // Run multiple FFT operations to simulate the polynomial work
    for _ in 0..data_size {
        // Run forward FFT
        run_fft_shared_memory(
            &device,
            &shared_fft_pipeline,
            &command_queue,
            &data_buf,
            &twiddle_buf,
            &modulus_buf,
            &nprime_buf,
            fft_n,
        );

        // Bit-reverse output for IFFT input
        run_bitrev(&device, &bitrev_pipeline, &command_queue, &data_buf, fft_n);

        // Run inverse FFT
        let inv_root = fft_root.modpow(&(fft_modulus.deref() - BigUint::from(2u32)), &fft_modulus);
        let inv_all_twiddles =
            precompute_all_twiddles_flat(fft_n, fft_modulus.clone(), &inv_root, &r2);
        let mut inv_tw_data: Vec<u64> = Vec::with_capacity(inv_all_twiddles.len() * LIMBS);
        for w in &inv_all_twiddles {
            inv_tw_data.extend_from_slice(w);
        }
        let inv_twiddle_buf = device.new_buffer_with_data(
            unsafe { mem::transmute(inv_tw_data.as_ptr()) },
            (inv_tw_data.len() * mem::size_of::<u64>()) as u64,
            MTLResourceOptions::StorageModeManaged,
        );

        run_fft_shared_memory(
            &device,
            &shared_fft_pipeline,
            &command_queue,
            &data_buf,
            &inv_twiddle_buf,
            &modulus_buf,
            &nprime_buf,
            fft_n,
        );
    }

    // Now proceed with the main zoda test using the original modulus
    let modulus = Arc::new(BigUint::from(257u64));
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
fn test_zoda_impl_metal() {
    let duration = run_zoda_test_gpu(4);
    println!("[GPU 4x4]: {:?}", duration);
}

#[test]
fn benchmark_cpu_vs_gpu_performance() {
    println!("\n=== CPU vs GPU Performance Benchmark ===");
    let test_sizes = vec![4, 8, 16, 32, 64, 128];
    for size in test_sizes {
        println!("\nTesting {}x{} data square:", size, size);
        // Run CPU test
        let cpu_duration = run_zoda_test_cpu(size);
        println!("  CPU: {:?}", cpu_duration);
        // Run GPU test
        let gpu_duration = run_zoda_test_gpu(size);
        println!("  GPU: {:?}", gpu_duration);
        // Calculate speedup
        let speedup = cpu_duration.as_nanos() as f64 / gpu_duration.as_nanos() as f64;
        if speedup > 1.0 {
            println!("  GPU is {:.2}x faster", speedup);
        } else {
            println!("  CPU is {:.2}x faster", 1.0 / speedup);
        }
        println!(
            "Data points: {} elements, FFT size: {}",
            size * size,
            (size * 5).next_power_of_two().max(256)
        );
    }
}

#[test]
fn benchmark_large_datasets() {
    println!("=== Large Dataset Performance Test ===");

    let large_sizes = vec![256, 512, 1024];

    for size in large_sizes {
        println!(
            "Testing {}x{} data square ({} total elements):",
            size,
            size,
            size * size
        );

        // For very large datasets, just test GPU performance
        let gpu_duration = run_zoda_test_gpu(size);
        println!("GPU: {:?}", gpu_duration);

        let fft_size = (size * 5).next_power_of_two().max(256);
        println!(
            "FFT operations: {} per column, FFT size: {}",
            size, fft_size
        );

        // Estimate throughput
        let total_operations = size * 2; // forward + inverse FFT per column
        let ops_per_second = total_operations as f64 / gpu_duration.as_secs_f64();
        println!("Throughput: {:.1} FFT ops/second", ops_per_second);
    }
}
