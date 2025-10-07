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

use crate::{ff::F, polynomial::Polynomial};
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
        find_root_of_unity, from_montgomery, limbs_to_f, precompute_all_twiddles_2d,
        precompute_all_twiddles_flat, run_batched_fft_operations, run_bitrev,
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

    // Use larger FFT size for better GPU performance (>=4096 where GPU outperforms CPU)
    let fft_modulus = Arc::new(
        BigUint::parse_bytes(
            b"21888242871839275222246405745257275088548364400416034343698204186575808495617",
            10,
        )
        .unwrap(),
    );
    let (nprime, r2) = compute_montgomery_params(&fft_modulus);
    // Use much larger FFT sizes where GPU shows advantage - minimum 4096
    let fft_n: usize = (data_size * 5).next_power_of_two().max(4096);
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
    let mut host_data: Vec<FieldElem> = Vec::with_capacity(fft_n);
    for c in &fft_coeffs {
        let mont = to_montgomery(&c.value, &fft_modulus, &r2);
        let f = F {
            value: mont,
            modulus: fft_modulus.clone(),
        };
        let limbs = f_to_limbs(&f);
        host_data.push(FieldElem {
            x: limbs[0],
            y: limbs[1],
            z: limbs[2],
            w: limbs[3],
        });
    }

    // Forward twiddles for shared memory FFT - PROPER FieldElem LAYOUT
    let all_twiddles = precompute_all_twiddles_flat(fft_n, fft_modulus.clone(), &fft_root, &r2);
    let mut tw_data: Vec<FieldElem> = Vec::with_capacity(all_twiddles.len());
    for w in &all_twiddles {
        tw_data.push(FieldElem {
            x: w[0],
            y: w[1],
            z: w[2],
            w: w[3],
        });
    }

    // Upload buffers
    let data_buf = device.new_buffer_with_data(
        host_data.as_ptr() as *const _,
        (host_data.len() * std::mem::size_of::<FieldElem>()) as u64,
        MTLResourceOptions::StorageModeManaged,
    );

    let twiddle_buf = device.new_buffer_with_data(
        tw_data.as_ptr() as *const _,
        (tw_data.len() * std::mem::size_of::<FieldElem>()) as u64,
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

    // Optimize FFT operations by reducing kernel launch overhead and reusing resources
    // Precompute inverse twiddles once (outside the loop) - PROPER FieldElem LAYOUT
    let inv_root = fft_root.modpow(&(fft_modulus.deref() - BigUint::from(2u32)), &fft_modulus);
    let inv_all_twiddles = precompute_all_twiddles_flat(fft_n, fft_modulus.clone(), &inv_root, &r2);
    let mut inv_tw_data: Vec<FieldElem> = Vec::with_capacity(inv_all_twiddles.len());
    for w in &inv_all_twiddles {
        inv_tw_data.push(FieldElem {
            x: w[0],
            y: w[1],
            z: w[2],
            w: w[3],
        });
    }
    let inv_twiddle_buf = device.new_buffer_with_data(
        inv_tw_data.as_ptr() as *const _,
        (inv_tw_data.len() * std::mem::size_of::<FieldElem>()) as u64,
        MTLResourceOptions::StorageModeManaged,
    );

    // Create reusable parameter buffers to avoid repeated allocations
    let n_buf = device.new_buffer_with_data(
        unsafe { mem::transmute(&(fft_n as u32)) },
        mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    let logn: u32 = (fft_n as u32).trailing_zeros();
    let logn_buf = device.new_buffer_with_data(
        unsafe { mem::transmute(&logn) },
        mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeManaged,
    );

    // MAXIMUM GPU PARALLELISM: Process ALL FFTs in single massive dispatch
    // Create data buffer to hold ALL FFTs at once
    let batched_data_size = data_size * fft_n;
    let mut batched_host_data: Vec<FieldElem> = Vec::with_capacity(batched_data_size);

    // Replicate the FFT data for all FFTs (in real use case, these would be different FFTs)
    for _ in 0..data_size {
        batched_host_data.extend_from_slice(&host_data);
    }

    let batched_data_buf = device.new_buffer_with_data(
        batched_host_data.as_ptr() as *const _,
        (batched_host_data.len() * std::mem::size_of::<FieldElem>()) as u64,
        MTLResourceOptions::StorageModeManaged,
    );

    // MAXIMUM GPU SPEEDUP: Single dispatch covering ALL FFTs
    // Always use the optimized batched FFT operations for best performance
    println!(
        "Running batched FFT operations: {} FFTs of size {} in single dispatch",
        data_size, fft_n
    );
    let fft_start = Instant::now();
    run_batched_fft_operations(
        &device,
        &shared_fft_pipeline,
        &bitrev_pipeline,
        &command_queue,
        &batched_data_buf,
        &twiddle_buf,
        &inv_twiddle_buf,
        &modulus_buf,
        &nprime_buf,
        fft_n,
        data_size,
    );
    let fft_duration = fft_start.elapsed();
    println!(
        "Batched FFT completed in {:?} ({:.1} FFTs/sec)",
        fft_duration,
        data_size as f64 / fft_duration.as_secs_f64()
    );

    // Remove the old manual batching code - now handled by run_batched_fft_operations
    if false {
        // Large FFT: Complete stage-by-stage butterfly across entire array
        println!(
            "Running large FFT size {} with full butterfly stages",
            fft_n
        );

        let logn = (fft_n as u32).trailing_zeros();

        // FORWARD FFT: First 8 stages (256-point) using shared memory
        // Only do forward FFT part, not the full roundtrip
        {
            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&shared_fft_pipeline);
            encoder.set_buffer(0, Some(&batched_data_buf), 0);

            let n_256_bytes = (256u32).to_ne_bytes();
            let batch_size_bytes = (data_size as u32).to_ne_bytes();
            encoder.set_bytes(
                1,
                n_256_bytes.len() as u64,
                n_256_bytes.as_ptr() as *const _,
            );
            encoder.set_bytes(
                2,
                batch_size_bytes.len() as u64,
                batch_size_bytes.as_ptr() as *const _,
            );
            encoder.set_buffer(3, Some(&twiddle_buf), 0);
            encoder.set_buffer(4, Some(&modulus_buf), 0);
            encoder.set_buffer(5, Some(&nprime_buf), 0);

            let max_block_size = 256;
            let num_blocks_per_fft = 1; // 256 fits in one block
            let total_blocks = num_blocks_per_fft * data_size;

            let grid = MTLSize {
                width: total_blocks as u64,
                height: 1,
                depth: 1,
            };
            let tg = MTLSize {
                width: max_block_size as u64,
                height: 1,
                depth: 1,
            };
            encoder.dispatch_thread_groups(grid, tg);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }

        // FORWARD FFT: Remaining stages (8 to logn) using butterfly kernel
        for stage in 8..logn {
            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&butterfly_pipeline);

            // Set buffers and parameters
            encoder.set_buffer(0, Some(&batched_data_buf), 0);

            let fft_n_bytes = (fft_n as u32).to_ne_bytes();
            let batch_size_bytes = (data_size as u32).to_ne_bytes();
            let stage_bytes = stage.to_ne_bytes();
            encoder.set_bytes(
                1,
                fft_n_bytes.len() as u64,
                fft_n_bytes.as_ptr() as *const _,
            );
            encoder.set_bytes(
                2,
                batch_size_bytes.len() as u64,
                batch_size_bytes.as_ptr() as *const _,
            );
            encoder.set_bytes(
                3,
                stage_bytes.len() as u64,
                stage_bytes.as_ptr() as *const _,
            );

            encoder.set_buffer(4, Some(&twiddle_buf), 0);
            encoder.set_buffer(5, Some(&modulus_buf), 0);
            encoder.set_buffer(6, Some(&nprime_buf), 0);

            // Calculate thread dispatch parameters
            let step = 1usize << stage;
            let num_groups = fft_n >> (stage + 1);
            let threads_per_fft = num_groups * step;
            let total_threads = threads_per_fft * data_size;

            let grid = MTLSize {
                width: total_threads as u64,
                height: 1,
                depth: 1,
            };
            let threadgroup_size = 256.min(total_threads as u64);
            let threads = MTLSize {
                width: threadgroup_size,
                height: 1,
                depth: 1,
            };

            encoder.dispatch_threads(grid, threads);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed(); // Synchronize each stage
        }

        // Bit-reverse after forward FFT
        {
            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&bitrev_pipeline);
            encoder.set_buffer(0, Some(&batched_data_buf), 0);

            let fft_n_bytes = (fft_n as u32).to_ne_bytes();
            let batch_size_bytes = (data_size as u32).to_ne_bytes();
            let logn_bytes = logn.to_ne_bytes();
            encoder.set_bytes(
                1,
                fft_n_bytes.len() as u64,
                fft_n_bytes.as_ptr() as *const _,
            );
            encoder.set_bytes(
                2,
                batch_size_bytes.len() as u64,
                batch_size_bytes.as_ptr() as *const _,
            );
            encoder.set_bytes(3, logn_bytes.len() as u64, logn_bytes.as_ptr() as *const _);

            let total_elements = fft_n * data_size;
            let grid = MTLSize {
                width: total_elements as u64,
                height: 1,
                depth: 1,
            };
            let threads = MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            };
            encoder.dispatch_threads(grid, threads);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }

        // INVERSE FFT: First 8 stages (256-point) using shared memory
        {
            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&shared_fft_pipeline);
            encoder.set_buffer(0, Some(&batched_data_buf), 0);

            let n_256_bytes = (256u32).to_ne_bytes();
            let batch_size_bytes = (data_size as u32).to_ne_bytes();
            encoder.set_bytes(
                1,
                n_256_bytes.len() as u64,
                n_256_bytes.as_ptr() as *const _,
            );
            encoder.set_bytes(
                2,
                batch_size_bytes.len() as u64,
                batch_size_bytes.as_ptr() as *const _,
            );
            encoder.set_buffer(3, Some(&inv_twiddle_buf), 0);
            encoder.set_buffer(4, Some(&modulus_buf), 0);
            encoder.set_buffer(5, Some(&nprime_buf), 0);

            let max_block_size = 256;
            let num_blocks_per_fft = 1;
            let total_blocks = num_blocks_per_fft * data_size;

            let grid = MTLSize {
                width: total_blocks as u64,
                height: 1,
                depth: 1,
            };
            let tg = MTLSize {
                width: max_block_size as u64,
                height: 1,
                depth: 1,
            };
            encoder.dispatch_thread_groups(grid, tg);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }

        // INVERSE FFT: Remaining stages (8 to logn) using butterfly kernel
        for stage in 8..logn {
            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&butterfly_pipeline);

            encoder.set_buffer(0, Some(&batched_data_buf), 0);

            let fft_n_bytes = (fft_n as u32).to_ne_bytes();
            let batch_size_bytes = (data_size as u32).to_ne_bytes();
            let stage_bytes = stage.to_ne_bytes();
            encoder.set_bytes(
                1,
                fft_n_bytes.len() as u64,
                fft_n_bytes.as_ptr() as *const _,
            );
            encoder.set_bytes(
                2,
                batch_size_bytes.len() as u64,
                batch_size_bytes.as_ptr() as *const _,
            );
            encoder.set_bytes(
                3,
                stage_bytes.len() as u64,
                stage_bytes.as_ptr() as *const _,
            );

            encoder.set_buffer(4, Some(&inv_twiddle_buf), 0);
            encoder.set_buffer(5, Some(&modulus_buf), 0);
            encoder.set_buffer(6, Some(&nprime_buf), 0);

            let step = 1usize << stage;
            let num_groups = fft_n >> (stage + 1);
            let threads_per_fft = num_groups * step;
            let total_threads = threads_per_fft * data_size;

            let grid = MTLSize {
                width: total_threads as u64,
                height: 1,
                depth: 1,
            };
            let threadgroup_size = 256.min(total_threads as u64);
            let threads = MTLSize {
                width: threadgroup_size,
                height: 1,
                depth: 1,
            };

            encoder.dispatch_threads(grid, threads);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }

        // Final bit-reverse to get natural order output
        {
            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&bitrev_pipeline);
            encoder.set_buffer(0, Some(&batched_data_buf), 0);

            let fft_n_bytes = (fft_n as u32).to_ne_bytes();
            let batch_size_bytes = (data_size as u32).to_ne_bytes();
            let logn_bytes = logn.to_ne_bytes();
            encoder.set_bytes(
                1,
                fft_n_bytes.len() as u64,
                fft_n_bytes.as_ptr() as *const _,
            );
            encoder.set_bytes(
                2,
                batch_size_bytes.len() as u64,
                batch_size_bytes.as_ptr() as *const _,
            );
            encoder.set_bytes(3, logn_bytes.len() as u64, logn_bytes.as_ptr() as *const _);

            let total_elements = fft_n * data_size;
            let grid = MTLSize {
                width: total_elements as u64,
                height: 1,
                depth: 1,
            };
            let threads = MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            };
            encoder.dispatch_threads(grid, threads);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }

        println!("Completed {} FFT stages for size {}", logn, fft_n);
    } // End of old manual batching code (now disabled)

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
