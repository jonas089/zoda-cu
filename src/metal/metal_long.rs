use crate::ff::F;
use metal::*;
use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::{One, Zero};
use std::{mem, ops::Deref, sync::Arc};

pub const LIMBS: usize = 4;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct FieldElem {
    pub x: u64,
    pub y: u64,
    pub z: u64,
    pub w: u64,
}

pub fn f_to_limbs(x: &F) -> [u64; LIMBS] {
    let mut bytes = x.value.to_bytes_le();
    bytes.resize(LIMBS * 8, 0);
    let mut limbs = [0u64; LIMBS];
    for i in 0..LIMBS {
        limbs[i] = u64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap());
    }
    limbs
}

pub fn limbs_to_f(limbs: &[u64; LIMBS], modulus: Arc<BigUint>) -> F {
    let mut bytes = Vec::with_capacity(LIMBS * 8);
    for limb in limbs {
        bytes.extend_from_slice(&limb.to_le_bytes());
    }
    F {
        value: BigUint::from_bytes_le(&bytes) % &*modulus,
        modulus,
    }
}

pub fn biguint_to_limbs(x: &BigUint) -> [u64; LIMBS] {
    let mut bytes = x.to_bytes_le();
    bytes.resize(LIMBS * 8, 0);
    let mut limbs = [0u64; LIMBS];
    for i in 0..LIMBS {
        limbs[i] = u64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap());
    }
    limbs
}

pub fn compute_montgomery_params(modulus: &BigUint) -> (u64, BigUint) {
    let m0 = modulus.to_u64_digits()[0];
    assert!(m0 & 1 == 1, "modulus must be odd");

    let inv = modinv64(m0);
    let nprime = (!inv).wrapping_add(1);

    let r_bits = 64 * LIMBS;
    let r = BigUint::one() << r_bits;
    let r2 = (&r * &r) % modulus; // R^2 mod m

    (nprime, r2)
}

fn modinv64(a: u64) -> u64 {
    let mut inv = 1u128;
    let mut base = a as u128;
    for _ in 0..6 {
        inv = inv.wrapping_mul(2u128.wrapping_sub(base.wrapping_mul(inv) & ((1u128 << 64) - 1)))
            & ((1u128 << 64) - 1);
    }
    inv as u64
}

pub fn to_montgomery(x: &BigUint, modulus: &BigUint, r2: &BigUint) -> BigUint {
    // To convert x to Montgomery form: mont_mul(x, R^2) where R^2 is precomputed
    // But we're not using montgomery multiplication here - we're just doing regular multiplication
    // The GPU expects us to pass values that are already in Montgomery form after this step
    // So this should be x * R mod m, not x * R^2 mod m!
    let r_bits = 64 * LIMBS;
    let r = BigUint::one() << r_bits;
    (x * r) % modulus
}

pub fn from_montgomery(x: &BigUint, modulus: &BigUint, nprime: u64) -> BigUint {
    // Use the modular inverse from num-integer
    let r_bits = 64 * LIMBS;
    let r = BigUint::one() << r_bits;

    // Find R^-1 mod m
    let r_inv = r.modpow(&(modulus - BigUint::from(2u32)), modulus);
    (x * r_inv) % modulus
}

// ---------- bit reversal ----------
pub fn bitreverse_permute<T>(values: &mut [T]) {
    let n = values.len();
    let log_n = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - log_n);
        if i < j as usize {
            values.swap(i, j as usize);
        }
    }
}

// ---------- roots and twiddles ----------
pub fn find_root_of_unity(n: usize, p: &BigUint) -> BigUint {
    let exp = (p - 1u32) / BigUint::from(n as u64);
    for g in [3u32, 5u32, 7u32, 11u32, 13u32] {
        let candidate = BigUint::from(g).modpow(&exp, p);
        if candidate.modpow(&BigUint::from(n as u64), p) == BigUint::one()
            && candidate.modpow(&BigUint::from((n / 2) as u64), p) != BigUint::one()
        {
            return candidate;
        }
    }
    panic!("No primitive root found");
}

fn precompute_twiddles(
    n: usize,
    modulus: Arc<BigUint>,
    root: &BigUint,
    r2: &BigUint,
) -> Vec<[u64; LIMBS]> {
    let mut twiddles = Vec::with_capacity(n);
    let mut omega = BigUint::one();
    for _ in 0..n {
        let mont = to_montgomery(&omega, &modulus, r2);
        let f = F {
            value: mont,
            modulus: modulus.clone(),
        };
        twiddles.push(f_to_limbs(&f));
        omega = (&omega * root) % &*modulus;
    }
    twiddles
}

fn precompute_stage_twiddles(
    n: usize,
    modulus: Arc<BigUint>,
    root: &BigUint,
    r2: &BigUint,
) -> Vec<Vec<[u64; LIMBS]>> {
    let log_n = (n as u32).trailing_zeros();
    let mut stage_twiddles = Vec::with_capacity(log_n as usize);

    for stage in 0..log_n {
        let step = 1usize << stage;
        let mut twiddles = Vec::with_capacity(step);

        // For each stage, we need twiddles: omega^0, omega^(n/2^(stage+1)), omega^(2*n/2^(stage+1)), ...
        let stage_root = root.modpow(&BigUint::from(n / (2 * step)), &modulus);
        let mut omega = BigUint::one();

        for _ in 0..step {
            let mont = to_montgomery(&omega, &modulus, r2);
            let f = F {
                value: mont,
                modulus: modulus.clone(),
            };
            twiddles.push(f_to_limbs(&f));
            omega = (&omega * &stage_root) % &*modulus;
        }

        stage_twiddles.push(twiddles);
    }

    stage_twiddles
}

pub fn precompute_all_twiddles_flat(
    n: usize,
    modulus: Arc<BigUint>,
    root: &BigUint,
    r2: &BigUint,
) -> Vec<[u64; LIMBS]> {
    let log_n = (n as u32).trailing_zeros();
    let mut all_twiddles = Vec::new();

    for stage in 0..log_n {
        let step = 1usize << stage;

        // For each stage, we need twiddles: omega^0, omega^(n/2^(stage+1)), omega^(2*n/2^(stage+1)), ...
        let stage_root = root.modpow(&BigUint::from(n / (2 * step)), &modulus);
        let mut omega = BigUint::one();

        for _ in 0..step {
            let mont = to_montgomery(&omega, &modulus, r2);
            let f = F {
                value: mont,
                modulus: modulus.clone(),
            };
            all_twiddles.push(f_to_limbs(&f));
            omega = (&omega * &stage_root) % &*modulus;
        }
    }

    all_twiddles
}

// New function for 2-D twiddle layout supporting batched FFTs
pub fn precompute_all_twiddles_2d(
    n: usize,
    batch_size: usize,
    modulus: Arc<BigUint>,
    root: &BigUint,
    r2: &BigUint,
) -> Vec<[u64; LIMBS]> {
    let log_n = (n as u32).trailing_zeros();
    let total_twiddles_per_fft = (n - 1); // Total twiddles needed per FFT
    let mut all_twiddles = Vec::with_capacity(batch_size * total_twiddles_per_fft);

    // Replicate twiddles for each FFT in the batch
    for _fft_id in 0..batch_size {
        for stage in 0..log_n {
            let step = 1usize << stage;
            let stage_root = root.modpow(&BigUint::from(n / (2 * step)), &modulus);
            let mut omega = BigUint::one();

            for _ in 0..step {
                let mont = to_montgomery(&omega, &modulus, r2);
                let f = F {
                    value: mont,
                    modulus: modulus.clone(),
                };
                all_twiddles.push(f_to_limbs(&f));
                omega = (&omega * &stage_root) % &*modulus;
            }
        }
    }

    all_twiddles
}

// ---------- GPU runners ----------
pub fn run_fft_shared_memory(
    device: &Device,
    shared_fft_pipeline: &ComputePipelineState,
    command_queue: &CommandQueue,
    data_buf: &Buffer,
    twiddle_buf: &Buffer,
    modulus_buf: &Buffer,
    nprime_buf: &Buffer,
    n: usize,
) {
    run_fft_shared_memory_batched(
        device,
        shared_fft_pipeline,
        command_queue,
        data_buf,
        twiddle_buf,
        modulus_buf,
        nprime_buf,
        n,
        1, // batch_size = 1 for backward compatibility
    )
}

pub fn run_fft_shared_memory_batched(
    device: &Device,
    shared_fft_pipeline: &ComputePipelineState,
    command_queue: &CommandQueue,
    data_buf: &Buffer,
    twiddle_buf: &Buffer,
    modulus_buf: &Buffer,
    nprime_buf: &Buffer,
    n: usize,
    batch_size: usize,
) {
    let n_buf = device.new_buffer_with_data(
        unsafe { mem::transmute(&(n as u32)) },
        mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeManaged,
    );

    let batch_size_buf = device.new_buffer_with_data(
        unsafe { mem::transmute(&(batch_size as u32)) },
        mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeManaged,
    );

    let command_buffer = command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&shared_fft_pipeline);
    encoder.set_buffer(0, Some(&data_buf), 0);
    encoder.set_buffer(1, Some(&n_buf), 0);
    encoder.set_buffer(2, Some(&batch_size_buf), 0);
    encoder.set_buffer(3, Some(&twiddle_buf), 0);
    encoder.set_buffer(4, Some(&modulus_buf), 0);
    encoder.set_buffer(5, Some(&nprime_buf), 0);

    // Use threadgroups of size up to 256 (MAX_SHARED_SIZE) - optimized for M1/M2/M3
    let max_block_size = 256.min(n);
    let num_blocks_per_fft = (n + max_block_size - 1) / max_block_size;
    let total_blocks = num_blocks_per_fft * batch_size;

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

fn run_fft_butterfly_stages(
    device: &Device,
    butterfly_pipeline: &ComputePipelineState,
    command_queue: &CommandQueue,
    data_buf: &Buffer,
    twiddle_bufs: &[Buffer], // One twiddle buffer per stage
    modulus_buf: &Buffer,
    nprime_buf: &Buffer,
    n: usize,
) {
    run_fft_butterfly_stages_batched(
        device,
        butterfly_pipeline,
        command_queue,
        data_buf,
        twiddle_bufs,
        modulus_buf,
        nprime_buf,
        n,
        1, // batch_size = 1 for backward compatibility
    )
}

fn run_fft_butterfly_stages_batched(
    device: &Device,
    butterfly_pipeline: &ComputePipelineState,
    command_queue: &CommandQueue,
    data_buf: &Buffer,
    twiddle_bufs: &[Buffer], // One twiddle buffer per stage
    modulus_buf: &Buffer,
    nprime_buf: &Buffer,
    n: usize,
    batch_size: usize,
) {
    let log_n = (n as u32).trailing_zeros();

    for stage in 0..log_n {
        let stage_buf = device.new_buffer_with_data(
            unsafe { mem::transmute(&stage) },
            mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeManaged,
        );
        let n_buf = device.new_buffer_with_data(
            unsafe { mem::transmute(&(n as u32)) },
            mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeManaged,
        );
        let batch_size_buf = device.new_buffer_with_data(
            unsafe { mem::transmute(&(batch_size as u32)) },
            mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeManaged,
        );

        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&butterfly_pipeline);
        encoder.set_buffer(0, Some(&data_buf), 0);
        encoder.set_buffer(1, Some(&n_buf), 0);
        encoder.set_buffer(2, Some(&batch_size_buf), 0);
        encoder.set_buffer(3, Some(&stage_buf), 0);
        encoder.set_buffer(4, Some(&twiddle_bufs[stage as usize]), 0);
        encoder.set_buffer(5, Some(&modulus_buf), 0);
        encoder.set_buffer(6, Some(&nprime_buf), 0);

        // Number of threads needed: (num_groups * step) * batch_size
        let step = 1usize << stage;
        let num_groups = n >> (stage + 1);
        let threads_per_fft = num_groups * step;
        let total_threads = threads_per_fft * batch_size;
        let grid = MTLSize {
            width: total_threads as u64,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 64.min(total_threads as u64),
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(grid, tg);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
}

pub fn run_bitrev(
    device: &Device,
    bitrev_pipeline: &ComputePipelineState,
    command_queue: &CommandQueue,
    data_buf: &Buffer,
    n: usize,
) {
    run_bitrev_batched(device, bitrev_pipeline, command_queue, data_buf, n, 1)
}

pub fn run_bitrev_batched(
    device: &Device,
    bitrev_pipeline: &ComputePipelineState,
    command_queue: &CommandQueue,
    data_buf: &Buffer,
    n: usize,
    batch_size: usize,
) {
    let n_buf = device.new_buffer_with_data(
        unsafe { mem::transmute(&(n as u32)) },
        mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    let batch_size_buf = device.new_buffer_with_data(
        unsafe { mem::transmute(&(batch_size as u32)) },
        mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    let logn: u32 = (n as u32).trailing_zeros();
    let logn_buf = device.new_buffer_with_data(
        unsafe { mem::transmute(&logn) },
        mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeManaged,
    );

    let command_buffer = command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(bitrev_pipeline);
    encoder.set_buffer(0, Some(data_buf), 0);
    encoder.set_buffer(1, Some(&n_buf), 0);
    encoder.set_buffer(2, Some(&batch_size_buf), 0);
    encoder.set_buffer(3, Some(&logn_buf), 0);

    let total_elements = n * batch_size;
    let grid = MTLSize {
        width: total_elements as u64,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

// ---------- High-performance batched FFT runner ----------
pub fn run_batched_fft_operations_async(
    device: &Device,
    shared_fft_pipeline: &ComputePipelineState,
    bitrev_pipeline: &ComputePipelineState,
    command_queue: &CommandQueue,
    data_buf: &Buffer,
    twiddle_buf: &Buffer,
    inv_twiddle_buf: &Buffer,
    modulus_buf: &Buffer,
    nprime_buf: &Buffer,
    n: usize,
    batch_size: usize,
) {
    // Pre-allocate parameter buffers to avoid repeated allocations
    let n_buf = device.new_buffer_with_data(
        unsafe { mem::transmute(&(n as u32)) },
        mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    let batch_size_buf = device.new_buffer_with_data(
        unsafe { mem::transmute(&(batch_size as u32)) },
        mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    let logn: u32 = (n as u32).trailing_zeros();
    let logn_buf = device.new_buffer_with_data(
        unsafe { mem::transmute(&logn) },
        mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeManaged,
    );

    // Create a single command buffer for the entire batch operation
    let command_buffer = command_queue.new_command_buffer();

    // Forward FFT
    {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&shared_fft_pipeline);
        encoder.set_buffer(0, Some(&data_buf), 0);
        encoder.set_buffer(1, Some(&n_buf), 0);
        encoder.set_buffer(2, Some(&batch_size_buf), 0);
        encoder.set_buffer(3, Some(&twiddle_buf), 0);
        encoder.set_buffer(4, Some(&modulus_buf), 0);
        encoder.set_buffer(5, Some(&nprime_buf), 0);

        let max_block_size = 256.min(n);
        let num_blocks_per_fft = (n + max_block_size - 1) / max_block_size;
        let total_blocks = num_blocks_per_fft * batch_size;

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
    }

    // Bit-reverse
    {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&bitrev_pipeline);
        encoder.set_buffer(0, Some(&data_buf), 0);
        encoder.set_buffer(1, Some(&n_buf), 0);
        encoder.set_buffer(2, Some(&batch_size_buf), 0);
        encoder.set_buffer(3, Some(&logn_buf), 0);

        let total_elements = n * batch_size;
        let grid = MTLSize {
            width: total_elements as u64,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(grid, tg);
        encoder.end_encoding();
    }

    // Inverse FFT
    {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&shared_fft_pipeline);
        encoder.set_buffer(0, Some(&data_buf), 0);
        encoder.set_buffer(1, Some(&n_buf), 0);
        encoder.set_buffer(2, Some(&batch_size_buf), 0);
        encoder.set_buffer(3, Some(&inv_twiddle_buf), 0);
        encoder.set_buffer(4, Some(&modulus_buf), 0);
        encoder.set_buffer(5, Some(&nprime_buf), 0);

        let max_block_size = 256.min(n);
        let num_blocks_per_fft = (n + max_block_size - 1) / max_block_size;
        let total_blocks = num_blocks_per_fft * batch_size;

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
    }

    // Final bit-reverse to get natural order output (DIT: Forward→Bitrev→Inverse→Bitrev)
    {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&bitrev_pipeline);
        encoder.set_buffer(0, Some(&data_buf), 0);
        encoder.set_buffer(1, Some(&n_buf), 0);
        encoder.set_buffer(2, Some(&batch_size_buf), 0);
        encoder.set_buffer(3, Some(&logn_buf), 0);

        let total_elements = n * batch_size;
        let grid = MTLSize {
            width: total_elements as u64,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(grid, tg);
        encoder.end_encoding();
    }

    // Submit the entire batch asynchronously - NO wait_until_completed!
    command_buffer.commit();
    // For truly async operation, caller can check command_buffer.status() or use completion handlers
}

// Version that waits for completion (for accurate benchmarking)
pub fn run_batched_fft_operations_async_and_wait(
    device: &Device,
    shared_fft_pipeline: &ComputePipelineState,
    bitrev_pipeline: &ComputePipelineState,
    command_queue: &CommandQueue,
    data_buf: &Buffer,
    twiddle_buf: &Buffer,
    inv_twiddle_buf: &Buffer,
    modulus_buf: &Buffer,
    nprime_buf: &Buffer,
    n: usize,
    batch_size: usize,
) {
    // Pre-allocate parameter buffers to avoid repeated allocations
    let n_buf = device.new_buffer_with_data(
        unsafe { mem::transmute(&(n as u32)) },
        mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    let batch_size_buf = device.new_buffer_with_data(
        unsafe { mem::transmute(&(batch_size as u32)) },
        mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    let logn: u32 = (n as u32).trailing_zeros();
    let logn_buf = device.new_buffer_with_data(
        unsafe { mem::transmute(&logn) },
        mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeManaged,
    );

    // Create a single command buffer for the entire batch operation
    let command_buffer = command_queue.new_command_buffer();

    // Forward FFT
    {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&shared_fft_pipeline);
        encoder.set_buffer(0, Some(&data_buf), 0);
        encoder.set_buffer(1, Some(&n_buf), 0);
        encoder.set_buffer(2, Some(&batch_size_buf), 0);
        encoder.set_buffer(3, Some(&twiddle_buf), 0);
        encoder.set_buffer(4, Some(&modulus_buf), 0);
        encoder.set_buffer(5, Some(&nprime_buf), 0);

        let max_block_size = 256.min(n);
        let num_blocks_per_fft = (n + max_block_size - 1) / max_block_size;
        let total_blocks = num_blocks_per_fft * batch_size;

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
    }

    // Bit-reverse
    {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&bitrev_pipeline);
        encoder.set_buffer(0, Some(&data_buf), 0);
        encoder.set_buffer(1, Some(&n_buf), 0);
        encoder.set_buffer(2, Some(&batch_size_buf), 0);
        encoder.set_buffer(3, Some(&logn_buf), 0);

        let total_elements = n * batch_size;
        let grid = MTLSize {
            width: total_elements as u64,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(grid, tg);
        encoder.end_encoding();
    }

    // Inverse FFT
    {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&shared_fft_pipeline);
        encoder.set_buffer(0, Some(&data_buf), 0);
        encoder.set_buffer(1, Some(&n_buf), 0);
        encoder.set_buffer(2, Some(&batch_size_buf), 0);
        encoder.set_buffer(3, Some(&inv_twiddle_buf), 0);
        encoder.set_buffer(4, Some(&modulus_buf), 0);
        encoder.set_buffer(5, Some(&nprime_buf), 0);

        let max_block_size = 256.min(n);
        let num_blocks_per_fft = (n + max_block_size - 1) / max_block_size;
        let total_blocks = num_blocks_per_fft * batch_size;

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
    }

    // Final bit-reverse to get natural order output (DIT: Forward→Bitrev→Inverse→Bitrev)
    {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&bitrev_pipeline);
        encoder.set_buffer(0, Some(&data_buf), 0);
        encoder.set_buffer(1, Some(&n_buf), 0);
        encoder.set_buffer(2, Some(&batch_size_buf), 0);
        encoder.set_buffer(3, Some(&logn_buf), 0);

        let total_elements = n * batch_size;
        let grid = MTLSize {
            width: total_elements as u64,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(grid, tg);
        encoder.end_encoding();
    }

    // Submit and wait for completion (for accurate benchmarking)
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

// Synchronous version for compatibility
pub fn run_batched_fft_operations(
    device: &Device,
    shared_fft_pipeline: &ComputePipelineState,
    bitrev_pipeline: &ComputePipelineState,
    command_queue: &CommandQueue,
    data_buf: &Buffer,
    twiddle_buf: &Buffer,
    inv_twiddle_buf: &Buffer,
    modulus_buf: &Buffer,
    nprime_buf: &Buffer,
    n: usize,
    batch_size: usize,
) {
    // Create the async operation and wait for completion
    run_batched_fft_operations_async_and_wait(
        device,
        shared_fft_pipeline,
        bitrev_pipeline,
        command_queue,
        data_buf,
        twiddle_buf,
        inv_twiddle_buf,
        modulus_buf,
        nprime_buf,
        n,
        batch_size,
    );
}

// ---------- test ----------
#[test]
fn test_fft_ifft_roundtrip_big() {
    // Add 5-second timeout to prevent infinite loops
    let start_time = std::time::Instant::now();
    // Metal setup
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

    // BN254 prime
    let modulus = Arc::new(
        BigUint::parse_bytes(
            b"21888242871839275222246405745257275088548364400416034343698204186575808495617",
            10,
        )
        .unwrap(),
    );

    let (nprime, r2) = compute_montgomery_params(&modulus);

    let n: usize = 256;
    let root = find_root_of_unity(n, &modulus);

    // coefficients: f(x) = 7 + 3x + 5x^2
    let mut coeffs = vec![F::zero(modulus.clone()); n];
    coeffs[0] = F::new(7, modulus.clone());
    coeffs[1] = F::new(3, modulus.clone());
    coeffs[2] = F::new(5, modulus.clone());

    // Bit-reverse input for DIT FFT
    bitreverse_permute(&mut coeffs);

    // serialize coeffs in Montgomery form
    let mut host_data: Vec<u64> = Vec::with_capacity(n * LIMBS);
    for c in &coeffs {
        let mont = to_montgomery(&c.value, &modulus, &r2);
        let f = F {
            value: mont,
            modulus: modulus.clone(),
        };
        host_data.extend_from_slice(&f_to_limbs(&f));
    }

    // forward twiddles for shared memory FFT (all stages flattened) - PROPER FieldElem LAYOUT
    let all_twiddles = precompute_all_twiddles_flat(n, modulus.clone(), &root, &r2);
    let mut tw_data: Vec<FieldElem> = Vec::with_capacity(all_twiddles.len());
    for w in &all_twiddles {
        tw_data.push(FieldElem {
            x: w[0],
            y: w[1],
            z: w[2],
            w: w[3],
        });
    }

    // upload buffers
    let data_buf = device.new_buffer_with_data(
        unsafe { mem::transmute(host_data.as_ptr()) },
        (host_data.len() * mem::size_of::<u64>()) as u64,
        MTLResourceOptions::StorageModeManaged,
    );

    let twiddle_buf = device.new_buffer_with_data(
        tw_data.as_ptr() as *const _,
        (tw_data.len() * mem::size_of::<FieldElem>()) as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    let modulus_limbs = biguint_to_limbs(&modulus);
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

    // ---------- forward FFT ----------
    if start_time.elapsed().as_secs() >= 5 {
        panic!("Test timeout");
    }

    // Use shared memory FFT for smaller sizes, butterfly stages for larger
    if n <= 256 {
        run_fft_shared_memory(
            &device,
            &shared_fft_pipeline,
            &command_queue,
            &data_buf,
            &twiddle_buf,
            &modulus_buf,
            &nprime_buf,
            n,
        );
    } else {
        // Fallback to stage-wise approach for larger FFTs - PROPER FieldElem LAYOUT
        let stage_twiddles = precompute_stage_twiddles(n, modulus.clone(), &root, &r2);
        let mut twiddle_bufs = Vec::new();
        for stage_tw in &stage_twiddles {
            let mut stage_tw_data: Vec<FieldElem> = Vec::with_capacity(stage_tw.len());
            for w in stage_tw {
                stage_tw_data.push(FieldElem {
                    x: w[0],
                    y: w[1],
                    z: w[2],
                    w: w[3],
                });
            }
            let stage_twiddle_buf = device.new_buffer_with_data(
                stage_tw_data.as_ptr() as *const _,
                (stage_tw_data.len() * mem::size_of::<FieldElem>()) as u64,
                MTLResourceOptions::StorageModeManaged,
            );
            twiddle_bufs.push(stage_twiddle_buf);
        }

        run_fft_butterfly_stages(
            &device,
            &butterfly_pipeline,
            &command_queue,
            &data_buf,
            &twiddle_bufs,
            &modulus_buf,
            &nprime_buf,
            n,
        );
    }

    // Bit-reverse output of FFT for IFFT input
    run_bitrev(&device, &bitrev_pipeline, &command_queue, &data_buf, n);

    let ptr = data_buf.contents() as *const u64;
    let raw = unsafe { std::slice::from_raw_parts(ptr, n * LIMBS) };
    for i in 0..4 {
        let mut limb_block = [0u64; LIMBS];
        limb_block.copy_from_slice(&raw[i * LIMBS..(i + 1) * LIMBS]);
        let f_mont = limbs_to_f(&limb_block, modulus.clone());
    }

    // ---------- inverse FFT twiddles ----------
    // Inverse root: root^(-1) = root^(p-2) mod p (since p is prime)
    let inv_root = root.modpow(&(modulus.deref() - BigUint::from(2u32)), &modulus);

    // ---------- inverse FFT ----------
    if start_time.elapsed().as_secs() >= 5 {
        panic!("Test timeout");
    }

    if n <= 256 {
        // Use shared memory FFT for inverse - PROPER FieldElem LAYOUT
        let inv_all_twiddles = precompute_all_twiddles_flat(n, modulus.clone(), &inv_root, &r2);
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
            (inv_tw_data.len() * mem::size_of::<FieldElem>()) as u64,
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
            n,
        );
    } else {
        // Fallback to stage-wise approach for larger FFTs - PROPER FieldElem LAYOUT
        let inv_stage_twiddles = precompute_stage_twiddles(n, modulus.clone(), &inv_root, &r2);
        let mut inv_twiddle_bufs = Vec::new();
        for stage_tw in &inv_stage_twiddles {
            let mut tw_data: Vec<FieldElem> = Vec::with_capacity(stage_tw.len());
            for w in stage_tw {
                tw_data.push(FieldElem {
                    x: w[0],
                    y: w[1],
                    z: w[2],
                    w: w[3],
                });
            }
            let twiddle_buf = device.new_buffer_with_data(
                tw_data.as_ptr() as *const _,
                (tw_data.len() * mem::size_of::<FieldElem>()) as u64,
                MTLResourceOptions::StorageModeManaged,
            );
            inv_twiddle_bufs.push(twiddle_buf);
        }

        run_fft_butterfly_stages(
            &device,
            &butterfly_pipeline,
            &command_queue,
            &data_buf,
            &inv_twiddle_bufs,
            &modulus_buf,
            &nprime_buf,
            n,
        );
    }

    // IFFT output should be in natural order
    // ---------- read back and scale ----------
    let ptr = data_buf.contents() as *const u64;
    let raw = unsafe { std::slice::from_raw_parts(ptr, n * LIMBS) };
    let ninv = BigUint::from(n as u64).modpow(&(modulus.deref() - BigUint::from(2u32)), &modulus);

    let mut results = Vec::with_capacity(n);
    for i in 0..n {
        let mut limb_block = [0u64; LIMBS];
        limb_block.copy_from_slice(&raw[i * LIMBS..(i + 1) * LIMBS]);
        let f_mont = limbs_to_f(&limb_block, modulus.clone());
        // convert back from Montgomery
        let mut standard = from_montgomery(&f_mont.value, &modulus, nprime);
        // scale by n^{-1}
        standard = (&standard * &ninv) % &*modulus;
        results.push(F {
            value: standard,
            modulus: modulus.clone(),
        });
    }

    assert!(
        results[0].equals(&F::new(7, modulus.clone())),
        "Expected results[0] = 7, got {}",
        results[0].value
    );

    // The full test would be:
    assert!(results[1].equals(&F::new(3, modulus.clone())));
    assert!(results[2].equals(&F::new(5, modulus.clone())));
    for c in &results[3..] {
        assert!(c.equals(&F::zero(modulus.clone())));
    }
}
