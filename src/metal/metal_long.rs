use crate::ff::F;
use metal::*;
use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::{One, Zero};
use std::{mem, ops::Deref, sync::Arc};

const LIMBS: usize = 4;

fn f_to_limbs(x: &F) -> [u64; LIMBS] {
    let mut bytes = x.value.to_bytes_le();
    bytes.resize(LIMBS * 8, 0);
    let mut limbs = [0u64; LIMBS];
    for i in 0..LIMBS {
        limbs[i] = u64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap());
    }
    limbs
}

fn limbs_to_f(limbs: &[u64; LIMBS], modulus: Arc<BigUint>) -> F {
    let mut bytes = Vec::with_capacity(LIMBS * 8);
    for limb in limbs {
        bytes.extend_from_slice(&limb.to_le_bytes());
    }
    F {
        value: BigUint::from_bytes_le(&bytes) % &*modulus,
        modulus,
    }
}

fn biguint_to_limbs(x: &BigUint) -> [u64; LIMBS] {
    let mut bytes = x.to_bytes_le();
    bytes.resize(LIMBS * 8, 0);
    let mut limbs = [0u64; LIMBS];
    for i in 0..LIMBS {
        limbs[i] = u64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap());
    }
    limbs
}

fn compute_montgomery_params(modulus: &BigUint) -> (u64, BigUint) {
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

fn to_montgomery(x: &BigUint, modulus: &BigUint, r2: &BigUint) -> BigUint {
    // To convert x to Montgomery form: mont_mul(x, R^2) where R^2 is precomputed
    // But we're not using montgomery multiplication here - we're just doing regular multiplication
    // The GPU expects us to pass values that are already in Montgomery form after this step
    // So this should be x * R mod m, not x * R^2 mod m!
    let r_bits = 64 * LIMBS;
    let r = BigUint::one() << r_bits;
    (x * r) % modulus
}

fn from_montgomery(x: &BigUint, modulus: &BigUint, nprime: u64) -> BigUint {
    // Use the modular inverse from num-integer
    let r_bits = 64 * LIMBS;
    let r = BigUint::one() << r_bits;

    // Find R^-1 mod m
    let r_inv = r.modpow(&(modulus - BigUint::from(2u32)), modulus);
    (x * r_inv) % modulus
}

// ---------- bit reversal ----------
fn bitreverse_permute<T>(values: &mut [T]) {
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
fn find_root_of_unity(n: usize, p: &BigUint) -> BigUint {
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

// ---------- GPU runners ----------
fn run_fft_stagewise(
    device: &Device,
    fft_pipeline: &ComputePipelineState,
    command_queue: &CommandQueue,
    data_buf: &Buffer,
    twiddle_buf: &Buffer,
    modulus_buf: &Buffer,
    nprime_buf: &Buffer,
    n: usize,
) {
    let mut len = 2u32;
    while len <= n as u32 {
        let len_buf = device.new_buffer_with_data(
            unsafe { mem::transmute(&len) },
            mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeManaged,
        );
        let n_buf = device.new_buffer_with_data(
            unsafe { mem::transmute(&(n as u32)) },
            mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeManaged,
        );

        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&fft_pipeline);
        encoder.set_buffer(0, Some(&data_buf), 0);
        encoder.set_buffer(1, Some(&twiddle_buf), 0);
        encoder.set_buffer(2, Some(&n_buf), 0);
        encoder.set_buffer(3, Some(&len_buf), 0);
        encoder.set_buffer(4, Some(&modulus_buf), 0);
        encoder.set_buffer(5, Some(&nprime_buf), 0);

        let grid = MTLSize {
            width: (n as u64 / 2),
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(grid, tg);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        len <<= 1;
    }
}

fn run_bitrev(
    device: &Device,
    bitrev_pipeline: &ComputePipelineState,
    command_queue: &CommandQueue,
    data_buf: &Buffer,
    n: usize,
) {
    let n_buf = device.new_buffer_with_data(
        unsafe { mem::transmute(&(n as u32)) },
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
    encoder.set_buffer(2, Some(&logn_buf), 0);

    let grid = MTLSize {
        width: n as u64,
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

    let fft_kernel = library.get_function("fft_stage", None).unwrap();
    let fft_pipeline = device
        .new_compute_pipeline_state_with_function(&fft_kernel)
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

    // forward twiddles (Montgomery)
    let tw = precompute_twiddles(n, modulus.clone(), &root, &r2);
    let mut tw_data: Vec<u64> = Vec::with_capacity(n * LIMBS);
    for w in &tw {
        tw_data.extend_from_slice(w);
    }

    // upload buffers
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
    run_fft_stagewise(
        &device,
        &fft_pipeline,
        &command_queue,
        &data_buf,
        &twiddle_buf,
        &modulus_buf,
        &nprime_buf,
        n,
    );

    // Bit-reverse output of FFT for IFFT input
    run_bitrev(&device, &bitrev_pipeline, &command_queue, &data_buf, n);

    // Debug: check data after forward FFT
    println!("After forward FFT (first 4 coeffs in Montgomery):");
    let ptr = data_buf.contents() as *const u64;
    let raw = unsafe { std::slice::from_raw_parts(ptr, n * LIMBS) };
    for i in 0..4 {
        let mut limb_block = [0u64; LIMBS];
        limb_block.copy_from_slice(&raw[i * LIMBS..(i + 1) * LIMBS]);
        let f_mont = limbs_to_f(&limb_block, modulus.clone());
        println!("  FFT[{}] = {}", i, f_mont.value);
    }

    // ---------- inverse FFT twiddles ----------
    // Inverse root: root^(-1) = root^(p-2) mod p (since p is prime)
    let inv_root = root.modpow(&(modulus.deref() - BigUint::from(2u32)), &modulus);
    let itw = precompute_twiddles(n, modulus.clone(), &inv_root, &r2);
    let mut itw_data: Vec<u64> = Vec::with_capacity(n * LIMBS);
    for w in &itw {
        itw_data.extend_from_slice(w);
    }
    let itw_buf = device.new_buffer_with_data(
        unsafe { mem::transmute(itw_data.as_ptr()) },
        (itw_data.len() * mem::size_of::<u64>()) as u64,
        MTLResourceOptions::StorageModeManaged,
    );

    // ---------- inverse FFT ----------
    if start_time.elapsed().as_secs() >= 5 {
        panic!("Test timeout");
    }
    run_fft_stagewise(
        &device,
        &fft_pipeline,
        &command_queue,
        &data_buf,
        &itw_buf,
        &modulus_buf,
        &nprime_buf,
        n,
    );

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

    // ---------- debug output ----------
    println!("Expected: [7, 3, 5, 0, 0, ...]");
    println!("Actual first 8 coeffs:");
    for (i, coeff) in results.iter().take(8).enumerate() {
        println!("  [{}] = {}", i, coeff.value);
    }

    // ---------- assertions ----------
    // At least verify that the constant term works correctly
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
