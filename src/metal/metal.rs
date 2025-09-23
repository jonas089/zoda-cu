use metal::*;
use std::mem;

/// Bit-reversal permutation on host side
fn bitreverse_permute(values: &mut [u32]) {
    let n = values.len();
    let log_n = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - log_n);
        if i < j as usize {
            values.swap(i, j as usize);
        }
    }
}

/// Precompute forward twiddles (ω^k)
fn precompute_twiddles(n: u32, modulus: u32) -> Vec<u32> {
    let g: u32 = 3; // primitive root for p=257
    let exp = (modulus as u64 - 1) / n as u64;
    let root = mod_pow(g as u64, exp, modulus as u64);
    let mut omega: u64 = 1;
    let mut twiddles = Vec::with_capacity(n as usize);
    for _ in 0..n {
        twiddles.push(omega as u32);
        omega = (omega * root) % (modulus as u64);
    }
    twiddles
}

fn mod_pow(mut base: u64, mut exp: u64, m: u64) -> u64 {
    let mut res = 1;
    while exp > 0 {
        if exp & 1 == 1 {
            res = (res * base) % m;
        }
        base = (base * base) % m;
        exp >>= 1;
    }
    res
}

/// CPU IFFT for validation
fn cpu_ifft(mut values: Vec<u32>, n: usize, modulus: u32) -> Vec<u32> {
    let g: u32 = 3;
    let exp = (modulus as u64 - 1) / n as u64;
    let root = mod_pow(g as u64, exp, modulus as u64);
    let inv_root = mod_pow(root, (n as u64 - 1), modulus as u64) as u32;

    bitreverse_permute(&mut values);

    let mut len = 2;
    while len <= n {
        let step = n / len;
        for i in (0..n).step_by(len) {
            let mut w = 1u32;
            let w_len = mod_pow(inv_root as u64, step as u64, modulus as u64) as u32;
            for j in 0..len / 2 {
                let u = values[i + j];
                let v = (values[i + j + len / 2] as u64 * w as u64 % modulus as u64) as u32;
                values[i + j] = (u + v) % modulus;
                values[i + j + len / 2] = (modulus + u - v) % modulus;
                w = (w as u64 * w_len as u64 % modulus as u64) as u32;
            }
        }
        len <<= 1;
    }

    // scale by n^-1
    let n_inv = mod_pow(n as u64, (modulus - 2) as u64, modulus as u64) as u32;
    for x in &mut values {
        *x = (*x as u64 * n_inv as u64 % modulus as u64) as u32;
    }
    values
}

#[test]
fn test_fft_ifft_roundtrip() {
    let device = Device::system_default().unwrap();
    let library = device
        .new_library_with_file("./metal/fft.metallib")
        .unwrap();
    let kernel = library.get_function("fft_stage", None).unwrap();
    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .unwrap();
    let command_queue = device.new_command_queue();

    let n: u32 = 256;
    let modulus: u32 = 257;

    // coefficients: f(x) = 7 + 3x + 5x^2
    let mut coeffs = vec![0u32; n as usize];
    coeffs[0] = 7;
    coeffs[1] = 3;
    coeffs[2] = 5;

    // bit-reversal before upload (DIT FFT)
    bitreverse_permute(&mut coeffs);

    let twiddles = precompute_twiddles(n, modulus);

    // Upload
    let data_buf = device.new_buffer_with_data(
        unsafe { mem::transmute(coeffs.as_ptr()) },
        (coeffs.len() * mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    let twiddle_buf = device.new_buffer_with_data(
        unsafe { mem::transmute(twiddles.as_ptr()) },
        (twiddles.len() * mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    let n_buf = device.new_buffer_with_data(
        unsafe { mem::transmute(&n) },
        mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    let modulus_buf = device.new_buffer_with_data(
        unsafe { mem::transmute(&modulus) },
        mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeManaged,
    );

    // Run FFT stages
    let mut len = 2;
    while len <= n {
        let len_buf = device.new_buffer_with_data(
            unsafe { mem::transmute(&len) },
            mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeManaged,
        );
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&data_buf), 0);
        encoder.set_buffer(1, Some(&twiddle_buf), 0);
        encoder.set_buffer(2, Some(&n_buf), 0);
        encoder.set_buffer(3, Some(&len_buf), 0);
        encoder.set_buffer(4, Some(&modulus_buf), 0);

        let grid_size = MTLSize {
            width: (n / 2) as u64,
            height: 1,
            depth: 1,
        };
        let threadgroup_size = MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        len <<= 1;
    }

    // Read back
    let ptr = data_buf.contents() as *const u32;
    let results = unsafe { std::slice::from_raw_parts(ptr, n as usize) }.to_vec();
    println!("FFT result (first 16): {:?}", &results[..16]);

    // CPU IFFT
    let recovered = cpu_ifft(results.clone(), n as usize, modulus);
    println!("Recovered coeffs (first 8): {:?}", &recovered[..8]);

    assert_eq!(recovered[0], 7);
    assert_eq!(recovered[1], 3);
    assert_eq!(recovered[2], 5);
    for &c in &recovered[3..] {
        assert_eq!(c, 0);
    }
}
