use metal::*;
use num_bigint::BigUint;
use std::{sync::Arc, time::Instant};
use zoda_rs::ff::F;
use zoda_rs::metal::metal_long::{
    FieldElem, biguint_to_limbs, bitreverse_permute, compute_montgomery_params, f_to_limbs,
    find_root_of_unity, precompute_all_twiddles_flat, run_batched_fft_operations, to_montgomery,
};

fn main() {
    println!("🚀 Apple GPU FFT Batching Demo");
    println!("==============================");

    // Metal setup
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

    // Test different batch sizes to show batching improvements
    let batch_sizes = vec![1, 4, 16, 64, 256, 1024, 4096];

    println!("\\nTesting FFT size: {}", n);
    println!("Modulus: BN254 prime (256-bit)");
    println!(
        "\\n{:<12} {:<15} {:<15} {:<15}",
        "Batch Size", "Time (ms)", "FFTs/sec", "Throughput"
    );
    println!("{}", "-".repeat(65));

    for &batch_size in &batch_sizes {
        // Prepare test data
        let mut coeffs = vec![
            F {
                value: BigUint::from(0u32),
                modulus: modulus.clone()
            };
            n
        ];
        coeffs[0] = F::new(7, modulus.clone());
        coeffs[1] = F::new(3, modulus.clone());
        coeffs[2] = F::new(5, modulus.clone());

        // Bit-reverse input for DIT FFT
        bitreverse_permute(&mut coeffs);

        // Prepare batched data
        let mut batched_host_data: Vec<FieldElem> = Vec::with_capacity(batch_size * n);
        for _ in 0..batch_size {
            for c in &coeffs {
                let mont = to_montgomery(&c.value, &modulus, &r2);
                let f = F {
                    value: mont,
                    modulus: modulus.clone(),
                };
                let limbs = f_to_limbs(&f);
                batched_host_data.push(FieldElem {
                    x: limbs[0],
                    y: limbs[1],
                    z: limbs[2],
                    w: limbs[3],
                });
            }
        }

        let data_buf = device.new_buffer_with_data(
            batched_host_data.as_ptr() as *const _,
            (batched_host_data.len() * std::mem::size_of::<FieldElem>()) as u64,
            MTLResourceOptions::StorageModeManaged,
        );

        // Twiddles
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
        let twiddle_buf = device.new_buffer_with_data(
            tw_data.as_ptr() as *const _,
            (tw_data.len() * std::mem::size_of::<FieldElem>()) as u64,
            MTLResourceOptions::StorageModeManaged,
        );

        // Inverse twiddles
        let inv_root = root.modpow(&(&*modulus - BigUint::from(2u32)), &modulus);
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
            (inv_tw_data.len() * std::mem::size_of::<FieldElem>()) as u64,
            MTLResourceOptions::StorageModeManaged,
        );

        let modulus_limbs = biguint_to_limbs(&modulus);
        let modulus_buf = device.new_buffer_with_data(
            modulus_limbs.as_ptr() as *const _,
            (modulus_limbs.len() * std::mem::size_of::<u64>()) as u64,
            MTLResourceOptions::StorageModeManaged,
        );

        let nprime_buf = device.new_buffer_with_data(
            unsafe { std::mem::transmute(&nprime) },
            std::mem::size_of::<u64>() as u64,
            MTLResourceOptions::StorageModeManaged,
        );

        // Benchmark batched FFT operations
        let start = Instant::now();
        run_batched_fft_operations(
            &device,
            &shared_fft_pipeline,
            &bitrev_pipeline,
            &command_queue,
            &data_buf,
            &twiddle_buf,
            &inv_twiddle_buf,
            &modulus_buf,
            &nprime_buf,
            n,
            batch_size,
        );
        let elapsed = start.elapsed();

        let time_ms = elapsed.as_secs_f64() * 1000.0;
        let ffts_per_sec = batch_size as f64 / elapsed.as_secs_f64();
        let throughput = if batch_size == 1 {
            1.0
        } else {
            ffts_per_sec
                / (batch_sizes[0] as f64
                    / batch_sizes.iter().find(|&&x| x == 1).map_or(1.0, |_| {
                        time_ms / batch_sizes.iter().find(|&&x| x == 1).map_or(1.0, |_| 1.0)
                    }))
        };

        println!(
            "{:<12} {:<15.2} {:<15.0} {:<15.2}x",
            batch_size, time_ms, ffts_per_sec, throughput
        );
    }
}
