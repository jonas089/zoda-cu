// CUDA NTT: FFI bindings and safe Rust wrapper.
//
// The C side (cuda/ntt_kernel.cu) owns all device memory. Each call copies the
// host slice to the GPU, runs the transform, and copies the result back.

use crate::field::babybear::{BabyBear, BABYBEAR_PRIME};
use crate::ntt::roots_of_unity;

type CudaError = i32;
const CUDA_SUCCESS: CudaError = 0;

#[link(name = "ntt_cuda", kind = "static")]
extern "C" {
    fn cuda_ntt(values: *mut u32, roots: *const u32, n: u32) -> CudaError;
    fn cuda_intt(values: *mut u32, roots: *const u32, n: u32) -> CudaError;
    fn cudaGetDeviceCount(count: *mut i32) -> CudaError;
}

/// Check if a CUDA device is present.
pub fn cuda_available() -> bool {
    let mut count = 0;
    let err = unsafe { cudaGetDeviceCount(&mut count) };
    err == CUDA_SUCCESS && count > 0
}

fn run(
    values: &mut [BabyBear],
    f: unsafe extern "C" fn(*mut u32, *const u32, u32) -> CudaError,
    name: &str,
) -> Result<(), String> {
    let n = values.len();
    assert!(n.is_power_of_two(), "NTT size must be power of 2");

    // The circle: roots[i] = omega^i for i in 0..n.
    let roots: Vec<u32> = roots_of_unity(n as u32, BABYBEAR_PRIME as u32);

    // BabyBear values are canonical (< 2^31), so they fit in u32 losslessly.
    let mut raw: Vec<u32> = values.iter().map(|v| v.value as u32).collect();

    let err = unsafe { f(raw.as_mut_ptr(), roots.as_ptr(), n as u32) };
    if err != CUDA_SUCCESS {
        return Err(format!("{name} failed with CUDA error {err}"));
    }

    for (v, r) in values.iter_mut().zip(raw) {
        *v = BabyBear::new(r as u64);
    }
    Ok(())
}

/// Forward NTT on the GPU, in place.
pub fn ntt_cuda(values: &mut [BabyBear]) -> Result<(), String> {
    run(values, cuda_ntt, "cuda_ntt")
}

/// Inverse NTT on the GPU, in place.
pub fn intt_cuda(values: &mut [BabyBear]) -> Result<(), String> {
    run(values, cuda_intt, "cuda_intt")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntt::{intt_babybear as cpu_intt, ntt_babybear as cpu_ntt};

    fn sample(n: usize) -> Vec<BabyBear> {
        (0..n).map(|i| BabyBear::new((i * 7 + 3) as u64)).collect()
    }

    #[test]
    fn test_cuda_ntt_vs_cpu() {
        for log_n in [1, 4, 8, 12] {
            let n = 1usize << log_n;
            let mut cpu_values = sample(n);
            let mut gpu_values = cpu_values.clone();

            cpu_ntt(&mut cpu_values);
            ntt_cuda(&mut gpu_values).unwrap();

            for (i, (c, g)) in cpu_values.iter().zip(&gpu_values).enumerate() {
                assert_eq!(c.value, g.value, "n={n}: mismatch at {i}: CPU={} GPU={}", c.value, g.value);
            }
        }
    }

    #[test]
    fn test_cuda_intt_vs_cpu() {
        for log_n in [1, 4, 8, 12] {
            let n = 1usize << log_n;
            let mut cpu_values = sample(n);
            let mut gpu_values = cpu_values.clone();

            cpu_intt(&mut cpu_values);
            intt_cuda(&mut gpu_values).unwrap();

            for (i, (c, g)) in cpu_values.iter().zip(&gpu_values).enumerate() {
                assert_eq!(c.value, g.value, "n={n}: mismatch at {i}: CPU={} GPU={}", c.value, g.value);
            }
        }
    }

    #[test]
    fn test_cuda_roundtrip() {
        let original = sample(1 << 10);
        let mut values = original.clone();
        ntt_cuda(&mut values).unwrap();
        intt_cuda(&mut values).unwrap();
        for (i, (o, r)) in original.iter().zip(&values).enumerate() {
            assert_eq!(o.value, r.value, "roundtrip failed at {i}");
        }
    }
}
