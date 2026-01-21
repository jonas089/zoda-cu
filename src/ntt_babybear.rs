// NTT implementation for BabyBear field

use crate::babybear::BabyBear;

/// Bit-reverse permutation
#[inline]
fn bit_reverse(mut x: usize, log_n: usize) -> usize {
    let mut result = 0;
    for _ in 0..log_n {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// In-place Cooley-Tukey NTT
pub fn ntt(values: &mut [BabyBear], omega: BabyBear) {
    let n = values.len();
    assert!(n.is_power_of_two(), "NTT size must be power of 2");
    let log_n = n.trailing_zeros() as usize;

    // Bit-reversal permutation
    for i in 0..n {
        let j = bit_reverse(i, log_n);
        if i < j {
            values.swap(i, j);
        }
    }

    // Cooley-Tukey butterfly iterations
    let mut len = 2;
    while len <= n {
        let step = n / len;
        let w_len = omega.pow(step as u64); // omega^(n/len)

        for i in (0..n).step_by(len) {
            let mut w = BabyBear::one();
            for j in 0..len / 2 {
                let u = values[i + j];
                let v = values[i + j + len / 2] * w;
                values[i + j] = u + v;
                values[i + j + len / 2] = u - v;
                w = w * w_len;
            }
        }
        len *= 2;
    }
}

/// In-place inverse NTT
pub fn intt(values: &mut [BabyBear], omega: BabyBear) {
    let n = values.len();

    // Use omega^-1 for inverse transform
    let inv_omega = omega.pow(n as u64 - 1);
    ntt(values, inv_omega);

    // Scale by 1/n
    let inv_n = BabyBear::new(n as u64).inverse();
    for v in values.iter_mut() {
        *v = *v * inv_n;
    }
}

/// Generate roots of unity domain
pub fn roots_of_unity_domain(n: usize) -> Vec<BabyBear> {
    assert!(n.is_power_of_two(), "Domain size must be power of 2");
    let log_n = n.trailing_zeros();
    let omega = BabyBear::get_root_of_unity(log_n);

    let mut domain = Vec::with_capacity(n);
    let mut cur = BabyBear::one();
    for _ in 0..n {
        domain.push(cur);
        cur = cur * omega;
    }
    domain
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::babybear::BabyBear;

    #[test]
    fn test_ntt_intt_roundtrip() {
        let n = 256usize;
        let omega = BabyBear::get_root_of_unity(n.trailing_zeros());

        // Create test data
        let mut values: Vec<BabyBear> = (0..n)
            .map(|i| BabyBear::new((i * 7 + 3) as u64))
            .collect();

        let original = values.clone();

        // Forward NTT
        ntt(&mut values, omega);

        // Inverse NTT
        intt(&mut values, omega);

        // Should recover original values
        for (a, b) in original.iter().zip(values.iter()) {
            assert_eq!(a.value, b.value, "NTT roundtrip failed");
        }
    }

    #[test]
    fn test_polynomial_evaluation() {
        // Test that NTT correctly evaluates polynomial at roots of unity
        let n = 8usize;
        let omega = BabyBear::get_root_of_unity(n.trailing_zeros());
        let domain = roots_of_unity_domain(n);

        // Polynomial: 1 + 2x + 3x^2
        let mut coeffs = vec![BabyBear::zero(); n];
        coeffs[0] = BabyBear::new(1);
        coeffs[1] = BabyBear::new(2);
        coeffs[2] = BabyBear::new(3);

        // Compute NTT (evaluations at roots of unity)
        let mut evals = coeffs.clone();
        ntt(&mut evals, omega);

        // Verify first evaluation: f(1) = 1 + 2 + 3 = 6
        assert_eq!(evals[0].value, 6);

        // Verify evaluation at omega
        let x = domain[1];
        let expected = coeffs[0] + coeffs[1] * x + coeffs[2] * x * x;
        assert_eq!(evals[1].value, expected.value);
    }

    #[test]
    fn test_roots_of_unity() {
        let n = 16usize;
        let domain = roots_of_unity_domain(n);

        // First element should be 1
        assert_eq!(domain[0].value, 1);

        // omega^n should equal 1
        let omega = domain[1];
        let result = omega.pow(n as u64);
        assert_eq!(result.value, 1);

        // All elements should be distinct (except the first and last which should both be 1 for a full cycle)
        // Actually, for n elements, we should have n distinct values
        for i in 0..n {
            for j in i+1..n {
                assert_ne!(
                    domain[i].value,
                    domain[j].value,
                    "domain[{}] = {} equals domain[{}] = {}",
                    i, domain[i].value, j, domain[j].value
                );
            }
        }
    }
}
