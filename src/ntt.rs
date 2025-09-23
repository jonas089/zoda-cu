use std::sync::Arc;

use num_bigint::BigUint;

use crate::ff::F;

fn bit_reverse(mut x: usize, log_n: usize) -> usize {
    let mut result = 0;
    for _ in 0..log_n {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

pub fn fft(values: &mut [F], root: &F) {
    let n = values.len();
    let log_n = n.trailing_zeros() as usize;
    // bit reversal
    for i in 0..n {
        let j = bit_reverse(i, log_n);
        if i < j {
            values.swap(i, j);
        }
    }
    // Cooley–Tukey iterative
    let mut len = 2;
    while len <= n {
        let step = n / len;
        let w_len = root.clone().pow(step as u32); // omega^(n/len)
        for i in (0..n).step_by(len) {
            let mut w = F::new(1, values[0].modulus.clone());
            for j in 0..len / 2 {
                let u = values[i + j].clone();
                let v = &values[i + j + len / 2] * &w;
                values[i + j] = &u + &v;
                values[i + j + len / 2] = &u - &v;
                w = &w * &w_len;
            }
        }
        len *= 2;
    }
}

pub fn ifft(values: &mut [F], root: &F) {
    let n = values.len();
    let inv_root = root.clone().pow((n as u32) - 1); // root^-1
    fft(values, &inv_root);
    let inv_n = F::new(1, values[0].modulus.clone()) / F::new(n as u64, values[0].modulus.clone());
    for v in values.iter_mut() {
        *v = v.clone() * inv_n.clone();
    }
}

/// Build a multiplicative subgroup (roots of unity domain) of size `n`.
/// Requires that `n` divides (modulus - 1).
pub fn roots_of_unity_domain(n: usize, modulus: Arc<BigUint>) -> Vec<F> {
    use num_bigint::BigUint;
    assert!(n.is_power_of_two(), "FFT needs power-of-two size");
    let p = modulus.as_ref();

    // Find a primitive root g (generator of F_p^*).
    // For small primes you can brute force, for large ones precompute.
    fn is_generator(g: &BigUint, p: &BigUint) -> bool {
        let phi = p - BigUint::from(1u32);
        // factor phi (here we just try small divisors — fine for toy primes)
        let mut d = BigUint::from(2u32);
        let mut factors = vec![];
        let mut tmp = phi.clone();
        while &d * &d <= tmp {
            if (&tmp % &d) == BigUint::ZERO {
                factors.push(d.clone());
                while (&tmp % &d) == BigUint::ZERO {
                    tmp /= &d;
                }
            }
            d += 1u32;
        }
        if tmp > BigUint::from(1u32) {
            factors.push(tmp);
        }

        for q in factors {
            let exp = &phi / q;
            if g.modpow(&exp, p) == BigUint::from(1u32) {
                return false;
            }
        }
        true
    }

    let mut g_int = BigUint::from(2u32);
    while !is_generator(&g_int, p) {
        g_int += 1u32;
    }
    let g = F::new(g_int.to_u64_digits()[0], modulus.clone());

    // compute ω = g^((p-1)/n), the primitive n-th root of unity
    let exp = (p - 1u32) / BigUint::from(n as u32);
    let omega_big = g.value.modpow(&exp, p);
    let omega = F {
        value: omega_big,
        modulus: modulus.clone(),
    };

    // collect [1, ω, ω^2, …, ω^(n-1)]
    let mut domain = Vec::with_capacity(n);
    let mut cur = F::new(1, modulus.clone());
    for _ in 0..n {
        domain.push(cur.clone());
        cur = &cur * &omega;
    }
    domain
}

#[test]
fn test_fft_interpolation() {
    use num_bigint::BigUint;
    use std::sync::Arc;

    let modulus = Arc::new(BigUint::from(257u64));
    let n = 256;

    let domain = roots_of_unity_domain(n, modulus.clone());

    let coeffs = vec![
        F::new(7, modulus.clone()),
        F::new(3, modulus.clone()),
        F::new(5, modulus.clone()),
    ];

    // === Interpolate by going coeffs -> evals -> coeffs ===
    // pad coeffs to length n
    let mut evals = coeffs.clone();
    evals.resize(n, F::zero(modulus.clone()));

    // primitive n-th root of unity is the domain[1]
    let omega = domain[1].clone();

    // FFT to evaluations
    fft(&mut evals, &omega);

    // Inverse FFT back to coefficients
    let mut recovered = evals.clone();
    ifft(&mut recovered, &omega);

    // Shrink recovered to original degree
    let recovered_trimmed: Vec<F> = recovered.into_iter().take(coeffs.len()).collect();

    // Check coefficients match
    for (a, b) in coeffs.iter().zip(recovered_trimmed.iter()) {
        assert!(a.equals(b), "coeff mismatch: {:?} vs {:?}", a, b);
    }
}
