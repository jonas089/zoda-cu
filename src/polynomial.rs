use std::{
    ops::{Add, Mul},
    sync::Arc,
};

use num_bigint::BigUint;

// This file contains an implementation of a Polynomial defined over our Finite Field Elements
use crate::ff::F;
pub struct Polynomial {
    pub coeffs: Vec<F>,
    pub modulus: Arc<BigUint>,
}

#[allow(unused)]
impl Polynomial {
    pub fn zero(modulus: Arc<BigUint>) -> Self {
        Self {
            coeffs: vec![],
            modulus,
        }
    }

    pub fn degree(&self) -> usize {
        if self.coeffs.is_empty() {
            0
        } else {
            self.coeffs.len() - 1
        }
    }

    pub fn new(mut coeffs: Vec<F>, modulus: Arc<BigUint>) -> Self {
        while coeffs
            .last()
            .is_some_and(|x| x.equals(&F::zero(x.modulus.clone())))
        {
            coeffs.pop();
        }
        Self { coeffs, modulus }
    }

    pub fn evaluate(&self, x: &F) -> F {
        match self.coeffs.split_last() {
            None => F::zero(x.modulus.clone()),
            Some((last, rest_rev)) => rest_rev
                .iter()
                .rev()
                .fold(last.clone(), |acc, c| &(&acc * x) + c),
        }
    }

    pub fn scale(&self, scalar: F) -> Self {
        let coeffs: Vec<F> = self.coeffs.iter().map(|c| c * &scalar).collect();
        Polynomial::new(coeffs, self.modulus.clone())
    }

    pub fn from_coeffs(coeffs: Vec<F>) -> Self {
        assert!(
            !coeffs.is_empty(),
            "Polynomial must have at least one coefficient"
        );
        let modulus = coeffs[0].modulus.clone();
        Polynomial::new(coeffs, modulus)
    }

    pub fn interpolate(xs: &[F], ys: &[F]) -> Self {
        assert_eq!(xs.len(), ys.len(), "Mismatched input lengths");
        let n = xs.len();
        let mut result = Polynomial::zero(xs[0].modulus.clone());

        for i in 0..n {
            let mut basis = Polynomial::new(
                vec![F {
                    value: BigUint::from(1u64),
                    modulus: xs[0].modulus.clone(),
                }],
                xs[0].modulus.clone(),
            );
            let mut denom = F {
                value: BigUint::from(1u64),
                modulus: xs[0].modulus.clone(),
            };

            for j in 0..n {
                if i == j {
                    continue;
                }
                let one = F {
                    value: BigUint::from(1u64),
                    modulus: xs[0].modulus.clone(),
                };
                let factor = Polynomial::new(
                    vec![(F::zero(xs[0].modulus.clone()) - xs[j].clone()), one],
                    xs[0].modulus.clone(),
                );

                basis = basis * factor;
                denom = denom * (xs[i].clone() - xs[j].clone());
            }

            let coeff = &ys[i] / &denom;
            basis = basis.scale(coeff);
            result = result + basis;
        }

        result
    }
}

impl Add for Polynomial {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let max_len = std::cmp::max(self.coeffs.len(), rhs.coeffs.len());
        let mut result = vec![F::zero(self.modulus.clone()); max_len];

        for (i, coeff) in result.iter_mut().enumerate().take(self.coeffs.len()) {
            *coeff += self.coeffs[i].clone();
        }

        for (i, coeff) in result.iter_mut().enumerate().take(rhs.coeffs.len()) {
            *coeff += rhs.coeffs[i].clone();
        }

        Self::new(result, self.modulus.clone())
    }
}

impl Mul for Polynomial {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        if self.coeffs.is_empty() || rhs.coeffs.is_empty() {
            return Polynomial::zero(self.modulus.clone());
        }

        let mut result = vec![F::zero(self.modulus.clone()); self.degree() + rhs.degree() + 1];

        for i in 0..self.coeffs.len() {
            for j in 0..rhs.coeffs.len() {
                result[i + j] += self.coeffs[i].clone() * rhs.coeffs[j].clone();
            }
        }

        Polynomial::new(result, self.modulus.clone())
    }
}

#[cfg(test)]
mod tests {
    use crate::{ff::F, polynomial::Polynomial};
    use num_bigint::BigUint;
    use std::sync::Arc;

    #[test]
    fn test_poly_eval() {
        let field_modulus: Arc<BigUint> = Arc::new(BigUint::from(10000u64));
        // 1 + 2x + 3x^2
        let f = Polynomial::new(
            vec![
                F {
                    value: BigUint::from(1u64),
                    modulus: field_modulus.clone(),
                },
                F {
                    value: BigUint::from(2u64),
                    modulus: field_modulus.clone(),
                },
                F {
                    value: BigUint::from(3u64),
                    modulus: field_modulus.clone(),
                },
            ],
            field_modulus.clone(),
        );

        let eval = f.evaluate(&F {
            value: BigUint::from(10u64),
            modulus: field_modulus.clone(),
        });

        eval.assert_eq(&F {
            value: BigUint::from(321u64),
            modulus: field_modulus,
        });
    }

    #[test]
    fn test_lagrange_interpolate() {
        let field_modulus: Arc<BigUint> = Arc::new(BigUint::from(10000u64));
        let xs = vec![F {
            value: BigUint::from(23u64),
            modulus: field_modulus.clone(),
        }];
        let ys = vec![F {
            value: BigUint::from(23u64),
            modulus: field_modulus.clone(),
        }];
        let poly = Polynomial::interpolate(&xs, &ys);
        let eval = poly.evaluate(&F {
            value: BigUint::from(10u64),
            modulus: field_modulus.clone(),
        });

        eval.assert_eq(&F {
            value: BigUint::from(23u64),
            modulus: field_modulus,
        });
    }
}
