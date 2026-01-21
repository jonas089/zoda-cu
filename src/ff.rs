// Thie file contains an implementation of Finite Field Elements
// that is generic w.r. to the Modulus

use num_bigint::BigUint;
use std::{
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub},
    sync::Arc,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct F {
    pub value: BigUint,
    pub modulus: Arc<BigUint>,
}

impl F {
    pub fn new(value: u64, modulus: Arc<BigUint>) -> Self {
        Self {
            value: BigUint::from(value) % &*modulus,
            modulus,
        }
    }
    pub fn from_biguint(value: BigUint, modulus: Arc<BigUint>) -> Self {
        Self {
            value: value % &*modulus,
            modulus,
        }
    }
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = self.value.to_bytes_be();
        bytes.append(&mut self.modulus.to_bytes_be());
        bytes
    }
}

impl Add for F {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        assert_eq!(self.modulus, rhs.modulus, "different fields");
        Self {
            value: (self.value + rhs.value) % &*self.modulus,
            modulus: self.modulus,
        }
    }
}

impl<'a, 'b> Add<&'b F> for &'a F {
    type Output = F;
    fn add(self, rhs: &'b F) -> F {
        assert_eq!(self.modulus, rhs.modulus);
        F {
            value: (&self.value + &rhs.value) % &*self.modulus,
            modulus: self.modulus.clone(),
        }
    }
}

impl Sub for F {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.modulus, rhs.modulus, "different fields");
        let m = &self.modulus;
        let val = if self.value >= rhs.value {
            &self.value - &rhs.value
        } else {
            (&self.value + &**m) - &rhs.value
        };
        Self {
            value: val % &**m,
            modulus: m.clone(),
        }
    }
}

impl<'a, 'b> Sub<&'b F> for &'a F {
    type Output = F;
    fn sub(self, rhs: &'b F) -> F {
        assert_eq!(self.modulus, rhs.modulus, "different fields");
        let m = &self.modulus;
        let val = if self.value >= rhs.value {
            &self.value - &rhs.value
        } else {
            (&self.value + &**m) - &rhs.value
        };
        F {
            value: val % &**m,
            modulus: m.clone(),
        }
    }
}

impl AddAssign for F {
    fn add_assign(&mut self, rhs: Self) {
        assert_eq!(self.modulus, rhs.modulus, "different fields");
        self.value = (self.value.clone() + rhs.value) % &*self.modulus;
    }
}

impl Mul for F {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        assert_eq!(self.modulus, rhs.modulus, "different fields");
        Self {
            value: (self.value * rhs.value) % &*self.modulus,
            modulus: self.modulus,
        }
    }
}

impl<'a, 'b> Mul<&'b F> for &'a F {
    type Output = F;
    fn mul(self, rhs: &'b F) -> F {
        assert_eq!(self.modulus, rhs.modulus, "different fields");
        F {
            value: (&self.value * &rhs.value) % &*self.modulus,
            modulus: self.modulus.clone(),
        }
    }
}

impl MulAssign for F {
    fn mul_assign(&mut self, rhs: Self) {
        assert_eq!(self.modulus, rhs.modulus, "different fields");
        self.value = (self.value.clone() * rhs.value) % &*self.modulus
    }
}

impl Div for F {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        assert_eq!(self.modulus, rhs.modulus, "different fields");
        assert_ne!(rhs.value, BigUint::ZERO, "division by zero");
        let exp = &*rhs.modulus - BigUint::from(2u32);
        let inv = rhs.value.modpow(&exp, &rhs.modulus);
        Self {
            value: (self.value * inv) % &*rhs.modulus,
            modulus: rhs.modulus,
        }
    }
}

impl<'a, 'b> Div<&'b F> for &'a F {
    type Output = F;
    fn div(self, rhs: &'b F) -> F {
        assert_eq!(self.modulus, rhs.modulus, "different fields");
        assert_ne!(rhs.value, BigUint::ZERO, "division by zero");
        let exp = &*rhs.modulus - BigUint::from(2u32);
        let inv = rhs.value.modpow(&exp, &rhs.modulus);
        F {
            value: (&self.value * &inv) % &*rhs.modulus,
            modulus: rhs.modulus.clone(),
        }
    }
}

impl DivAssign for F {
    fn div_assign(&mut self, rhs: Self) {
        assert_eq!(self.modulus, rhs.modulus, "different fields");
        let exp = &*rhs.modulus - BigUint::from(2u32);
        let inv = rhs.value.modpow(&exp, &rhs.modulus);
        self.value = (self.value.clone() * inv) % &*self.modulus;
    }
}

impl Neg for F {
    type Output = Self;
    fn neg(self) -> Self {
        if self.value == BigUint::ZERO {
            self
        } else {
            Self {
                value: &*self.modulus - &self.value,
                modulus: self.modulus,
            }
        }
    }
}

impl<'a> Neg for &'a F {
    type Output = F;
    fn neg(self) -> F {
        if self.value == BigUint::ZERO {
            self.clone()
        } else {
            F {
                value: &*self.modulus - &self.value,
                modulus: self.modulus.clone(),
            }
        }
    }
}

impl F {
    pub fn pow(self, power: u32) -> Self {
        Self {
            value: self.value.modpow(&BigUint::from(power), &self.modulus),
            modulus: self.modulus,
        }
    }
    pub fn equals(&self, rhs: &Self) -> bool {
        if self.value == rhs.value && self.modulus == rhs.modulus {
            return true;
        }
        false
    }
    pub fn assert_eq(&self, rhs: &Self) -> bool {
        if self.value == rhs.value && self.modulus == rhs.modulus {
            return true;
        }
        panic!(
            "Field element mismatch: lhs: {:?}, rhs: {:?}",
            &self.value, rhs.value
        );
    }
    pub fn zero(modulus: Arc<BigUint>) -> Self {
        Self {
            value: BigUint::ZERO,
            modulus,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigUint;
    use std::sync::Arc;

    #[test]
    fn test_negation() {
        let modulus = Arc::new(BigUint::from(7u64));

        let zero = F::zero(modulus.clone());
        let neg_zero = -&zero;
        assert!(zero.equals(&neg_zero));

        let three = F {
            value: BigUint::from(3u64),
            modulus: modulus.clone(),
        };
        let neg_three = -&three;
        let expected = F {
            value: BigUint::from(4u64),
            modulus: modulus.clone(),
        };
        assert!(neg_three.equals(&expected));

        let sum = &three + &neg_three;
        assert!(sum.equals(&zero));
    }
}
