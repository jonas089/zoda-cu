// BabyBear field: p = 2^31 - 2^27 + 1 = 2013265921
//
// Values are kept in canonical form (0 <= value < p). This is the CPU-side
// type; the GPU kernel works on the raw u32 representation.

use std::ops::{Add, Mul, Sub};

pub const BABYBEAR_PRIME: u64 = 2013265921; // 2^31 - 2^27 + 1

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BabyBear {
    pub value: u64,
}

impl BabyBear {
    #[inline]
    pub fn new(value: u64) -> Self {
        Self {
            value: value % BABYBEAR_PRIME,
        }
    }

    #[inline]
    pub fn zero() -> Self {
        Self { value: 0 }
    }

    #[inline]
    pub fn one() -> Self {
        Self { value: 1 }
    }

    #[inline]
    pub fn to_bytes(&self) -> [u8; 8] {
        self.value.to_le_bytes()
    }
}

impl Add for BabyBear {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        // Both operands are < p, so one conditional subtraction reduces the sum.
        let sum = self.value + rhs.value;
        Self {
            value: if sum >= BABYBEAR_PRIME { sum - BABYBEAR_PRIME } else { sum },
        }
    }
}

impl Sub for BabyBear {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let diff = if self.value >= rhs.value {
            self.value - rhs.value
        } else {
            self.value + BABYBEAR_PRIME - rhs.value
        };
        Self { value: diff }
    }
}

impl Mul for BabyBear {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // Both operands are < 2^31, so the product fits in u64.
        Self {
            value: (self.value * rhs.value) % BABYBEAR_PRIME,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_arithmetic() {
        let a = BabyBear::new(100);
        let b = BabyBear::new(200);

        assert_eq!((a + b).value, 300);
        assert_eq!((b - a).value, 100);
        assert_eq!((a - b).value, BABYBEAR_PRIME - 100);
        assert_eq!((a * b).value, 20000);
    }

    #[test]
    fn test_modular_reduction() {
        assert_eq!(BabyBear::new(BABYBEAR_PRIME + 5).value, 5);
        assert_eq!((BabyBear::new(BABYBEAR_PRIME - 1) + BabyBear::one()).value, 0);
    }

    #[test]
    fn test_mul_wraps() {
        let a = BabyBear::new(BABYBEAR_PRIME - 1); // -1
        assert_eq!((a * a).value, 1);
    }
}
