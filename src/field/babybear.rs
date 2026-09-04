// BabyBear field: p = 2^31 - 2^27 + 1 = 2013265921
//
// Values are kept in canonical form (0 <= value < p), which fits in a u32.
// The struct is repr(C) with a single u32, so a slice of BabyBear has the same
// memory layout as a slice of u32 and can be handed to the GPU as is.

use std::ops::{Add, Mul, Sub};

pub const BABYBEAR_PRIME: u32 = 2013265921; // 2^31 - 2^27 + 1

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BabyBear {
    pub value: u32,
}

impl BabyBear {
    #[inline]
    pub fn new(value: u64) -> Self {
        Self {
            value: (value % BABYBEAR_PRIME as u64) as u32,
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
    pub fn to_bytes(&self) -> [u8; 4] {
        self.value.to_le_bytes()
    }
}

impl Add for BabyBear {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        // Both operands are < 2^31, so the sum fits in u32 and one conditional subtraction reduces it.
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
            value: ((self.value as u64 * rhs.value as u64) % BABYBEAR_PRIME as u64) as u32,
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
        assert_eq!(BabyBear::new(BABYBEAR_PRIME as u64 + 5).value, 5);
        assert_eq!((BabyBear::new(BABYBEAR_PRIME as u64 - 1) + BabyBear::one()).value, 0);
    }

    #[test]
    fn test_mul_wraps() {
        let a = BabyBear::new(BABYBEAR_PRIME as u64 - 1); // -1
        assert_eq!((a * a).value, 1);
    }
}
