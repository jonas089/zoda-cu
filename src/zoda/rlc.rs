//! Random linear combinations with multi-limb challenges.
//!
//! Shared by the encoder in `zoda::babybear` and the benchmark validation, so
//! both derive and apply coefficients the same way.

use crate::field::babybear::BabyBear;
use sha2::{Digest, Sha256};

/// Base-field limbs in one RLC challenge.
///
/// The coefficients live in `F_p^RLC_LIMBS`, so a forged RLC passes one check
/// with probability `p^-RLC_LIMBS`. For BabyBear that is `2^-123.6` at four
/// limbs, against `2^-30.9` with a single base-field coefficient. rsema1d does
/// the same thing with `GF(2^16)^8 = GF(2^128)`.
///
/// Four is also the most we can take from one SHA-256: the digest is 32 bytes
/// and each limb consumes an 8-byte window.
pub const RLC_LIMBS: usize = 4;

/// One RLC value: an element of `F_p^RLC_LIMBS` held as its base-field limbs.
///
/// We never multiply two of these, so no extension-field arithmetic is needed.
/// A data element is a base-field scalar and a coefficient is a limb vector,
/// and a scalar times a limb vector is component-wise.
pub type Rlc = [BabyBear; RLC_LIMBS];

pub fn rlc_zero() -> Rlc {
    [BabyBear::zero(); RLC_LIMBS]
}

/// `sum over columns of row[col] * coeff[col]`, accumulated per limb.
pub fn rlc_row(row: &[BabyBear], coefficients: &[Rlc]) -> Rlc {
    let mut acc = rlc_zero();
    for (&value, coeff) in row.iter().zip(coefficients.iter()) {
        for limb in 0..RLC_LIMBS {
            acc[limb] = acc[limb] + (value * coeff[limb]);
        }
    }
    acc
}

/// One coefficient per column, each `RLC_LIMBS` independent base-field elements.
///
/// A single SHA-256 per column gives all the limbs: the digest's four disjoint
/// 8-byte windows are independent uniform `u64` under the random oracle model,
/// which is what the limbs have to be for the soundness argument to multiply
/// out to `p^-RLC_LIMBS`.
pub fn generate_deterministic_coefficients(data_root: &str, num_columns: usize) -> Vec<Rlc> {
    (0..num_columns)
        .map(|i| {
            let mut hasher = Sha256::new();
            hasher.update(data_root.as_bytes());
            hasher.update(i.to_le_bytes());
            let digest = hasher.finalize();

            let mut coeff = rlc_zero();
            for (limb, slot) in coeff.iter_mut().enumerate() {
                let window: [u8; 8] = digest[limb * 8..(limb + 1) * 8]
                    .try_into()
                    .expect("8 byte window");
                *slot = BabyBear::new(u64::from_be_bytes(window));
            }
            coeff
        })
        .collect()
}
