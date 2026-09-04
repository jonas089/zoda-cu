//! GPU benchmarks. These are tests that need a CUDA device, so the whole
//! module only exists under `cargo test --features cuda`.

pub mod utils;
mod celestia_squares;
mod zoda_validated;
