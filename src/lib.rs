#![allow(unused)]
mod metal;
mod ntt;

use num_bigint::BigUint;
use rand::Rng;
use std::{cmp::max, sync::Arc};

use crate::{ff::F, polynomial::Polynomial};
mod ff;
mod polynomial;
mod types;
use num_traits::ToPrimitive;
use sha2::{Digest, Sha256};

pub struct DataSquare {
    pub cells: Vec<Cell>,
    pub columns: usize,
    pub rows: usize,
}
pub struct Cell {
    pub value: F,
    pub column: usize,
    pub row: usize,
}

impl DataSquare {
    pub fn new(cells: Vec<Cell>, columns: usize, rows: usize) -> Self {
        Self {
            cells,
            columns,
            rows,
        }
    }

    pub fn get_cell(&self, column: usize, row: usize) -> Option<&Cell> {
        self.cells
            .iter()
            .find(|cell| cell.column == column && cell.row == row)
    }

    pub fn set_cell(&mut self, column: usize, row: usize, value: F) {
        if let Some(cell) = self
            .cells
            .iter_mut()
            .find(|c| c.column == column && c.row == row)
        {
            cell.value = value;
        } else {
            self.cells.push(Cell { value, column, row });
        }
        self.rows = max(self.rows, row + 1); // store dimensions as counts
        self.columns = max(self.columns, column + 1);
    }

    pub fn get_row(&self, row: usize) -> Vec<F> {
        let mut row_cells: Vec<_> = self.cells.iter().filter(|cell| cell.row == row).collect();
        row_cells.sort_by_key(|c| c.column);
        row_cells.into_iter().map(|c| c.value.clone()).collect()
    }

    pub fn get_column(&self, column: usize) -> Vec<F> {
        let mut col_cells: Vec<_> = self
            .cells
            .iter()
            .filter(|cell| cell.column == column)
            .collect();
        col_cells.sort_by_key(|c| c.row);
        col_cells.into_iter().map(|c| c.value.clone()).collect()
    }

    pub fn hash_root(&self) -> String {
        let mut hasher = Sha256::new();
        let all_bytes: Vec<u8> = self
            .cells
            .iter()
            .flat_map(|cell| cell.value.to_bytes())
            .collect();
        hasher.update(&all_bytes);
        format!("{:x}", hasher.finalize())
    }
}

#[test]
fn test_zoda_impl() {
    // some NTT friendly modulus
    let modulus = Arc::new(BigUint::from(257u64));
    let mut data_square = DataSquare::new(vec![], 0, 0);
    data_square.set_cell(0, 0, F::new(1, modulus.clone()));
    data_square.set_cell(0, 1, F::new(2, modulus.clone()));
    data_square.set_cell(0, 2, F::new(3, modulus.clone()));
    data_square.set_cell(0, 3, F::new(4, modulus.clone()));
    data_square.set_cell(1, 0, F::new(1, modulus.clone()));
    data_square.set_cell(1, 1, F::new(2, modulus.clone()));
    data_square.set_cell(1, 2, F::new(3, modulus.clone()));
    data_square.set_cell(1, 3, F::new(4, modulus.clone()));

    let domain: Vec<F> = (0..data_square.rows)
        .map(|i| F::new(i as u64, modulus.clone()))
        .collect();

    // 1:4 parity data
    let extended_domain: Vec<F> = (0..data_square.columns * 5)
        .map(|i| F::new(i as u64, modulus.clone()))
        .collect();

    let mut column_polys = Vec::new();
    for column_idx in 0..data_square.columns {
        let column = data_square.get_column(column_idx);
        // interpolate each column into a polynomial
        let column_poly = Polynomial::interpolate(&domain, &column);
        column_polys.push(column_poly);
    }

    // evaluate the column polynomials over the extended domain and create new cells
    let mut extended_data_square = DataSquare::new(vec![], 0, 0);
    for (col_idx, column_poly) in column_polys.into_iter().enumerate() {
        for i in 0..extended_domain.len() {
            let x = &extended_domain[i];
            let y = column_poly.evaluate(&x);
            extended_data_square.set_cell(col_idx, i, y); // (column, row)
        }
    }

    let encoded_data_square_root = extended_data_square.hash_root();

    // compute running sum row-wise for the encoded data (original + parity), using random
    // linear combinations
    let mut y: Vec<F> = Vec::new();

    // compute y using the original data in the extended data square,
    // computing running sum of random linear combinations
    // column-wise
    // generate deterministic coefficients using encoded_data_square_root (fiat shamir)
    let mut deterministic_coefficients: Vec<F> = (0..extended_data_square.rows)
        .map(|i| {
            // hash root + index with SHA256
            let mut hasher = Sha256::new();
            hasher.update(encoded_data_square_root.as_bytes());
            hasher.update(&i.to_le_bytes());
            let digest = hasher.finalize();
            // interpret the whole 256-bit digest as a BigUint
            let big = BigUint::from_bytes_be(&digest);
            // fold it into u64 for your F::new constructor
            // (still deterministic, but using all digest bits)
            let val = (big % u64::MAX).to_u64().unwrap();
            F::new(val, modulus.clone())
        })
        .collect();

    // deterministically derive random coefficients from the root
    for i in 0..deterministic_coefficients.len() {
        deterministic_coefficients[i] =
            deterministic_coefficients[i].clone() + F::new(i as u64, modulus.clone());
    }

    for row_idx in 0..data_square.rows {
        let row_data = extended_data_square.get_row(row_idx);

        // compute running sum of random coefficients * row data
        let running_sum = row_data
            .iter()
            .zip(deterministic_coefficients.iter())
            .map(|(x, y)| x * y)
            .fold(F::zero(modulus.clone()), |acc, x| acc + x);
        y.push(running_sum);
    }

    // now interpolate y the same way as the columns over the original domain (because we only used rows in range 0..data_square.rows)
    let y_poly = Polynomial::interpolate(&domain, &y);
    let mut y_encoded: Vec<F> = Vec::new();
    for x in extended_domain {
        let y_val = y_poly.evaluate(&x);
        y_encoded.push(y_val);
    }

    // 64 queries
    for _ in 0..64 {
        let random_row = rand::rng().random_range(0..extended_data_square.rows);
        let row_data = extended_data_square.get_row(random_row);
        let running_sum = row_data
            .iter()
            .zip(deterministic_coefficients.iter())
            .map(|(x, y)| x * y)
            .fold(F::zero(modulus.clone()), |acc, x| acc + x);

        assert_eq!(running_sum, y_encoded[random_row]);
    }
}
