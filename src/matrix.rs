use core::clone::Clone;
use rand::prelude::*;

#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, values: Vec<f64>) -> Matrix {
        if values.len() != rows * cols {
            panic!("Incorrect number of values for the matrix dimensions.")
        }

        let data = values
            .chunks(cols)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<Vec<f64>>>();
        Matrix { rows, cols, data }
    }

    pub fn random(rows: usize, cols: usize) -> Matrix {
        let mut rng = rand::thread_rng();
        let values = (0..rows * cols)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        Matrix::new(rows, cols, values)
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Matrix dimensions do not match for addition.");
        }
        
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(r1, r2)| r1.iter().zip(r2).map(|(a, b)| a + b).collect())
            .collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    pub fn subtract(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Matrix dimensions do not match for subtraction.");
        }

        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(r1, r2)| r1.iter().zip(r2).map(|(a, b)| a - b).collect())
            .collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    pub fn multiply_by_scalar(&self, scalar: f64) -> Matrix {
        let data = self
            .data
            .iter()
            .map(|row| row.iter().map(|&x| x * scalar).collect())
            .collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    pub fn dot(&self, other: &Matrix) -> Matrix {
        if self.cols != other.rows {
            panic!("Matrix dimensions do not match for dot product.");
        }

        let mut result = Matrix::new(self.rows, other.cols, vec![0.0; self.rows * other.cols]);

        for i in 0..self.rows {
            for j in 0..other.cols {
                for k in 0..self.cols {
                    result.data[i][j] += self.data[i][k] * other.data[k][j]; 
                }
            }
        }
        result
    }

    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::new(self.cols, self.rows, vec![0.0; self.rows * self.cols]);

        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j][i] = self.data[i][j];
            }
        }
        result
    }

    pub fn apply_activation(&self) -> Matrix {
        let data = self
            .data
            .iter()
            .map(|row| row.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect())
            .collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    pub fn apply_activation_derivative(&self) -> Matrix {
        let data = self
            .data
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&x| x * (1.0 - x))
                    .collect()
            })
            .collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    pub fn square_sum(&self) -> f64 {
        self
            .data
            .iter()
            .flat_map(|row| row.iter())
            .map(|&x| x * x)
            .sum()
    }
}