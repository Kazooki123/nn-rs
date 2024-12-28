use core::clone::Clone;

use crate::matrix::Matrix;

pub struct Layer {
    pub weights: Matrix,
    pub biases: Matrix,
}

pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize) -> Layer {
        Layer {
            weights: Matrix::random(output_size, input_size),
            biases: Matrix::random(output_size, 1)
        }
    }

    pub fn forward(&self, input: &Matrix) -> Matrix {
        let weighted_sum = self.weights.dot(input).add(&self.biases);
        weighted_sum.apply_activation()
    }
}

impl NeuralNetwork {
    pub fn new(layer_sizes: &[usize]) -> NeuralNetwork {
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            layers.push(Layer::new(layer_sizes[i], layer_sizes[i + 1]));
        }
        NeuralNetwork { layers }
    }

    pub fn predict(&self, input: &Matrix) -> Matrix {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward(&output);
        }
        output
    }

    pub fn train(&mut self, input: &Matrix, target: &Matrix, learning_rate: f64) {
        let mut outputs = vec![input.clone()];

        for layer in &self.layers {
            outputs.push(layer.forward(outputs.last().unwrap()));
        }

        let mut error = target.subtract(outputs.last().unwrap());

        for i in (0..self.layers.len()).rev() {
            let gradients = error
                .multiply_by_scalar(learning_rate)
                .multiply_by_scalar(learning_rate);

            let deltas = gradients.dot(&outputs[i].transpose());

            self.layers[i].weights = self.layers[i].weights.add(&deltas);
            self.layers[i].biases = self.layers[i].biases.add(&gradients);
            
            error = self.layers[i].weights.transpose().dot(&error);
        }
    }
}