use nn::{Matrix, NeuralNetwork};

fn main() {
    let inputs = vec![
        Matrix::new(2, 1, vec![0.0, 0.0]),
        Matrix::new(2, 1, vec![0.0, 1.0]),
        Matrix::new(2, 1, vec![1.0, 0.0]),
        Matrix::new(2, 1, vec![1.0, 1.0]),
    ];

    let targets = vec![
        Matrix::new(1, 1, vec![0.0]),
        Matrix::new(1, 1, vec![1.0]),
        Matrix::new(1, 1, vec![1.0]),
        Matrix::new(1, 1, vec![0.0]),
    ];

    let mut nn = NeuralNetwork::new(&[2, 2, 1]);

    let epochs = 10_000;
    let learning_rate = 0.1;

    for epoch in 0..epochs {
        let mut total_error = 0.0;

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let prediction = nn.predict(input);

            let error = target.subtract(&prediction);
            total_error += error.square_sum();

            nn.train(input, target, learning_rate);
        }

        if epoch % 1000 == 0 {
            println!("Epoch {}: Error = {}", epoch, total_error);
        }
    }

    println!("\nTesting Trained Neural Network");
    for input in &inputs {
        let output = nn.predict(input);
        println!("Input: {:?} -> Output: {:?}", input.data, output.data);
    }
}