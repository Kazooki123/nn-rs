#[macro_export]
macro_rules! create_layer {
    ($input:expr, $output:expr) => {
        Layer::new($input, $output)
    };
}