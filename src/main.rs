use matrix::Matrix;
use mnist::{display, load_images, load_labels};


fn main() {
    let images : Matrix<f32> = load_images("dataset/train-images.idx3-ubyte");
    let labels = load_labels("dataset/train-labels.idx1-ubyte");
    for i in 20..30 {
        display(&images, &labels, i);
        println!("\n\n");
    }
}