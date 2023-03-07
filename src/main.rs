use matrix::Matrix;
use mnist::{display, load_images, load_labels, select_specific_numbers};


fn main() {
    let images : Matrix<f32> = load_images("dataset/train-images.idx3-ubyte");
    let labels = load_labels("dataset/train-labels.idx1-ubyte");
    let (images, labels) = select_specific_numbers(images, labels, vec![9]);
    for i in 20..30 {
        display(&images, &labels, i);
        println!("\n\n");
    }
}