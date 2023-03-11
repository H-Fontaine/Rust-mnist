use matrix::Matrix;
use mnist::{load_labels, select_specific_numbers, load_images_normalized, save_image_normalized};


fn main() {
    let images : Matrix<f32> = load_images_normalized("dataset/t10k-images.idx3-ubyte");
    let labels = load_labels("dataset/t10k-labels.idx1-ubyte");
    let (images, labels) = select_specific_numbers(&images, &labels, vec![9]);
    save_image_normalized("C:/Users/Hugo/Documents", "test".to_string(), &images, &labels, 4);
}