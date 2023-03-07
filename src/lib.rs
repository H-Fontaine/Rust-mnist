use std::borrow::Borrow;
use std::collections::HashSet;
use matrix::Matrix;
use std::fs::File;
use std::io::{Cursor, Read};
use std::ops::Div;
use byteorder::{BigEndian, ReadBytesExt};
use num_traits::{Zero, NumCast};


/*
Load images from the mnist dataset stored at the provided path in a matrix
 - path : &str          Where the file is stored
*/
pub fn load_images<T>(path : &str) -> Matrix<T> where T : Zero + Copy + NumCast + Div<T, Output = T> {
    let mut file = File::open(path).unwrap();
    let mut buffer: Vec<u8> = Vec::new();
    file.read_to_end(&mut buffer).unwrap(); //read the entire file
    let mut reader = Cursor::new(buffer);

    let _magic_number = reader.read_u32::<BigEndian>().unwrap();
    let nb_images = reader.read_u32::<BigEndian>().unwrap();
    let nb_rows = reader.read_u32::<BigEndian>().unwrap();
    assert_eq!(nb_rows, 28);
    let nb_columns = reader.read_u32::<BigEndian>().unwrap();
    assert_eq!(nb_columns, 28);

    let mut res : Matrix<T> = Matrix::zeros(nb_images as usize, (nb_rows * nb_columns) as usize);

    for i in 0..(nb_images as usize) {
        for j in 0..res.columns() {
            res[i][j] = NumCast::from(reader.read_u8().unwrap()).unwrap();
        }
    }
    res / NumCast::from(255).unwrap()
}


/*
Load labels from the mnist dataset stored at the provided path in a vec
 - path : &str          Where the file is stored
*/
pub fn load_labels(path : &str) -> Vec<usize> {
    let mut file = File::open(path).unwrap();
    let mut buffer: Vec<u8> = Vec::new();
    file.read_to_end(&mut buffer).unwrap(); //read the entire file
    let mut reader = Cursor::new(buffer);

    let _magic_number = reader.read_u32::<BigEndian>().unwrap();
    let nb_labels = reader.read_u32::<BigEndian>().unwrap() as usize;

    let mut res = Vec::<usize>::with_capacity(nb_labels);
    for _i in 0..nb_labels {
        res.push(reader.read_u8().unwrap() as usize);
    }
    res
}

/*
Select the provided numbers according to the provided vector
 */
pub fn select_specific_numbers<B, V, W, T>(images : B, labels : V, wanted_numbers : W) -> (Matrix<T>, Vec<usize>) where T : Copy, B : Borrow<Matrix<T>>, V : Borrow<Vec<usize>>, W : Borrow<Vec<usize>> {
    let wanted_number_set : HashSet<usize> = wanted_numbers.borrow().clone().into_iter().collect();
    let wanted_images = images.borrow().chose_lines_by_bool(labels.borrow().into_iter().map(|a| {wanted_number_set.contains(a)}).collect::<Vec<bool>>());
    let mut wanted_label = Vec::new();
    for (bool, label) in labels.borrow().into_iter().map(|a| {wanted_number_set.contains(a)}).zip(labels.borrow().into_iter()) {
        if bool {
            wanted_label.push(*label);
        }
    }
    (wanted_images, wanted_label)
}

/*
Translate the provided labels in matrix of response understandable for the neural network :
The matrix returned res = (a)<i,j>    i in [0, label size]        is for all i, j : (a)<i,j>  = | 1 if j = label[i]
                                      j in [0, 9]                                               | 0 if not
 - labels : &Vec<usize>         The labels
*/
pub fn labels_to_responses<T>(labels : &Vec<usize>) -> Matrix<T> where T : Zero + Copy + NumCast {
    let mut res : Matrix<T> = Matrix::zeros(labels.len(), 10);
    for i in 0..labels.len() {
        res[i][labels[i]] = NumCast::from(1).unwrap();
    }
    res
}


/*
Display the chosen labeled image in the console
 - images : &Matrix<T>          The images that can be display
 - labels : &Vec<usize>         The labels associated to the images
 - index_image : usize          The index of the image to display
*/
pub fn display<T>(images : &Matrix<T>, labels : &Vec<usize>, index_image : usize) where T : NumCast + PartialOrd {
    println!("This number is a {}", labels[index_image]);
    for i in 0..28 {
        for j in 0..28 {
            if images[index_image][i * 28 + j] >= NumCast::from(0.45).unwrap() {
                print!("#")
            } else {
                print!(" ")
            }
        }
        println!();
    }
}