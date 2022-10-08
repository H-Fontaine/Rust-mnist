use matrix::Matrix;
use std::fs::File;
use std::io::{Cursor, Read};
use byteorder::{BigEndian, ReadBytesExt};
use num_traits::NumCast;

pub fn load_images(path : &str) -> Matrix<f32> {
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

    let mut res : Matrix<f32> = Matrix::zeros(nb_images as usize, (nb_rows * nb_columns) as usize);

    for i in 0..(nb_images as usize) {
        for j in 0..res.columns() {
            res[i][j] = NumCast::from(reader.read_u8().unwrap()).unwrap();
        }
    }
    res / NumCast::from(255).unwrap()
}

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

pub fn display(images : &Matrix<f32>, labels : &Vec<usize>, index_image : usize) {
    println!("This number is a {}", labels[index_image]);
    for i in 0..28 {
        for j in 0..28 {
            if images[index_image][i * 28 + j] >= 0.45f32 {
                print!("#")
            } else {
                print!(" ")
            }
        }
        println!();
    }
}