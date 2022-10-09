use matrix::Matrix;
use std::fs::File;
use std::io::{Cursor, Read};
use byteorder::{BigEndian, ReadBytesExt};
use num_traits::{Float, Num, NumCast};

pub fn load_images<T : Num>(path : &str) -> Matrix<T> where T : Copy + NumCast {
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

pub fn labels_to_responses<T : Num>(labels : &Vec<usize>) -> Matrix<T> where T : Copy + NumCast {
    let mut res : Matrix<T> = Matrix::zeros(labels.len(), 10);
    for i in 0..labels.len() {
        res[i][labels[i]] = NumCast::from(1).unwrap();
    }
    res
}

pub fn display<F : Float, T : Num>(images : &Matrix<T>, labels : &Vec<usize>, index_image : usize) where T : PartialOrd<F> {
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