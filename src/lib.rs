use std::collections::HashSet;
use matrix::Matrix;
use std::fs::File;
use std::io::{Cursor, Read};
use std::ops::Div;
use byteorder::{BigEndian, ReadBytesExt};
use num_traits::{Zero, NumCast};
use plotters::drawing::IntoDrawingArea;
use plotters::prelude::BitMapBackend;
use plotters::style::RGBColor;


/*
Load images from the mnist dataset stored at the provided path in a matrix
 - path : &str          Where the file is stored
*/
pub fn load_images<T>(path : &str) -> Matrix<T> where T : Zero + Copy + NumCast {
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
    res
}

pub fn load_images_normalized<T>(path : &str) -> Matrix<T> where T : Zero + Copy + NumCast + Div<T, Output = T> {
    load_images(path) / NumCast::from(255).unwrap()
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
pub fn select_specific_numbers<T>(images : &Matrix<T>, labels : &Vec<usize>, wanted_numbers : Vec<usize>) -> (Matrix<T>, Vec<usize>) where T : Copy {
    let wanted_number_set : HashSet<usize> = wanted_numbers.into_iter().collect();
    let wanted_images = images.chose_lines_by_bool(labels.into_iter().map(|a| {wanted_number_set.contains(a)}).collect::<Vec<bool>>());
    let mut wanted_label = Vec::new();
    for (bool, label) in labels.into_iter().map(|a| {wanted_number_set.contains(a)}).zip(labels.into_iter()) {
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
pub fn print<T>(images : &Matrix<T>, labels : &Vec<usize>, index_image : usize) where T : NumCast + PartialOrd {
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

/*
Save the image at the provided location
 - images : &Matrix<T>          The images that can be saved
 - labels : &Vec<usize>         The labels associated to the images
 - index_image : usize          The index of the image to display
*/
pub fn save_image<T>(path : &str, images : &Matrix<T>, labels : &Vec<usize>, index_image : usize) where T : NumCast + PartialOrd + Copy {
    let path = path.to_owned() + &"/Image of a ".to_owned() + &labels[index_image].to_string() + &" .png".to_owned();
    let path = path.as_str();
    let drawing_area = BitMapBackend::new(path, (500, 500)).into_drawing_area();
    let child_drowing_areas = drawing_area.split_evenly(((images.columns() as f64).sqrt() as usize, (images.columns() as f64).sqrt() as usize));
    for (area, &grey_value) in child_drowing_areas.into_iter().zip(images[index_image].into_iter()) {
        let mut value : u8 = NumCast::from(grey_value).unwrap();
        value = 255 - value;
        area.fill(&RGBColor(value, value, value)).unwrap();
    }
    drawing_area.present().unwrap();
}


/*
Save the normalized image at the provided location
 - images : &Matrix<T>          The images that can be saved
 - labels : &Vec<usize>         The labels associated to the images
 - index_image : usize          The index of the image to display
*/
pub fn save_image_normalized<T>(path : &str, images : &Matrix<T>, labels : &Vec<usize>, index_image : usize) where T : NumCast + PartialOrd + Copy {
    let path = path.to_owned() + &"/Image of a ".to_owned() + &labels[index_image].to_string() + &" .png".to_owned();
    let path = path.as_str();
    let drawing_area = BitMapBackend::new(path, (500, 500)).into_drawing_area();
    let child_drowing_areas = drawing_area.split_evenly(((images.columns() as f64).sqrt() as usize, (images.columns() as f64).sqrt() as usize));
    for (area, &grey_value) in child_drowing_areas.into_iter().zip(images[index_image].into_iter()) {
        let temp : f32 = NumCast::from(grey_value).unwrap();
        let value = 255 - (temp * 255f32) as u8;
        area.fill(&RGBColor(value, value, value)).unwrap();
    }
    drawing_area.present().unwrap();
}



















