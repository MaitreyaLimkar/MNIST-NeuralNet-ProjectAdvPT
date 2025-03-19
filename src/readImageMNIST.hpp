//
// Created by Maitreya Limkar on 25-01-2025.
//

#ifndef MNIST_IMAGE_READER_HPP
#define MNIST_IMAGE_READER_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>

class DataSetImages
{
private:
    size_t batch_size;
    size_t number_of_images;
    size_t number_of_rows;
    size_t number_of_columns;
    std::vector<Eigen::MatrixXd> batches;

public:
    // Declaring constructor and destructor
    explicit DataSetImages(size_t batch_size);
    ~DataSetImages();

    // Declaring public member functions
    void readImageData(const std::string &filepath);
    void writeImageToFile(const std::string &filepath, size_t index);
    Eigen::MatrixXd getBatch(size_t index);
    size_t getNoOfBatches();
};

// Initializing constructor with batch size
inline DataSetImages::DataSetImages(size_t batch_size) :
    batch_size(batch_size), number_of_images(0),
    number_of_rows(0), number_of_columns(0) {}

DataSetImages::~DataSetImages() {}

// Returning a batch matrix at the given index
inline Eigen::MatrixXd DataSetImages::getBatch(size_t index)
{
    return batches[index];
}

// Returning the total number of batches
inline size_t DataSetImages::getNoOfBatches()
{
    return batches.size();
}

// Reading MNIST image data and storing it in batches
inline void DataSetImages::readImageData(const std::string& input_filepath)
{
    // Opening input file in binary mode
    std::ifstream input_file(input_filepath, std::ios::binary);
    // Checking if file is open
    if (!input_file.is_open())
    {
        std::cerr << "Error: Unable to open file: " << input_filepath << std::endl;
        return;
    }

    // Declaring variables for reading header information
    char bin_data[4];

    // Reading and reversing magic number
    input_file.read(bin_data, 4);
    std::reverse(bin_data, bin_data + 4);
    int magic_number = 0;
    std::memcpy(&magic_number, bin_data, sizeof(int));

    // Reading and reversing number of images
    input_file.read(bin_data, 4);
    std::reverse(bin_data, bin_data + 4);
    int number_of_images_ = 0;
    std::memcpy(&number_of_images_, bin_data, sizeof(int));
    number_of_images = number_of_images_;

    // Reading and reversing number of rows
    input_file.read(bin_data, 4);
    std::reverse(bin_data, bin_data + 4);
    int number_of_rows_ = 0;
    std::memcpy(&number_of_rows_, bin_data, sizeof(int));
    number_of_rows = number_of_rows_;

    // Reading and reversing number of columns
    input_file.read(bin_data, 4);
    std::reverse(bin_data, bin_data + 4);
    int number_of_columns_ = 0;
    std::memcpy(&number_of_columns_, bin_data, sizeof(int));
    number_of_columns = number_of_columns_;

    // Calculating image size and batch-related values
    size_t image_size = number_of_rows * number_of_columns;
    size_t images_in_last_batch = number_of_images % batch_size;
    unsigned char *image_bin = new unsigned char[image_size];
    double *image_doubles = new double[image_size];

    Eigen::MatrixXd image_matrix(batch_size, image_size);
    size_t batch_filler = 0;

    // Looping through each image
    for (size_t i = 0; i < number_of_images; i++)
    {
        // Reading one image
        input_file.read(reinterpret_cast<char*>(image_bin), image_size);

        // Normalizing pixel values to [0, 1]
        std::transform(image_bin, image_bin + image_size, image_doubles,
                       [](unsigned char c) { return static_cast<double>(c) / 255.0; });

        // Converting image to Eigen row vector
        image_matrix.row(batch_filler) = Eigen::Map<Eigen::VectorXd>(image_doubles, image_size);
        batch_filler++;

        // Storing batch when full
        if (batch_filler == batch_size || i == number_of_images - 1)
        {
            size_t valid_rows = batch_filler;
            batches.emplace_back(image_matrix.topRows(valid_rows));
            batch_filler = 0;
        }
    }
    delete[] image_bin;
    delete[] image_doubles;
    input_file.close();
}

// Writing an image to an output file
inline void DataSetImages::writeImageToFile(const std::string& output_filepath, size_t index)
{
    // Calculating batch and image index
    size_t batch_no = index / batch_size;
    size_t image_index = index % batch_size;

    // Checking if image index is out of range
    if (index >= number_of_images)
    {
        std::cerr << "Error: Image index out of range" << std::endl;
        return;
    }

    // Checking if batch index is out of range
    if (batch_no >= batches.size())
    {
        std::cerr << "Error: Image Batch index out of range" << std::endl;
        return;
    }

    // Opening output file for writing
    std::ofstream output_file(output_filepath);
    // Checking if file is open
    if (!output_file.is_open())
    {
        std::cerr << "Error: Unable to open file for writing: " << output_filepath << std::endl;
        return;
    }

    // Writing metadata
    output_file << "2\n"; // Tensor rank
    output_file << number_of_rows << "\n";
    output_file << number_of_columns << "\n";

    // Writing image pixel values
    size_t image_size = number_of_rows * number_of_columns;
    for (size_t k = 0; k < image_size; k++)
    {
        output_file << batches[batch_no](image_index, k) << "\n";
    }
    output_file.close();
}

#endif // MNIST_IMAGE_READER_HPP