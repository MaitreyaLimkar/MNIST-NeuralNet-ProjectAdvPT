#pragma once
#include <iostream>
#include <memory>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <Eigen/Dense>

class readImageMNIST {
private:
    size_t batch_size_temp, number_of_images_temp,
    number_of_rows_temp, number_of_columns_temp;
    std::vector<Eigen::MatrixXd> batches_temp;

public:
    explicit readImageMNIST(size_t batch_size);
    ~readImageMNIST();
    void readImageData(const std::string &input_filepath);
    void writeImageToFile(const std::string &output_filepath, size_t index);
    Eigen::MatrixXd getBatch(size_t index);
    size_t getNumOfBatches();
};

inline readImageMNIST::readImageMNIST(size_t batch_size)
    : batch_size_temp(batch_size), number_of_images_temp(0),
      number_of_rows_temp(0), number_of_columns_temp(0) {}

inline readImageMNIST::~readImageMNIST() {}

inline Eigen::MatrixXd readImageMNIST::getBatch(size_t index) {
    return batches_temp[index];
}

inline size_t readImageMNIST::getNumOfBatches() {
    return batches_temp.size();
}

inline void readImageMNIST::readImageData(const std::string &input_filepath) {
    std::ifstream input_file(input_filepath, std::ios::binary);
    if(!input_file.is_open()) {
        std::cerr << "Unable to open file: " << input_filepath << std::endl;
        return;
    }

    char binary_data[4];
    // Read magic number
    input_file.read(binary_data, 4);
    std::reverse(binary_data, binary_data + 4);
    int magic_number = 0;
    std::memcpy(&magic_number, binary_data, sizeof(int));

    // Read number of images
    input_file.read(binary_data, 4);
    std::reverse(binary_data, binary_data + 4);
    int number_of_images = 0;
    std::memcpy(&number_of_images, binary_data, sizeof(int));
    number_of_images_temp = number_of_images;

    // Read number of rows
    input_file.read(binary_data, 4);
    std::reverse(binary_data, binary_data + 4);
    int number_of_rows = 0;
    std::memcpy(&number_of_rows, binary_data, sizeof(int));
    number_of_rows_temp = number_of_rows;

    // Read number of columns
    input_file.read(binary_data, 4);
    std::reverse(binary_data, binary_data + 4);
    int number_of_columns = 0;
    std::memcpy(&number_of_columns, binary_data, sizeof(int));
    number_of_columns_temp = number_of_columns;

    // Calculating size of a single image (rows * columns)
    size_t image_size = number_of_rows_temp * number_of_columns_temp;
    std::vector<unsigned char> image_bin(image_size);
    std::vector<double> image_doubles(image_size);

    // Matrix for batch storage
    Eigen::MatrixXd image_matrix(batch_size_temp, image_size);
    size_t batch_filler = 0;

    for (size_t i = 0; i < number_of_images_temp; i++) {
        input_file.read(reinterpret_cast<char*>(image_bin.data()), image_size);

        // Normalize pixel values to range [0, 1]
        std::transform(image_bin.begin(), image_bin.end(), image_doubles.begin(),
                       [](unsigned char c) { return static_cast<double>(c) / 255.0; });

        // Map raw image data to Eigen matrix row
        image_matrix.row(batch_filler) = Eigen::Map<Eigen::VectorXd>(image_doubles.data(), image_size);
        batch_filler++;

        // Store batch and reset when full
        if (batch_filler == batch_size_temp || i == number_of_images_temp - 1) {
            batches_temp.push_back(image_matrix.topRows(batch_filler));
            batch_filler = 0;
        }
    }
    input_file.close();
}

inline void readImageMNIST::writeImageToFile(const std::string &output_filepath, size_t index) {
    size_t batch_no = index / batch_size_temp;
    size_t row_in_batch = index % batch_size_temp;

    // Check if the index is out of range
    if (batch_no >= batches_temp.size() || row_in_batch >= batches_temp[batch_no].rows()) {
        std::cerr << "Error: Image index " << index << " out of range." << std::endl;
        return;
    }

    // Open output file for writing
    std::ofstream output_file(output_filepath);
    if (!output_file.is_open()) {
        std::cerr << "Error: Unable to open file: " << output_filepath << " for writing." << std::endl;
        return;
    }

    // Use stringstream for optimized file writing
    std::ostringstream buffer;
    buffer << "2\n"  // Tensor rank
           << number_of_rows_temp << "\n"
           << number_of_columns_temp << "\n";

    size_t image_size = number_of_rows_temp * number_of_columns_temp;

    // Write pixel values more efficiently
    for (size_t i = 0; i < image_size; i++) {
        buffer << batches_temp.at(batch_no)(row_in_batch, i) << "\n";
    }

    // Write everything to file in one operation (faster I/O)
    output_file << buffer.str();
    output_file.close();
}