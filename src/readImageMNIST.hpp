#pragma once
#include <iostream>
#include <memory>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <Eigen/Dense>

class DataSetImages {
private:
    size_t batch_size_temp;
    size_t number_of_images_temp;
    size_t number_of_rows_temp;
    size_t number_of_columns_temp;
    std::vector<Eigen::MatrixXd> batches_temp;

public:
    explicit DataSetImages(size_t batch_size);
    ~DataSetImages();
    void readImageData(const std::string &filepath);
    void writeImageToFile(const std::string &filepath, size_t index);
    Eigen::MatrixXd getBatch(size_t index);
    size_t getNoOfBatches();
};

DataSetImages::DataSetImages(size_t batch_size)
    : batch_size_temp(batch_size), number_of_images_temp(0),
      number_of_rows_temp(0), number_of_columns_temp(0) {}

DataSetImages::~DataSetImages() {}

Eigen::MatrixXd DataSetImages::getBatch(size_t index) {
    return batches_temp[index];
}

size_t DataSetImages::getNoOfBatches() {
    return batches_temp.size();
}

void DataSetImages::readImageData(const std::string &input_filepath) {
    std::ifstream input_file(input_filepath, std::ios::binary);
    if(!input_file.is_open()) {
        std::cerr << "Unable to open file: " << input_filepath << std::endl;
        return;
    }

    char bin_data[4];
    // Read magic number
    input_file.read(bin_data, 4);
    std::reverse(bin_data, bin_data + 4);
    int magic_number = 0;
    std::memcpy(&magic_number, bin_data, sizeof(int));

    // Read number of images
    input_file.read(bin_data, 4);
    std::reverse(bin_data, bin_data + 4);
    int number_of_images = 0;
    std::memcpy(&number_of_images, bin_data, sizeof(int));
    number_of_images_temp = number_of_images;

    // Read number of rows
    input_file.read(bin_data, 4);
    std::reverse(bin_data, bin_data + 4);
    int number_of_rows = 0;
    std::memcpy(&number_of_rows, bin_data, sizeof(int));
    number_of_rows_temp = number_of_rows;

    // Read number of columns
    input_file.read(bin_data, 4);
    std::reverse(bin_data, bin_data + 4);
    int number_of_columns = 0;
    std::memcpy(&number_of_columns, bin_data, sizeof(int));
    number_of_columns_temp = number_of_columns;

    size_t image_size = number_of_rows_temp * number_of_columns_temp;
    size_t images_in_last_batch = number_of_images_temp % batch_size_temp;

    unsigned char *image_bin = new unsigned char[image_size];
    double *image_doubles = new double[image_size];

    Eigen::MatrixXd image_matrix(batch_size_temp, image_size);
    size_t batch_filler = 0;

    for(size_t i = 0; i < number_of_images_temp; i++)
    {
        input_file.read(reinterpret_cast<char*>(image_bin), image_size);
        std::transform(image_bin, image_bin + image_size, image_doubles,
                       [](unsigned char c){ return static_cast<double>(c) / 255.0; });
        image_matrix.row(batch_filler) = Eigen::Map<Eigen::VectorXd>(image_doubles, image_size);
        batch_filler++;

        if(batch_filler == batch_size_temp || (i == number_of_images_temp - 1)) {
            size_t valid_rows = batch_filler;
            batches_temp.push_back(image_matrix.topRows(valid_rows));
            batch_filler = 0;
        }
    }
    delete[] image_bin;
    delete[] image_doubles;
    input_file.close();
}

void DataSetImages::writeImageToFile(const std::string &output_filepath, size_t index) {
    size_t batch_no = index / batch_size_temp;
    size_t row_in_batch = index % batch_size_temp;

    if(batch_no >= batches_temp.size() || row_in_batch >= batches_temp[batch_no].rows()) {
        std::cerr << "Index out of range." << std::endl;
        return;
    }

    std::ofstream output_file(output_filepath);
    if(!output_file.is_open()) {
        std::cerr << "Error: Unable to open file for writing: " << output_filepath << std::endl;
        return;
    }

    output_file << 2 << "\n"; // rank
    output_file << number_of_rows_temp << "\n";
    output_file << number_of_columns_temp << "\n";

    size_t image_size = number_of_rows_temp * number_of_columns_temp;
    for(size_t i = 0; i < image_size; i++) {
        output_file << batches_temp[batch_no](row_in_batch, i) << "\n";
    }
    output_file.close();
}
