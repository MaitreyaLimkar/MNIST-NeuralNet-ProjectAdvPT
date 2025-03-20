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
    size_t batch_size_;
    size_t number_of_images_;
    size_t number_of_rows_;
    size_t number_of_columns_;
    std::vector<Eigen::MatrixXd> batches_;

public:
    explicit DataSetImages(size_t batch_size);
    ~DataSetImages();

    void readImageData(const std::string &filepath);
    void writeImageToFile(const std::string &filepath, size_t index);
    Eigen::MatrixXd getBatch(size_t index);
    size_t getNoOfBatches();
};

DataSetImages::DataSetImages(size_t batch_size)
    : batch_size_(batch_size), number_of_images_(0), number_of_rows_(0), number_of_columns_(0)
{}

DataSetImages::~DataSetImages() {}

Eigen::MatrixXd DataSetImages::getBatch(size_t index)
{
    return batches_[index];
}

size_t DataSetImages::getNoOfBatches()
{
    return batches_.size();
}

void DataSetImages::readImageData(const std::string &input_filepath)
{
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
    number_of_images_ = number_of_images;

    // Read number of rows
    input_file.read(bin_data, 4);
    std::reverse(bin_data, bin_data + 4);
    int number_of_rows = 0;
    std::memcpy(&number_of_rows, bin_data, sizeof(int));
    number_of_rows_ = number_of_rows;

    // Read number of columns
    input_file.read(bin_data, 4);
    std::reverse(bin_data, bin_data + 4);
    int number_of_columns = 0;
    std::memcpy(&number_of_columns, bin_data, sizeof(int));
    number_of_columns_ = number_of_columns;

    size_t image_size = number_of_rows_ * number_of_columns_;
    size_t images_in_last_batch = number_of_images_ % batch_size_;

    unsigned char *image_bin = new unsigned char[image_size];
    double *image_doubles = new double[image_size];

    Eigen::MatrixXd image_matrix(batch_size_, image_size);
    size_t batchFillCount = 0;

    for(size_t i = 0; i < number_of_images_; i++)
    {
        input_file.read(reinterpret_cast<char*>(image_bin), image_size);
        std::transform(image_bin, image_bin + image_size, image_doubles,
                       [](unsigned char c){ return static_cast<double>(c) / 255.0; });
        image_matrix.row(batchFillCount) = Eigen::Map<Eigen::VectorXd>(image_doubles, image_size);
        batchFillCount++;

        if(batchFillCount == batch_size_ || (i == number_of_images_ - 1))
        {
            size_t validRows = batchFillCount;
            batches_.push_back(image_matrix.topRows(validRows));
            batchFillCount = 0;
        }
    }

    delete[] image_bin;
    delete[] image_doubles;
    input_file.close();
}

void DataSetImages::writeImageToFile(const std::string &output_filepath, size_t index)
{
    size_t batch_no = index / batch_size_;
    size_t row_in_batch = index % batch_size_;

    if(batch_no >= batches_.size() || row_in_batch >= batches_[batch_no].rows()) {
        std::cerr << "Index out of range." << std::endl;
        return;
    }

    std::ofstream output_file(output_filepath);
    if(!output_file.is_open()) {
        std::cerr << "Error: Unable to open file for writing: " << output_filepath << std::endl;
        return;
    }

    output_file << 2 << "\n"; // rank
    output_file << number_of_rows_ << "\n";
    output_file << number_of_columns_ << "\n";

    size_t image_size = number_of_rows_ * number_of_columns_;
    for(size_t i = 0; i < image_size; i++) {
        output_file << batches_[batch_no](row_in_batch, i) << "\n";
    }
    output_file.close();
}
