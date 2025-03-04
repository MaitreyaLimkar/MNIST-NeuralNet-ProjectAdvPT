//
// Created by Maitreya Limkar on 05-02-2025.
//

#ifndef MNIST_LABEL_READER_HPP
#define MNIST_LABEL_READER_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>

// Declaring DatasetLabels class
class DatasetLabels
{
private:
    // Declaring private member variables
    size_t batch_size_;
    size_t number_of_labels_;
    std::vector<Eigen::MatrixXd> batches_;

public:
    // Declaring constructor and destructor
    explicit DatasetLabels(size_t batch_size);
    ~DatasetLabels() = default;

    // Declaring public member functions
    void readLabelData(const std::string &);
    void writeLabelToFile(const std::string &, size_t);
    [[nodiscard]] Eigen::MatrixXd getBatch(size_t) const;
};

// Initializing constructor with batch size
inline DatasetLabels::DatasetLabels(size_t batch_size) : batch_size_(batch_size), number_of_labels_(0) {}

// Returning a batch matrix at the given index
inline Eigen::MatrixXd DatasetLabels::getBatch(size_t index) const
{
    // Checking if index is out of range
    if (index >= batches_.size())
    {
        throw std::out_of_range("Batch index out of range");
    }
    return batches_[index];
}

// Reading MNIST label data and storing it in batches
inline void DatasetLabels::readLabelData(const std::string &input_filepath)
{
    // Opening input file in binary mode
    std::ifstream input_file(input_filepath, std::ios::binary);
    // Checking if file is open
    if (!input_file.is_open())
    {
        std::cerr << "Error: Unable to open file " << input_filepath << std::endl;
        return;
    }

    // Declaring variables for reading header information
    char bin_data[4];

    // Reading and reversing magic number
    int magic_number;
    input_file.read(bin_data, 4);
    std::reverse(bin_data, bin_data + 4);
    std::memcpy(&magic_number, bin_data, sizeof(int));

    // Reading and reversing number of labels
    input_file.read(bin_data, 4);
    std::reverse(bin_data, bin_data + 4);
    std::memcpy(&number_of_labels_, bin_data, sizeof(int));

    // Initializing label matrix for one-hot encoding
    Eigen::MatrixXd label_matrix(batch_size_, 10);
    label_matrix.setZero();

    // Looping through each label
    for (size_t i = 0; i < number_of_labels_; i++)
    {
        // Reading one label
        uint8_t byte;
        input_file.read(reinterpret_cast<char *>(&byte), sizeof(byte));

        // Converting label to one-hot encoding
        const int label = static_cast<int>(byte);
        label_matrix(i % batch_size_, label) = 1;

        // Storing batch when full
        if ((i + 1) % batch_size_ == 0)
        {
            batches_.push_back(label_matrix);
            label_matrix.setZero();
        }
        // Handling last batch correctly
        else if (i == number_of_labels_ - 1)
        {
            batches_.emplace_back(label_matrix.topRows(i % batch_size_ + 1));
        }
    }

    input_file.close();
}

// Writing label data to file
inline void DatasetLabels::writeLabelToFile(const std::string &output_filepath, size_t index)
{
    // Calculating batch and image index
    size_t batch_no = index / batch_size_;
    size_t image_index = index % batch_size_;

    // Checking if batch index is out of range
    if (batch_no >= batches_.size())
    {
        std::cerr << "Error: Batch index out of range" << std::endl;
        return;
    }

    // Opening output file for writing
    std::ofstream output_file(output_filepath, std::ios::binary);
    if (!output_file.is_open())
    {
        std::cerr << "Error: Unable to open file " << output_filepath << std::endl;
        return;
    }

    // Writing metadata
    output_file << 1 << "\n"; // Tensor rank
    output_file << 10 << "\n"; // One-hot encoding size

    // Writing label data
    for (size_t i = 0; i < 10; ++i)
    {
        output_file << batches_[batch_no](image_index, i) << "\n";
    }
    output_file.close();
}

#endif // MNIST_LABEL_READER_HPP