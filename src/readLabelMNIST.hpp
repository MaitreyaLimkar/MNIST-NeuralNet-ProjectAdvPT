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
    size_t batch_size;
    size_t number_of_labels;
    std::vector<Eigen::MatrixXd> batches;

public:
    // Declaring constructor and destructor
    explicit DatasetLabels(size_t batch_size);
    ~DatasetLabels() = default;

    // Declaring public member functions
    void readLabelData(const std::string &);
    void writeLabelToFile(const std::string &, size_t);
    Eigen::MatrixXd getBatch(size_t) const;
    size_t getNumBatches() const { return batches.size(); }
};

// Initializing constructor with batch size
inline DatasetLabels::DatasetLabels(size_t batch_size) :
batch_size(batch_size), number_of_labels(0) {}

// Returning a batch matrix at the given index
inline Eigen::MatrixXd DatasetLabels::getBatch(size_t index) const
{
    // Checking if index is out of range
    if (index >= batches.size())
    {
        throw std::out_of_range("Batch index out of range");
    }
    return batches[index];
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
    int number_of_labels = 0;
    input_file.read(bin_data, 4);
    std::reverse(bin_data, bin_data + 4);
    std::memcpy(&number_of_labels, bin_data, sizeof(int));

    // Initializing label matrix for one-hot encoding
    Eigen::MatrixXd label_matrix(batch_size, 10);
    label_matrix.setZero();
    size_t batch_filler = 0;
    // Looping through each label
    for (size_t i = 0; i < number_of_labels; i++)
    {
        // Reading one label
        uint8_t byte = 0;
        input_file.read(reinterpret_cast<char *>(&byte), 1);

        // Converting label to one-hot encoding
        const int label = byte;
        label_matrix(batch_filler, label) = 1.0;
        batch_filler++;
        // Storing batch when full
        if (batch_filler == batch_size || i == number_of_labels - 1)
        {
            size_t valid_rows = batch_filler;
            batches.emplace_back(label_matrix.topRows(valid_rows));
            label_matrix.setZero();
            batch_filler = 0;
        }
    }
    input_file.close();
}

// Writing label data to file
inline void DatasetLabels::writeLabelToFile(const std::string &output_filepath, size_t index)
{
    // Calculating batch and image index
    size_t batch_no = index / batch_size;
    size_t image_index = index % batch_size;

    // Checking if batch index is out of range
    if (batch_no >= batches.size())
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
    for (int j = 0; j < 10; j++)
    {
        output_file << batches[batch_no](image_index, j) << "\n";
    }
    output_file.close();
}

#endif // MNIST_LABEL_READER_HPP