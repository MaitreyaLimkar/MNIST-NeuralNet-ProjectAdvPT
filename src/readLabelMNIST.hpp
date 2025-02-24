//
// Created by Maitreya Limkar on 05-02-2025.
//

#ifndef MNIST_LABEL_READER_HPP
#define MNIST_LABEL_READER_HPP

#include <iostream>
#include <memory>
#include <fstream>
#include <vector>
#include <Eigen/Dense>

// Class to store MNIST label data
class DatasetLabels
{
private:
    size_t batch_size_;
    size_t number_of_labels_{};
    std::vector<Eigen::MatrixXd> batches_;

public:
    explicit DatasetLabels(size_t batch_size);
    ~DatasetLabels();
    void readLabelData(const std::string &);
    void writeLabelToFile(const std::string &, const size_t &);
    Eigen::MatrixXd getBatch(const size_t &);
};

inline DatasetLabels::DatasetLabels(size_t batch_size)
{
    batch_size_ = batch_size;
}

inline DatasetLabels::~DatasetLabels() = default;

inline Eigen::MatrixXd DatasetLabels::getBatch(const size_t &index)
{
    return batches_[index];
}

// Reads file containing MNIST label data and stores it in batches
// Makes use of Eigen library to store the data in a matrix
inline void DatasetLabels::readLabelData(const std::string &input_filepath)
{
    std::ifstream input_file(input_filepath, std::ios::binary);
    if (input_file.is_open())
    {
        // Read magic number
        char bin_data[4];
        int magic_number;
        input_file.read(bin_data, 4);
        std::reverse(bin_data, bin_data + 4); // reverse magic number
        std::memcpy(&magic_number, bin_data, sizeof(int));

        // Read number of labels
        int number_of_labels = 0;
        input_file.read(bin_data, 4);
        std::reverse(bin_data, bin_data + 4); // reverse magic number
        std::memcpy(&number_of_labels, bin_data, sizeof(int));
        number_of_labels_ = number_of_labels;

        // Read label data
        int label = 0;

        // Create a Eigen::Matrix of size (batch_size,10) and then push it to batches_.
        Eigen::MatrixXd label_matrix(batch_size_, 10);
        label_matrix.setZero();
        for (size_t i = 0; i < number_of_labels_; i++)
        {
            // Read an unsigned byte from the file
            uint8_t byte;
            input_file.read(reinterpret_cast<char *>(&byte), sizeof(byte));
            // Convert byte to int
            label = static_cast<int>(byte);

            // Set corresponding position to 1 (one-hot incoding)
            label_matrix(i % batch_size_, label) = 1;

            // Push the batch to batches_ if it is full
            if ((i + 1) % batch_size_ == 0)
            {
                batches_.push_back(label_matrix);
                label_matrix.setZero();
            }
            // or else push the last batch to batches_
            else if (i == number_of_labels_ - 1)
            {
                batches_.push_back(label_matrix);
            }
        }
    }
    else
    {
        std::cout << "Unable to open file" << std::endl;
    }
}

// Writes label data to file
inline void DatasetLabels::writeLabelToFile(const std::string &output_filepath, const size_t &index)
{
    // Write in this format to be compatible with this format
    // 1
    // 10
    // 0 if label is 0
    // 1 if label is 1
    // label 0, 1 should be written for respective digit entry
    std::ofstream output_file(output_filepath, std::ios::binary);
    if (output_file.is_open())
    {

        // Write the rank of the tensor to the file
        output_file << 1 << "\n";

        // Write the shape of the tensor to the file
        output_file << 10 << "\n";

        // Write the tensor elements to the file
        size_t batch_no = index / batch_size_;
        size_t image_index = index % batch_size_;
        size_t image_size = 10;
        for (size_t i = 0; i < image_size; ++i)
        {
            output_file << batches_[batch_no](image_index, i) << "\n";
        }
        output_file.close();
    }
    else
    {
        std::cout << "Unable to open file" << std::endl;
    }
}

#endif // MNIST_LABEL_READER_HPP
