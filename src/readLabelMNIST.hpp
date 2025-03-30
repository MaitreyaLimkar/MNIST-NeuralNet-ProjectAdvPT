#pragma once
#include <iostream>
#include <memory>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <Eigen/Dense>

class readLabelMNIST {
private:
    size_t batch_size_temp;
    size_t number_of_labels_temp;
    std::vector<Eigen::MatrixXd> batches_temp;

public:
    explicit readLabelMNIST(size_t batch_size);
    ~readLabelMNIST();
    void readLabelData(const std::string &input_filepath);
    void writeLabelToFile(const std::string &output_filepath, size_t index);
    Eigen::MatrixXd getBatch(size_t index);
    size_t getNumBatches() const { return batches_temp.size(); }
};

inline readLabelMNIST::readLabelMNIST(size_t batch_size)
    : batch_size_temp(batch_size), number_of_labels_temp(0) {}

inline readLabelMNIST::~readLabelMNIST() {}

inline Eigen::MatrixXd readLabelMNIST::getBatch(size_t index) {
    return batches_temp[index];
}

inline void readLabelMNIST::readLabelData(const std::string &input_filepath) {
    std::ifstream input_file(input_filepath, std::ios::binary);
    if (!input_file.is_open()) {
        std::cerr << "Unable to open file " << input_filepath << std::endl;
        return;
    }

    char binary_data[4];
    // Read and reverse magic number
    input_file.read(binary_data, 4);
    std::reverse(std::begin(binary_data), std::end(binary_data));
    int magic_number = 0;
    std::memcpy(&magic_number, binary_data, sizeof(int));

    // Read and reverse number of labels
    input_file.read(binary_data, 4);
    std::reverse(std::begin(binary_data), std::end(binary_data));
    std::memcpy(&number_of_labels_temp, binary_data, sizeof(int));

    constexpr int num_classes = 10;
    Eigen::MatrixXd label_matrix(batch_size_temp, num_classes);
    label_matrix.setZero();
    size_t batch_filler = 0;

    for (size_t i = 0; i < number_of_labels_temp; ++i) {
        uint8_t label_byte = 0;
        input_file.read(reinterpret_cast<char*>(&label_byte), 1);
        int label = static_cast<int>(label_byte);

        if (label >= 0 && label < num_classes) {
            label_matrix(batch_filler, label) = 1.0;
        } else {
            std::cerr << "Warning: Invalid label " << label << " at index " << i << std::endl;
        }

        batch_filler++;
        // Push full or final batch
        if (batch_filler == batch_size_temp || i == number_of_labels_temp - 1) {
            batches_temp.emplace_back(label_matrix.topRows(batch_filler));
            label_matrix.setZero();
            batch_filler = 0;
        }
    }
    input_file.close();
}

inline void readLabelMNIST::writeLabelToFile(const std::string &output_filepath, size_t index) {
    size_t batch_no = index / batch_size_temp;
    size_t row_in_batch = index % batch_size_temp;

    // Error handling for out-of-range indexes
    if (batch_no >= batches_temp.size() || row_in_batch >= batches_temp[batch_no].rows()) {
        std::cerr << "Error: Label index " << index << " is out of range." << std::endl;
        return;
    }

    // Open output file for writing
    std::ofstream output_file(output_filepath);
    if (!output_file.is_open()) {
        std::cerr << "Error: Unable to open file for writing: " << output_filepath << std::endl;
        return;
    }

    // Use stringstream for efficient writing
    std::ostringstream buffer;
    buffer << "1\n" // Tensor rank
           << "10\n"; // One-hot encoding size

    // Write label data
    for (int i = 0; i < 10; ++i) {
        buffer << batches_temp.at(batch_no)(row_in_batch, i) << "\n";
    }

    // Write buffer content to file in one operation (faster I/O)
    output_file << buffer.str();
    output_file.close();
}