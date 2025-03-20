#pragma once
#include <iostream>
#include <memory>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <Eigen/Dense>

class DatasetLabels {
private:
    size_t batch_size_temp;
    size_t number_of_labels_temp;
    std::vector<Eigen::MatrixXd> batches_temp;

public:
    explicit DatasetLabels(size_t batch_size);
    ~DatasetLabels();
    void readLabelData(const std::string &filepath);
    void writeLabelToFile(const std::string &filepath, size_t index);
    Eigen::MatrixXd getBatch(size_t index);
    size_t getNumBatches() const { return batches_temp.size(); }
};

DatasetLabels::DatasetLabels(size_t batch_size)
    : batch_size_temp(batch_size), number_of_labels_temp(0) {}

DatasetLabels::~DatasetLabels() {}

Eigen::MatrixXd DatasetLabels::getBatch(size_t index) {
    return batches_temp[index];
}

void DatasetLabels::readLabelData(const std::string &input_filepath) {
    std::ifstream input_file(input_filepath, std::ios::binary);
    if (!input_file.is_open()) {
        std::cerr << "Unable to open file " << input_filepath << std::endl;
        return;
    }

    char bin_data[4];
    // Magic number
    input_file.read(bin_data, 4);
    std::reverse(bin_data, bin_data + 4);
    int magic_number = 0;
    std::memcpy(&magic_number, bin_data, sizeof(int));

    // Number of labels
    input_file.read(bin_data, 4);
    std::reverse(bin_data, bin_data + 4);
    int number_of_labels = 0;
    std::memcpy(&number_of_labels, bin_data, sizeof(int));
    number_of_labels_temp = number_of_labels;

    // Build batches: each batch is [batch_size_temp x 10]
    Eigen::MatrixXd label_matrix(batch_size_temp, 10);
    label_matrix.setZero();
    size_t batch_filler = 0;

    for(size_t i = 0; i < number_of_labels_temp; i++) {
        uint8_t byte = 0;
        input_file.read(reinterpret_cast<char *>(&byte), 1);
        int label = static_cast<int>(byte);
        label_matrix(batch_filler, label) = 1.0;
        batch_filler++;

        if(batch_filler == batch_size_temp || (i == number_of_labels_temp - 1)) {
            size_t validRows = batch_filler;
            batches_temp.push_back(label_matrix.topRows(validRows));
            label_matrix.setZero();
            batch_filler = 0;
        }
    }
    input_file.close();
}

void DatasetLabels::writeLabelToFile(const std::string &output_filepath, size_t index) {
    size_t batch_no = index / batch_size_temp;
    size_t row_in_batch = index % batch_size_temp;

    if (batch_no >= batches_temp.size() || row_in_batch >= batches_temp[batch_no].rows()) {
        std::cerr << "Index out of range." << std::endl;
        return;
    }

    std::ofstream output_file(output_filepath);
    if(!output_file.is_open()) {
        std::cerr << "Unable to open file " << output_filepath << std::endl;
        return;
    }
    output_file << 1 << "\n";
    output_file << 10 << "\n";

    for(int i = 0; i < 10; i++) {
        output_file << batches_temp[batch_no](row_in_batch, i) << "\n";
    }
    output_file.close();
}
