//
// Created by Maitreya Limkar on 25-01-2025.
//

#include <fstream>
#include <string>
#include "readLabelMNIST.hpp"

int main(int arg, char** argv){
    // Assigning command-line arguments to variables
    const std::string input_filepath = argv[1];
    const std::string output_filepath = argv[2];
    const size_t index = std::stoi(argv[3]);
    // Creating DatasetLabels object with batch size
    DatasetLabels labels(500); // batch size set to 5000
    // Reading label data from input file
    labels.readLabelData(input_filepath);
    // Writing specified label to output file
    labels.writeLabelToFile(output_filepath, index);
    return 0;
}