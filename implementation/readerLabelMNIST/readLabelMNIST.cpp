//
// Created by Maitreya Limkar on 25-01-2025.
//

#include <fstream>
#include <string>
#include "readLabelMNIST.hpp"

int main(const int argc, char const *argv[]){
    const std::string input_filepath = argv[1];
    const std::string output_filepath = argv[2];
    const size_t index = std::stoi(argv[3]);
    DatasetLabels labels(5000); //batch size set to 5000
    labels.readLabelData(input_filepath);
    labels.writeLabelToFile(output_filepath,index);
    return 0;
}