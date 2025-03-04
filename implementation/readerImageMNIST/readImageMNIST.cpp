//
// Created by Maitreya Limkar on 25-01-2025.
//

#include <fstream>
#include <string>
#include "readImageMNIST.hpp"

int main(int argc,char* argv[]){
    // Assigning command-line arguments to variables
    const std::string input_filepath = argv[1];
    const std::string output_filepath = argv[2];
    const size_t index = std::stoi(argv[3]);
    // Creating DataSetImages object with batch size
    DataSetImages data(5000);
    // Reading image data from input file
    data.readImageData(input_filepath);
    // Writing specified image to output file
    data.writeImageToFile(output_filepath, index);
    return 0;
}