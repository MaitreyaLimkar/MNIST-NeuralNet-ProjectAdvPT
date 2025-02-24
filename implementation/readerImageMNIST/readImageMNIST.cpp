//
// Created by Maitreya Limkar on 25-01-2025.
//

#include <fstream>
#include <string>
#include "readImageMNIST.hpp"

int main(int argc,char* argv[]){
    const std::string input_filepath = argv[1];
    const std::string output_filepath = argv[2];
    const size_t index = std::stoi(argv[3]);
    DataSetImages data(5000);
    data.readImageData(input_filepath);
    data.writeImageToFile(output_filepath,index);
    return 0;
}