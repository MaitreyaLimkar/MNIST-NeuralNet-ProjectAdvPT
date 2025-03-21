#include <fstream>
#include <string>
#include "readImageMNIST.hpp"

int main(int count, char** argvect) {
    // Assigning command-line arguments to variables
    // Checking for required command-line arguments
    if (count < 4) {
        std::cerr << "Usage: " << argvect[0]
                  << " <input_filepath> <output_filepath> <index>"
                  << std::endl;
        return 1;
    }
    const std::string input_filepath = argvect[1];
    const std::string output_filepath = argvect[2];
    const size_t index = std::stoi(argvect[3]);
    // Creating readImageMNIST object with batch size
    readImageMNIST info(50);
    // Reading image data from input file
    info.readImageData(input_filepath);
    // Writing specified image to output file
    info.writeImageToFile(output_filepath, index);
    return 0;
}