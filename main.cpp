//
// Created by Maitreya Limkar on 25-01-2025.
//

#include <iostream>
#include <cstdlib>
#include "readImageMNIST.hpp"

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <image_dataset_input> <image_tensor_output> <image_index>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string image_dataset_input = argv[1];
    std::string image_tensor_output = argv[2];
    int image_index = std::stoi(argv[3]);

    // Read the image data
    std::vector<uint8_t> image_data = readMNISTImages(image_dataset_input, image_index);

    // Convert the image to an Eigen tensor (double precision)
    Eigen::Tensor<double, 2> tensor(28, 28);
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            tensor(i, j) = static_cast<double>(image_data[i * 28 + j]) / 255.0; // Normalize [0,1]
        }
    }

    // Write tensor to output file
    writeTensorToFile(tensor, image_tensor_output);

    std::cout << "Image at index " << image_index << " successfully saved to " << image_tensor_output << std::endl;

    return EXIT_SUCCESS;
}