//
// Created by Maitreya Limkar on 25-01-2025.
//

#include <iostream>
#include <string>
#include "readImageMNIST.hpp"

int main(int argc, char* argv[]) {
    // Validate command-line arguments
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <image_dataset_input> <image_tensor_output> <image_index>" << std::endl;
        return 1;
    }

    // Parse command-line arguments
    std::string input_path = argv[1];
    std::string output_path = argv[2];
    int image_index = std::stoi(argv[3]);

    // Read the specified MNIST image using the namespace function
    Eigen::Tensor<double, 2> mnist_image =
        MNISTReader::readMNISTImage(input_path, image_index);

    // Check if image was successfully read
    if (mnist_image.size() > 0) {
        // Write the tensor to output file using namespace function
        MNISTReader::writeTensorToFile(mnist_image, output_path);
        std::cout << "Successfully processed MNIST image at index "
                  << image_index << std::endl;
    } else {
        std::cerr << "Failed to read MNIST image." << std::endl;
        return 1;
    }

    return 0;
}