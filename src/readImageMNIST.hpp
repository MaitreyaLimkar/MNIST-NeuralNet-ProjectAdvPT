//
// Created by Maitreya Limkar on 25-01-2025.
//

#ifndef MNIST_IMAGE_READER_H
#define MNIST_IMAGE_READER_H

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>
#include <fstream>

// Namespace to avoid potential naming conflicts
namespace MNISTReader {
    // Function to read a 32-bit big-endian integer from a file
    // This helps handle the specific binary format of MNIST datasets
    uint32_t read_32bit_int(std::ifstream& file);

    // Template function for writing tensor to file
    // Works with different component types while maintaining the required output format
    template<typename ComponentType>
    void writeTensorToFile(const Eigen::Tensor<ComponentType, 2>& tensor, const std::string& filename) {
        std::ofstream outfile(filename);
        if (!outfile) {
            std::cerr << "Error: Unable to open output file " << filename << std::endl;
            return;
        }

        // Write tensor dimensions first
        outfile << tensor.dimension(0) << std::endl;
        outfile << tensor.dimension(1) << std::endl;

        // Write tensor values with space-separated format
        for (int i = 0; i < tensor.dimension(0); ++i) {
            for (int j = 0; j < tensor.dimension(1); ++j) {
                outfile << tensor(i, j);
                if (j < tensor.dimension(1) - 1) outfile << " ";
            }
            outfile << std::endl;
        }
    }

    // Function to read a specific MNIST image from the dataset
    // Returns a normalized tensor representing the image
    Eigen::Tensor<double, 2> readMNISTImage(const std::string& filename, int index);
}
#endif // MNIST_IMAGE_READER_H