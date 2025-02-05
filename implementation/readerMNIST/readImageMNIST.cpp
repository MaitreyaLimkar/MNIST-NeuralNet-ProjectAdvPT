//
// Created by Maitreya Limkar on 25-01-2025.
//

#include <iostream>
#include <fstream>
#include <string>
#include "readImageMNIST.hpp"

std::vector<uint8_t> readMNISTImages(const std::string& filepath, int index) {
    std::ifstream file(filepath, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filepath << std::endl;
        exit(EXIT_FAILURE);
    }

    // Read header (magic number, num_images, rows, cols)
    int32_t magic_number, num_images, rows, cols;
    file.read(reinterpret_cast<char*>(&magic_number), 4);
    file.read(reinterpret_cast<char*>(&num_images), 4);
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);

    // Convert from big-endian to little-endian
    magic_number = __builtin_bswap32(magic_number);
    num_images = __builtin_bswap32(num_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    if (magic_number != 2051) {
        std::cerr << "Error: Invalid MNIST image file format!" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (index >= num_images || index < 0) {
        std::cerr << "Error: Index out of range!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Seek to the image position in the file
    file.seekg(16 + index * 28 * 28, std::ios::beg);

    // Read the image pixel data
    std::vector<uint8_t> image_data(28 * 28);
    file.read(reinterpret_cast<char*>(image_data.data()), 28 * 28);

    return image_data;
}

template <typename ComponentType>
void writeTensorToFile(const Eigen::Tensor<ComponentType, 2>& tensor, const std::string& filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open output file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    // Write header (data type, rows, cols)
    file << "2\n";  // Assuming "2" denotes double precision
    file << tensor.dimension(0) << "\n";
    file << tensor.dimension(1) << "\n";

    // Write pixel values, each on a new line
    for (int i = 0; i < tensor.dimension(0); ++i) {
        for (int j = 0; j < tensor.dimension(1); ++j) {
            file << tensor(i, j) << "\n";
        }
    }

    file.close();
}

// Explicit template instantiation to avoid linker issues
template void writeTensorToFile<double>(const Eigen::Tensor<double, 2>& tensor, const std::string& filename);