//
// Created by Maitreya Limkar on 25-01-2025.
//

#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>
#include "readImageMNIST.hpp"

namespace MNISTReader {
    // Implementation of reading 32-bit big-endian integers
    uint32_t read_32bit_int(std::ifstream& file) {
        uint32_t value;
        file.read(reinterpret_cast<char*>(&value), sizeof(value));

        // Convert from big-endian to host byte order
        // This ensures correct interpretation across different systems
        return (value >> 24) |
               ((value << 8) & 0x00FF0000) |
               ((value >> 8) & 0x0000FF00) |
               (value << 24);
    }

    // Implementation of reading MNIST image
    Eigen::Tensor<double, 2> readMNISTImage(const std::string& filename, int index) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Error: Unable to open file " << filename << std::endl;
            return {0, 0};
        }

        // Read magic number and verify it's an image file
        uint32_t magic_number = read_32bit_int(file);
        if (magic_number != 0x00000803) {
            std::cerr << "Invalid MNIST image file: Incorrect magic number." << std::endl;
            return {0, 0};
        }

        // Read dataset metadata
        uint32_t num_images = read_32bit_int(file);
        uint32_t num_rows = read_32bit_int(file);
        uint32_t num_cols = read_32bit_int(file);

        // Validate requested image index
        if (index < 0 || index >= num_images) {
            std::cerr << "Invalid image index. Must be between 0 and "
                      << (num_images - 1) << std::endl;
            return {0, 0};
        }

        // Seek to the specific image in the dataset
        file.seekg(16 + index * (num_rows * num_cols));

        // Create tensor to store the image
        Eigen::Tensor<double, 2> image(num_rows, num_cols);

        // Read pixel values and normalize to [0.0, 1.0]
        for (int i = 0; i < num_rows; ++i) {
            for (int j = 0; j < num_cols; ++j) {
                uint8_t pixel_value;
                file.read(reinterpret_cast<char*>(&pixel_value), 1);

                // Linear mapping from [0, 255] to [0.0, 1.0]
                image(i, j) = static_cast<double>(pixel_value) / 255.0;
            }
        }

        return image;
    }
}