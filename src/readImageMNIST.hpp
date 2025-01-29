//
// Created by Maitreya Limkar on 25-01-2025.
//

#ifndef MNIST_IMAGE_READER_H
#define MNIST_IMAGE_READER_H

#include <vector>
#include <unsupported/Eigen/CXX11/Tensor>

// Reads a single image from an MNIST dataset file.
std::vector<uint8_t> readMNISTImages(const std::string& filepath, int index);

// Template function to write an Eigen tensor to a file.
template <typename ComponentType>
void writeTensorToFile(const Eigen::Tensor<ComponentType, 2>& tensor, const std::string& filename);

#endif // MNIST_IMAGE_READER_H