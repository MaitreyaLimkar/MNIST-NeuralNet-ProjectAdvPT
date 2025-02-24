//
// Created by Maitreya Limkar on 25-01-2025.
//

#ifndef MNIST_IMAGE_READER_HPP
#define MNIST_IMAGE_READER_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>

class DataSetImages
{
private:
    size_t batch_size;
    size_t number_of_images{};
    size_t number_of_rows{};
    size_t number_of_columns{};
    std::vector<Eigen::MatrixXd> batches;

public:
    explicit DataSetImages(size_t batch_size);
    ~DataSetImages() = default;

    void readImageData(const std::string&);
    void writeImageToFile(const std::string&, size_t);
    [[nodiscard]] Eigen::MatrixXd getBatch(size_t) const;
    [[nodiscard]] size_t getNoOfBatches() const;
};

// Constructor: Initializes batch size
inline DataSetImages::DataSetImages(size_t batch_size) : batch_size(batch_size) {}

// Returns a batch matrix at the given index
inline Eigen::MatrixXd DataSetImages::getBatch(size_t index) const
{
    if (index >= batches.size())
    {
        throw std::out_of_range("Batch index out of range");
    }
    return batches[index];
}

// Returns the total number of batches
inline size_t DataSetImages::getNoOfBatches() const
{
    return batches.size();
}

// Reads MNIST image data and stores it in batches
inline void DataSetImages::readImageData(const std::string& input_filepath)
{
    std::ifstream input_file(input_filepath, std::ios::binary);
    if (!input_file.is_open())
    {
        std::cerr << "Error: Unable to open file: " << input_filepath << std::endl;
        return;
    }

    // Read MNIST header information
    char bin_data[4];
    int temp_value;

    // Read magic number (not needed but useful for validation)
    input_file.read(bin_data, 4);
    std::reverse(bin_data, bin_data + 4);
    std::memcpy(&temp_value, bin_data, sizeof(int));

    // Read number of images
    input_file.read(bin_data, 4);
    std::reverse(bin_data, bin_data + 4);
    std::memcpy(&number_of_images, bin_data, sizeof(int));

    // Read number of rows
    input_file.read(bin_data, 4);
    std::reverse(bin_data, bin_data + 4);
    std::memcpy(&number_of_rows, bin_data, sizeof(int));

    // Read number of columns
    input_file.read(bin_data, 4);
    std::reverse(bin_data, bin_data + 4);
    std::memcpy(&number_of_columns, bin_data, sizeof(int));

    // Compute image size and batch-related values
    size_t image_size = number_of_rows * number_of_columns;
    size_t images_in_last_batch = number_of_images % batch_size;
    Eigen::MatrixXd image_matrix(batch_size, image_size);

    std::vector<unsigned char> image_bin(image_size);
    std::vector<double> image(image_size);

    for (size_t i = 0; i < number_of_images; ++i)
    {
        // Read one image
        input_file.read(reinterpret_cast<char*>(image_bin.data()), image_size);

        // Normalize pixel values to [0, 1]
        std::transform(image_bin.begin(), image_bin.end(), image.begin(),
                       [](unsigned char c) { return static_cast<double>(c) / 255.0; });

        // Convert image to Eigen row vector
        image_matrix.row(i % batch_size) = Eigen::Map<Eigen::VectorXd>(image.data(), image_size);

        // Store batch when full
        if ((i + 1) % batch_size == 0 || i == number_of_images - 1)
        {
            size_t batch_rows = (i == number_of_images - 1) ? images_in_last_batch : batch_size;
            batches.emplace_back(image_matrix.topRows(batch_rows));
        }
    }

    input_file.close();
}

// Writes an image to an output file
inline void DataSetImages::writeImageToFile(const std::string& output_filepath, size_t index)
{
    if (index >= number_of_images)
    {
        std::cerr << "Error: Image index out of range" << std::endl;
        return;
    }

    size_t batch_no = index / batch_size;
    size_t image_index = index % batch_size;

    if (batch_no >= batches.size())
    {
        std::cerr << "Error: Batch index out of range" << std::endl;
        return;
    }

    std::ofstream output_file(output_filepath);
    if (!output_file.is_open())
    {
        std::cerr << "Error: Unable to open file for writing: " << output_filepath << std::endl;
        return;
    }

    // Write metadata
    output_file << "2\n"; // Tensor rank
    output_file << number_of_rows << "\n";
    output_file << number_of_columns << "\n";

    // Write image pixel values
    size_t image_size = number_of_rows * number_of_columns;
    for (size_t i = 0; i < image_size; ++i)
    {
        output_file << batches[batch_no](image_index, i) << "\n";
    }

    output_file.close();
}

#endif // MNIST_IMAGE_READER_HPP