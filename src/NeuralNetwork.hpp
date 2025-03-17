//
// Created by Maitreya Limkar on 15-03-2025.
//

#pragma once

#include <iostream>
#include <utility>
#include <omp.h>  // Added OpenMP header
#include "FCLayer.hpp"
#include "ReLU.hpp"
#include "Softmax.hpp"
#include "Loss.hpp"
#include "SGD.hpp"
#include "readImageMNIST.hpp"
#include "readLabelMNIST.hpp"

class NeuralNetwork
{
    public:
    NeuralNetwork(double, int, int, int, std::string, std::string, std::string, std::string, std::string);
    ~NeuralNetwork();

    // Performs a forward pass on the input tensor.
    Eigen::MatrixXd forward(const Eigen::MatrixXd &);

    // Performs a backward pass with the given error tensor.
    Eigen::MatrixXd backward(const Eigen::MatrixXd &);

    // Trains the network using the training data and labels.
    void train();

    // Tests the network on the test data and writes predictions to a log file.
    void test();

private:
    double learning_rate;
    int num_epochs, batch_size,
        hidden_size, input_size = 784;
    FCLayer fc1; // Layer from input to hidden
    FCLayer fc2; // Layer from hidden to output
    ReLU relu;
    Softmax softmax;
    CrossEntropyLoss ce_loss;
    SGD sgd;
    std::string train_images_path, train_labels_path,
    test_images_path, test_labels_path, prediction_log_file_path;
};

inline NeuralNetwork::NeuralNetwork(double learning_rate,
                                    int num_epochs,
                                    int batch_size,
                                    int hidden_size,
                                    std::string train_images_path,
                                    std::string train_labels_path,
                                    std::string test_images_path,
                                    std::string test_labels_path,
                                    std::string prediction_log_file_path): fc1(input_size, hidden_size), fc2(hidden_size, 10)
{
    this->learning_rate = learning_rate;
    this->num_epochs = num_epochs;
    this->batch_size = batch_size;
    this->hidden_size = hidden_size;
    this->train_images_path = std::move(train_images_path);
    this->train_labels_path = std::move(train_labels_path);
    this->test_images_path = std::move(test_images_path);
    this->test_labels_path = std::move(test_labels_path);
    this->prediction_log_file_path = std::move(prediction_log_file_path);
    sgd = SGD(learning_rate);
    fc1 = FCLayer(input_size, hidden_size);
    fc2 = FCLayer(hidden_size, 10);
}

inline NeuralNetwork::~NeuralNetwork() = default;

inline Eigen::MatrixXd NeuralNetwork::forward(const Eigen::MatrixXd &input)
{
    // First layer: Fully Connected (input -> hidden)
    Eigen::MatrixXd fc1_forward = fc1.forward(input);
    // Apply ReLU activation.
    Eigen::MatrixXd relu_forward = relu.forward(fc1_forward);
    // Second layer: Fully Connected (hidden -> output)
    Eigen::MatrixXd fc2_forward = fc2.forward(relu_forward);
    // Apply softmax to get probabilities.
    Eigen::MatrixXd softmax_forward = softmax.forward(fc2_forward);
    return softmax_forward;
}

inline Eigen::MatrixXd NeuralNetwork::backward(const Eigen::MatrixXd &error)
{
    // Backward pass through softmax.
    Eigen::MatrixXd softmax_backward = softmax.backward(error);
    // Backward pass through second FC layer, using SGD optimizer.
    Eigen::MatrixXd fc2_backward = fc2.backward(softmax_backward, sgd);
    // Backward pass through ReLU.
    Eigen::MatrixXd relu_backward = relu.backward(fc2_backward);
    // Backward pass through first FC layer.
    Eigen::MatrixXd fc1_backward = fc1.backward(relu_backward, sgd);
    return fc1_backward;
}

// Parallelizing batch processing using 4 threads
inline void NeuralNetwork::train()
{
    // Load training images and labels.
    DataSetImages train_images(batch_size);
    train_images.readImageData(train_images_path);
    DatasetLabels train_labels(batch_size);
    train_labels.readLabelData(train_labels_path);
    // For each epoch.
    for (int epoch = 0; epoch < num_epochs; epoch++)
    {
        // For each batch in training data.
        #pragma omp parallel for num_threads(4)
        for (int batch = 0; batch < static_cast<int>(train_images.getNoOfBatches()); batch++)
        {
            // Each thread processes different batches in parallel
            Eigen::MatrixXd predicted_output = forward(train_images.getBatch(batch));
            double loss = ce_loss.forward(predicted_output, train_labels.getBatch(batch));
            Eigen::MatrixXd loss_backward = ce_loss.backward(train_labels.getBatch(batch));

            // Critical section to avoid race conditions when updating model parameters
            #pragma omp critical
            {
                backward(loss_backward);
            }
        }
    }
}

inline void NeuralNetwork::test()
{
    // Load testing images and labels.
    DataSetImages test_images(batch_size);
    test_images.readImageData(test_images_path);
    DatasetLabels test_labels(batch_size);
    test_labels.readLabelData(test_labels_path);
    std::ofstream predictionLogFile(prediction_log_file_path);
    if (!predictionLogFile.is_open())
    {
        std::cerr << "Error: Cannot open prediction log file: " << prediction_log_file_path << std::endl;
        return;
    }

    // For each batch in the test data.
    #pragma omp parallel for num_threads(4) ordered
    for (int j = 0; j < static_cast<int>(test_images.getNoOfBatches()); j++)
    {
        Eigen::MatrixXd batchOutput = forward(test_images.getBatch(j));

        // Use ordered directive to maintain the same output order as the original
        #pragma omp ordered
        {
            predictionLogFile << "Current batch: " << j << std::endl;
            // For each image in the batch.
            for (int i = 0; i < batchOutput.rows(); i++)
            {
                Eigen::Index predLabel;
                batchOutput.row(i).maxCoeff(&predLabel);
                Eigen::Index actualLabel;
                test_labels.getBatch(j).row(i).maxCoeff(&actualLabel);
                predictionLogFile << " - image " << j * batch_size + i
                                << ": Prediction=" << predLabel
                                << ". Label=" << actualLabel << std::endl;
            }
        }
    }
    predictionLogFile.close();
}