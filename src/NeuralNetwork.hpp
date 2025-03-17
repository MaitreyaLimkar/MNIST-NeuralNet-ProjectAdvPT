//
// Created by Maitreya Limkar on 15-03-2025.
//

#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <iostream>
#include <utility>
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
                                    std::string prediction_log_file_path)
    : learning_rate(learning_rate),
      num_epochs(num_epochs),
      batch_size(batch_size),
      hidden_size(hidden_size),
      fc1(input_size, hidden_size),
      fc2(hidden_size, 10),
      sgd(learning_rate),
      train_images_path(std::move(train_images_path)),
      train_labels_path(std::move(train_labels_path)),
      test_images_path(std::move(test_images_path)),
      test_labels_path(std::move(test_labels_path)),
      prediction_log_file_path(std::move(prediction_log_file_path)){}

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
    Eigen::MatrixXd softmax_forward = Softmax::forward(fc2_forward);
    return softmax_forward;
}

inline Eigen::MatrixXd NeuralNetwork::backward(const Eigen::MatrixXd &error)
{
    // Backward pass through softmax.
    Eigen::MatrixXd softmax_backward = Softmax::backward(error);
    // Backward pass through second FC layer, using SGD optimizer.
    Eigen::MatrixXd fc2_backward = fc2.backward(softmax_backward, sgd);
    // Backward pass through ReLU.
    Eigen::MatrixXd relu_backward = relu.backward(fc2_backward);
    // Backward pass through first FC layer.
    Eigen::MatrixXd fc1_backward = fc1.backward(relu_backward, sgd);
    return fc1_backward;
}

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
        for (size_t batch = 0; batch < train_images.getNoOfBatches(); batch++)
        {
            // Forward pass on current batch.
            Eigen::MatrixXd predicted_output = forward(train_images.getBatch(batch));
            // Compute loss using cross-entropy.
            //double loss = celoss.forward(predicted_output, train_labels.getBatch(batch));
            // Backpropagate: compute gradient of loss and update parameters.
            Eigen::MatrixXd loss_backward = ce_loss.backward(train_labels.getBatch(batch));
            backward(loss_backward);
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
    for (size_t j = 0; j < test_images.getNoOfBatches(); j++)
    {
        predictionLogFile << "Current batch: " << j << std::endl;
        // For each image in the batch.
        Eigen::MatrixXd batchOutput = forward(test_images.getBatch(j));
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
    predictionLogFile.close();
}
#endif //NEURALNETWORK_HPP
