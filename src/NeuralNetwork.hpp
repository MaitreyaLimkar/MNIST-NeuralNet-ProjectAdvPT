//
// Created by Maitreya Limkar on 15-03-2025.
//

#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <Eigen/Dense>
#include "FCLayer.hpp"
#include "ReLU.hpp"
#include "Softmax.hpp"
#include "Loss.hpp"

class NeuralNetwork
{
    public:
    NeuralNetwork(int input_size, int output_size, int hidden_size, double learning_rate)
        : learning_rate(learning_rate), fc1(input_size, hidden_size), fc2(hidden_size, output_size){}

    Eigen::VectorXd forward(const Eigen::VectorXd& input)
    {
        fc1_output = fc1.forward(input);        // fc1 linear output
        relu_output = relu(fc1_output);         // applying ReLU
        fc2_output = fc2.forward(relu_output);  // fc2 linear output
        softmax_output = softmax(fc2_output);   // Softmax activation
        log_softmax_output = softmax_output.array().log();
        return log_softmax_output;
    }

    void backward(const Eigen::VectorXd& target)
    {
        // Using softmax output (not the log-softmax)
        Eigen::VectorXd deltaOutput_1 = softmax_output - target;
        // Backprop through second FC layer
        Eigen::VectorXd deltaOutput_2 = fc2.backward(deltaOutput_1, learning_rate);
        // Backprop through ReLU: multiply by derivative of ReLU
        Eigen::VectorXd delta_ReLU = deltaOutput_2.array() * reluDerivative(fc1_output).array();
        // Backprop through first FC layer.
        fc1.backward(delta_ReLU, learning_rate);
    }

    [[nodiscard]] double computeLoss(const Eigen::VectorXd& target) const
    {
        // Use softmax output (not log-softmax) for loss computation.
        return crossEntropyLoss(softmax_output, target);
    }

    private:
    double learning_rate;
    FCLayer fc1; // Layer from input to hidden
    FCLayer fc2; // Layer from hidden to output
    Eigen::VectorXd fc1_output, relu_output, fc2_output,
                    softmax_output, log_softmax_output;
    /*std::string train_images_path;
    std::string train_labels_path;
    std::string test_images_path;
    std::string test_labels_path;
    std::string prediction_log_file_path;
    double learningRate{};
    int num_epochs{}, batch_size{},
        hidden_size{}, input_size = 784;*/
};

#endif //NEURALNETWORK_HPP
