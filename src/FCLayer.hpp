//
// Created by Maitreya Limkar on 14-03-2025.
//

#ifndef FCLAYER_HPP
#define FCLAYER_HPP

#include <Eigen/Dense>
#include <cstdlib>
#include <cmath>

// Fully Connected Layer class
class FCLayer
{
    public:
    FCLayer(int input_size, int output_size):input_size(input_size), output_size(output_size)
    {
        // Calculating the Xavier factor initialization.
        double scale = std::sqrt(6.0 / (input_size + output_size));
        weights = Eigen::MatrixXd::Random(output_size,input_size) * scale;
        biases = Eigen::VectorXd::Random(output_size) * scale;
    }
    ~FCLayer() = default;

    Eigen::VectorXd forward(const Eigen::VectorXd& input)
    {
        prev_input = input;
        prev_output = weights * input + biases;
        return prev_output;
    }

    Eigen::VectorXd backward(const Eigen::VectorXd& grad_output, double learning_rate)
    {
        // Computing gradients for the weights and biases
        Eigen::MatrixXd grad_weights = grad_output * prev_input.transpose();
        Eigen::MatrixXd grad_biases = grad_output;
        // Updating parameters using SGD
        weights -= grad_weights * learning_rate;
        biases -= grad_biases * learning_rate;
        // Computing gradients to pass to previous layer
        Eigen::MatrixXd grad_input = weights.transpose() * grad_output;
        return grad_input;
    }

    // Getters for debugging
    [[nodiscard]] const Eigen::MatrixXd& get_weights() const { return weights; }
    [[nodiscard]] const Eigen::MatrixXd& get_biases() const { return biases; }

    private:
    int input_size;
    int output_size;
    Eigen::MatrixXd weights;
    Eigen::VectorXd biases;
    Eigen::VectorXd prev_input;
    Eigen::VectorXd prev_output;
};

#endif //FCLAYER_HPP
