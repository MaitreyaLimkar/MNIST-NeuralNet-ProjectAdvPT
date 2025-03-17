//
// Created by Maitreya Limkar on 14-03-2025.
//

#ifndef FCLAYER_HPP
#define FCLAYER_HPP

#include <Eigen/Dense>
#include <cmath>
#include "SGD.hpp"

// Fully Connected Layer class
class FCLayer
{
    public:
    FCLayer(size_t input_size, size_t output_size):input_size(input_size), output_size(output_size)
    {
        // Calculating the Xavier factor initialization.
        double limit = std::sqrt(6.0 / (input_size + output_size));
        weights = Eigen::MatrixXd::Random(input_size, output_size) * limit;
        biases = Eigen::VectorXd::Zero(output_size);
    }

    void weight_setter(const Eigen::MatrixXd &set_weights)
    {
        weights = set_weights;
    }
    void bias_setter(const Eigen::VectorXd &set_biases)
    {
        biases = set_biases;
    }

    Eigen::VectorXd forward(const Eigen::VectorXd& input)
    {
        prev_input = input;  // Save input for use in backpropagation.
        prev_output = weights * input + biases;
        return prev_output;
    }

    Eigen::VectorXd backward(const Eigen::VectorXd& error, const SGD &sgd)
    {
        Eigen::MatrixXd grad_weights = error * prev_input.transpose();
        const Eigen::VectorXd& grad_biases = error;
        weights = sgd.update_weights(weights, grad_weights);
        biases = sgd.update_biases(biases, grad_biases);

        // Propagate error to previous layer: dL/d(input) = weights^T * error.
        return weights.transpose() * error;
    }
    ~FCLayer() = default;

    private:
    size_t input_size, output_size;
    Eigen::MatrixXd weights;
    Eigen::VectorXd biases, prev_input, prev_output;
};

#endif //FCLAYER_HPP
