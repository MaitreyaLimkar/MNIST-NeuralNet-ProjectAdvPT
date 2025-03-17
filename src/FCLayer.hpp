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
        weights = Eigen::MatrixXd::Random(input_size + 1, output_size) * limit; // 1 row added for biases

    }

    void weight_setter(const Eigen::MatrixXd &set_weights)
    {
        weights = set_weights;
    }

    Eigen::MatrixXd forward(Eigen::MatrixXd input) {
        prev_input = Eigen::MatrixXd(input.rows(), input.cols()+1);
        auto ones = Eigen::MatrixXd::Constant(input.rows(), 1, 1.0);
        prev_input << input, ones;
        Eigen::MatrixXd output = prev_input*weights;
        return output;
    }

    Eigen::MatrixXd backward(const Eigen::MatrixXd& error, SGD sgd)
    {
        grad_weights.resize(weights.rows(), weights.cols());  // Proper resizing
        grad_weights = prev_input.transpose() * error;
        weights = sgd.update_weights(weights, grad_weights);
        return error * weights.transpose();
    }
    ~FCLayer() = default;

private:
    size_t input_size, output_size;
    Eigen::MatrixXd weights, biases, prev_input,
                    prev_output, grad_weights;
};

#endif //FCLAYER_HPP