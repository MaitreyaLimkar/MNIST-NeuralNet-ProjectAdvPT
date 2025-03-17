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
        double scale = std::sqrt(6.0 / (static_cast<double>(input_size) +
                                            static_cast<double>(output_size)));
        weights = Eigen::MatrixXd::Random(static_cast<Eigen::Index>(output_size),
                                        static_cast<Eigen::Index>(input_size)) * scale;
        biases = Eigen::VectorXd::Random(static_cast<Eigen::Index>(output_size)) * scale;
    }

    void weight_setter(Eigen::MatrixXd &set_weights)
    {
        weights = set_weights;
    }

    Eigen::VectorXd forward(const Eigen::VectorXd& input)
    {
        prev_input.resize(input.rows(), input.cols() + 1);
        prev_input.block(0, 0, input.rows(), input.cols()) = input;
        prev_input.col(input.cols()) = Eigen::VectorXd::Ones(input.rows());
        prev_output = prev_input * weights;
        return prev_output;
    }

    Eigen::VectorXd backward(const Eigen::MatrixXd& error, SGD sgd)
    {
        grad_weights(weights.rows(), weights.cols());
        grad_weights = prev_input.transpose()*error;
        weights = sgd.update(weights, grad_weights);
        return error*weights.transpose();
    }
    ~FCLayer() = default;

    private:
    size_t input_size, output_size;
    Eigen::MatrixXd weights, biases, prev_input,
                    prev_output, grad_weights;
};

#endif //FCLAYER_HPP
