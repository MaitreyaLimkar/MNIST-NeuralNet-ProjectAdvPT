//
// Created by Maitreya Limkar on 14-03-2025.
//

#ifndef FCLAYER_HPP
#define FCLAYER_HPP

#include <Eigen/Dense>
#include "SGD.hpp"

// Fully Connected Layer class
class FCLayer
{
public:
    FCLayer(){}
    FCLayer(size_t input_size, size_t output_size):input_size(input_size), output_size(output_size)
    {
        // Calculating the Xavier factor initialization.
        limit = sqrt( 6.0 /(input_size + output_size)) ;
        weights = Eigen::MatrixXd::Random(input_size + 1, output_size); // 1 row added for biases
        weights = weights * limit;
    }

    void weight_setter(const Eigen::MatrixXd &set_weights)
    {
        weights = set_weights;
    }

    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) {
        size_t batch_size = input.rows();
        prev_input.resize(batch_size, input_size + 1);
        prev_input.block(0, 0, batch_size, input_size) = input;
        prev_input.col(input_size) = Eigen::VectorXd::Ones(batch_size);
        Eigen::MatrixXd output = prev_input * weights;
        return output;
    }

    Eigen::MatrixXd backward(const Eigen::MatrixXd& error, SGD sgd)
    {
        Eigen::MatrixXd grad_weights = prev_input.transpose() * error;
        weights = sgd.update_weights(weights, grad_weights);
        Eigen::MatrixXd next_error = error * weights.transpose();
        Eigen::MatrixXd output = next_error.leftCols(input_size);
        return output;
    }
    ~FCLayer() = default;

private:
    size_t input_size, output_size;
    double limit;
    Eigen::MatrixXd weights, prev_input;
};


#endif //FCLAYER_HPP