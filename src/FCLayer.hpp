#ifndef ADVPT_MPU_FULLY_CONNECTED_HPP
#define ADVPT_MPU_FULLY_CONNECTED_HPP
#include "Eigen/Dense"
#include "SGD.hpp"

class FullyConnected {
private:
    Eigen::MatrixXd weights, input_tensor;
    size_t input_size, output_size;

public:
    FullyConnected() {}
    // Setting input and output dimensions, resizing weight matrix, initializing weights using
    // Xavier uniform initialization, and setting bias row to zero
    FullyConnected(size_t in, size_t out) : input_size(in), output_size(out) {
        weights.resize(input_size + 1, output_size);
        weights.topRows(input_size) = xavierUniformInit(input_size, output_size);
        weights.row(input_size).setZero(); }

    void setWeights(const Eigen::MatrixXd &w) { // Setting weight matrix explicitly
        weights = w;
    }

    // Augmenting input with bias column, computing linear combination using weights
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input) {
        size_t batch_size = input.rows();
        input_tensor.resize(batch_size, input_size + 1);
        input_tensor.block(0, 0, batch_size, input_size) = input;
        input_tensor.col(input_size) = Eigen::VectorXd::Ones(batch_size);
        Eigen::MatrixXd output = input_tensor * weights;
        return output;
    }

    // Computing gradient for weights, propagating error to previous layer, updating weights using SGD
    Eigen::MatrixXd backward(const Eigen::MatrixXd &error_tensor, SGD &sgd) {
        Eigen::MatrixXd gradient_weights = input_tensor.transpose() * error_tensor;
        Eigen::MatrixXd passed_error = error_tensor * weights.topRows(input_size).transpose();
        weights = sgd.update_weights(weights, gradient_weights);
        return passed_error;
    }
    ~FullyConnected() {}
};
#endif // ADVPT_MPU_FULLY_CONNECTED_HPP