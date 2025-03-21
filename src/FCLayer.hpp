#pragma once
/* ---- Fully Connected Layer ---- */
#include "Eigen/Dense"
#include "SGD.hpp"

extern Eigen::MatrixXd XavierUniformInit(int rows, int cols, unsigned int seed);
class FullyConnected {
private:
    Eigen::MatrixXd weights, input_tensor;
    size_t input_size{}, output_size{};

public:
    FullyConnected() = default;
    // Setting input and output dimensions, resizing weight matrix, initializing weights using
    // Xavier uniform initialization, and setting bias row to zero
    FullyConnected(size_t in, size_t out) : input_size(in), output_size(out) {
        weights.resize(input_size + 1, output_size);
        weights.topRows(input_size) = XavierUniformInit(input_size, output_size);
        weights.row(input_size).setZero(); }
    ~FullyConnected() = default;

    void setWeights(const Eigen::MatrixXd &weights_matrix) {
        weights = weights_matrix;
    }

    // Augmenting input with bias column, computing linear combination using weights
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input) {
        const size_t batch_size = input.rows();
        // Augment input with column of ones for the bias
        input_tensor.resize(batch_size, input_size + 1);
        input_tensor.block(0, 0, batch_size, input_size) = input;
        input_tensor.col(input_size) = Eigen::VectorXd::Ones(batch_size);
        // Multiply by weights => output shape: (batch_size x output_size)
        return input_tensor * weights;
    }

    // Computing gradient w.r.t. weights, updates weights, returns gradient for previous layer
    Eigen::MatrixXd backward(const Eigen::MatrixXd &grad_output, SGD &sgd) {
        // Compute dW = X^T * dY, shape: [ (input_size+1) x output_size ]
        Eigen::MatrixXd grad_weights = input_tensor.transpose() * grad_output;
        //   dX = dY * W^T, but ignoring the last row of W (the bias row).
        Eigen::MatrixXd grad_input = grad_output * weights.topRows(input_size).transpose();
        // Update weights in place with the computed gradient
        weights = sgd.update_weights(weights, grad_weights);
        return grad_input;
    }
};