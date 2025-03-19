#ifndef ADVPT_MPU_FULLY_CONNECTED_HPP
#define ADVPT_MPU_FULLY_CONNECTED_HPP

#include "Eigen/Dense"
#include "SGD.hpp"

// FullyConnected layer with bias incorporated.
// Uses explicit block operations to form an augmented input.
class FullyConnected
{
private:
    Eigen::MatrixXd weights; // shape: (input_size + 1) x output_size
    size_t input_size;
    size_t output_size;

    // Cache the augmented input (with bias column) for backward pass.
    Eigen::MatrixXd input_tensor;

public:
    FullyConnected() {}
    FullyConnected(size_t in, size_t out) : input_size(in), output_size(out)
    {
        // Use He (Kaiming) uniform initialization for non-bias weights.
        weights.resize(input_size + 1, output_size);
        weights.topRows(input_size) = heUniformInit(input_size, output_size);
        // Initialize bias row to zeros.
        weights.row(input_size).setZero();
    }

    // Setter for weights (useful for testing)
    void setWeights(const Eigen::MatrixXd &w) {
        weights = w;
    }

    // Forward pass:
    // Input: [batch_size x input_size]
    // Augment input with a column of ones to account for bias.
    // Output: [batch_size x output_size]
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input) {
        size_t batch_size = input.rows();
        input_tensor.resize(batch_size, input_size + 1);
        // Copy input into first columns.
        input_tensor.block(0, 0, batch_size, input_size) = input;
        // Set last column to ones (bias).
        input_tensor.col(input_size) = Eigen::VectorXd::Ones(batch_size);
        // Compute output.
        Eigen::MatrixXd output = input_tensor * weights;
        return output;
    }

    // Backward pass:
    // error_tensor: [batch_size x output_size]
    // Returns: gradient for previous layer [batch_size x input_size] (excluding bias derivative)
    Eigen::MatrixXd backward(const Eigen::MatrixXd &error_tensor, SGD &sgd) {
        Eigen::MatrixXd gradient_weights = input_tensor.transpose() * error_tensor;
        // Compute propagated error using the current weights (exclude bias row)
        Eigen::MatrixXd propagated_error = error_tensor * weights.topRows(input_size).transpose();
        // Update weights after computing the propagated error.
        weights = sgd.updateWeights(weights, gradient_weights);
        return propagated_error;
    }

    ~FullyConnected() {}
};

#endif // ADVPT_MPU_FULLY_CONNECTED_HPP
