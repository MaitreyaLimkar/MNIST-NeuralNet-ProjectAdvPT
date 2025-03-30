#pragma once
/* ---- Softmax Activation ---- */
#include <Eigen/Dense>

class Softmax {
private:
    Eigen::MatrixXd input_cache;  // Stores input for backward pass
    Eigen::MatrixXd softmax_output;  // Stores softmax output for gradient computation

public:
    Softmax() = default;
    ~Softmax() = default;
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input_tensor);
    Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient);
};

inline Eigen::MatrixXd Softmax::forward(const Eigen::MatrixXd& input_tensor) {
    // Save input for use in backward pass
    input_cache = input_tensor;
    // Compute max per row (for numerical stability)
    Eigen::VectorXd row_max = input_tensor.rowwise().maxCoeff();
    // Shift input tensor for numerical stability (broadcasting)
    Eigen::MatrixXd shifted_inputs = input_tensor.array().colwise() - row_max.array();
    // Compute exponentials
    Eigen::MatrixXd exp_values = shifted_inputs.array().exp();
    // Compute row-wise sum of exponentials
    Eigen::VectorXd row_sums = exp_values.rowwise().sum();
    // Compute final softmax output
    softmax_output = (exp_values.array().colwise() / row_sums.array()).matrix();
    return softmax_output;
}

inline Eigen::MatrixXd Softmax::backward(const Eigen::MatrixXd& gradient) {
    // Compute element-wise product of gradient and softmax output, then sum each row
    Eigen::VectorXd weighted_sum = (gradient.array() *
        softmax_output.array()).rowwise().sum();
    // Broadcast sum over all columns and subtract from gradient
    Eigen::MatrixXd adjusted_gradient = gradient.array() -
        weighted_sum.replicate(1, gradient.cols()).array();
    // Multiply by softmax output element-wise to compute gradient
    return softmax_output.array() * adjusted_gradient.array();
}