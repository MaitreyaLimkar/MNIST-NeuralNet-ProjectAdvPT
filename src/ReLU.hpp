#ifndef RELU_HPP
#define RELU_HPP
/* ---- ReLU Activation Function ---- */
#include <Eigen/Dense>

class ReLU {
private:
    Eigen::MatrixXd input_cache;  // Stores input for use in backward pass

public:
    ReLU() = default;
    ~ReLU() = default;
    // Forward pass: applies ReLU activation
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input);
    // Backward pass: computes gradient w.r.t. input
    Eigen::MatrixXd backward(const Eigen::MatrixXd& grad_output);
};

inline Eigen::MatrixXd ReLU::forward(const Eigen::MatrixXd& input) {
    input_cache = input;
    return input.cwiseMax(0.0);  // Element-wise max with 0 (ReLU)
}

inline Eigen::MatrixXd ReLU::backward(const Eigen::MatrixXd& grad_output) {
    // Gradient mask: 1 where input was > 0, else 0
    Eigen::MatrixXd relu_derivative = (input_cache.array() > 0.0).cast<double>();
    return grad_output.array() * relu_derivative.array();  // Element-wise product
}

#endif // RELU_HPP