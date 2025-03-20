/* ---- Softmax Activation ---- */
#pragma once
#include <Eigen/Dense>

class Softmax {
private:
    Eigen::MatrixXd temp_input;
    Eigen::MatrixXd softmax_out;
public:
    Softmax();
    ~Softmax();
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input_tensor);
    Eigen::MatrixXd backward(const Eigen::MatrixXd &error_tensor);
};

Softmax::Softmax() {}
Softmax::~Softmax() {}

Eigen::MatrixXd Softmax::forward(const Eigen::MatrixXd &input_tensor) {
    // Saving input for backward pass
    temp_input = input_tensor;
    // Computing maximum value in each row for numerical stabilization
    Eigen::VectorXd max_x = input_tensor.rowwise().maxCoeff();
    // Subtracting row-wise maximum from each element (using broadcasting)
    Eigen::MatrixXd shifted = input_tensor.array().colwise() - max_x.array();
    // Computing exponential for each element in the shifted matrix
    Eigen::MatrixXd exp_tensor = shifted.array().exp();
    // Summing exponentials along each row to obtain normalization factor
    Eigen::VectorXd row_sum = exp_tensor.rowwise().sum();
    // Normalizing each element by dividing by the corresponding row sum
    softmax_out = (exp_tensor.array().colwise() / row_sum.array()).matrix();
    return softmax_out;
}

Eigen::MatrixXd Softmax::backward(const Eigen::MatrixXd &error_tensor) {
    // Computing element-wise product of error and softmax output, then summing each row
    Eigen::VectorXd weightedErrorSum = (error_tensor.array() * softmax_out.array()).rowwise().sum();
    // Replicating row-wise error sums to match dimensions of error tensor
    Eigen::MatrixXd matrix_sum = weightedErrorSum.replicate(1, error_tensor.cols());
    // Subtracting replicated sum from error tensor to adjust each error term
    Eigen::MatrixXd shifter_error = error_tensor.array() - matrix_sum.array();
    // Multiplying adjusted error by softmax output element-wise to compute gradient
    return softmax_out.array() * shifter_error.array();
}