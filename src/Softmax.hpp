#pragma once

#include <Eigen/Dense>

class Softmax
{
private:
    Eigen::MatrixXd inputTensorCache; // Logits
    Eigen::MatrixXd yHat;             // Softmax outputs
public:
    Softmax();
    ~Softmax();

    Eigen::MatrixXd forward(const Eigen::MatrixXd &inputTensor);
    Eigen::MatrixXd backward(const Eigen::MatrixXd &errorTensor); // (Optional; may not be used if loss returns gradient)
};

Softmax::Softmax() {}
Softmax::~Softmax() {}

Eigen::MatrixXd Softmax::forward(const Eigen::MatrixXd &inputTensor)
{
    inputTensorCache = inputTensor;
    // Subtract max per row for numerical stability.
    // rowMax is an (m x 1) vector; subtracting it column-wise works properly.
    Eigen::VectorXd rowMax = inputTensor.rowwise().maxCoeff();
    Eigen::MatrixXd shifted = inputTensor.array().colwise() - rowMax.array();

    Eigen::MatrixXd expTensor = shifted.array().exp();
    Eigen::VectorXd rowSum = expTensor.rowwise().sum();
    // Divide each element by the sum of its row (using colwise division).
    yHat = (expTensor.array().colwise() / rowSum.array()).matrix();
    return yHat;
}

Eigen::MatrixXd Softmax::backward(const Eigen::MatrixXd &errorTensor)
{
    // Not used if combined with loss.
    Eigen::VectorXd weightedErrorSum = (errorTensor.array() * yHat.array()).rowwise().sum();
    Eigen::MatrixXd sumMatrix = weightedErrorSum.replicate(1, errorTensor.cols());
    Eigen::MatrixXd adjustedError = errorTensor.array() - sumMatrix.array();
    return yHat.array() * adjustedError.array();
}