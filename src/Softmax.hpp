//
// Created by Maitreya Limkar on 15-03-2025.
//

#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include <Eigen/Dense>

// Softmax activation function for numerical stability
class Softmax {
public:
    Softmax();
    ~Softmax();
    Eigen::MatrixXd forward(const Eigen::MatrixXd &);
    Eigen::MatrixXd backward(const Eigen::MatrixXd &);

private:
    Eigen::MatrixXd lastInput;
    Eigen::MatrixXd lastOutput;
};

inline Softmax::Softmax() = default;
inline Softmax::~Softmax() = default;

inline Eigen::MatrixXd Softmax::forward(const Eigen::MatrixXd &inputTensor)
{
    lastInput = inputTensor;
    auto shifted = inputTensor.colwise() - inputTensor.rowwise().maxCoeff();
    Eigen::MatrixXd expTensor = shifted.array().exp();
    lastOutput = expTensor.array().colwise() / expTensor.array().rowwise().sum();
    return lastOutput;
}

inline Eigen::MatrixXd Softmax::backward(const Eigen::MatrixXd &errorTensor)
{
    Eigen::MatrixXd weightedErrorSum = (errorTensor.array() * lastOutput.array()).rowwise().sum();
    Eigen::MatrixXd adjustedError = errorTensor.array() - (weightedErrorSum.replicate(1, errorTensor.cols())).array();
    Eigen::MatrixXd gradInput = lastOutput.array() * adjustedError.array();
    return gradInput;
}

#endif //SOFTMAX_HPP
