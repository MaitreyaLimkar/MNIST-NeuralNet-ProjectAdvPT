//
// Created by Maitreya Limkar on 15-03-2025.
//

#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include <Eigen/Dense>

// Softmax activation function for numerical stability
inline Eigen::VectorXd softmax(const Eigen::VectorXd& x)
{
    double maxCoeff = x.maxCoeff();
    Eigen::VectorXd shifted = x.array() - maxCoeff;
    Eigen::VectorXd exponent = shifted.array().exp();
    double sum = exponent.sum();
    return exponent / sum;
}

#endif //SOFTMAX_HPP
