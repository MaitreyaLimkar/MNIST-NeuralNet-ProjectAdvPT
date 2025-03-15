//
// Created by Maitreya Limkar on 15-03-2025.
//

#ifndef RELU_HPP
#define RELU_HPP

#include <Eigen/Dense>

// ReLU activation: element-wise max(0, x)
inline Eigen::VectorXd relu(const Eigen::VectorXd& x)
{
    return x.cwiseMax(0.0);
}

// ReLU derivative: 1 if x > 0, else 0
inline Eigen::VectorXd reluDerivative(const Eigen::VectorXd& x)
{
    Eigen::VectorXd dx = x;
    for (int i = 0; i < dx.size(); ++i)
    {
        dx(i) = (x(i) > 0) ? 1.0 : 0;
    }
    return dx;
}

#endif //RELU_HPP
