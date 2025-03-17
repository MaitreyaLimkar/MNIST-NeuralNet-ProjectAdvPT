//
// Created by Maitreya Limkar on 15-03-2025.
//

#ifndef LOSS_HPP
#define LOSS_HPP

#include <Eigen/Dense>
#include <cmath>

#define EPS 1e-10

class CrossEntropyLoss {
private:
    // Cache the predicted tensor from the forward pass for use in backward propagation.
    Eigen::MatrixXd prev_prediction;

public:
    CrossEntropyLoss() = default;
    ~CrossEntropyLoss() = default;

    double forward(const Eigen::MatrixXd &input_tensor, const Eigen::MatrixXd &label_tensor) {
        // Cache predictions for use in backward pass.
        prev_prediction = input_tensor;
        // Compute the element-wise log with an epsilon to prevent log(0)
        Eigen::MatrixXd log_prediction = (input_tensor.array() + EPS).log();
        // Compute the loss (summing over all elements).
        double loss = - (label_tensor.array() * log_prediction.array()).sum();
        return loss;
    }

    Eigen::MatrixXd backward(const Eigen::MatrixXd &label_tensor) {
        // Compute gradient safely using EPSILON to avoid division by zero.
        Eigen::MatrixXd grad_output = -(label_tensor.array() / (prev_prediction.array() + EPS));
        return grad_output;
    }
};


#endif //LOSS_HPP
