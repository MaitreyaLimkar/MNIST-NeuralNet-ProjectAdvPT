//
// Created by Maitreya Limkar on 15-03-2025.
//

#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include <Eigen/Dense>

// Softmax activation function for numerical stability
class Softmax {
public:
    // Forward pass: compute softmax for each row in the input matrix.
    static Eigen::MatrixXd forward(const Eigen::MatrixXd &input) {
        Eigen::MatrixXd output(input.rows(), input.cols());
        for (int i = 0; i < input.rows(); i++) {
            double maxVal = input.row(i).maxCoeff();
            Eigen::RowVectorXd exps = (input.row(i).array() - maxVal).exp();
            double sumExps = exps.sum();
            output.row(i) = exps / sumExps;
        }
        return output;
    }
    // Backward pass: if using softmax with cross-entropy, the gradient is typically computed as (softmax - target).
    // Here we simply return the error tensor unchanged.
    static Eigen::MatrixXd backward(const Eigen::MatrixXd &errorTensor) {
        return errorTensor;
    }
};
#endif //SOFTMAX_HPP
