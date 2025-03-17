//
// Created by Maitreya Limkar on 15-03-2025.
//

#ifndef SGD_HPP
#define SGD_HPP

#include <Eigen/Dense>

class SGD {
public:
    explicit SGD(const double lr) : lr(lr) {}
    ~SGD() = default;

    [[nodiscard]] Eigen::MatrixXd update_weights(const Eigen::MatrixXd &weights, const Eigen::MatrixXd &gradient) const
    {
        return weights - lr * gradient;
    }
    [[nodiscard]] Eigen::VectorXd update_biases(const Eigen::VectorXd &biases, const Eigen::VectorXd &grad) const {
        return biases - lr * grad;
    }
    [[nodiscard]] double getLearningRate() const { return lr; }

private:
    double lr; // Learning rate.
};

#endif //SGD_HPP
