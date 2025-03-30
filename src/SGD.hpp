#pragma once
/* ---- SGD Optimizer ---- */
#include <Eigen/Dense>
#include <random>
#include <cmath>

class SGD {
private:
    double learning_rate = 0.001;

public:
    SGD() = default;
    explicit SGD(double lr);
    ~SGD() = default;

    Eigen::MatrixXd update_weights(const Eigen::MatrixXd& weights, const Eigen::MatrixXd& gradients) const;
};

inline SGD::SGD(double lr) : learning_rate(lr) {}

inline Eigen::MatrixXd SGD::update_weights(const Eigen::MatrixXd& weights, const Eigen::MatrixXd& gradients) const {
    // Basic SGD weight update rule: w = w - lr * grad
    return weights - learning_rate * gradients;
}


/* ---- Xavier Uniform Initialization ---- */
inline Eigen::MatrixXd XavierUniformInit(int rows, int cols, unsigned int seed = 1337) {
    static std::mt19937 rng(seed);
    double limit = std::sqrt(6.0 / static_cast<double>(rows + cols));
    std::uniform_real_distribution<double> dist(-limit, limit);

    Eigen::MatrixXd W(rows, cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            W(r, c) = dist(rng);
        }
    }
    return W;
}