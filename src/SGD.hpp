#pragma once
/* ---- SGD Optimizer ---- */
#include <Eigen/Dense>
#include <random>
#include <cmath>

class SGD {
private:
    double learningRate;
public:
    SGD();
    explicit SGD(double lr);
    ~SGD();
    Eigen::MatrixXd update_weights(Eigen::MatrixXd &weights, const Eigen::MatrixXd &gradient);
};

SGD::SGD() : learningRate(0.001) {} // Initializing with default value 0.001
SGD::SGD(double lr) : learningRate(lr) {} // Initializing with provided value
SGD::~SGD() {}

Eigen::MatrixXd SGD::update_weights(Eigen::MatrixXd &weights, const Eigen::MatrixXd &gradient) {
    // Subtracting product of learning rate and gradient from weights to update them
    return weights - learningRate * gradient;
}

/* ---- Xavier Uniform Initialization ---- */
inline Eigen::MatrixXd XavierUniformInit(int outDim, int inDim, unsigned int seed = 1337)
{
    static std::mt19937 rng(seed);
    double limit = std::sqrt(6.0 / double(inDim + outDim));
    std::uniform_real_distribution<double> dist(-limit, limit);
    Eigen::MatrixXd W(outDim, inDim);
    for (int r = 0; r < outDim; ++r) {
        for (int c = 0; c < inDim; ++c) {
            W(r, c) = dist(rng);
        }
    }
    return W;
}