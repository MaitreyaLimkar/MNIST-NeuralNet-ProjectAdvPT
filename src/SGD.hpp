/* ---- SGD Optimizer ---- */
#pragma once
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

/* ---- Xavier (Glorot) Initialization ---- */
inline Eigen::MatrixXd xavierUniformInit(int outDim, int inDim, unsigned int seed = 1337) {
    static std::mt19937 rng(seed); // Initializing random number generator with provided seed
    double limit = std::sqrt(6.0 / double(inDim + outDim));
    // Setting up uniform distribution ranging from -limit to limit
    std::uniform_real_distribution<double> dist(-limit, limit);
    Eigen::MatrixXd weight(outDim, inDim);
    for (int row = 0; row < outDim; ++row) { // Iterating over each row
        for (int col = 0; col < inDim; ++col) { // Iterating over each column
            // Sampling random value from distribution and assigning to matrix element
            weight(row, col) = dist(rng);
        }
    }
    return weight;
}