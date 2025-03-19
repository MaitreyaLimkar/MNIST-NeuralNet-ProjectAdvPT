#pragma once

#include <Eigen/Dense>
#include <random>
#include <cmath>

// Simple SGD optimizer.
class SGD {
private:
    double learningRate;
public:
    SGD();
    explicit SGD(double lr);
    ~SGD();

    // Updates weights: returns weights - learningRate * gradient.
    Eigen::MatrixXd updateWeights(Eigen::MatrixXd &weights, const Eigen::MatrixXd &gradient);
};

SGD::SGD() : learningRate(0.001) {}

SGD::SGD(double lr) : learningRate(lr) {}

SGD::~SGD() {}

Eigen::MatrixXd SGD::updateWeights(Eigen::MatrixXd &weights, const Eigen::MatrixXd &gradient) {
    return weights - learningRate * gradient;
}

// He (Kaiming) uniform initialization utility.
// outDim: number of output neurons; inDim: number of input neurons.
inline Eigen::MatrixXd heUniformInit(int outDim, int inDim, unsigned int seed = 1337)
{
    static std::mt19937 rng(seed);
    double limit = std::sqrt(6.0 / double(inDim));
    std::uniform_real_distribution<double> dist(-limit, limit);
    Eigen::MatrixXd W(outDim, inDim);
    for (int r = 0; r < outDim; ++r) {
        for (int c = 0; c < inDim; ++c) {
            W(r, c) = dist(rng);
        }
    }
    return W;
}
