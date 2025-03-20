//
// Created by Maitreya Limkar on 15-03-2025.
//

#ifndef LOSS_HPP
#define LOSS_HPP

#include <Eigen/Dense>
#include <cmath>

#define EPSILON 1e-10

class CrossEntropyLoss {
private:
    Eigen::MatrixXd lastPrediction;
public:
    CrossEntropyLoss();
    ~CrossEntropyLoss();
    double forward(const Eigen::MatrixXd &, const Eigen::MatrixXd &);
    Eigen::MatrixXd backward(const Eigen::MatrixXd &);
};

CrossEntropyLoss::CrossEntropyLoss() = default;
CrossEntropyLoss::~CrossEntropyLoss() = default;

double CrossEntropyLoss::forward(const Eigen::MatrixXd &inputTensor, const Eigen::MatrixXd &labelTensor)
{
    lastPrediction = inputTensor;
    double loss = -(labelTensor.array() * (inputTensor.array() + EPSILON).log()).sum();
    loss /= static_cast<double>(inputTensor.rows());
    return loss;
}

Eigen::MatrixXd CrossEntropyLoss::backward(const Eigen::MatrixXd &labelTensor)
{
    // target output/predicted output
    auto output = (lastPrediction - labelTensor) / static_cast<double>(lastPrediction.rows());
    return output;
}
#endif //LOSS_HPP
