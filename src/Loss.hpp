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
    Eigen::MatrixXd lastPrediction;
public:
    CrossEntropyLoss();
    ~CrossEntropyLoss();
    double forward(const Eigen::MatrixXd &, const Eigen::MatrixXd &);
    Eigen::MatrixXd backward(const Eigen::MatrixXd &);
};

inline CrossEntropyLoss::CrossEntropyLoss() = default;
inline CrossEntropyLoss::~CrossEntropyLoss() = default;

inline double CrossEntropyLoss::forward(const Eigen::MatrixXd &inputTensor, const Eigen::MatrixXd &labelTensor)
{
    lastPrediction = inputTensor;
    return -((labelTensor.array() * (inputTensor.array().log())).sum());
}

inline Eigen::MatrixXd CrossEntropyLoss::backward(const Eigen::MatrixXd &labelTensor)
{
    // target output/predicted output
    auto output = -(labelTensor.array() / lastPrediction.array());
    return output;
}
#endif //LOSS_HPP
