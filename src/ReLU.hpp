//
// Created by Maitreya Limkar on 15-03-2025.
//

#ifndef RELU_HPP
#define RELU_HPP

#include <Eigen/Dense>

class ReLU
{
private:

    Eigen::MatrixXd lastInput;

public:
    ReLU();
    ~ReLU();

    Eigen::MatrixXd forward(const Eigen::MatrixXd &);
    Eigen::MatrixXd backward(const Eigen::MatrixXd &);
};
inline ReLU::ReLU()= default;
inline ReLU::~ReLU()= default;

inline Eigen::MatrixXd ReLU::forward(const Eigen::MatrixXd& input)
{
    lastInput = input;
    Eigen::MatrixXd output = input.cwiseMax(0.0);
    return output;
}

inline Eigen::MatrixXd ReLU::backward(const Eigen::MatrixXd& error)
{
    Eigen::MatrixXd mask = (lastInput.array() >= 0.0).cast<double>();
    Eigen::MatrixXd output = error.block(0,0,lastInput.rows(),lastInput.cols()).array() * mask.array();
    return output;
}

#endif //RELU_HPP