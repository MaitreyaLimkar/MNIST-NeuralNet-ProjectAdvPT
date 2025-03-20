#ifndef RELU_HPP
#define RELU_HPP
#include <Eigen/Dense>

class ReLU {
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

inline Eigen::MatrixXd ReLU::forward(const Eigen::MatrixXd& input) {
    lastInput = input; // Saving input for use in backward computation
    // Computing element-wise maximum with zero
    Eigen::MatrixXd output = input.cwiseMax(0.0);
    return output;
}

inline Eigen::MatrixXd ReLU::backward(const Eigen::MatrixXd& error) {
    // Computing mask by checking positive elements
    Eigen::MatrixXd mask = (lastInput.array() >= 0.0).cast<double>();
    // Multiplying error with mask element-wise
    Eigen::MatrixXd output = error.block(0,0,
        lastInput.rows(),lastInput.cols()).array() * mask.array();
    return output;
}
#endif //RELU_HPP