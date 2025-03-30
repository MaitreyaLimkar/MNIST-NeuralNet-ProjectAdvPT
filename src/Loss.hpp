#ifndef LOSS_HPP
#define LOSS_HPP

/* ---- Cross Entropy Loss ---- */
#include <Eigen/Dense>
#include <cmath>

constexpr double EPS = 1e-10; // To avoid log(0) issues

class CrossEntropyLoss {
private:
    Eigen::MatrixXd prediction_cache; // Store prediction for backward pass

public:
    CrossEntropyLoss() = default;
    ~CrossEntropyLoss() = default;
    double forward(const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &labels);
    Eigen::MatrixXd backward(const Eigen::MatrixXd &labels);
};

// Cross Entropy Forward Pass: Computes the loss
inline double CrossEntropyLoss::forward(const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &labels) {
    prediction_cache = predictions; // Save predictions for backward pass
    // Compute element-wise cross-entropy loss
    Eigen::MatrixXd log_preds = (predictions.array() + EPS).log();
    double loss = -(labels.array() * log_preds.array()).sum();
    // Normalize loss by batch size
    return loss / predictions.rows();
}

// Cross Entropy Backward Pass: Computes gradient for backpropagation
inline Eigen::MatrixXd CrossEntropyLoss::backward(const Eigen::MatrixXd &labels) {
    // Compute gradient: dL/dp = (p - y) / batch_size
    return (prediction_cache - labels) / labels.rows();
}

#endif // LOSS_HPP