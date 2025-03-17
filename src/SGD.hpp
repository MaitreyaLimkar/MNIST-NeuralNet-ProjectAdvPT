//
// Created by Maitreya Limkar on 15-03-2025.
//

#ifndef SGD_HPP
#define SGD_HPP

#include <Eigen/Dense>

class SGD {
private:
    double lr; // Learning rate

public:
    // Default constructor with standard learning rate
    SGD() : lr(0.001) {}

    // Constructor with custom learning rate
    explicit SGD(double learning_rate) : lr(learning_rate) {}

    // Virtual destructor for proper inheritance
    virtual ~SGD() = default;

    // Weight update method
    Eigen::MatrixXd update_weights(const Eigen::MatrixXd& weights, const Eigen::MatrixXd& gradient) const {
        return weights - lr * gradient;
    }

    // Getter for learning rate
    double getLearningRate() const {
        return lr;
    }

    // Setter for learning rate
    void setLearningRate(double learning_rate) {
        lr = learning_rate;
    }
};

#endif //SGD_HPP
