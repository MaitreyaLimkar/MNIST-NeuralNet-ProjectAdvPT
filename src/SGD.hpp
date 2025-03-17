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
    SGD();
    explicit SGD(double);
    ~SGD() = default;
    [[nodiscard]] Eigen::MatrixXd update_weights(const Eigen::MatrixXd &, const Eigen::MatrixXd &) const;
};

inline SGD::SGD(){
    this->lr = 0.001;
}

inline SGD::SGD(const double lr){
    this->lr = lr;
}



inline Eigen::MatrixXd SGD::update_weights(const Eigen::MatrixXd &weights, const Eigen::MatrixXd &gradient) const{
    return (weights - lr * gradient);
}

#endif //SGD_HPP
