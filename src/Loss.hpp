//
// Created by Maitreya Limkar on 15-03-2025.
//

#ifndef LOSS_HPP
#define LOSS_HPP

#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

// Cross-entropy loss between the prediction and the one-hot target
inline double crossEntropyLoss(const Eigen::VectorXd& prediction,
                               const Eigen::VectorXd& target)
{
    double loss = 0.0;
    for (int i = 0; i < prediction.size(); ++i) {
        double p = std::max(prediction(i), 1e-10); // avoiding log(0)
        loss -= target(i) * std::log(p);
    }
    return loss;
}

#endif //LOSS_HPP
