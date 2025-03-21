#pragma once
/* ---- Neural Network ---- */
#include <iostream>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include "Loss.hpp"
#include "SGD.hpp"
#include "ReLU.hpp"
#include "Softmax.hpp"
#include "FCLayer.hpp"
#include "readImageMNIST.hpp"
#include "readLabelMNIST.hpp"
// Very important. Stay focused. All the best for the exam.
class NeuralNetwork
{
private:
    double learning_rate;
    int num_epochs, batch_size,
    hidden_layer_size, input_size = 784;

    // Layers
    FullyConnected fc1, fc2;

    ReLU relu;
    Softmax softmax;
    CrossEntropyLoss ce_loss;
    SGD sgd;

    // File paths
    std::string train_data_path, train_labels_path,
    test_data_path, test_labels_path, prediction_log_file_path;

public:
    NeuralNetwork(double lr, int numberOfEpochs, int sizeBatch, int sizeHidden,
                  std::string pathImageTrain, std::string pathLabelTrain,
                  std::string pathImageTest, std::string pathLabelTest,
                  std::string predLogPath) :
      learning_rate(lr), num_epochs(numberOfEpochs), batch_size(sizeBatch),
      hidden_layer_size(sizeHidden), train_data_path(std::move(pathImageTrain)),
      train_labels_path(std::move(pathLabelTrain)),
      test_data_path(std::move(pathImageTest)),
      test_labels_path(std::move(pathLabelTest)),
      prediction_log_file_path(std::move(predLogPath)), sgd(lr)
    {   // Initialize FullyConnected layers via Xavier
        fc1 = FullyConnected(input_size, hidden_layer_size);
        fc2 = FullyConnected(hidden_layer_size, 10);
    }
    ~NeuralNetwork() = default;

    // Forward pass through FC1 -> ReLU -> FC2 -> Softmax.
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input_tensor)
    {
        Eigen::MatrixXd out_fc1 = fc1.forward(input_tensor);
        Eigen::MatrixXd out_relu = relu.forward(out_fc1);
        Eigen::MatrixXd out_fc2 = fc2.forward(out_relu);
        Eigen::MatrixXd out_softmax = softmax.forward(out_fc2);
        return out_softmax;
    }

    // Backward pass: Propagate the loss gradient through FC2, ReLU, then FC1.
    Eigen::MatrixXd backward(const Eigen::MatrixXd &deriv_loss)
    {
        Eigen::MatrixXd grad_fc2 = fc2.backward(deriv_loss, sgd);
        Eigen::MatrixXd grad_relu = relu.backward(grad_fc2);
        Eigen::MatrixXd grad_fc1 = fc1.backward(grad_relu, sgd);
        return grad_fc1;
    }

    // Training routine: Loads training data and labels, then performs forward/backward passes.
       void train()
    {
        auto start_time = std::chrono::steady_clock::now();
        const double time_limit_seconds = 1200.0; // Limit of 20 mins for CI
        // Load MNIST data
        readImageMNIST train_data_obj(batch_size);
        train_data_obj.readImageData(train_data_path);
        readLabelMNIST train_labels_obj(batch_size);
        train_labels_obj.readLabelData(train_labels_path);
        size_t num_batches = train_data_obj.getNumOfBatches();

        for (int epoch = 0; epoch < num_epochs; ++epoch)
        {
            // std::cout << "Epoch " << (epoch + 1) << " / " << num_epochs << "..." << std::endl;
            // Shuffle batch indices for better generalization
            std::vector<size_t> batch_indices(num_batches);
            std::iota(batch_indices.begin(), batch_indices.end(), 0);
            std::shuffle(batch_indices.begin(), batch_indices.end(),
                         std::default_random_engine(static_cast<unsigned>(epoch)));

            for (size_t idx = 0; idx < num_batches; ++idx)
            {
                size_t b = batch_indices[idx];
                // Forward pass
                Eigen::MatrixXd batch_images = train_data_obj.getBatch(b);
                Eigen::MatrixXd batch_labels = train_labels_obj.getBatch(b);
                Eigen::MatrixXd predictions = forward(batch_images);
                // Compute cross-entropy loss for debug NN
                double loss_val = ce_loss.forward(predictions, batch_labels);
                // Backprop
                Eigen::MatrixXd dLoss = ce_loss.backward(batch_labels);
                backward(dLoss);
                // Time check to stop early if needed
                auto now_time = std::chrono::steady_clock::now();
                std::chrono::duration<double> elapsed = now_time - start_time;
                if (elapsed.count() >= time_limit_seconds)
                {
                    std::cout << "Time limit reached (" << elapsed.count() <<
                        " seconds). Stopping training." << std::endl;
                    return;
                }
            }
        }
        auto end_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> total_time = end_time - start_time;
        std::cout << "Training completed in " << total_time.count() << " seconds." << std::endl;
    }

    // Testing routine: Loads test data and labels, logs predictions, and computes accuracy.
    void test()
    {
        readImageMNIST test_data_obj(batch_size);
        test_data_obj.readImageData(test_data_path);
        readLabelMNIST test_labels_obj(batch_size);
        test_labels_obj.readLabelData(test_labels_path);
        std::ofstream prediction_log(prediction_log_file_path);
        if (!prediction_log.is_open())
        {
            std::cerr << "Error: Cannot open prediction log file: " <<
                prediction_log_file_path << std::endl;
            return;
        }

        size_t num_test_batches = test_data_obj.getNumOfBatches();
        int total_samples = 0;
        int correct_predictions = 0;

        for (size_t b = 0; b < num_test_batches; ++b)
        {
            prediction_log << "Current batch: " << b << "\n";
            Eigen::MatrixXd batch_images = test_data_obj.getBatch(b);
            Eigen::MatrixXd predictions = forward(batch_images);
            Eigen::MatrixXd batch_labels = test_labels_obj.getBatch(b);

            for (int i = 0; i < predictions.rows(); ++i)
            {
                Eigen::Index pred_label;
                predictions.row(i).maxCoeff(&pred_label);
                Eigen::Index actual_label;
                batch_labels.row(i).maxCoeff(&actual_label);
                prediction_log << " - image " << (b * batch_size + i)
                               << ": Prediction=" << pred_label
                               << ". Label=" << actual_label << "\n";
                total_samples++;
                if (pred_label == actual_label) {
                    correct_predictions++;
                }
            }
        }

        prediction_log.close();
        double accuracy = 100.0 * static_cast<double>(correct_predictions) / static_cast<double>(total_samples);
        std::cout << "Test accuracy: " << accuracy << "%" << std::endl;
    }
};