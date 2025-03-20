#pragma once
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

class NeuralNetwork
{
private:
    double learningRate;
    int num_epochs, batch_size,
    hidden_size, input_size = 784;
    FullyConnected fc1;  
    FullyConnected fc2;  
    ReLU relu;
    Softmax softmax;
    CrossEntropyLoss celoss;
    SGD sgd;
    std::string train_images_path, train_labels_path,
    test_images_path, test_labels_path, pred_log_path;

public:
    NeuralNetwork(double lr, int num_of_epochs, int size_of_batch, int size_of_hidden,
    std::string tr_images_path, std::string tr_labels_path, std::string tst_images_path,
    std::string tst_labels_path, std::string pred_logfile_path)
    : learningRate(lr), num_epochs(num_of_epochs), batch_size(size_of_batch), hidden_size(size_of_hidden),
    train_images_path(tr_images_path), train_labels_path(tr_labels_path),
    test_images_path(tst_images_path), test_labels_path(tst_labels_path),
    pred_log_path(pred_logfile_path), sgd(lr) {
        fc1 = FullyConnected(input_size, hidden_size);
        fc2 = FullyConnected(hidden_size, 10); }
    ~NeuralNetwork() {}

    Eigen::MatrixXd forward(const Eigen::MatrixXd &input_tensor) {
        Eigen::MatrixXd out_fc1 = fc1.forward(input_tensor);
        Eigen::MatrixXd out_relu = relu.forward(out_fc1);
        Eigen::MatrixXd out_fc2 = fc2.forward(out_relu);
        Eigen::MatrixXd out_softmax = softmax.forward(out_fc2);
        return out_softmax;
    }
    
    Eigen::MatrixXd backward(const Eigen::MatrixXd &deriv_loss) {
        Eigen::MatrixXd grad_fc2 = fc2.backward(deriv_loss, sgd);
        Eigen::MatrixXd grad_relu = relu.backward(grad_fc2);
        Eigen::MatrixXd grad_fc1 = fc1.backward(grad_relu, sgd);
        return grad_fc1;
    }
    
    void train() {
        auto start_time = std::chrono::steady_clock::now();
        const double time_limit_seconds = 20.0 * 60.0;
        DataSetImages trainData(batch_size);
        trainData.readImageData(train_images_path);
        DatasetLabels trainLabels(batch_size);
        trainLabels.readLabelData(train_labels_path);
        size_t numBatches = trainData.getNoOfBatches();
        
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            std::cout << "Epoch " << epoch << " / " << num_epochs << std::endl;
            std::vector<size_t> batchIndices(numBatches);
            std::iota(batchIndices.begin(), batchIndices.end(), 0);
            std::shuffle(batchIndices.begin(), batchIndices.end(), std::default_random_engine(epoch));

            for (size_t idx = 0; idx < numBatches; idx++) {
                size_t b = batchIndices[idx];
                Eigen::MatrixXd batchImages = trainData.getBatch(b);   
                Eigen::MatrixXd batchLabels = trainLabels.getBatch(b);   
                Eigen::MatrixXd predictions = forward(batchImages);
                Eigen::MatrixXd deriv_loss = celoss.backward(batchLabels);
                backward(deriv_loss);
                auto current_time = std::chrono::steady_clock::now();
                std::chrono::duration<double> elapsed = current_time - start_time;
                if (elapsed.count() >= time_limit_seconds) {
                    std::cout << "Time limit reached (" << elapsed.count() << " seconds). Stopping training early." << std::endl;
                    return; }
            }
        }
        auto end_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> total_training_time = end_time - start_time;
        std::cout << "Total training time: " << total_training_time.count() << " seconds" << std::endl;
    }

    void test() {
        DataSetImages testDataObj(batch_size);
        testDataObj.readImageData(test_images_path);
        DatasetLabels testLabelsObj(batch_size);
        testLabelsObj.readLabelData(test_labels_path);
        std::ofstream predictionLogFile(pred_log_path);
        if (!predictionLogFile.is_open()) {
            std::cerr << "Error: Cannot open prediction log file: " << pred_log_path << std::endl;
            return; }

        size_t numTestBatches = testDataObj.getNoOfBatches();
        int totalSamples = 0, correctPredictions = 0;
        for (size_t b = 0; b < numTestBatches; b++) {
            predictionLogFile << "Current batch: " << b << "\n";
            Eigen::MatrixXd batchImages = testDataObj.getBatch(b);
            Eigen::MatrixXd predictions = forward(batchImages);
            Eigen::MatrixXd batchLabels = testLabelsObj.getBatch(b);
            for (int i = 0; i < predictions.rows(); i++) {
                Eigen::Index predLabel, actualLabel;
                predictions.row(i).maxCoeff(&predLabel);
                batchLabels.row(i).maxCoeff(&actualLabel);
                predictionLogFile << " - image " << (b * batch_size + i)
                                  << ": Prediction=" << predLabel
                                  << ". Label=" << actualLabel << "\n";
                totalSamples++;
                if (predLabel == actualLabel)
                    correctPredictions++;
            }
        }
        predictionLogFile.close();
        double accuracy = 100.0 * correctPredictions / totalSamples;
        std::cout << "Test accuracy: " << accuracy << "%" << std::endl;
    }
};