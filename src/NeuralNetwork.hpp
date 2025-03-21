#pragma once

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>

#include "Loss.hpp"
#include "SGD.hpp"
#include "ReLU.hpp"      // Ensure the case matches the actual filename (e.g., "relu.hpp")
#include "Softmax.hpp"
#include "FCLayer.hpp"
#include "readImageMNIST.hpp"
#include "readLabelMNIST.hpp"

class NeuralNetwork
{
private:
    double learningRate;
    int numEpochs;
    int batchSize;
    int hiddenLayerSize;
    int inputSize = 784; // MNIST images (28x28)

    // Layers
    FullyConnected fc1;  // [784 -> hiddenLayerSize]
    FullyConnected fc2;  // [hiddenLayerSize -> 10]

    ReLU relu;
    Softmax softmax;
    CrossEntropyLoss celoss;
    SGD sgd;

    // File paths for data
    std::string trainDataPath;
    std::string trainLabelsPath;
    std::string testDataPath;
    std::string testLabelsPath;
    std::string predictionLogFilePath;

public:
    NeuralNetwork(double lr,
                  int nEpochs,
                  int bSize,
                  int hLayerSize,
                  std::string tDataPath,
                  std::string tLabelsPath,
                  std::string tstDataPath,
                  std::string tstLabelsPath,
                  std::string predLogPath)
        : learningRate(lr), numEpochs(nEpochs), batchSize(bSize), hiddenLayerSize(hLayerSize),
          trainDataPath(tDataPath), trainLabelsPath(tLabelsPath),
          testDataPath(tstDataPath), testLabelsPath(tstLabelsPath),
          predictionLogFilePath(predLogPath), sgd(lr)
    {
        // Initialize FullyConnected layers with He initialization
        fc1 = FullyConnected(inputSize, hiddenLayerSize);
        fc2 = FullyConnected(hiddenLayerSize, 10);
    }

    ~NeuralNetwork() {
        // Cleanup if necessary.
    }

    // Forward pass through FC1 -> ReLU -> FC2 -> Softmax.
    // Input: [batch_size x 784], Output: [batch_size x 10]
    Eigen::MatrixXd forward(const Eigen::MatrixXd &inputTensor)
    {
        Eigen::MatrixXd out_fc1 = fc1.forward(inputTensor);
        Eigen::MatrixXd out_relu = relu.forward(out_fc1);
        Eigen::MatrixXd out_fc2 = fc2.forward(out_relu);
        Eigen::MatrixXd out_softmax = softmax.forward(out_fc2);
        return out_softmax;
    }

    // Backward pass: Propagate the loss gradient through FC2, ReLU, then FC1.
    // dLoss is assumed to be (yhat - y)/N, shape [batch_size x 10]
    Eigen::MatrixXd backward(const Eigen::MatrixXd &dLoss)
    {
        Eigen::MatrixXd grad_fc2 = fc2.backward(dLoss, sgd);
        Eigen::MatrixXd grad_relu = relu.backward(grad_fc2);
        Eigen::MatrixXd grad_fc1 = fc1.backward(grad_relu, sgd);
        return grad_fc1;
    }

    // Training routine: Loads training data and labels, then performs forward/backward passes.
    void train()
    {
        auto start_time = std::chrono::steady_clock::now();
        const double time_limit_seconds = 20.0 * 60.0; // 20 minutes

        readImageMNIST trainData(batchSize);
        trainData.readImageData(trainDataPath);

        readLabelMNIST trainLabels(batchSize);
        trainLabels.readLabelData(trainLabelsPath);

        size_t numBatches = trainData.getNumOfBatches();

        for (int epoch = 0; epoch < numEpochs; epoch++)
        {
            std::cout << "Epoch " << epoch << " / " << numEpochs << std::endl;

            // Optionally, shuffle batches to improve generalization.
            std::vector<size_t> batchIndices(numBatches);
            std::iota(batchIndices.begin(), batchIndices.end(), 0);
            std::shuffle(batchIndices.begin(), batchIndices.end(), std::default_random_engine(epoch));

            for (size_t idx = 0; idx < numBatches; idx++)
            {
                size_t b = batchIndices[idx];

                Eigen::MatrixXd batchImages = trainData.getBatch(b);   // [miniBatchSize x 784]
                Eigen::MatrixXd batchLabels = trainLabels.getBatch(b);   // [miniBatchSize x 10]

                Eigen::MatrixXd predictions = forward(batchImages);
                double lossVal = celoss.forward(predictions, batchLabels);
                //std::cout << "  Batch " << b << " loss: " << lossVal << std::endl;

                Eigen::MatrixXd dLoss = celoss.backward(batchLabels);
                backward(dLoss);

                // Check elapsed time after each batch
                auto current_time = std::chrono::steady_clock::now();
                std::chrono::duration<double> elapsed = current_time - start_time;
                if (elapsed.count() >= time_limit_seconds)
                {
                    std::cout << "Time limit reached (" << elapsed.count() << " seconds). Stopping training early." << std::endl;
                    return; // or break out of outer loop if you prefer
                }
            }
        }

        auto end_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> total_training_time = end_time - start_time;
        std::cout << "Total training time: " << total_training_time.count() << " seconds" << std::endl;
    }

    // Testing routine: Loads test data and labels, logs predictions, and computes accuracy.
    void test()
    {
        readImageMNIST testDataObj(batchSize);
        testDataObj.readImageData(testDataPath);

        readLabelMNIST testLabelsObj(batchSize);
        testLabelsObj.readLabelData(testLabelsPath);

        std::ofstream predictionLogFile(predictionLogFilePath);
        if (!predictionLogFile.is_open())
        {
            std::cerr << "Error: Cannot open prediction log file: " << predictionLogFilePath << std::endl;
            return;
        }

        size_t numTestBatches = testDataObj.getNumOfBatches();
        int totalSamples = 0;
        int correctPredictions = 0;


        for (size_t b = 0; b < numTestBatches; b++)
        {
            predictionLogFile << "Current batch: " << b << "\n";
            Eigen::MatrixXd batchImages = testDataObj.getBatch(b);
            Eigen::MatrixXd predictions = forward(batchImages);
            Eigen::MatrixXd batchLabels = testLabelsObj.getBatch(b);

            for (int i = 0; i < predictions.rows(); i++)
            {
                Eigen::Index predLabel;
                predictions.row(i).maxCoeff(&predLabel);

                Eigen::Index actualLabel;
                batchLabels.row(i).maxCoeff(&actualLabel);

                predictionLogFile << " - image " << (b * batchSize + i)
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

