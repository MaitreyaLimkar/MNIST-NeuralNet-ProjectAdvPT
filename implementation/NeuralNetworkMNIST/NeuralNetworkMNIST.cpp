#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <cstdlib>      // For std::srand, std::exit
#include "NeuralNetwork.hpp"   // Your neural network implementation
#include "readImageMNIST.hpp"  // Provides DataSetImages class
#include "readLabelMNIST.hpp"  // Provides DatasetLabels class
#include <Eigen/Dense>

// Helper function: trim whitespace from both ends of a string.
std::string trim(const std::string &s) {
    size_t start = s.find_first_not_of(" \t");
    size_t end = s.find_last_not_of(" \t");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

// Simple configuration parser: expects lines "key = value"
std::map<std::string, std::string> parseConfig(const std::string &configPath) {
    std::map<std::string, std::string> config;
    std::ifstream file(configPath.c_str());
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open config file: " << configPath << std::endl;
        std::exit(1);
    }
    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comments.
        if (line.empty() || line[0] == '#')
            continue;
        size_t pos = line.find('=');
        if (pos == std::string::npos)
            continue;
        std::string key = trim(line.substr(0, pos));
        std::string value = trim(line.substr(pos + 1));
        config[key] = value;
    }
    file.close();
    return config;
}

int main(int argc, char* argv[]) {
    // Ensure a configuration file is provided.
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_config>" << std::endl;
        return 1;
    }
    std::string configPath = argv[1];
    auto config = parseConfig(configPath);

    // Retrieve file paths.
    std::string trainImagePath = config["rel_path_train_images"];
    std::string trainLabelPath = config["rel_path_train_labels"];
    std::string testImagePath  = config["rel_path_test_images"];
    std::string testLabelPath  = config["rel_path_test_labels"];
    std::string logFilePath    = config["rel_path_log_file"];

    // Retrieve hyperparameters.
    int numEpochs    = std::stoi(config["num_epochs"]);
    int batchSize    = std::stoi(config["batch_size"]);
    int hiddenSize   = std::stoi(config["hidden_size"]);
    double learningRate = std::stod(config["learning_rate"]);

    // Fixed dimensions for MNIST.
    int inputSize  = 784;   // 28x28 flattened.
    int outputSize = 10;    // 10 classes.

    // For demonstration, use subsets.
    int trainSamples = 1000;  // Adjust as needed.
    int testSamples  = 200;    // Adjust as needed.

    // Fix the random seed for reproducibility.
    std::srand(42);

    // Load training and testing data.
    DataSetImages trainImages(batchSize);
    trainImages.readImageData(trainImagePath);
    DatasetLabels trainLabels(batchSize);
    trainLabels.readLabelData(trainLabelPath);

    DataSetImages testImages(batchSize);
    testImages.readImageData(testImagePath);
    DatasetLabels testLabels(batchSize);
    testLabels.readLabelData(testLabelPath);

    // Create the neural network.
    NeuralNetwork nn(inputSize, hiddenSize, outputSize, learningRate);

    std::cout << "Starting training for " << numEpochs << " epochs..." << std::endl;
    // -------------------------
    // Training Phase
    // -------------------------
    for (int epoch = 0; epoch < numEpochs; ++epoch) {
        double epochLoss = 0.0;
        for (int i = 0; i < trainSamples; ++i) {
            int batchNo = i / batchSize;
            int sampleIndex = i % batchSize;
            // Get training image.
            Eigen::MatrixXd imageBatch = trainImages.getBatch(batchNo);
            Eigen::VectorXd image = imageBatch.row(sampleIndex).transpose();
            // Get corresponding one-hot label.
            Eigen::MatrixXd labelBatch = trainLabels.getBatch(batchNo);
            Eigen::VectorXd label = labelBatch.row(sampleIndex).transpose();

            // Forward pass.
            nn.forward(image);
            // Compute loss.
            double loss = nn.computeLoss(label);
            epochLoss += loss;
            // Backward pass: update parameters (note: only target is passed).
            nn.backward(label);
        }
        std::cout << "Epoch " << epoch
                  << " Average Loss: " << (epochLoss / trainSamples)
                  << std::endl;
    }

    // -------------------------
    // Testing Phase & Logging Predictions
    // -------------------------
    std::ofstream logFile(logFilePath.c_str());
    if (!logFile.is_open()) {
        std::cerr << "Error: Cannot open log file: " << logFilePath << std::endl;
        return 1;
    }
    logFile << "Prediction Log\n";
    int numBatches = testSamples / batchSize;
    int globalImageIndex = 0;
    for (int batch = 0; batch < numBatches; ++batch) {
        logFile << "Current batch: " << batch << "\n";
        Eigen::MatrixXd testImageBatch = testImages.getBatch(batch);
        Eigen::MatrixXd testLabelBatch = testLabels.getBatch(batch);
        for (int j = 0; j < batchSize; ++j) {
            // Get test image and label.
            Eigen::VectorXd image = testImageBatch.row(j).transpose();
            Eigen::VectorXd label = testLabelBatch.row(j).transpose();

            // Forward pass.
            Eigen::VectorXd logSoftmaxOutput = nn.forward(image);
            // Convert log-softmax to probabilities.
            Eigen::VectorXd probabilities = logSoftmaxOutput.array().exp();
            int predicted;
            probabilities.maxCoeff(&predicted);
            int trueLabel;
            label.maxCoeff(&trueLabel);

            logFile << " - image " << globalImageIndex
                    << ": Prediction=" << predicted
                    << ". Label=" << trueLabel << "\n";
            globalImageIndex++;
        }
    }
    logFile.close();
    std::cout << "Testing complete. Predictions logged to " << logFilePath << std::endl;

    return 0;
}
