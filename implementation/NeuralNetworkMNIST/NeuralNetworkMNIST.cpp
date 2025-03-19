#include <string>
#include "NeuralNetwork.hpp"
#include <chrono>

int main(int arg, char** argv)
{
    double learning_rate = std::stod(argv[1]);
    int num_epochs = std::stoi(argv[2]);
    int batch_size = std::stoi(argv[3]);
    int hidden_size = std::stoi(argv[4]);
    std::string train_images_path = argv[5];
    std::string train_labels_path = argv[6];
    std::string test_images_path = argv[7];
    std::string test_labels_path = argv[8];
    std::string prediction_log_file_path = argv[9];

    NeuralNetwork NN(learning_rate, num_epochs, batch_size, hidden_size,
            train_images_path, train_labels_path, test_images_path,
            test_labels_path, prediction_log_file_path);

    std::cout << "Starting training with the following hyperparameters:\n"
              << "  Learning rate: " << learning_rate << "\n"
              << "  Epochs:        " << num_epochs    << "\n"
              << "  Batch size:    " << batch_size   << "\n"
              << "  Hidden size:   " << hidden_size << "\n\n";

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    NN.train();
    // End timing after training
    auto end = std::chrono::high_resolution_clock::now();
    // Calculate elapsed time in seconds
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "\nTraining completed. Now running test phase...\n";
    std::cout << "Training time: " << elapsed.count() << " seconds" << std::endl;
    NN.test();
    std::cout << "Test completed. Predictions logged to: " << prediction_log_file_path << "\n";
    return 0;
}