#include <string>
#include "NeuralNetwork.hpp"
#include <chrono>

int main(int count, char** argvect)
{
    // Expected arguments: <learning_rate> <num_epochs> <batch_size> <hidden_size>
    // <train_images_path> <train_labels_path> <test_images_path> <test_labels_path> <prediction_log_file_path>
    if (count != 10) {
        std::cerr << "Usage:\n  " << argvect[0]
                  << " <learning_rate> <num_epochs> <batch_size> <hidden_size>"
                  << " <train_images_path> <train_labels_path>"
                  << " <test_images_path> <test_labels_path> <prediction_log_file_path>\n";
        return 1;
    }

    // Parse hyperparameters
    double learning_rate;
    int num_epochs, batch_size, hidden_size;
    try {
        learning_rate = std::stod(argvect[1]);
        num_epochs    = std::stoi(argvect[2]);
        batch_size    = std::stoi(argvect[3]);
        hidden_size   = std::stoi(argvect[4]);
    } catch (const std::exception& e) {
        std::cerr << "Error: Invalid numeric argument.\n"
                  << "Caught exception: " << e.what() << std::endl;
        return 1;
    }

    // Parse file paths
    std::string train_images_path        = argvect[5];
    std::string train_labels_path        = argvect[6];
    std::string test_images_path         = argvect[7];
    std::string test_labels_path         = argvect[8];
    std::string prediction_log_file_path = argvect[9];

    // Display hyperparameters
    std::cout << "Starting training with the following hyperparameters:\n"
              << "  Learning rate : " << learning_rate << "\n"
              << "  Epochs        : " << num_epochs    << "\n"
              << "  Batch size    : " << batch_size    << "\n"
              << "  Hidden size   : " << hidden_size   << "\n"
              << "Train images path   : " << train_images_path   << "\n"
              << "Train labels path   : " << train_labels_path   << "\n"
              << "Test images path    : " << test_images_path    << "\n"
              << "Test labels path    : " << test_labels_path    << "\n"
              << "Prediction log file : " << prediction_log_file_path << "\n\n";

    // Neural network created
    NeuralNetwork NN( learning_rate, num_epochs, batch_size, hidden_size,
        train_images_path, train_labels_path,
        test_images_path, test_labels_path,
        prediction_log_file_path);

    // Time ttaken training phase
    auto start_time = std::chrono::high_resolution_clock::now();
    NN.train();
    auto end_time = std::chrono::high_resolution_clock::now();

    // Compute elapsed time
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    std::cout << "Training completed in " << elapsed_seconds.count() << " seconds.\n";

    // Test phase
    std::cout << "\nNow running test phase...\n";
    NN.test();
    std::cout << "Test completed. Predictions logged to: "
              << prediction_log_file_path << "\n";
    return 0;
} // It ends here! Lots of hardwork ;)