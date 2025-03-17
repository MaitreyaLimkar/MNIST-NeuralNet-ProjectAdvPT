#include <string>
#include "NeuralNetwork.hpp"   

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
    NN.train();
    NN.test();
    return 0;
}