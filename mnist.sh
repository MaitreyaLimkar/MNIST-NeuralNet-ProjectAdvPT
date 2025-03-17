#!/bin/bash
set -euo pipefail

# Check if config file is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input_config>"
    exit 1
fi

CONFIG_FILE="$1"

# Read the config file line by line
while IFS= read -r line || [ -n "$line" ]; do
    # Trim leading/trailing whitespace.
    line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    # Skip empty lines or lines starting with '#'
    if [ -z "$line" ] || [[ "$line" == \#* ]]; then
        continue
    fi

    # Only process lines that contain an '=' character.
    if [[ "$line" != *"="* ]]; then
        continue
    fi

    # Split the line into key and value.
    IFS='=' read -r key value <<< "$line"
    # Remove any spaces from key and value.
    key=$(echo "$key" | tr -d '[:space:]')
    value=$(echo "$value" | tr -d '[:space:]')
    declare "$key"="$value"
done < "$CONFIG_FILE"

# Define the required variables.
required_vars=("learning_rate" "num_epochs" "batch_size" "hidden_size" \
               "rel_path_train_images" "rel_path_train_labels" \
               "rel_path_test_images" "rel_path_test_labels" "rel_path_log_file")

# Check that all required variables are set.
for var in "${required_vars[@]}"; do
    if [ -z "${!var:-}" ]; then
        echo "Error: Variable '$var' is not set in the config file."
        exit 1
    fi
done
# Usage: ./build/NeuralNetworkMNIST "mnist-configs/input.config"
# Run the executable with the parsed arguments.
./build/NeuralNetworkMNIST "$learning_rate" "$num_epochs" "$batch_size" "$hidden_size" \
    "$rel_path_train_images" "$rel_path_train_labels" \
    "$rel_path_test_images" "$rel_path_test_labels" "$rel_path_log_file"
