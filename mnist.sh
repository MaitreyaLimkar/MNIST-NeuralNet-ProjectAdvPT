if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input_config>"
    exit 1
fi

# Read the config file
while IFS= read -r line || [ -n "$line" ]
do
    # Skip lines without an equal sign
    if [[ $line != *"="* ]]; then
        continue
    fi

    # Split the line into key and value
    IFS='=' read -r key value <<< "$line"

    key=$(echo $key | tr -d '[:space:]')
    value=$(echo $value | tr -d '[:space:]')
    declare $key=$value

    # Print the key-value pair
    # echo "$key=$value"
done < "$1"

# Run the build/mnist executable with the appropriate arguments
./build/mnist $learning_rate $num_epochs $batch_size $hidden_size $rel_path_train_images $rel_path_train_labels $rel_path_test_images $rel_path_test_labels $rel_path_log_file