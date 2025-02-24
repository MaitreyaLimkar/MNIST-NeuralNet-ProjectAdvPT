#!/bin/bash
#./build/readLabelMNIST "mnist-datasets/train-labels.idx1-ubyte" "label_output_tensor.txt" "0"
./build/readLabelMNIST "$1" "$2" "$3"