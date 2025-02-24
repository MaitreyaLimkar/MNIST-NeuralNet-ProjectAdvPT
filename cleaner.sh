[ -d build ] && rm -rf build
[ -d bin ] && rm -rf bin
[ -e image_output_tensor.txt ] && rm image_output_tensor.txt
[ -e label_output_tensor.txt ] && rm label_output_tensor.txt
[ -e log_predictions-ci.txt ] && rm log_predictions-ci.txt
[ -e log_predictions.txt ] && rm log_predictions.txt