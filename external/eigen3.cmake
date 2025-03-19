FetchContent_Declare(eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG master
)
FetchContent_Populate(eigen3) # we do not need add_subdirectory() here since we only include the header
add_library(eigen INTERFACE
        ../src/FCLayer.hpp
        ../src/ReLU.hpp
        ../src/Softmax.hpp
        ../src/Loss.hpp
        ../src/NeuralNetwork.hpp)
add_library(eigen3::eigen ALIAS eigen)
target_include_directories(eigen INTERFACE ${eigen3_SOURCE_DIR})