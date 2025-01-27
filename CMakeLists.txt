cmake_minimum_required(VERSION 3.26)
project(MNIST_NN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Explicitly set Eigen include directory
set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/eigen/Eigen")

add_executable(MNIST_NN implementation/readImageMNIST/readImageMNIST.cpp
        main.cpp
        src/readImageMNIST.hpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
        ${EIGEN3_INCLUDE_DIR}
)

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(external/eigen)
add_subdirectory(implementation)