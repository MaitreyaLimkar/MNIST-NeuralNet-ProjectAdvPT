cmake_minimum_required(VERSION 3.26)
project(MNIST_NN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Explicitly set Eigen include directory
set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/eigen/Eigen")

add_executable(MNIST_NN main.cpp
        implementation/readerMNIST/readImageMNIST.cpp
        src/readImageMNIST.hpp
        src/readLabelMNIST.hpp
        implementation/readerMNIST/readLabelMNIST.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
        ${EIGEN3_INCLUDE_DIR}
)

set_target_properties(MNIST_NN PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(external/eigen)
add_subdirectory(implementation)