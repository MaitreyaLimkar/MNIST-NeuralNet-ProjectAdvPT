cmake_minimum_required(VERSION 3.26)
project(MNIST_NN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)

include(FetchContent)
include_directories(${CMAKE_SOURCE_DIR}/src)
include(external/eigen3.cmake)
add_subdirectory(implementation)