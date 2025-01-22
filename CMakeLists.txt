cmake_minimum_required(VERSION 3.14)
project(mnist_data_recognition)

# Include the external Eigen setup
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/executable)
include(external/eigen3.cmake)

include_directories(${CMAKE_SOURCE_DIR}/src)

# Link Eigen to your executable
target_link_libraries(MyApp PRIVATE Eigen3::Eigen)
