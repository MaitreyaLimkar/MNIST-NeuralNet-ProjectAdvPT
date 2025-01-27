project(readImageMNIST)
add_executable(${PROJECT_NAME} readImageMNIST.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_link_libraries(${PROJECT_NAME} PRIVATE eigen)