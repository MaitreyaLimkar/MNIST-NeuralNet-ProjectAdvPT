FetchContent_Declare(eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 9441d94dccccd5db8d64179516fdc5b53994a047
)
FetchContent_Populate(eigen3) # we do not need add_subdirectory() here since we only include the header

add_library(eigen INTERFACE)
add_library(eigen3::eigen ALIAS eigen)
target_include_directories(eigen INTERFACE ${eigen3_SOURCE_DIR})