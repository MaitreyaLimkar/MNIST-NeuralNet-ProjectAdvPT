# Cmake file to fetch Eigen 3 from remote repo

include(Fetch_content)
# Fetch Eigen
FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
)

# Make Eigen available
FetchContent_MakeAvailable(eigen)