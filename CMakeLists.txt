cmake_minimum_required(VERSION 3.10)

project(cuda_tools_study LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED) # Comment: for obtaining CMake
                            # variable CUDA_TOOLKIT_ROOT_DIR
find_library(NVTOOLS_EXT nvToolsExt 
    PATHS ${CUDA_TOOLKIT_ROOT_DIR} 
    PATH_SUFFIXES lib64
)

add_subdirectory(hello_world)
add_subdirectory(kernel_analysis)