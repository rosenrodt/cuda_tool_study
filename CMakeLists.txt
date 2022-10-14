cmake_minimum_required(VERSION 3.10)

project(cuda_tools_study LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
set(NVTOOLS_EXT CUDA::nvToolsExt)

add_subdirectory(hello_world)
add_subdirectory(kernel_analysis)