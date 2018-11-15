# cmake 3.8 natively support CUDA,
# refer to https://devblogs.nvidia.com/parallelforall/building-cuda-applications-cmake/
cmake_minimum_required(VERSION 3.8)
project(resnext LANGUAGES CXX CUDA)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
add_subdirectory(trt-se-resnext)
