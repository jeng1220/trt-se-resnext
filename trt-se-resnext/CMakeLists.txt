cmake_minimum_required(VERSION 3.8)
file(GLOB resnext_src
  common.cpp
  common.h
  resnext.cpp
  resnext.h
)
add_library(resnext STATIC ${resnext_src})
target_compile_features(resnext PUBLIC cxx_std_11)

add_executable(trt_se_resnext main.cpp)
target_link_libraries(trt_se_resnext PRIVATE resnext nvinfer nvparsers nvinfer_plugin cudnn cublas cudart_static nvToolsExt cudart rt dl pthread)
