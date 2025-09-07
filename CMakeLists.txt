cmake_minimum_required(VERSION 3.14)
project(wef)

# enable_language(CUDA)
# set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 90) # 52=Maxwell, 60/61=Pascal, 70=Volta, 75=Turing, 80/86=Ampere, 90=Hopper

add_executable(
    wef
    train.cpp
    src/kernel.cu
    src/img_class.cpp
    src/layers_GPU.cpp
    # src/layers_CUDA.cu
    src/matrix_ops_GPU.cpp
    # src/matrix_ops_CUDA.cu
    src/matrix_ops.cpp
    src/tensor.cpp
    src/useGPU.cpp
    src/layers.cpp
    src/layers_multithreaded.cpp
    src/model.cpp
    src/transformer.cpp
)

# set(VULKAN_SDK "${CMAKE_SOURCE_DIR}/dependencies/vulkan/x86_64")
set(VULKAN_SDK "${CMAKE_SOURCE_DIR}/dependencies/vulkan/macOS")

find_library(VULKAN_LIB vulkan HINTS ${VULKAN_SDK}/lib)

target_link_libraries(
    wef
    PRIVATE 
        ${VULKAN_LIB}
        pthread
    )

target_include_directories(
    wef
    PRIVATE
        include
        ${VULKAN_SDK}/include
    )