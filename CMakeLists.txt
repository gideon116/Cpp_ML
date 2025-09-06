cmake_minimum_required(VERSION 3.14)
project(wef)

add_executable(
    wef
    train.cpp
    src/img_class.cpp
    src/layers_GPU.cpp
    src/matrix_operations.cpp
    src/tensor.cpp
    src/useGPU.cpp
    src/layers.cpp
    src/layers_multithreaded.cpp
    src/model.cpp
    src/transformer.cpp
)

target_link_libraries(
    wef
    PRIVATE 
        # ${CMAKE_SOURCE_DIR}/dependencies/vulkan/x86_64/lib/libvulkan.so
        ${CMAKE_SOURCE_DIR}/dependencies/vulkan/macOS/lib/libvulkan.1.4.321.dylib
        pthread
    )

target_include_directories(
    wef
    PRIVATE
        include
        # dependencies/vulkan/x86_64/include
        dependencies/vulkan/macOS/include
    )