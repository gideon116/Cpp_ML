cmake_minimum_required(VERSION 3.15)
project(wef LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Option: build Python module instead of standalone exe ──
option(BUILD_PYTHON "Build pybind11 Python module instead of standalone executable" OFF)

# ── CUDA detection ────────────────────────────────────────
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 90)
    endif()
    set(HAS_CUDA TRUE)
    message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER}")
else()
    set(HAS_CUDA FALSE)
    message(STATUS "CUDA not found")
endif()

# ── Vulkan SDK (only when CUDA is NOT available) ──────────
# Both backends implement the same GPU class names, so only one can be compiled.
# Priority: CUDA > Vulkan > CPU-only
set(HAS_VULKAN FALSE)
if(NOT HAS_CUDA)
    if(APPLE)
        set(VULKAN_SDK "${CMAKE_SOURCE_DIR}/dependencies/vulkan/macOS")
    elseif(WIN32)
        set(VULKAN_SDK "${CMAKE_SOURCE_DIR}/dependencies/vulkan/x86_64")
    else()
        set(VULKAN_SDK "${CMAKE_SOURCE_DIR}/dependencies/vulkan/x86_64")
    endif()

    find_library(VULKAN_LIB vulkan HINTS ${VULKAN_SDK}/lib)

    if(VULKAN_LIB)
        message(STATUS "Vulkan found: ${VULKAN_LIB}")
        set(HAS_VULKAN TRUE)
    else()
        message(WARNING "Neither CUDA nor Vulkan found – building CPU-only")
    endif()
endif()

# ── Common C++ sources ───────────────────────────────────
set(LEARNN_SOURCES
    src/tensor.cpp
    src/matrix_ops.cpp
    src/layers.cpp
    src/layers_multithreaded.cpp
    src/model.cpp
    src/transformer.cpp
    src/img_class.cpp
)

if(HAS_CUDA)
    list(APPEND LEARNN_SOURCES
        src/layers_CUDA.cu
        src/matrix_ops_CUDA.cu
    )
elseif(HAS_VULKAN)
    list(APPEND LEARNN_SOURCES
        src/use_GPU.cpp
        src/layers_GPU.cpp
        src/matrix_ops_GPU.cpp
    )
endif()

# ══════════════════════════════════════════════════════════
# MODE 1: Standalone C++ executable (default)
# ══════════════════════════════════════════════════════════
if(NOT BUILD_PYTHON)

    add_executable(wef
        train.cpp
        ${LEARNN_SOURCES}
    )

    target_include_directories(wef PRIVATE include)

    if(HAS_CUDA)
        target_compile_definitions(wef PRIVATE LEARNN_HAS_CUDA)
        target_link_libraries(wef PRIVATE pthread)
    elseif(HAS_VULKAN)
        target_compile_definitions(wef PRIVATE LEARNN_HAS_VULKAN)
        target_include_directories(wef PRIVATE ${VULKAN_SDK}/include)
        target_link_libraries(wef PRIVATE ${VULKAN_LIB} pthread)
    else()
        target_link_libraries(wef PRIVATE pthread)
    endif()

# ══════════════════════════════════════════════════════════
# MODE 2: pybind11 Python module (pip install)
# ══════════════════════════════════════════════════════════
else()

    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(_learnn_core
        python/bindings.cpp
        ${LEARNN_SOURCES}
    )

    target_include_directories(_learnn_core PRIVATE include)

    if(HAS_CUDA)
        target_compile_definitions(_learnn_core PRIVATE LEARNN_HAS_CUDA)
    elseif(HAS_VULKAN)
        target_compile_definitions(_learnn_core PRIVATE LEARNN_HAS_VULKAN)
        target_include_directories(_learnn_core PRIVATE ${VULKAN_SDK}/include)
        target_link_libraries(_learnn_core PRIVATE ${VULKAN_LIB})
        if(APPLE)
            set_target_properties(_learnn_core PROPERTIES
                BUILD_RPATH "${VULKAN_SDK}/lib"
                INSTALL_RPATH "${VULKAN_SDK}/lib"
            )
        endif()
    endif()

    target_link_libraries(_learnn_core PRIVATE pthread)

    install(TARGETS _learnn_core DESTINATION wefml)

endif()