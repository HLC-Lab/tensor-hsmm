cmake_minimum_required(VERSION 3.20)

project(tensor-viterbi VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Build options ---
# ON  → GPU backend (CUDA or ROCm, detected from environment)
# OFF → CPU/OpenMP backend only
option(BUILD_GPU "Build GPU (CUDA/ROCm) backend instead of CPU/OpenMP" ON)

# --- OpenMP (always required — used by both CPU and GPU builds) ---
find_package(OpenMP REQUIRED)
message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")

# --- GPU or CPU backend ---
if(BUILD_GPU)
    if(NOT DEFINED GPU_PLATFORM)
        if(DEFINED ENV{ROCM_PATH} OR EXISTS "/opt/rocm")
            set(GPU_PLATFORM "ROCM")
        else()
            set(GPU_PLATFORM "CUDA")
        endif()
    endif()
    message(STATUS "GPU Platform: ${GPU_PLATFORM}")

    if(GPU_PLATFORM STREQUAL "ROCM")
        if(NOT DEFINED ROCM_PATH)
            if(DEFINED ENV{ROCM_PATH})
                set(ROCM_PATH "$ENV{ROCM_PATH}")
            else()
                message(FATAL_ERROR "ROCM_PATH is not set. Load the rocm module before configuring.")
            endif()
        endif()
        message(STATUS "ROCm path: ${ROCM_PATH}")
        list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib/cmake/hip" "${ROCM_PATH}")

        find_package(hip CONFIG REQUIRED PATHS ${ROCM_PATH})
        enable_language(HIP)
        set(CMAKE_HIP_STANDARD 17)
        set(CMAKE_HIP_ARCHITECTURES "" CACHE STRING "HIP GPU architectures (e.g. gfx90a)")

        set(CUDA_SOURCES src/hsmm.cu src/kernels.cu)
        set(CUDA_HEADERS src/kernels.cuh)
        set(HIPIFY_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/hipified_src")
        file(MAKE_DIRECTORY ${HIPIFY_OUT_DIR})

        file(WRITE ${HIPIFY_OUT_DIR}/cuda_runtime.h "
#ifndef FAKE_CUDA_RUNTIME_H
#define FAKE_CUDA_RUNTIME_H
#include <hip/hip_runtime.h>
#if !defined(__syncwarp)
    #define __syncwarp() __builtin_amdgcn_wave_barrier()
#endif
#if !defined(__shfl_down_sync)
    #define __shfl_down_sync(mask, var, delta) __shfl_down(var, delta)
#endif
#define hipLaunchCooperativeKernel(f, g, b, p, sh) \\
        (hipLaunchCooperativeKernel)((f), (g), (b), (p), (sh), (hipStream_t)0)
#endif
")

        set(NATIVE_SOURCES "")
        foreach(FILE ${CUDA_SOURCES} ${CUDA_HEADERS})
            get_filename_component(FILENAME ${FILE} NAME)
            set(OUT_FILE "${HIPIFY_OUT_DIR}/${FILENAME}")
            add_custom_command(
                OUTPUT ${OUT_FILE}
                COMMAND ${ROCM_PATH}/bin/hipify-perl ${CMAKE_CURRENT_SOURCE_DIR}/${FILE} > ${OUT_FILE}
                DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${FILE}
                COMMENT "Hipifying: ${FILE}"
            )
            if(${FILE} MATCHES ".*\\.cu$")
                list(APPEND NATIVE_SOURCES ${OUT_FILE})
            endif()
        endforeach()

        set_source_files_properties(${NATIVE_SOURCES} PROPERTIES LANGUAGE HIP)
        add_definitions(-DUSE_ROCM)

    else()  # CUDA
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_ARCHITECTURES "" CACHE STRING "CUDA GPU architectures (e.g. 80)")
        set(NATIVE_SOURCES src/hsmm.cu src/kernels.cu)
        add_definitions(-DUSE_CUDA)
    endif()

else()
    message(STATUS "Building CPU/OpenMP backend only")
    set(NATIVE_SOURCES "src/hsmm.cu")
    set_source_files_properties(src/hsmm.cu PROPERTIES LANGUAGE CXX)
    add_definitions(-DNO_GPU)
endif()

# --- Pybind11 ---
if(NOT PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE "python3")
endif()
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('include'))" OUTPUT_VARIABLE PYTHON_INC OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import pybind11; print(pybind11.get_include())" OUTPUT_VARIABLE PYBIND_INC OUTPUT_STRIP_TRAILING_WHITESPACE)

# --- Extension module ---
add_library(_native SHARED src/bindings.cpp ${NATIVE_SOURCES})

set_target_properties(_native PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_native"
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tensor_viterbi/viterbi
)

target_compile_options(_native PRIVATE
    # CXX Logic (Host Code)
    $<$<COMPILE_LANGUAGE:CXX>:
        $<$<CXX_COMPILER_ID:NVHPC>:-O3 -Ofast -tp=native -Wall>
        $<$<NOT:$<CXX_COMPILER_ID:NVHPC>>:-O3 -march=native -mtune=native -ffast-math -Wall -Wextra -Wno-unused-parameter>
    >

    # HIP Logic
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -ffast-math -Wall -Wextra -Wno-unused-parameter -fopenmp>

    # CUDA Logic (Device Code)
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3 
        # Note: CUDA (nvcc) accepts -use_fast_math instead of -ffast-math
        -use_fast_math 
        -Xptxas=-O3 
        -Xcompiler=-Wall,-Wextra,-Wno-unused-parameter,-fopenmp
    >
)

if(BUILD_GPU AND GPU_PLATFORM STREQUAL "ROCM")
    target_include_directories(_native BEFORE PRIVATE ${HIPIFY_OUT_DIR})
endif()

target_include_directories(_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(_native SYSTEM PRIVATE ${PYTHON_INC} ${PYBIND_INC})

# For ROCm (clang-based): link with -fopenmp so clang picks up its own LLVM
# OpenMP runtime (libomp, which provides __kmpc_fork_call etc.) rather than
# the GCC/Cray libgomp found by find_package(OpenMP) for the CXX compiler.
if(BUILD_GPU AND GPU_PLATFORM STREQUAL "ROCM")
    target_link_options(_native PRIVATE -fopenmp)
    target_link_libraries(_native PRIVATE hip::host)
else()
    target_link_libraries(_native PRIVATE OpenMP::OpenMP_CXX)
endif()


