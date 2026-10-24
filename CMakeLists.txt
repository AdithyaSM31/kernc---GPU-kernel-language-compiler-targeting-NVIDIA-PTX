cmake_minimum_required(VERSION 3.16)
project(kernc VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler warnings
if (MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find CUDA / ptxas / libcuda
find_path(CUDA_INCLUDE_DIRS cuda.h
    PATHS
        /usr/local/cuda-13.0/include
        /usr/local/cuda/include
        /usr/include
)

find_library(CUDA_DRIVER_LIB cuda
    PATHS
        /usr/lib/wsl/lib
        /usr/local/cuda-13.0/lib64/stubs
        /usr/local/cuda/lib64/stubs
        /usr/lib/x86_64-linux-gnu
)

find_program(PTXAS_BIN ptxas
    PATHS
        /usr/local/cuda-13.0/bin
        /usr/local/cuda/bin
        /usr/bin
)

if (CUDA_INCLUDE_DIRS AND CUDA_DRIVER_LIB)
    set(CUDA_FOUND TRUE)
    message(STATUS "Found CUDA headers: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "Found CUDA driver: ${CUDA_DRIVER_LIB}")
else()
    message(WARNING "CUDA Driver API not found. GPU tests will be disabled.")
endif()

if (PTXAS_BIN)
    message(STATUS "Found ptxas: ${PTXAS_BIN}")
else()
    message(WARNING "ptxas not found. PTX acceptance tests will be disabled.")
endif()

# Core Compiler Library
add_library(kernc_lib STATIC
    src/Diagnostics.cpp
    src/Lexer.cpp
    src/AST.cpp
    src/Parser.cpp
    src/Sema.cpp
    src/IR.cpp
    src/IRGen.cpp
    src/Passes.cpp
    src/PTXEmitter.cpp
)
target_include_directories(kernc_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# CLI Compiler Executable
add_executable(kernc src/main.cpp)
target_link_libraries(kernc PRIVATE kernc_lib)

# Standalone Driver API Runner Tool
if (CUDA_FOUND)
    add_executable(kernc-runner tools/runner/runner.cpp)
    target_include_directories(kernc-runner PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(kernc-runner PRIVATE ${CUDA_DRIVER_LIB})
endif()

# Testing
enable_testing()
add_subdirectory(tests)
