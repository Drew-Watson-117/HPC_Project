cmake_minimum_required(VERSION 3.12)

# Project Name
set(PROJECT_NAME gpu_shared)
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

# Source and Header Files
set(SOURCE_FILES main.cpp helper_functions.cpp)
set(HEADER_FILES helper_functions.hpp)
set(CUDA_SOURCE_FILES kernel.cu)

# Add Executable
add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES} ${CUDA_SOURCE_FILES})

# Set C++ Standard
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)

# Compiler Options
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${PROJECT_NAME} PRIVATE /W4 /permissive-)
    set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "/STACK:10000000")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic -Wl,--stack,10000000, -O3)
endif()

# Find CUDA Package
find_package(CUDAToolkit 12.2 REQUIRED)
target_link_libraries(${PROJECT_NAME} CUDA::cudart)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES OFF)
