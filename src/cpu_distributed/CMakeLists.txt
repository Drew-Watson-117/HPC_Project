cmake_minimum_required(VERSION 3.12)

set(PROJECT_NAME cpu_distributed)

project(${PROJECT_NAME})

set(SOURCE_FILES cpu_distributed.cpp helper_functions.cpp)
set(HEADER_FILES helper_functions.hpp)
add_executable(cpu_distributed ${HEADER_FILES} ${SOURCE_FILES})
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INLCUDE_PATH})
target_include_directories(cpu_distributed PRIVATE${MPI_CXX_INLCUDE_PATH})
target_link_libraries(cpu_distributed PUBLIC MPI::MPI_CXX)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${PROJECT_NAME} PRIVATE /W4 /permissive-)
    set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "/STACK:10000000")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic -Wl,--stack,10000000, -O3)
endif()
