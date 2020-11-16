cmake_minimum_required(VERSION 3.10)
project(cuq LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CUDA_FLAGS "-lineinfo")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Werror") #flags to host compiler behind the nvcc
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror cross-execution-space-call,reorder,deprecated-declarations")

set(CMAKE_CXX_FLAGS "-O2 -Wall -Wextra -Wshadow -Wpedantic -Wnon-virtual-dtor -Wunused -Wdouble-promotion")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -rdynamic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

# Library
add_library(cuq
        src/cuq.cpp
        src/device_selection.cu
        )
target_include_directories(cuq PUBLIC include/)
target_include_directories(cuq PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(cuq cudart nvidia-ml)
set_target_properties(
        cuq
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
)

install(TARGETS cuq DESTINATION lib)
install(FILES include/cuq.h DESTINATION include)

# Example
add_executable(concurrent_gpu_demo
        examples/concurrent_gpu_demo.cu
        )
target_link_libraries(concurrent_gpu_demo cuq)

# Tests
include(CTest)
add_subdirectory(tests)
