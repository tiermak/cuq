cmake_minimum_required(VERSION 3.10)

project(cuq LANGUAGES CXX)
project(cuq LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 11)

set(CMAKE_CUDA_FLAGS "-lineinfo")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Werror") #flags to host compiler behind the nvcc
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror cross-execution-space-call,reorder,deprecated-declarations")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-O2 -Wall -Wextra -Wshadow -Wpedantic -Wnon-virtual-dtor -Wunused -Wdouble-promotion")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

include_directories("${PROJECT_SOURCE_DIR}/include")

#
# Library
#
add_library(cuq SHARED 
    src/cuq.cpp
  )

target_compile_features(cuq PUBLIC cxx_std_11)

add_executable(concurrent_gpu_test
  examples/concurrent_gpu_demo.cu
)
set_target_properties( concurrent_gpu_test
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                       POSITION_INDEPENDENT_CODE ON
                      )
target_link_libraries(concurrent_gpu_test PRIVATE cudart cuq)

# set(CMAKE_BUILD_TYPE Debug)
