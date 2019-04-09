include_directories("${PROJECT_SOURCE_DIR}/tests")

add_executable(tests
  device_selection/device_selection_tests.cu
)

set_target_properties(tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        )
target_link_libraries(tests PRIVATE cuq cudart)
