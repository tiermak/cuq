include_directories("${PROJECT_SOURCE_DIR}/tests")

add_executable(all_tests
  device_selection/device_selection_tests.cu
)

set_target_properties(all_tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        )
target_link_libraries(all_tests PRIVATE cuq cudart)
