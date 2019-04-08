include_directories("${PROJECT_SOURCE_DIR}/tests")

add_executable(tests
  device_selection/device_selection_tests.cu
)

set_target_properties(tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        )
target_link_libraries(tests PRIVATE cuq cudart)


add_executable(concurrent_processes_test
  device_selection/concurrent_processes_device_selection.cu
)

set_target_properties(concurrent_processes_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        )
target_link_libraries(concurrent_processes_test PRIVATE cuq cudart)