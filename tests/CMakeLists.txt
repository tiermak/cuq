cmake_minimum_required(VERSION 3.10)

add_executable(device_selection_test
        device_selection/device_selection_tests.cu
        )
target_include_directories(device_selection_test PRIVATE catch2)

target_link_libraries(device_selection_test cuq)
add_test(
        NAME device_selection_test
        COMMAND device_selection_test
)
