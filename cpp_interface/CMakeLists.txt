cmake_minimum_required(VERSION 3.10)
project(ML_test)

set(CMAKE_PREFIX_PATH "/usr/local/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(test_torch_model test_torch_model.cpp)
target_link_libraries(test_torch_model "${TORCH_LIBRARIES}")
set_property(TARGET test_torch_model PROPERTY CXX_STANDARD 14)
