cmake_minimum_required(VERSION 2.8)

project(optical_flow_test)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

INCLUDE(FindPkgConfig)

pkg_check_modules(OpenCV REQUIRED opencv)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

add_executable(optical_flow_test optical_flow_test.cpp)

target_link_libraries(optical_flow_test ${OpenCV_LIBRARIES})
