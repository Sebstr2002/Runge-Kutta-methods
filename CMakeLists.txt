cmake_minimum_required(VERSION 3.14)
project(hamsolver)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)

# Source files
add_library(hamsolver MODULE
    binding.cpp
    src/butcher-tableau.cpp
    src/runge-kutta.cpp
)

target_include_directories(hamsolver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(hamsolver PRIVATE pybind11::module)

# Ensure proper module naming on all platforms
set_target_properties(hamsolver PROPERTIES PREFIX "" SUFFIX ".so")
