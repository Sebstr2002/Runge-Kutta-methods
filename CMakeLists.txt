cmake_minimum_required(VERSION 3.14)
project(hamsolver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type fallback so single-config generators don't ship a debug build.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(HAMSOLVER_BUILD_TESTS "Build the C++ unit tests" ON)

find_package(pybind11 REQUIRED)

# --- Core integrator as a static library so tests and the python module share it ---
add_library(hamsolver_core STATIC
    src/butcher-tableau.cpp
    src/runge-kutta.cpp
    src/physics.cpp
)
target_include_directories(hamsolver_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_target_properties(hamsolver_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- Python module ---
add_library(hamsolver MODULE binding.cpp)
target_link_libraries(hamsolver PRIVATE hamsolver_core pybind11::module)

# Ensure proper module naming on all platforms (see CLAUDE.md)
set_target_properties(hamsolver PROPERTIES PREFIX "" SUFFIX ".so")

# --- Tests ---
if(HAMSOLVER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)

    # Wire pytest into ctest. Skipped automatically if pytest is missing.
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME python_tests
            COMMAND ${Python3_EXECUTABLE} -m pytest
                    ${CMAKE_SOURCE_DIR}/tests/python -q
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        )
        # Make the freshly built module importable in the pytest run.
        set_tests_properties(python_tests PROPERTIES
            ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:hamsolver>"
        )
    endif()
endif()
