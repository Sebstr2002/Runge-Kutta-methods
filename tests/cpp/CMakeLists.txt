include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.4
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

add_executable(hamsolver_tests
    test_butcher_tableau.cpp
    test_runge_kutta.cpp
)
target_link_libraries(hamsolver_tests PRIVATE hamsolver_core Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(hamsolver_tests)
