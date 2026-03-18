cmake_minimum_required(VERSION 3.13)
project(pattern_matching)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

include(cmake/Dependencies.cmake)
find_package(OpenCL REQUIRED)

add_subdirectory(pattern_matching)
add_subdirectory(tests)

option(ENABLE_ASAN OFF)
option(ENABLE_BENCHMARK OFF)

if(ENABLE_BENCHMARK)
    add_subdirectory(benchmark)
endif()

target_compile_options(pattern_matching PRIVATE
$<$<CONFIG:Debug>:
    -Wall
    -Wextra
    -Wpedantic
    -DDEBUG
    -Wmissing-declarations
    -Wcast-align
    -Wshadow
    -Wsign-conversion>
)

if(ENABLE_ASAN)
    foreach(target pattern_matching)
        target_compile_options(${target} PRIVATE $<$<CONFIG:Debug>: -fsanitize=address>)
        target_link_options(${target} PRIVATE $<$<CONFIG:Debug>: -fsanitize=address>)
    endforeach()
endif()