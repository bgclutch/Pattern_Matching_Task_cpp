add_executable(pattern_matching)
find_package(OpenCL REQUIRED)

target_compile_features(pattern_matching PUBLIC cxx_std_23)
target_link_libraries(pattern_matching PRIVATE OpenCL::OpenCL)

target_compile_definitions(pattern_matching PRIVATE
    KERNELS_ABS_PATH="${CMAKE_SOURCE_DIR}/kernels/"
    CL_TARGET_OPENCL_VERSION=200
    CL_HPP_TARGET_OPENCL_VERSION=200
)

target_include_directories(pattern_matching PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${OpenCL_INCLUDE_DIRS}
)

target_sources(pattern_matching PRIVATE
    src/main.cpp
)
