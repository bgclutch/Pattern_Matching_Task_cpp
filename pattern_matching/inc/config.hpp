#pragma once
#include <string>

namespace config {
#ifdef KERNELS_ABS_PATH
    const std::string KERNELS_PATH = KERNELS_ABS_PATH;
#else
    const std::string KERNELS_PATH = "kernels/";
#endif

const std::string NAIVE_PATTERN_KERNEL           = "naive_kernel.cl";
const std::string NAIVE_PATTERN_KERNEL_NAME      = "naive_pattern_kernel";
const std::string FAST_PATTERN_KERNEL            = "fast_kernel.cl";
const std::string FAST_PATTERN_KERNEL_NAME       = "fast_pattern_kernel";
} // namespace config