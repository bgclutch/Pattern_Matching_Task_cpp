#include "pattern_matching.hpp"
#include "config.hpp"
#include "utils.hpp"

namespace detail {
std::vector<cl_uint> createShiftTable(const std::string& patternData, const size_t patternSize) {
    std::vector<cl_uint> table(match::DICT_SIZE, std::numeric_limits<cl_uint>::max());
    auto pattern = reinterpret_cast<const unsigned char*>(patternData.data());

    for (size_t i = 0; i < patternSize; ++i)
        table[pattern[i]] = patternSize - 1 - i;

    return table;
}
} // namespace detail

namespace match {
namespace cpu {
void findMatchesCPU(FlatPatterns& patternSoA, const std::string& string) {
    auto offsets = patternSoA.getOffsets();
    auto lengths = patternSoA.getLengths();
    auto amount = patternSoA.getAmount();
    auto patterns = patternSoA.getPatterns();
    std::vector<cl_uint> matches(amount, 0);
    for (size_t i = 0; i < amount; ++i) {
            std::string curPattern = patterns.substr(offsets[i], lengths[i]);
            matches[i] = matchPatterns(string, curPattern);
        }
}

size_t matchPatterns(const std::string& stringData, const std::string& patternData) {
    size_t stringSize = stringData.size();
    size_t patternSize = patternData.size();

    if (patternSize > stringSize)
        return 0;

    std::vector<cl_uint> table = detail::createShiftTable(patternData, patternSize);
    size_t matches = 0;

    auto string = reinterpret_cast<const unsigned char*>(stringData.data());
    auto pattern = reinterpret_cast<const unsigned char*>(patternData.data());

    for (size_t shift = 0; shift <= (stringSize - patternSize); ) {
        size_t j = patternSize - 1;
        for (; j != std::numeric_limits<cl_uint>::max() && pattern[j] == string[shift + j]; --j) {}

        if (j == std::numeric_limits<cl_uint>::max()) {
            matches++;
            if (shift + patternSize < stringSize) {
                size_t followingChar = string[shift + patternSize];
                shift += patternSize - table[followingChar];
            } else {
                shift += 1;
            }
        }
        else {
            size_t followingChar = table[string[shift + j]];
            shift += std::max(1ul, j - followingChar);
        }
    }
    return matches;
}
} // namespace cpu

namespace gpu {
void findMatchesGPU(const ocl_utils::Kernel_Names& currentKernel, FlatPatterns& patternSoA, const std::string& string) {
    if (currentKernel == ocl_utils::Kernel_Names::naive) {
        ocl_utils::Environment env(config::KERNELS_PATH + config::NAIVE_PATTERN_KERNEL, config::NAIVE_PATTERN_KERNEL_NAME);
        naiveMatching(env, patternSoA, string);
    }
    else if (currentKernel == ocl_utils::Kernel_Names::fast) {
        ocl_utils::Environment env(config::KERNELS_PATH + config::FAST_PATTERN_KERNEL, config::FAST_PATTERN_KERNEL_NAME);
        fastMatching(env,patternSoA);
    }
    else {
        throw std::runtime_error("no kernel");
    }
}

void naiveMatching(ocl_utils::Environment& env, FlatPatterns& patternSoA, const std::string& string) {
    size_t stringLen = string.size();
    size_t amount = patternSoA.getAmount();
    cl::Context curContext = env.get_context();

    cl::Buffer string_buf   = ocl_utils::createBuffer(curContext, string);
    cl::Buffer patterns_buf = ocl_utils::createBuffer(curContext, patternSoA.getPatterns());
    cl::Buffer lengths_buf  = ocl_utils::createBuffer(curContext, patternSoA.getLengths());
    cl::Buffer offsets_buf  = ocl_utils::createBuffer(curContext, patternSoA.getOffsets());
    cl::Buffer matches_buf  = ocl_utils::createBuffer(curContext, patternSoA.getMatches());

    auto matchCall = cl::KernelFunctor<cl::Buffer, uint, cl::Buffer, uint,
                                       cl::Buffer, cl::Buffer, cl::Buffer>(env.get_kernel());

    matchCall(cl::EnqueueArgs(env.get_queue(), cl::NDRange(stringLen)),
            string_buf,
            stringLen,
            patterns_buf,
            amount,
            lengths_buf,
            offsets_buf,
            matches_buf);

    std::vector<cl_uint> tmpMatches(amount, 0);
    void* matchesHost = tmpMatches.data();
    size_t bytes = amount * sizeof(cl_uint);

    env.get_queue().enqueueReadBuffer(matches_buf, CL_TRUE, 0,
                                      bytes, matchesHost);
    patternSoA.setMatches(std::move(tmpMatches));
}

void fastMatching(ocl_utils::Environment& env, FlatPatterns& patternSoA) {

}
} // namespace gpu
} // namespace match