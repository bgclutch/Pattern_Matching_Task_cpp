#include "pattern_matching.hpp"
#include "config.hpp"
#include "utils.hpp"
#include <chrono>

namespace match {
namespace detail {
std::vector<size_t> createShiftTable(const std::string& patternData, const size_t patternSize) {
    std::vector<size_t> table(match::DICT_SIZE, patternSize);
    auto pattern = reinterpret_cast<const unsigned char*>(patternData.data());

    for (size_t i = 0; i < patternSize - 1; ++i)
        table[pattern[i]] = patternSize - 1 - i;

    return table;
}
} // namespace detail

namespace cpu {
benchmark::BenchTimes findMatchesCPU(FlatPatterns& patternSoA, const std::string& string) {
    auto offsets = patternSoA.getOffsets();
    auto lengths = patternSoA.getLengths();
    auto amount = patternSoA.getAmount();
    auto patterns = patternSoA.getPatterns();
    std::vector<cl_uint> matches(amount, 0);

    benchmark::BenchTimes result{};
    auto begin = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < amount; ++i) {
            std::string curPattern = patterns.substr(offsets[i], lengths[i]);
            matches[i] = detail::matchPatterns(string, curPattern);
        }
    patternSoA.setMatches(std::move(matches));

    auto end = std::chrono::high_resolution_clock::now();
    result.CPUTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    return result;
}

namespace detail {
size_t matchPatterns(const std::string& stringData, const std::string& patternData) {
    size_t stringSize = stringData.size();
    size_t patternSize = patternData.size();

    if (patternSize > stringSize || patternSize == 0)
        return 0;

    std::vector<size_t> table = match::detail::createShiftTable(patternData, patternSize);
    size_t matches = 0;

    auto text = reinterpret_cast<const unsigned char*>(stringData.data());
    auto pattern = reinterpret_cast<const unsigned char*>(patternData.data());

    size_t shift = 0;
    while (shift <= (stringSize - patternSize)) {
        size_t j = patternSize - 1;

        while (j < patternSize && pattern[j] == text[shift + j]) {
            --j;
        }

        if (j >= patternSize) {
            matches++;
        }

        shift += table[text[shift + patternSize - 1]];
    }
    return matches;
}
} // namespace cpu::detail
} // namespace cpu

namespace gpu {
benchmark::BenchTimes findMatchesGPU(const ocl_utils::Kernel_Names& currentKernel, FlatPatterns& patternSoA, const std::string& string) {
    benchmark::BenchTimes result{};
    if (currentKernel == ocl_utils::Kernel_Names::naive) {
        ocl_utils::Environment env(config::KERNELS_PATH + config::NAIVE_PATTERN_KERNEL, config::NAIVE_PATTERN_KERNEL_NAME);
        result = detail::naiveMatching(env, patternSoA, string);
    }
    else if (currentKernel == ocl_utils::Kernel_Names::fast) {
        ocl_utils::Environment env(config::KERNELS_PATH + config::FAST_PATTERN_KERNEL, config::FAST_PATTERN_KERNEL_NAME);
        result = detail::fastMatching(env, patternSoA, string);
    }
    else {
        throw std::runtime_error("no kernel");
    }
    return result;
}

namespace detail {
benchmark::BenchTimes naiveMatching(ocl_utils::Environment& env, FlatPatterns& patternSoA, const std::string& string) {
    size_t stringLen = string.size();
    size_t amount = patternSoA.getAmount();
    cl::Context& curContext = env.get_context();
    std::vector<cl_uint> tmpMatches(amount, 0);
    void* matchesHost = tmpMatches.data();
    size_t bytes = amount * sizeof(cl_uint);

    benchmark::BenchTimes result{};
    auto matchCall = cl::KernelFunctor<cl::Buffer, cl_uint, cl::Buffer, cl_uint,
                                       cl::Buffer, cl::Buffer, cl::Buffer>(env.get_kernel());

    auto wall_begin = std::chrono::high_resolution_clock::now();
    auto transfer_begin = std::chrono::high_resolution_clock::now();

    cl::Buffer string_buf   = ocl_utils::createBuffer(curContext, string);
    cl::Buffer patterns_buf = ocl_utils::createBuffer(curContext, patternSoA.getPatterns());
    cl::Buffer lengths_buf  = ocl_utils::createBuffer(curContext, patternSoA.getLengths());
    cl::Buffer offsets_buf  = ocl_utils::createBuffer(curContext, patternSoA.getOffsets());
    cl::Buffer matches_buf  = ocl_utils::createBuffer(curContext, patternSoA.getMatches());

    auto transfer_end = std::chrono::high_resolution_clock::now();
    result.TransferTime += std::chrono::duration_cast<std::chrono::nanoseconds>(transfer_end - transfer_begin);

    auto kernel_begin = std::chrono::high_resolution_clock::now();

    matchCall(cl::EnqueueArgs(env.get_queue(), cl::NDRange(stringLen)),
            string_buf,
            stringLen,
            patterns_buf,
            amount,
            lengths_buf,
            offsets_buf,
            matches_buf);

    env.get_queue().finish();
    auto kernel_end = std::chrono::high_resolution_clock::now();
    result.kernelTime = std::chrono::duration_cast<std::chrono::nanoseconds>(kernel_end - kernel_begin);

    transfer_begin = std::chrono::high_resolution_clock::now();

    env.get_queue().enqueueReadBuffer(matches_buf, CL_TRUE, 0,
                                      bytes, matchesHost);
    patternSoA.setMatches(std::move(tmpMatches));

    transfer_end = std::chrono::high_resolution_clock::now();
    result.TransferTime += std::chrono::duration_cast<std::chrono::nanoseconds>(transfer_end - transfer_begin);

    auto wall_end = std::chrono::high_resolution_clock::now();
    result.WallTime = std::chrono::duration_cast<std::chrono::nanoseconds>(wall_end - wall_begin);
    return result;
}

benchmark::BenchTimes fastMatching(ocl_utils::Environment& env, match::FlatPatterns& patternSoA, const std::string& string) {
    cl::Context& context = env.get_context();
    size_t stringLen = string.size();
    size_t amount = patternSoA.getAmount();
    const auto& lengths = patternSoA.getLengths();

    cl_uint maxPatternLen = 0;

    for (auto len : lengths) {
        if (len > maxPatternLen)
            maxPatternLen = len;
    }

    auto device = env.get_device();
    size_t maxDeviceWorkItemSize = env.get_device().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    size_t maxKernelWorkItemSize = env.get_kernel().getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
    cl_ulong maxLocalMemSize = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    size_t localSize = std::min(maxDeviceWorkItemSize, maxKernelWorkItemSize);
    size_t bytes = amount * sizeof(cl_int);

    while (localSize > 0 && (localSize + maxPatternLen) * sizeof(char) > maxLocalMemSize) {
        localSize /= 2;
    }

    size_t globalSize = ((stringLen + localSize - 1) / localSize) * localSize;
    size_t localMemArgSize = (localSize + maxPatternLen) * sizeof(char);
    std::vector<cl_uint> tmpMatches(amount, 0);

    auto fastCall = cl::KernelFunctor<cl::Buffer, cl_uint, cl::Buffer,
                                      cl_uint, cl::Buffer, cl::Buffer,
                                      cl::Buffer, cl::LocalSpaceArg, cl_uint>(env.get_kernel());

    benchmark::BenchTimes result{};
    auto wall_begin = std::chrono::high_resolution_clock::now();
    auto transfer_begin = std::chrono::high_resolution_clock::now();

    cl::Buffer string_buf   = ocl_utils::createBuffer(context, string);
    cl::Buffer patterns_buf = ocl_utils::createBuffer(context, patternSoA.getPatterns());
    cl::Buffer lengths_buf  = ocl_utils::createBuffer(context, lengths);
    cl::Buffer offsets_buf  = ocl_utils::createBuffer(context, patternSoA.getOffsets());
    cl::Buffer matches_buf  = ocl_utils::createBuffer(context, tmpMatches);

    auto transfer_end = std::chrono::high_resolution_clock::now();
    result.TransferTime += std::chrono::duration_cast<std::chrono::nanoseconds>(transfer_end - transfer_begin);

    auto kernel_begin = std::chrono::high_resolution_clock::now();

    fastCall(cl::EnqueueArgs(env.get_queue(), cl::NDRange(globalSize), cl::NDRange(localSize)),
            string_buf, stringLen,
            patterns_buf, amount,
            lengths_buf, offsets_buf, matches_buf,
            cl::Local(localMemArgSize),
            (cl_uint)maxPatternLen);

    env.get_queue().finish();
    auto kernel_end = std::chrono::high_resolution_clock::now();
    result.kernelTime = std::chrono::duration_cast<std::chrono::nanoseconds>(kernel_end - kernel_begin);

    transfer_begin = std::chrono::high_resolution_clock::now();
    env.get_queue().enqueueReadBuffer(matches_buf, CL_TRUE, 0, bytes, tmpMatches.data());
    patternSoA.setMatches(std::move(tmpMatches));
    transfer_end = std::chrono::high_resolution_clock::now();

    result.TransferTime += std::chrono::duration_cast<std::chrono::nanoseconds>(transfer_end - transfer_begin);

    auto wall_end = std::chrono::high_resolution_clock::now();
    result.WallTime = std::chrono::duration_cast<std::chrono::nanoseconds>(wall_end - wall_begin);
    return result;
}
} // namespace gpu::detail
} // namespace gpu
} // namespace match