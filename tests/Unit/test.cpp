#include "pattern_matching.hpp"
#include "utils.hpp"
#include "config.hpp"
#include <string>
#include <vector>
#include <CL/opencl.hpp>
#include <gtest/gtest.h>

bool verifyResults(const std::vector<cl_uint>&, const std::vector<cl_uint>&);

TEST(NAIVECPU, first_test) {
    cl_uint size = 11;
    std::string str = "abracadabra";
    size_t patternsAmount = 3;
    std::vector<cl_uint> lengths{3, 5, 2};
    std::vector<cl_uint> ans{1, 0, 2};
    std::vector<std::string> patterns{"rac", "barac", "ab"};
    match::FlatPatterns patternsSoA(patterns, lengths, patternsAmount);

    match::cpu::findMatchesCPU(ocl_utils::CPU_Names::naive, patternsSoA, str);
    auto matches = patternsSoA.getMatches();

    ASSERT_EQ(matches[2], 2);
}

TEST(NAIVECPU, second_test) {
    cl_uint size = 100;
    std::string str = "9ngqSi5LxnifvvDaxLUjdqzajlgfkeCeAns7MqjVpVxsl5QnV9Fr7BLNv2ih0LUqNmEvLUpsUnOnEFdTufY0RcXtoKA465DcLUj6";
    size_t patternsAmount = 5;
    std::vector<cl_uint> lengths{3, 3, 2, 6, 1};
    std::vector<cl_uint> ans{0, 1, 4, 1, 4};
    std::vector<std::string> patterns{"42!", "Ans", "LU", "MqjVpV", "v"};
    match::FlatPatterns patternsSoA(patterns, lengths, patternsAmount);

    match::cpu::findMatchesCPU(ocl_utils::CPU_Names::naive, patternsSoA, str);
    auto matches = patternsSoA.getMatches();

    ASSERT_EQ(matches[2], 4);
}

TEST(NAIVECPU, third_test) {
    cl_uint size = 24;
    std::string str = "Us$3$ZhUD>U&0YmfgUs$3$Zj";
    size_t patternsAmount = 3;
    std::vector<cl_uint> lengths{5, 3, 2};
    std::vector<cl_uint> ans{1, 2, 1};
    std::vector<std::string> patterns{"$3$Zh", "Us$", "mf"};
    match::FlatPatterns patternsSoA(patterns, lengths, patternsAmount);

    match::cpu::findMatchesCPU(ocl_utils::CPU_Names::naive, patternsSoA, str);
    auto matches = patternsSoA.getMatches();

    ASSERT_EQ(matches[0], 1);
}

TEST(FASTCPU, first_test) {
    cl_uint size = 11;
    std::string str = "abracadabra";
    size_t patternsAmount = 3;
    std::vector<cl_uint> lengths{3, 5, 2};
    std::vector<cl_uint> ans{1, 0, 2};
    std::vector<std::string> patterns{"rac", "barac", "ab"};
    match::FlatPatterns patternsSoA(patterns, lengths, patternsAmount);

    match::cpu::findMatchesCPU(ocl_utils::CPU_Names::fast, patternsSoA, str);
    auto matches = patternsSoA.getMatches();

    ASSERT_TRUE(verifyResults(ans, patternsSoA.getMatches()));

}

TEST(FASTCPU, second_test) {
    cl_uint size = 100;
    std::string str = "9ngqSi5LxnifvvDaxLUjdqzajlgfkeCeAns7MqjVpVxsl5QnV9Fr7BLNv2ih0LUqNmEvLUpsUnOnEFdTufY0RcXtoKA465DcLUj6";
    size_t patternsAmount = 5;
    std::vector<cl_uint> lengths{3, 3, 2, 6, 1};
    std::vector<cl_uint> ans{0, 1, 4, 1, 4};
    std::vector<std::string> patterns{"42!", "Ans", "LU", "MqjVpV", "v"};
    match::FlatPatterns patternsSoA(patterns, lengths, patternsAmount);

    match::cpu::findMatchesCPU(ocl_utils::CPU_Names::fast, patternsSoA, str);
    auto matches = patternsSoA.getMatches();

    ASSERT_TRUE(verifyResults(ans, patternsSoA.getMatches()));
}

TEST(FASTCPU, third_test) {
    cl_uint size = 24;
    std::string str = "Us$3$ZhUD>U&0YmfgUs$3$Zj";
    size_t patternsAmount = 3;
    std::vector<cl_uint> lengths{5, 3, 2};
    std::vector<cl_uint> ans{1, 2, 1};
    std::vector<std::string> patterns{"$3$Zh", "Us$", "mf"};
    match::FlatPatterns patternsSoA(patterns, lengths, patternsAmount);

    match::cpu::findMatchesCPU(ocl_utils::CPU_Names::fast, patternsSoA, str);
    auto matches = patternsSoA.getMatches();

    ASSERT_TRUE(verifyResults(ans, patternsSoA.getMatches()));
}

TEST(NaiveGpu, first_test) {
    cl_uint size = 11;
    std::string str = "abracadabra";
    size_t patternsAmount = 3;
    std::vector<cl_uint> lengths{3, 5, 2};
    std::vector<cl_uint> ans{1, 0, 2};
    std::vector<std::string> patterns{"rac", "barac", "ab"};
    match::FlatPatterns patternsSoA(patterns, lengths, patternsAmount);

    match::gpu::findMatchesGPU(ocl_utils::Kernel_Names::naive, patternsSoA, str);
    auto matches = patternsSoA.getMatches();

    ASSERT_TRUE(verifyResults(ans, patternsSoA.getMatches()));
}

TEST(NaiveGpu, second_test) {
    cl_uint size = 100;
    std::string str = "9ngqSi5LxnifvvDaxLUjdqzajlgfkeCeAns7MqjVpVxsl5QnV9Fr7BLNv2ih0LUqNmEvLUpsUnOnEFdTufY0RcXtoKA465DcLUj6";
    size_t patternsAmount = 5;
    std::vector<cl_uint> lengths{3, 3, 2, 6, 1};
    std::vector<cl_uint> ans{0, 1, 4, 1, 4};
    std::vector<std::string> patterns{"42!", "Ans", "LU", "MqjVpV", "v"};
    match::FlatPatterns patternsSoA(patterns, lengths, patternsAmount);

    match::gpu::findMatchesGPU(ocl_utils::Kernel_Names::naive, patternsSoA, str);
    auto matches = patternsSoA.getMatches();

    ASSERT_TRUE(verifyResults(ans, patternsSoA.getMatches()));
}

TEST(NaiveGpu, third_test) {
    cl_uint size = 24;
    std::string str = "Us$3$ZhUD>U&0YmfgUs$3$Zj";
    size_t patternsAmount = 3;
    std::vector<cl_uint> lengths{5, 3, 2};
    std::vector<cl_uint> ans{1, 2, 1};
    std::vector<std::string> patterns{"$3$Zh", "Us$", "mf"};
    match::FlatPatterns patternsSoA(patterns, lengths, patternsAmount);

    match::gpu::findMatchesGPU(ocl_utils::Kernel_Names::naive, patternsSoA, str);
    auto matches = patternsSoA.getMatches();

    ASSERT_TRUE(verifyResults(ans, patternsSoA.getMatches()));
}

TEST(FastGpu, first_test) {
    cl_uint size = 11;
    std::string str = "abracadabra";
    size_t patternsAmount = 3;
    std::vector<cl_uint> lengths{3, 5, 2};
    std::vector<cl_uint> ans{1, 0, 2};
    std::vector<std::string> patterns{"rac", "barac", "ab"};
    match::FlatPatterns patternsSoA(patterns, lengths, patternsAmount);

    match::gpu::findMatchesGPU(ocl_utils::Kernel_Names::fast, patternsSoA, str);
    auto matches = patternsSoA.getMatches();

    ASSERT_TRUE(verifyResults(ans, patternsSoA.getMatches()));
}

TEST(FastGpu, second_test) {
    cl_uint size = 100;
    std::string str = "K2YcvliDn6v8bDHtMd9lln3NVCaOnNMyG9avKo28dphyvjs7eHMk1cvlP4L5XFJ5ct3dGMLq5zDi6tPbVuD8KM6WKeaaLAbnrAKd";
    size_t patternsAmount = 5;
    std::vector<cl_uint> lengths{3, 9, 3, 4, 1};
    std::vector<cl_uint> ans{0, 1, 2, 1, 4};
    std::vector<std::string> patterns{"42!", "eHMk1cvlP", "cvl", "js7e", "l"};
    match::FlatPatterns patternsSoA(patterns, lengths, patternsAmount);

    match::gpu::findMatchesGPU(ocl_utils::Kernel_Names::fast, patternsSoA, str);
    auto matches = patternsSoA.getMatches();

    ASSERT_TRUE(verifyResults(ans, patternsSoA.getMatches()));
}

TEST(FastGpu, third_test) {
    cl_uint size = 24;
    std::string str = "AMH3$ZhUD>U&0Ymfg%Twfmf";
    size_t patternsAmount = 3;
    std::vector<cl_uint> lengths{5, 3, 2};
    std::vector<cl_uint> ans{0, 0, 2};
    std::vector<std::string> patterns{"$3$Zh", "Us$", "mf"};
    match::FlatPatterns patternsSoA(patterns, lengths, patternsAmount);

    match::gpu::findMatchesGPU(ocl_utils::Kernel_Names::fast, patternsSoA, str);
    auto matches = patternsSoA.getMatches();

    ASSERT_TRUE(verifyResults(ans, patternsSoA.getMatches()));
}

int main (int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

bool verifyResults(const std::vector<cl_uint>& cpuResults, const std::vector<cl_uint>& gpuResults) {
    if (cpuResults.size() != gpuResults.size()) {
        std::cerr << "Verification FAILED: Results size mismatch!" << std::endl;
        std::cerr << "CPU size: " << cpuResults.size() << ", GPU size: " << gpuResults.size() << std::endl;
        return false;
    }

    bool isOk = true;
    for (size_t i = 0; i < cpuResults.size(); ++i) {
        if (cpuResults[i] != gpuResults[i]) {
            std::cerr << "Verification FAILED at pattern #" << (i + 1) << "!" << std::endl;
            std::cerr << "  CPU found: " << cpuResults[i] << " matches" << std::endl;
            std::cerr << "  GPU found: " << gpuResults[i] << " matches" << std::endl;
            isOk = false;
        }
    }

    if (isOk) {
        std::cerr << "Verification PASSED: CPU and GPU results are identical." << std::endl;
    }

    return isOk;
}