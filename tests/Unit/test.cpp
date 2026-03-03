#include "pattern_matching.hpp"
#include "utils.hpp"
#include "config.hpp"
#include <string>
#include <vector>
#include <CL/opencl.hpp>
#include <gtest/gtest.h>

TEST(NaiveGpu, first_test) {
    cl_uint size = 11;
    std::string str = "abracadabra";
    size_t patternsAmount = 3;
    std::vector<cl_uint> lengths{3, 5, 2};
    std::vector<std::string> patterns{"rac", "barac", "ab"};
    match::FlatPatterns patternsSoA(patterns, lengths, patternsAmount);

    match::gpu::findMatchesGPU(ocl_utils::Kernel_Names::naive, patternsSoA, str);
    auto matches = patternsSoA.getMatches();

    ASSERT_EQ(matches[2], 2);
}

TEST(NaiveGpu, second_test) {
    cl_uint size = 20;
    std::string str = "SnnPyX,?T&/'*:j`Mn}s";
    size_t patternsAmount = 5;
    std::vector<cl_uint> lengths{3, 5, 2, 10, 1};
    std::vector<std::string> patterns{"42!", "SnnPyl", "{s", "X,?T&/'*:j", "n"};
    match::FlatPatterns patternsSoA(patterns, lengths, patternsAmount);

    match::gpu::findMatchesGPU(ocl_utils::Kernel_Names::naive, patternsSoA, str);
    auto matches = patternsSoA.getMatches();
    std::cerr << patternsSoA;

    ASSERT_EQ(matches[4], 3);
}

TEST(NaiveGpu, third_test) {
    cl_uint size = 24;
    std::string str = "Us$3$ZhUD>U&0YmfgUs$3$Zj";
    size_t patternsAmount = 3;
    std::vector<cl_uint> lengths{5, 3, 2};
    std::vector<std::string> patterns{"$3$Zh", "Us$", "mf"};
    match::FlatPatterns patternsSoA(patterns, lengths, patternsAmount);

    match::gpu::findMatchesGPU(ocl_utils::Kernel_Names::naive, patternsSoA, str);
    auto matches = patternsSoA.getMatches();

    ASSERT_EQ(matches[0], 1);
}

TEST(FastGpu, first_test) {
    cl_uint size = 11;
    std::string str = "abracadabra";
    size_t patternsAmount = 3;
    std::vector<cl_uint> lengths{3, 5, 2};
    std::vector<std::string> patterns{"rac", "barac", "ab"};
    match::FlatPatterns patternsSoA(patterns, lengths, patternsAmount);

    match::gpu::findMatchesGPU(ocl_utils::Kernel_Names::fast, patternsSoA, str);
    auto matches = patternsSoA.getMatches();

    ASSERT_EQ(matches[2], 2);
}

TEST(FastGpu, second_test) {
    cl_uint size = 20;
    std::string str = "Sz!PyX,?T&/'*:j`Mn}s";
    size_t patternsAmount = 3;
    std::vector<cl_uint> lengths{3, 5, 2, 10, 1};
    std::vector<std::string> patterns{"42!", "Sz!Pyl", "{s", "X,?T&/'*:j", "n"};
    match::FlatPatterns patternsSoA(patterns, lengths, patternsAmount);

    match::gpu::findMatchesGPU(ocl_utils::Kernel_Names::fast, patternsSoA, str);
    auto matches = patternsSoA.getMatches();

    ASSERT_EQ(matches[4], 1);
}

TEST(FastGpu, third_test) {
    cl_uint size = 24;
    std::string str = "AMH3$ZhUD>U&0Ymfg%Twfmf";
    size_t patternsAmount = 3;
    std::vector<cl_uint> lengths{5, 3, 2};
    std::vector<std::string> patterns{"$3$Zh", "Us$", "mf"};
    match::FlatPatterns patternsSoA(patterns, lengths, patternsAmount);

    match::gpu::findMatchesGPU(ocl_utils::Kernel_Names::fast, patternsSoA, str);
    auto matches = patternsSoA.getMatches();

    ASSERT_EQ(matches[2], 2);
}

int main (int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}