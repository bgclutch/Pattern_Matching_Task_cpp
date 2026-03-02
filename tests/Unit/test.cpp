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

#if 0
TEST(NaiveGpu, first_test) {


    ASSERT_EQ();
}

TEST(NaiveGpu, first_test) {


    ASSERT_EQ();
}
#endif


int main (int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}