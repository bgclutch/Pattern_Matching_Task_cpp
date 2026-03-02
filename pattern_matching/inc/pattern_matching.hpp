#pragma once
#include "utils.hpp"
#include "config.hpp"
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <CL/opencl.hpp>

namespace match {
namespace detail {
std::vector<cl_uint> createShiftTable(const std::string&, const size_t);
} // namespace detail

class FlatPatterns final {
 private:
    std::string patterns_;
    size_t patternsAmount_;
    std::vector<cl_uint> lengths_;
    std::vector<cl_uint> offsets_;
    std::vector<cl_uint> matches_;

 public:
    FlatPatterns(const std::vector<std::string>& patterns,
                 const std::vector<cl_uint>& lengths, size_t& patternsAmount) :
        patterns_{createPatterns(patterns)},
        patternsAmount_(patternsAmount),
        lengths_{lengths},
        offsets_{createOffsets(lengths_, patternsAmount_)},
        matches_{createMatches(patternsAmount_)} {}

    const std::vector<cl_uint>& getMatches() const noexcept {
        return matches_;
    }

    size_t getAmount() {
        return patternsAmount_;
    }

    size_t getAmount() const {
        return patternsAmount_;
    }

    size_t getPatternsLen() {
        return patterns_.size();
    }

    size_t getPatternsLen() const {
        return patterns_.size();
    }

    const std::string& getPatterns() const {
        return patterns_;
    }

    const std::vector<cl_uint>& getLengths() const {
        return lengths_;
    }

    const std::vector<cl_uint>& getOffsets() const {
        return offsets_;
    }

    void setMatches(std::vector<cl_uint>&& matches) {
        matches_ = std::move(matches);
    }

 private:
    std::string createPatterns(const std::vector<std::string>& patterns) {
        std::string data;

        for (auto& pattern : patterns)
            data += pattern;

        return data;
    }

    std::vector<cl_uint> createOffsets(const std::vector<cl_uint>& length, const size_t size) {
        std::vector<cl_uint> offsets;
        offsets.reserve(size);
        size_t tmp = 0;

        for (auto& len: length) {
            offsets.push_back(tmp);
            tmp += len;
        }
        return offsets;
    }

    std::vector<cl_uint> createMatches(const size_t size) {
        std::vector<cl_uint> matches;
        matches.resize(size);
        return matches;
    }
};

inline std::ostream& operator<<(std::ostream& outStream, const match::FlatPatterns& patterns) {
    size_t size = patterns.getAmount();
    auto matches = patterns.getMatches();

    for (size_t i = 0; i < size; ++i)
        outStream << i + 1 << " " << matches[i] << "\n";

    return outStream;
}

namespace cpu {
void findMatchesCPU(FlatPatterns&, const std::string&);
size_t matchPatterns(const std::string&, const std::string&);
} // namespace cpu

namespace gpu {
void findMatchesGPU(const ocl_utils::Kernel_Names&, FlatPatterns&, const std::string&);
void naiveMatching(ocl_utils::Environment&, FlatPatterns&, const std::string&);
void fastMatching(ocl_utils::Environment&, FlatPatterns&);
} // namespace gpu
} // namespace match