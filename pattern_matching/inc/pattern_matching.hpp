#pragma once
#include "utils.hpp"
#include "config.hpp"
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <CL/opencl.hpp>

namespace match {
class FlatPatterns {
 private:
    std::string string_;
    std::string patterns_;
    size_t patternsAmount_;
    std::vector<size_t> lengths_;
    std::vector<size_t> offsets_;
    std::vector<size_t> matches_;

 public:
    FlatPatterns(const std::string& string, const std::vector<std::string>& patterns,
                 const std::vector<size_t>& lengths, size_t& patternsAmount) :
        string_{string},
        patterns_{createPatterns(patterns)},
        patternsAmount_(patternsAmount),
        lengths_{lengths},
        offsets_{createOffsets(lengths_, patternsAmount_)},
        matches_{createMatches(patternsAmount_)} {}

    std::vector<size_t> getMatches() noexcept{
        return matches_;
    }

    const std::vector<size_t> getMatches() const noexcept {
        return matches_;
    }

    size_t getAmount() {
        return patternsAmount_;
    }

    size_t getAmount() const {
        return patternsAmount_;
    }

 private:
    std::string createPatterns(const std::vector<std::string>& patterns) {
        std::string data;

        for (auto& pattern : patterns)
            data += pattern;

        return data;
    }

    std::vector<size_t> createOffsets(const std::vector<size_t>& length, const size_t size) {
        std::vector<size_t> offsets;
        offsets.reserve(size);
        size_t tmp = 0;

        for (auto& len: length) {
            offsets.push_back(tmp);
            tmp += len;
        }

        return offsets;
    }

    std::vector<size_t> createMatches(const size_t size) {
        std::vector<size_t> matches;
        matches.resize(size);

        return matches;
    }

 public:
    void findMatches() {
        for (size_t i = 0; i < patternsAmount_; ++i) {
                std::string curPattern = patterns_.substr(offsets_[i], lengths_[i]);
                matches_[i] = matchPatterns(string_, curPattern);
            }
    }

 private:
    std::vector<size_t> createShiftTable(const std::string& patternData, const size_t patternSize) {
        std::vector<size_t> table(DICT_SIZE, std::numeric_limits<size_t>::max());
        auto pattern = reinterpret_cast<const unsigned char*>(patternData.data());

        for (size_t i = 0; i < patternSize; ++i)
            table[pattern[i]] = patternSize - 1 - i;

        return table;
    }

    size_t matchPatterns(const std::string& stringData, const std::string& patternData) {
        size_t stringSize = stringData.size();
        size_t patternSize = patternData.size();

        if (patternSize > stringSize)
            return 0;

        std::vector<size_t> table = createShiftTable(patternData, patternSize);
        size_t matches = 0;

        auto string = reinterpret_cast<const unsigned char*>(stringData.data());
        auto pattern = reinterpret_cast<const unsigned char*>(patternData.data());

        for (size_t shift = 0; shift <= (stringSize - patternSize); ) {
            size_t j = patternSize - 1;
            for (; j != std::numeric_limits<size_t>::max() && pattern[j] == string[shift + j]; --j) {}

            if (j == std::numeric_limits<size_t>::max()) {
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
};


inline std::ostream& operator<<(std::ostream& outStream, const match::FlatPatterns& patterns) {
    size_t size = patterns.getAmount();
    auto matches = patterns.getMatches();

    for (size_t i = 0; i < size; ++i)
        outStream << i + 1 << " " << matches[i] << "\n";

    return outStream;
}
} // namespace pattern