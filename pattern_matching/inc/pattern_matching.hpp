#pragma once
#include <vector>
#include <string>

namespace match {
class FlatPatterns {
 private:
    std::string patterns_;
    size_t patternsAmount_;
    std::vector<size_t> lengths_;
    std::vector<size_t> offsets_;
    std::vector<size_t> matches_;

 public:
    FlatPatterns(const std::vector<std::string>& patterns, const std::vector<size_t>& lengths, size_t& patternsAmount) :
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
};

// void matchPatterns(FlatPatterns& patterns) {

// }

inline std::ostream& operator<<(std::ostream& outStream, const match::FlatPatterns& patterns) {
    size_t size = patterns.getAmount();
    auto matches = patterns.getMatches();

    for (size_t i = 0; i < size; ++i)
        outStream << i + 1 << " " << matches[i] << "\n";

    return outStream;
}
} // namespace pattern