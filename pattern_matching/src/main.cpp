#include "utils.hpp"
#include "config.hpp"
#include "pattern_matching.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <CL/opencl.hpp>

int main() {
    size_t size;
    std::cin >> size;

    if (!std::cin.good() || size <= 0) {
        std::cerr << "wrong string size";
        return EXIT_FAILURE;
    }

    std::string data;
    std::cin >> data;

    size_t patternsCount;
    std::cin >> patternsCount;

    if (!std::cin.good() || size <= 0) {
        std::cerr << "wrong patterns amount";
        return EXIT_FAILURE;
    }

    std::vector<size_t> lengths(patternsCount);
    std::vector<std::string> patterns(patternsCount);

    for (size_t i = 0; i != patternsCount; ++i) {
        std::cin >> lengths[i];
        if (!std::cin.good()) {
            std::cerr << "wrong len\n" << lengths[i] << "\n";
            return EXIT_FAILURE;
        }

        if (!(std::cin >> patterns[i])) {
            std::cerr << "wrong pattern\n" << patterns[i];
            return EXIT_FAILURE;
        }
    }

    match::FlatPatterns patternSoA(data, patterns, lengths, patternsCount);
    patternSoA.findMatches();

    std::cout << patternSoA;

    return EXIT_SUCCESS;
}