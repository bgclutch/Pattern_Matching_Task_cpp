#include "pattern_matching.hpp"
#include "benchmark.hpp"
#include "utils.hpp"
#include "config.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <CL/opencl.hpp>

int main(int argc, char** argv) {
    std::ifstream input_data;

    if (argc > 1) {
        input_data.open(argv[1]);
        std::cout << "benchmark data loaded from " << argv[1] << std::endl;
    }
    else {
        input_data.open("benchmark/default_benchmark.in");
        std::cout << "default benchmark data loaded" << std::endl;
    }

    if (!input_data.is_open()) {
        std::cerr << "Error opening input_data\n";
        return EXIT_FAILURE;
    }

    benchmark::BenchDataSet dataSet = benchmark::getBenchmarkData(input_data);

    try {
//-------------------------------naive kernel benchmark--------------------------------------//
    benchmark::BenchDataSet naiveData = dataSet;
    benchmark::BenchTimes resultNaive = benchmark::runMatching(benchmark::DEVICE_TYPE::NAIVEGPU, naiveData);
    benchmark::printRes("naive GPU", resultNaive);

//-------------------------------fast kernel benchmark---------------------------------------//
    benchmark::BenchDataSet fastData = dataSet;
    benchmark::BenchTimes resultFast = benchmark::runMatching(benchmark::DEVICE_TYPE::FASTGPU, fastData);
    benchmark::printRes("fast GPU", resultFast);

//-------------------------------cpu matching benchmark--------------------------------------//
    benchmark::BenchDataSet data = dataSet;
    benchmark::BenchTimes resultCPU = benchmark::runMatching(benchmark::DEVICE_TYPE::CPU, data);
    benchmark::printRes("matching CPU", resultCPU);
    } catch (const std::runtime_error& e) {
        std::cerr << "Standard Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "Critical error!" << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Unknown critical error!" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

namespace benchmark {
benchmark::BenchDataSet getBenchmarkData(std::istream& input_data) {
    size_t size;
    input_data >> size;

    if (!input_data.good() || size <= 0) {
        std::cerr << "wrong string size";
        throw std::runtime_error("wrong input");
    }

    std::string data;
    input_data >> data;

    size_t patternsCount;
    input_data >> patternsCount;

    if (!input_data.good() || size <= 0) {
        std::cerr << "wrong patterns amount";
        throw std::runtime_error("wrong input");
    }

    std::vector<cl_uint> lengths(patternsCount);
    std::vector<std::string> patterns(patternsCount);

    for (size_t i = 0; i != patternsCount; ++i) {
        input_data >> lengths[i];
        if (!input_data.good()) {
            std::cerr << "wrong len\n" << lengths[i] << "\n";
            throw std::runtime_error("wrong input");
        }

        if (!(input_data >> patterns[i])) {
            std::cerr << "wrong pattern\n" << patterns[i];
            throw std::runtime_error("wrong input");
        }
    }

    match::FlatPatterns patternSoA(patterns, lengths, patternsCount);
    BenchDataSet dataSet{data, patternSoA};

    return dataSet;
}

benchmark::BenchTimes runMatching(DEVICE_TYPE deviceType, benchmark::BenchDataSet& dataSet) {
    benchmark::BenchTimes result{};

    if (deviceType == DEVICE_TYPE::CPU) {
        result = match::cpu::findMatchesCPU(dataSet.patterns, dataSet.data);
    }
    else {
        if (deviceType == DEVICE_TYPE::NAIVEGPU) {
            result = match::gpu::findMatchesGPU(ocl_utils::Kernel_Names::naive, dataSet.patterns, dataSet.data);
        }
        else if (deviceType == DEVICE_TYPE::FASTGPU) {
            result = match::gpu::findMatchesGPU(ocl_utils::Kernel_Names::fast, dataSet.patterns, dataSet.data);
        }
        else {
            throw std::runtime_error("wrong matcher called");
        }
    }
    return result;
}
} // namespace benchmark