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
    benchmark::BenchDataSet naiveGPUData = dataSet;
    benchmark::BenchTimes resultNaiveGPU = benchmark::runMatching(benchmark::DEVICE_TYPE::NAIVEGPU, naiveGPUData);
    benchmark::printRes("naive GPU", resultNaiveGPU);

//-------------------------------fast kernel benchmark---------------------------------------//
    benchmark::BenchDataSet fastGPUData = dataSet;
    benchmark::BenchTimes resultFastGPU = benchmark::runMatching(benchmark::DEVICE_TYPE::FASTGPU, fastGPUData);
    benchmark::printRes("fast GPU", resultFastGPU);

//-------------------------------naive cpu matching benchmark--------------------------------//
    benchmark::BenchDataSet naiveCPUData = dataSet;
    benchmark::BenchTimes resultNaiveCPU = benchmark::runMatching(benchmark::DEVICE_TYPE::NAIVECPU, naiveCPUData);
    benchmark::printRes("Naive CPU", resultNaiveCPU);

//-------------------------------fast cpu matching benchmark---------------------------------//
    benchmark::BenchDataSet fastCPUData = dataSet;
    benchmark::BenchTimes resultFastCPU = benchmark::runMatching(benchmark::DEVICE_TYPE::FASTCPU, fastCPUData);
    benchmark::printRes("Fast CPU", resultFastCPU);
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

    if (deviceType == DEVICE_TYPE::NAIVECPU || deviceType == DEVICE_TYPE::FASTCPU) {
        if (deviceType == DEVICE_TYPE::NAIVECPU) {
            result = match::cpu::findMatchesCPU(ocl_utils::CPU_Names::naive, dataSet.patterns, dataSet.data);
        }
        else if (deviceType == DEVICE_TYPE::FASTCPU) {
            result = match::cpu::findMatchesCPU(ocl_utils::CPU_Names::fast, dataSet.patterns, dataSet.data);
        }
        else {
            throw std::runtime_error("wrong GPU matcher called");
        }
    }
    else {
        if (deviceType == DEVICE_TYPE::NAIVEGPU) {
            result = match::gpu::findMatchesGPU(ocl_utils::Kernel_Names::naive, dataSet.patterns, dataSet.data);
        }
        else if (deviceType == DEVICE_TYPE::FASTGPU) {
            result = match::gpu::findMatchesGPU(ocl_utils::Kernel_Names::fast, dataSet.patterns, dataSet.data);
        }
        else {
            throw std::runtime_error("wrong GPU matcher called");
        }
    }
    return result;
}
} // namespace benchmark