#pragma once
#include "pattern_matching.hpp"
#include "utils.hpp"
#include "config.hpp"
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <CL/opencl.hpp>

namespace benchmark {
enum class DEVICE_TYPE {
    CPU,
    NAIVEGPU,
    FASTGPU
};

struct BenchDataSet {
    std::string data;
    match::FlatPatterns patterns;
};

void printRes(const std::string& nameRes, const BenchTimes& res) {
    std::cout << nameRes << " {\n"
              << "CPU time:      " << res.CPUTime.count() / 1000.      << " us\n"
              << "Wall time:     " << res.WallTime.count() / 1000.     << " us\n"
              << "Kernel time:   " << res.kernelTime.count() / 1000.   << " us\n"
              << "Transfer time: " << res.TransferTime.count() / 1000. << " us\n}\n";
}

benchmark::BenchDataSet getBenchmarkData(std::istream& input_data);
benchmark::BenchTimes runMatching(DEVICE_TYPE deviceType, benchmark::BenchDataSet& dataSet);
} // namespace benchmark
