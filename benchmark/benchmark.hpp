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
    NAIVECPU,
    FASTCPU,
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

benchmark::BenchDataSet getBenchmarkData(std::istream&);
benchmark::BenchTimes runMatching(DEVICE_TYPE, benchmark::BenchDataSet&);
bool verifyResults(const std::vector<cl_uint>&, const std::vector<cl_uint>&);
} // namespace benchmark
