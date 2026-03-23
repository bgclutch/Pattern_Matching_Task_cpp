# Pattern matching (HWB level 2)
This project provides an implementation of a Pattern matching algorithm in C++ and OpenCL.
It also includes automated tests comparing the results of the Pattern matching with CPU implementation of Boyer-Moore-Horspool algorithm with average O(N) complexity.

## Features:
1. An implementation of a Bitonic sort algorithm with OpenCL library
2. Comparison of results of GPU parallel algorithm and CPU-based Boyer-Moore-Horspool algorithm for correctness
3. Python scripts for automated testing and output verification

## Installation:
Clone this repository, then reach the project directory:
```sh
git clone git@github.com:bgclutch/Pattern_Matching_Task_cpp.git
cd Pattern_Matching_Task_cpp
```

## Building:
1. Build the project:
 ```sh
cmake -B build
cmake --build build
```

## Usage:
1. Navigate to the ```build``` folder:
```sh
cd build
```
2. Choose tree to run:
```sh
./bitonic_sort/bitonic_sort
```

## Running tests:
For End To End tests:
1.1 Navigate to the ```tests``` directory:
```sh
cd tests/End_To_End
```
1.2 Run default tests with the Python script:
```sh
python3 testrun.py
```
1.3 (Optional) Or ```regenerate``` test cases:
```sh
python3 testgen.py
```
And run it as in step 2.

For unit tests:
2.1 Navigate to the ```build``` folder:
```sh
cd build
```
2. Run unit tests:
```sh
./tests/tests
```
## Benchmark run
1. To build the project in benchmark mode:
```sh
cmake -DENABLE_BECHMARK=ON -B build
cmake --build build
```
2.1 Run benchmark with default data:
```sh
./build/benchmark/benchmark
```
2.2 Or use your data:
```sh
./build/benchmark/benchmark "USER'S FILE"
```

## Benchmark results
Benchmark for 5 benchmark tests with different data set size in each,
using -O2 optimisation

- Device: Huawei MateBook XPro 2022
- CPU: Intel Core i7 1260-P
- Memory: 16 GB Unified Memory
- Graphics: Intel Iris Xe Graphics

**Fast GPU Pattern matching algorithm compared with CPU Boyer-Moore-Horspool algorithm**

* **Lower** ratio means **better** result

| Elements amount| GPU Total time (Wall time) fast | Kernel Execution time fast | Data Transfer time fast | CPU time | Kernel time to CPU time ratio fast | Wall time to CPU time ratio fast |
|-----------------------|------------------|------------------|---------------|----------|-------------------------|------------------------|
| 8536091              |  3825800 us  | 3820900  us  |  4891 us    | 10430700 us    | 0.37                  | 0.37                  |
| 1233701            |  125847 us   | 123781 us  | 2065.98 us    | 496438 us  | 0.25                    | 0.25                    |
| 5603587           | 1992000  us  | 1988820  us  | 3572.44 us  | 6597610 us  | 0.3                    | 0.3                   |
| 9697390            | 8636360  us  |  8627610 us      | 8757.19 us  | 23647200 us  | 0.36                    | 0.36                   |
| 6877577            | 4673400  us  | 4670050 us      | 3361.81 us  | 20344200 us  | 0.23                    | 0.23                   |

**Naive GPU Pattern matching algorithm compared with Naive CPU algorithm**

* **Lower** ratio means **better** result

| Elements amount| GPU Total time (Wall time) naive | Kernel Execution time naive | Data Transfer time naive | CPU time | Kernel time to CPU time ratio naive | Wall time to CPU time ratio naive |
|-----------------------|------------------|------------------|---------------|----------|---------------|----------|
| 8536091              |  4391300 us  | 4385600 us  | 5700000 us    | 96266000 us   | 0.06                 | 0.05                |
| 6877577           | 9344150 us  | 9342000 us  | 2151.61 us  | 82258700 us  | 0.11                  | 0.11                  |
| 9697390            | 12091000 us  | 12082000 us      | 9803.75 us  | 223347700 us  | 0.05            | 0.05                    |
| 3197218            | 4564730 us  | 4563220 us      | 1508.01 us  | 29447900 us  | 0.15               | 0.15                   |
| 1040639            | 260652 us  | 260016 us      | 635.012 us  | 10629500 us  | 0.02                   | 0.02                  |


## Result consideration

The Fast kernel’s performance is heavily penalized by the "tail" (or halo) region overhead when the maximum pattern length is large relative to the work-group size, leading to excessive redundant data copying. While the Fast kernel aggressively prefetches data, the Naive kernel utilizes lazy evaluation, breaking early on mismatches and saving significant bandwidth. On Intel Integrated GPUs, this advantage is amplified because the hardware L3 cache efficiently manages coalesced global memory access, often making the manual overhead of local memory and synchronization barriers slower than the naive approach for datasets with long patterns. For example, the final test used 8147 patterns, none of which exceeded 200 characters. However, as we generate sequences with a wider range of lengths, the Fast kernel will eventually lose to the Naive kernel due to the increasing cost of tail data management.
