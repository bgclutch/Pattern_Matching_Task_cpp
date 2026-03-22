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

**Fast GPU Pattern matching algorithm comared with CPU Boyer-Moore-Horspool algorithm**

* **Lower** ratio means **better** result

| Elements amount| GPU Total time (Wall time) naive/fast | Kernel Execution time naive/fast | Data Transfer time  naive/fast | CPU time | Kernel time to CPU time ratio naive/fast | Wall time to CPU time ratio  naive/fast |
|-----------------------|------------------|------------------|---------------|----------|-------------------------|------------------------|
| 8536091              |  4391300 / 3825800 us  | 4385600 / 3820900  us  | 5700000 / 4891000 us    | 10430700 us    | 0.42 / 0.37                  | 0.42 / 0.37                  |
| 1233701            |  258880 / 125847 us   | 256614 / 123781 us  | 2265.81 / 2065.98 us    | 496438 us  | 0.52 / 0.25                    | 0.52 / 0.25                    |
| 5603587           | 2574400 / 1992000  us  | 2569200 / 1988820  us  | 5136.82 / 3572.44 us  | 6597610 us  | 0.39 / 0.3                    | 0.39 / 0.3                   |
| 9697390            | 12091000 / 8636360  us  | 12082000 / 8627610 us      | 9803.75 / 8757.19 us  | 23647200 us  | 0.51 / 0.36                    | 0.51 / 0.36                   |
| 6877577            | 10410400 / 4673400  us  | 10406200 / 4670050 us      | 4268.62 / 3361.81 us  | 20344200 us  | 0.51 / 0.23                    | 0.51 / 0.23                   |

**Naive GPU Pattern matching algorithm comared with Naive CPU algorithm**

* **Lower** ratio means **better** result

| Elements amount| GPU Total time (Wall time) naive/fast | Kernel Execution time naive/fast | Data Transfer time  naive/fast | CPU time | Kernel time to CPU time ratio naive/fast | Wall time to CPU time ratio  naive/fast |
|-----------------------|------------------|------------------|---------------|----------|-------------------------|------------------------|
| 8536091              |  4391300 / 3825800 us  | 4385600 / 3820900  us  | 5700000 / 4891000 us    | 10430700 us    | 0.42 / 0.37                  | 0.42 / 0.37                  |
| 1233701            |  258880 / 125847 us   | 256614 / 123781 us  | 2265.81 / 2065.98 us    | 496438 us  | 0.52 / 0.25                    | 0.52 / 0.25                    |
| 5603587           | 2574400 / 1992000  us  | 2569200 / 1988820  us  | 5136.82 / 3572.44 us  | 6597610 us  | 0.39 / 0.3                    | 0.39 / 0.3                   |
| 9697390            | 12091000 / 8636360  us  | 12082000 / 8627610 us      | 9803.75 / 8757.19 us  | 23647200 us  | 0.51 / 0.36                    | 0.51 / 0.36                   |
| 6877577            | 10410400 / 4673400  us  | 10406200 / 4670050 us      | 4268.62 / 3361.81 us  | 20344200 us  | 0.51 / 0.23                    | 0.51 / 0.23                   |


## Result consideration

The Fast kernel’s performance is heavily penalized by the "tail" (or halo) region overhead when the maximum pattern length is large relative to the work-group size, leading to excessive redundant data copying. While the Fast kernel aggressively prefetches data, the Naive kernel utilizes lazy evaluation, breaking early on mismatches and saving significant bandwidth. On Intel Integrated GPUs, this advantage is amplified because the hardware L3 cache efficiently manages coalesced global memory access, often making the manual overhead of local memory and synchronization barriers slower than the naive approach for datasets with long patterns. For example, the final test used 8147 patterns, none of which exceeded 200 characters. However, as we generate sequences with a wider range of lengths, the Fast kernel will eventually lose to the Naive kernel due to the increasing cost of tail data management.
