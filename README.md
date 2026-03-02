# Pattern matching
This project provides an implementation of a Pattern matching algorithm in C++ and OpenCL.
It also includes automated tests comparing the results of the Pattern matching with CPU implementation of Boyer-Moore algorithm.

## Features:
1. An implementation of a Bitonic sort algorithm with OpenCL library
2. Comparison of results of GPU parallel algorithm and CPU-based Boyer-Moore algorithm for correctness
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
Benchmark for 6 benchmark tests with different data set size in each,
using -O2 optimisation

- Device: Huawei MateBook XPro 2022
- CPU: Intel Core i7 1260-P
- Memory: 16 GB Unified Memory
- Graphics: Intel Iris Xe Graphics

**Fast GPU Pattern matching algorithm comared with CPU Boyer-Moore algorithm**

| Elements amount| GPU Total time (Wall time) | Kernel Execution time | Data Transfer time | CPU time | Kernel time to CPU time ratio | Wall time to CPU time ratio |
|-----------------------|------------------|------------------|---------------|----------|-------------------------|------------------------|
| 0                    |   us      |  us      |  us    |  us    | 0.                  | 0.                  |
| 0                  |   us      |  us      |  us    |  us  | 0.                    | 0.                    |
| 0                 |   us      |  us      |  us    |  us  | 8.                    | 0.                   |
| 0                |   us      |  us      |  us    |  us   | 0.                   | 0.                   |
| 0               |   us      |  us      |  us    |  us   | 0.                   | 0.                    |
| 0             |   us      |  us      |  us    |  us | 0.                   | 0.                  |