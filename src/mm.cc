// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>

#include "command_line.h"
#include "pvector.h"
#include "timer.h"
#include "util.h"
#include "pthreadpp.h"

/*
GAP Benchmark Suite
Kernel: Matrix Multiplication (MM)
Author: Custom Implementation

Performs matrix multiplication C = A * B where:
- Matrix size is 2^g x 2^g
- A and B are generated deterministically based on indices
- Multiplication is performed in parallel using PthreadPP
- Result is reduced to a single sum value
*/

using namespace std;

// Custom command line parser for matrix multiplication
class CLMM : public CLBase {
  int num_trials_ = 16;
  bool do_verify_ = false;

 public:
  CLMM(int argc, char** argv, std::string name) : CLBase(argc, argv, name) {
    // Override get_args_ to only include g, n, v, h
    get_args_ = "g:n:vh";
    // Clear help strings and add only relevant ones
    help_strings_.clear();
    AddHelpLine('h', "", "print this help message");
    AddHelpLine('g', "scale", "matrix size is 2^scale x 2^scale");
    AddHelpLine('n', "n", "perform n trials", std::to_string(num_trials_));
    AddHelpLine('v', "", "verify the output of each run", "false");
  }

  bool ParseArgs() {
    signed char c_opt;
    extern char *optarg;
    while ((c_opt = getopt(argc_, argv_, get_args_.c_str())) != -1) {
      HandleArg(c_opt, optarg);
    }
    if (scale_ == -1) {
      std::cout << "No matrix size specified. Use -g <scale> (Use -h for help)" << std::endl;
      return false;
    }
    return true;
  }

  void HandleArg(signed char opt, char* opt_arg) override {
    switch (opt) {
      case 'g': scale_ = atoi(opt_arg);                     break;
      case 'n': num_trials_ = atoi(opt_arg);               break;
      case 'v': do_verify_ = true;                         break;
      case 'h': PrintUsage();                              break;
      default: break;
    }
  }

  int num_trials() const { return num_trials_; }
  bool do_verify() const { return do_verify_; }
};

// Deterministic hash function for generating matrix A values
// Based on row, col, and scale to ensure determinism
inline unsigned long generateA(size_t row, size_t col, int scale) {
  // Use a hash-like function to generate deterministic "random" values
  unsigned long seed = (row * 2654435761UL) ^ (col * 2246822519UL) ^ (scale * 3266489917UL);
  seed = seed ^ (seed >> 15);
  seed = seed * 2246822507UL;
  seed = seed ^ (seed >> 13);
  seed = seed * 3266489917UL;
  seed = seed ^ (seed >> 16);
  return seed;
}

// Deterministic hash function for generating matrix B values
// Different pattern from A to ensure different values
inline unsigned long generateB(size_t row, size_t col, int scale) {
  // Use a different hash pattern for B
  unsigned long seed = (col * 2654435761UL) ^ (row * 2246822519UL) ^ (scale * 3266489917UL);
  seed = seed ^ (seed >> 12);
  seed = seed * 2654435769UL;
  seed = seed ^ (seed >> 14);
  seed = seed * 2246822507UL;
  seed = seed ^ (seed >> 15);
  return seed;
}

// Generate matrix A
pvector<unsigned long> generateMatrixA(size_t size, int scale) {
  pvector<unsigned long> A(size * size);
  P3_PARALLEL_FOR(size * size, [&](size_t idx) {
    size_t row = idx / size;
    size_t col = idx % size;
    A[idx] = generateA(row, col, scale);
  });
  return A;
}

// Generate matrix B
pvector<unsigned long> generateMatrixB(size_t size, int scale) {
  pvector<unsigned long> B(size * size);
  P3_PARALLEL_FOR(size * size, [&](size_t idx) {
    size_t row = idx / size;
    size_t col = idx % size;
    B[idx] = generateB(row, col, scale);
  });
  return B;
}

// Parallel matrix multiplication: C = A * B
// Each cell in C is computed modulo 100
unsigned long matrixMultiply(const pvector<unsigned long> &A,
                            const pvector<unsigned long> &B,
                            pvector<unsigned long> &C,
                            size_t size) {
  // Parallel multiplication: assign rows to threads
  P3_PARALLEL_FOR(size, [&](size_t i) {
    for (size_t j = 0; j < size; j++) {
      unsigned long sum = 0;
      for (size_t k = 0; k < size; k++) {
        sum += A[i * size + k] * B[k * size + j];
      }
      // Apply modulo 100 to each cell
      C[i * size + j] = sum % 100;
    }
  });

  // Reduction: sum all values in C
  int64_t total_sum = 0;
  P3_PARALLEL_FOR_REDUCTION(size * size, [&](size_t idx) -> int64_t {
    return static_cast<int64_t>(C[idx]);
  }, total_sum);

  return static_cast<unsigned long>(total_sum);
}

// Serial matrix multiplication for verification
unsigned long matrixMultiplySerial(const pvector<unsigned long> &A,
                                   const pvector<unsigned long> &B,
                                   size_t size) {
  pvector<unsigned long> C(size * size, 0);
  
  // Serial multiplication
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      unsigned long sum = 0;
      for (size_t k = 0; k < size; k++) {
        sum += A[i * size + k] * B[k * size + j];
      }
      // Apply modulo 100 to each cell
      C[i * size + j] = sum % 100;
    }
  }

  // Sum all values
  unsigned long total_sum = 0;
  for (size_t i = 0; i < size * size; i++) {
    total_sum += C[i];
  }

  return total_sum;
}

// Verifier: runs single-threaded to get ground truth
bool MMVerifier(const pvector<unsigned long> &A,
                const pvector<unsigned long> &B,
                unsigned long parallel_result,
                size_t size) {
  unsigned long serial_result = matrixMultiplySerial(A, B, size);
  bool match = (parallel_result == serial_result);
  if (!match) {
    cout << "Verification FAILED: parallel=" << parallel_result 
         << " serial=" << serial_result << endl;
  }
  return match;
}

int main(int argc, char* argv[]) {
  CLMM cli(argc, argv, "matrix-multiplication");
  if (!cli.ParseArgs())
    return -1;

  // Set number of threads if specified via environment variable
  const char* p3_threads = getenv("P3_NUM_THREADS");
  if (p3_threads) {
    int threads = std::atoi(p3_threads);
    if (threads > 0) {
      p3_set_num_threads(threads);
      std::cout << "Using " << threads << " threads (from P3_NUM_THREADS)" << std::endl;
    }
  }

  int scale = cli.scale();
  size_t size = 1UL << scale;  // 2^scale
  std::cout << "Matrix size: " << size << "x" << size << " (2^" << scale << ")" << std::endl;

  Timer t;
  
  // Generate matrices A and B
  t.Start();
  pvector<unsigned long> A = generateMatrixA(size, scale);
  pvector<unsigned long> B = generateMatrixB(size, scale);
  t.Stop();
  PrintTime("Matrix Generation Time", t.Seconds());

  // Compute ground truth for verification if needed
  unsigned long ground_truth = 0;
  if (cli.do_verify()) {
    std::cout << "Computing ground truth (serial)..." << std::endl;
    t.Start();
    ground_truth = matrixMultiplySerial(A, B, size);
    t.Stop();
    PrintTime("Ground Truth Computation Time", t.Seconds());
    // PrintStep("Ground Truth Sum", static_cast<int64_t>(ground_truth));
  }

  // Barrier: wait for user input before starting matrix multiplication trials
  std::cout << "Matrix construction complete. Press Enter to start Matrix Multiplication benchmark..." << std::endl;
  std::string input;
  std::getline(std::cin, input);

  // Benchmark loop
  double total_seconds = 0;
  Timer trial_timer;
  
  for (int iter = 0; iter < cli.num_trials(); iter++) {
    pvector<unsigned long> C(size * size, 0);
    
    trial_timer.Start();
    unsigned long result = matrixMultiply(A, B, C, size);
    trial_timer.Stop();
    
    PrintTime("Trial Time", trial_timer.Seconds());
    // PrintStep("Result Sum", static_cast<int64_t>(result));
    total_seconds += trial_timer.Seconds();

    // Verify if requested
    if (cli.do_verify()) {
      bool verified = (result == ground_truth);
      PrintLabel("Verification", verified ? "PASS" : "FAIL");
      if (!verified) {
        cout << "Expected: " << ground_truth << ", Got: " << result << endl;
      }
    }
  }

  PrintTime("Average Time", total_seconds / cli.num_trials());
  return 0;
}

