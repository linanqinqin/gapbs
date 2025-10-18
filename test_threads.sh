#!/bin/bash

# Test script to demonstrate different ways to specify number of threads
# for the pthreads version of GAPBS

echo "=== GAPBS Pthreads Thread Control Test ==="
echo

# Build the pthreads version
echo "Building pthreads version..."
make pthreads

if [ ! -f "./bfs_pthreads" ]; then
    echo "Error: Failed to build bfs_pthreads"
    exit 1
fi

echo "Build successful!"
echo

# Test 1: Default number of threads (hardware concurrency)
echo "=== Test 1: Default threads (hardware concurrency) ==="
./bfs_pthreads -g 10 -n 1
echo

# Test 2: Using OMP_NUM_THREADS environment variable
echo "=== Test 2: Using OMP_NUM_THREADS=4 ==="
OMP_NUM_THREADS=4 ./bfs_pthreads -g 10 -n 1
echo

# Test 3: Using GAPBS_NUM_THREADS environment variable
echo "=== Test 3: Using GAPBS_NUM_THREADS=2 ==="
GAPBS_NUM_THREADS=2 ./bfs_pthreads -g 10 -n 1
echo

# Test 4: Using both (GAPBS_NUM_THREADS should take precedence)
echo "=== Test 4: Both set (GAPBS_NUM_THREADS=8, OMP_NUM_THREADS=4) ==="
OMP_NUM_THREADS=4 GAPBS_NUM_THREADS=8 ./bfs_pthreads -g 10 -n 1
echo

# Test 5: Single thread
echo "=== Test 5: Single thread ==="
GAPBS_NUM_THREADS=1 ./bfs_pthreads -g 10 -n 1
echo

# Test 6: Many threads
echo "=== Test 6: Many threads (16) ==="
GAPBS_NUM_THREADS=16 ./bfs_pthreads -g 10 -n 1
echo

echo "=== All tests completed ==="
