// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include "gapbs_pthreads.h"

// Global instance
GAPBSPthreads gapbs_pthreads;

// Initialize with specific number of threads
void gapbs_set_num_threads(int num_threads) {
    gapbs_pthreads.set_num_threads(num_threads);
}
