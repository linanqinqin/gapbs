// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include "pthreadpp.h"

// Global instance
PthreadPP p3;

// Initialize with specific number of threads
void p3_set_num_threads(int num_threads) {
    p3.set_num_threads(num_threads);
}
