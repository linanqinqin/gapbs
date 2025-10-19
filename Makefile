# See LICENSE.txt for license details.

CXX_FLAGS += -std=c++11 -O3 -Wall -lpthread src/pthreadpp.cc

# P3 (pthreadpp) is now the default parallelization framework
# OpenMP has been completely replaced with P3

KERNELS = bc bfs cc cc_sv pr pr_spmv sssp tc
SUITE = $(KERNELS) converter

.PHONY: all
all: $(SUITE)

% : src/%.cc src/*.h src/pthreadpp.cc
	$(CXX) $(CXX_FLAGS) $< -o $@

# Testing
include test/test.mk

# Benchmark Automation
include benchmark/bench.mk


.PHONY: clean
clean:
	rm -f $(SUITE) test/out/*