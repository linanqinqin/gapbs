# See LICENSE.txt for license details.

ROOT_PATH=../..
include $(ROOT_PATH)/build/shared.mk

# P3 (pthreadpp) is now the default parallelization framework
# OpenMP has been completely replaced with P3

KERNELS = bc bfs cc cc_sv pr pr_spmv sssp tc mm
SUITE = $(KERNELS) converter

# Collect all source files
kernel_src = $(addprefix src/, $(addsuffix .cc, $(KERNELS)))
converter_src = src/converter.cc
all_src = $(kernel_src) $(converter_src)

# Object files and dependencies
kernel_obj = $(kernel_src:.cc=.o)
converter_obj = $(converter_src:.cc=.o)
all_obj = $(kernel_obj) $(converter_obj)
dep = $(all_obj:.o=.d)

lib_shim = -Wl,--whole-archive $(ROOT_PATH)/shim/libshim.a -ldl -Wl,--no-whole-archive

.PHONY: all
all: $(SUITE)

# Pattern rule for compiling .cc files to .o files
src/%.o: src/%.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Pattern rule for Caladan targets (kernels and converter)
$(SUITE): %: src/%.o $(RUNTIME_DEPS)
	$(LDXX) $(LDFLAGS) -o $@ $< \
	-Wl,--wrap=main $(lib_shim) $(RUNTIME_LIBS)
	@$(ROOT_PATH)/scripts/verify_shim.sh $@

ifneq ($(MAKECMDGOALS),clean)
-include $(dep)   # include all dep files in the makefile
endif

# rule to generate a dep file by using the C++ preprocessor
src/%.d: src/%.cc
	@$(CXX) $(CXXFLAGS) $< -MM -MT $(@:.d=.o) >$@

# Testing
include test/test.mk

# Benchmark Automation
include benchmark/bench.mk


.PHONY: clean
clean:
	rm -f $(all_obj) $(dep) $(SUITE) test/out/*