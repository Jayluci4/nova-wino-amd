# NOVA Winograd F(6,3) — HIP Kernel Build
#
# Usage:
#   make              Build shared library (links against PyTorch's ROCm)
#   make standalone   Build standalone test binary (links against system ROCm)
#   make test         Run correctness tests
#   make bench        Run performance benchmarks
#   make clean        Remove build artifacts

HIPCC         ?= hipcc
GPU_ARCH      ?= gfx942
CXX_STD       ?= -std=c++17

# Auto-detect PyTorch lib directory
TORCH_LIB     := $(shell python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null)
ROCM_DIR      ?= /opt/rocm

# Shared library flags (link against PyTorch's bundled ROCm 6.3)
SO_LDFLAGS    := -shared -fPIC
SO_LIBS       := -L$(TORCH_LIB) -L$(ROCM_DIR)/lib -lrocblas -lamdhip64
SO_RPATH      := -Wl,-rpath,$(TORCH_LIB)

# Standalone flags (link against system ROCm)
BIN_LIBS      := -L$(ROCM_DIR)/lib -lrocblas -I$(ROCM_DIR)/include

SRC           := csrc/nova_winograd.hip
LIB_OUT       := nova_winograd/lib/libnova_winograd.so
BIN_OUT       := build/nova_winograd_test

.PHONY: all standalone test bench clean

all: $(LIB_OUT)

$(LIB_OUT): $(SRC)
	@mkdir -p $(dir $@)
	$(HIPCC) $(SO_LDFLAGS) -o $@ $< $(CXX_STD) $(SO_LIBS) $(SO_RPATH) \
		-I$(ROCM_DIR)/include --offload-arch=$(GPU_ARCH)
	@echo "Built: $@"

standalone: $(SRC)
	@mkdir -p build
	$(HIPCC) -DSTANDALONE_TEST -o $(BIN_OUT) $< $(CXX_STD) $(BIN_LIBS) \
		--offload-arch=$(GPU_ARCH)
	@echo "Built: $(BIN_OUT)"

test: $(LIB_OUT)
	python3 -m pytest tests/ -v --tb=short

bench: $(LIB_OUT)
	python3 benchmarks/bench_layers.py

clean:
	rm -rf build/
	rm -f nova_winograd/lib/*.so
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
