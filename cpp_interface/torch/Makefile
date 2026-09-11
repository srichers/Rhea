# The LibTorch consumer. Loads the exported TorchScript directly and never reads a .rhea
# file. Builds on its own: no Kokkos anywhere in this directory.
#
#   make LIBTORCH_DIR=$HOME/libtorch
ifeq ($(LIBTORCH_DIR),)
LIBTORCH_DIR = /usr/local/libtorch
endif

# .. is for rhea_test_cells.hpp, which both consumers share
CXXFLAGS = -g -O2 -std=c++20 -D_GLIBCXX_USE_CXX11_ABI=1 -I. -I..
BUILD    = $(CXXFLAGS) -I$(LIBTORCH_DIR)/include -I$(LIBTORCH_DIR)/include/torch/csrc/api/include
LDLIBS   = -Wl,-rpath,$(LIBTORCH_DIR)/lib -L$(LIBTORCH_DIR)/lib -ltorch_cpu -lc10 -lkineto

all: predict_torch

predict_torch: predict_torch.cpp FFISubgridModel.h ../rhea_test_cells.hpp
	g++ $(BUILD) -o predict_torch predict_torch.cpp $(LDLIBS)

clean:
	rm -rf predict_torch
