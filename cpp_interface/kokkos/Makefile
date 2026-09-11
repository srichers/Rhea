# The standalone evaluator and its Kokkos driver. Depends on neither LibTorch nor e3nn,
# and builds on its own: a machine that has Kokkos but no LibTorch can build this
# directory alone, which is the situation on Aurora.
#
#   make KOKKOS_DIR=$HOME/kokkos-install
ifeq ($(KOKKOS_DIR),)
KOKKOS_DIR = /usr/local/kokkos
endif

# a CUDA-enabled Kokkos has to be consumed through nvcc_wrapper, and the lambda inside
# RheaPredictFunctor needs --extended-lambda. A Serial or OpenMP build compiles with g++.
KOKKOS_HAS_CUDA := $(shell grep -c '^#define KOKKOS_ENABLE_CUDA$$' $(KOKKOS_DIR)/include/KokkosCore_config.h 2>/dev/null || echo 0)

ifeq ($(KOKKOS_HAS_CUDA),1)
COMP        = $(KOKKOS_DIR)/bin/nvcc_wrapper
KOKKOS_FLAGS= --extended-lambda --expt-relaxed-constexpr -Wno-deprecated-gpu-targets
KOKKOS_LINK = -lcuda -lcudart
else
COMP        = g++
KOKKOS_FLAGS=
KOKKOS_LINK =
endif

# .. is for rhea_test_cells.hpp, which both consumers share
CXXFLAGS = -g -O2 -std=c++20 -I. -I..
BUILD    = $(CXXFLAGS) $(KOKKOS_FLAGS) -I$(KOKKOS_DIR)/include
LDLIBS   = -Wl,-rpath,$(KOKKOS_DIR)/lib -L$(KOKKOS_DIR)/lib \
           -lkokkoscore -lkokkoscontainers -lkokkossimd $(KOKKOS_LINK) -ldl

HEADERS = rhea_model.hpp rhea_loader.hpp RheaKokkos.hpp ../rhea_test_cells.hpp

all: predict_kokkos

predict_kokkos: predict_kokkos.cpp $(HEADERS)
	$(COMP) $(BUILD) -o predict_kokkos predict_kokkos.cpp $(LDLIBS)

bench_rhea_kokkos: bench_rhea_kokkos.cpp $(HEADERS)
	$(COMP) $(BUILD) -o bench_rhea_kokkos bench_rhea_kokkos.cpp $(LDLIBS)

# the same evaluator with no Kokkos either - builds anywhere in a couple of seconds, and
# writes the same predictions file so it can be diffed against either of the other two
predict_cpu: predict_cpu.cpp $(HEADERS)
	g++ $(CXXFLAGS) -o predict_cpu predict_cpu.cpp

clean:
	rm -rf predict_kokkos predict_cpu bench_rhea_kokkos
