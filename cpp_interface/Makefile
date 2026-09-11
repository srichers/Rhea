# Two consumers of a trained model, each with its own Makefile and its own binary:
#
#   torch/   loads the exported TorchScript through LibTorch and reads the .pt directly
#   kokkos/  reads a .rhea written by its own export_rhea.py, and needs neither
#            LibTorch nor e3nn. This is the one AthenaK wants.
#
# They share no code and are never linked together, which is what lets the Kokkos side be
# compiled by nvcc_wrapper: LibTorch's headers do not survive nvcc. Each subdirectory
# builds on its own, so a machine with only one of the two frameworks can still use it.
#
# Each writes its predictions for the same deterministic cells (rhea_test_cells.hpp), and
# compare_predictions.py checks they agree - including that they really saw the same cells.
#
#   make compare KOKKOS_DIR=$HOME/kokkos-install LIBTORCH_DIR=$HOME/libtorch
#
# Command-line variables reach the sub-makes through MAKEFLAGS, so KOKKOS_DIR and
# LIBTORCH_DIR need naming only once, here.

# which models the comparison runs on; override on the command line
PT   = ../model_training/model10_cpu.pt
RHEA = ../model_training/model10.rhea

all:
	$(MAKE) -C torch
	$(MAKE) -C kokkos

# run both consumers and diff their predictions. This is the regression gate.
compare: all
	./torch/predict_torch   $(PT)   torch_predictions.bin
	./kokkos/predict_kokkos $(RHEA) kokkos_predictions.bin
	python3 compare_predictions.py torch_predictions.bin kokkos_predictions.bin

# the other kokkos/ targets, forwarded for convenience
predict_cpu bench_rhea_kokkos:
	$(MAKE) -C kokkos $@

clean:
	$(MAKE) -C torch  clean
	$(MAKE) -C kokkos clean
	rm -rf torch_predictions.bin kokkos_predictions.bin

.PHONY: all compare clean predict_cpu bench_rhea_kokkos
