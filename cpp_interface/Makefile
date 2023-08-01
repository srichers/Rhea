COMP = g++
LIBTORCH_DIR = /usr/local/libtorch

INCLUDE_FLAGS = -I$(LIBTORCH_DIR)/include -I$(LIBTORCH_DIR)/include/torch/csrc/api/include
BUILD_FLAGS = -g -DUSE_C10D_GLOO -DUSE_DISTRIBUTED -DUSE_RPC -DUSE_TENSORPIPE -D_GLIBCXX_USE_CXX11_ABI=1 -g -std=gnu++14
LINK_FLAGS = -Wl,-rpath,$(LIBTORCH_DIR)/lib -L$(LIBTORCH_DIR)/lib -ltorch_cpu -lc10 -lkineto

test_torch_model:
	$(COMP) $(BUILD_FLAGS) -o test_torch_model test_torch_model.cpp $(INCLUDE_FLAGS) $(LINK_FLAGS)

clean:
	rm -rf test_torch_model.o test_torch_model

