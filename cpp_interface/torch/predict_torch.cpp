/*
Author: Sherwood Richers

License: GPLv3 (see LICENSE file)

Runs the deterministic test cells through the exported TorchScript model and writes the
predictions to a file. Links LibTorch and nothing else - the Kokkos consumer writes the
same file format from its own binary, and compare_predictions.py checks the two agree.
*/
#include "FFISubgridModel.h"
#include "rhea_test_cells.hpp"

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char* argv[]){
    if(argc < 3){
        std::fprintf(stderr, "usage: %s <model.pt> <out.bin> [ncells]\n", argv[0]);
        return -1;
    }
    const int ncells = (argc > 3) ? atoi(argv[3]) : 4096;

    // The Makefile links -ltorch_cpu, so this runs on the host whatever the Kokkos side
    // was built for. Different devices is a feature here: it makes the comparison
    // independent rather than a check that one kernel is deterministic.
    FFISubgridModel<3> model(std::string(argv[1]), torch::kCPU);
    std::printf("#  libtorch on CPU, %d cells\n", ncells);

    const std::vector<float> inputs = rhea_test_cells(ncells);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    torch::Tensor F4_in = torch::from_blob((void*)inputs.data(), {ncells,2,3,4}, options).clone();
    auto out = model.model.get_method("predict_all")({F4_in}).toTuple()->elements();
    torch::Tensor F4_out = out[0].toTensor().contiguous();
    torch::Tensor gr     = out[1].toTensor().contiguous();
    torch::Tensor st     = out[2].toTensor().contiguous();

    rhea_write_predictions(argv[2], ncells, inputs.data(), F4_out.data_ptr<float>(),
                           gr.data_ptr<float>(), st.data_ptr<float>());
    return 0;
}
