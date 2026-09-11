/*
Author: Sherwood Richers; Claude AI

License: GPLv3 (see LICENSE file)

Runs the deterministic test cells through the standalone evaluator with neither Kokkos nor
LibTorch in the build - just the loop in rhea_model.hpp, one cell at a time. It writes the
same predictions file as predict_torch and predict_kokkos, so compare_predictions.py can
diff any pair of the three.

Two things that buys. It builds in a couple of seconds with plain g++, which is the right
developer loop when editing rhea_model.hpp against nvcc_wrapper's half a minute. And
comparing it against predict_kokkos isolates a bug in the Kokkos driver from a bug in the
evaluator, which comparing either one against LibTorch cannot do.
*/
#include "rhea_loader.hpp"
#include "rhea_test_cells.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv){
    if(argc < 3){
        std::fprintf(stderr, "usage: %s <model.rhea> <out.bin> [ncells]\n", argv[0]);
        return -1;
    }
    const int ncells = (argc > 3) ? atoi(argv[3]) : 4096;

    RheaModelHost M(argv[1]);
    RheaTables T = M.tables();
    std::printf("#  plain C++ evaluator, %d cells, per-cell scratch %d bytes\n",
                ncells, (int)(T.scratch_floats*sizeof(float)));

    const std::vector<float> inputs = rhea_test_cells(ncells);
    std::vector<float> F4_out((size_t)ncells*24), gr(ncells), st(ncells);
    for(int c=0; c<ncells; c++)
        rhea_predict_cell(T, inputs.data() + (size_t)c*24, F4_out.data() + (size_t)c*24,
                          &gr[c], &st[c]);

    rhea_write_predictions(argv[2], ncells, inputs.data(), F4_out.data(), gr.data(), st.data());
    return 0;
}
