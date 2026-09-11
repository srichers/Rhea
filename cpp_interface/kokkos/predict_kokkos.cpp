/*
Author: Sherwood Richers; Claude AI
NOTE: This file was AI generated and tested against Python Rhea driver, but the logic was not human-vetted.

License: GPLv3 (see LICENSE file)

Runs the deterministic test cells through the standalone evaluator and writes the
predictions to a file. Links Kokkos and nothing else - no LibTorch, which is what lets
this be compiled by nvcc_wrapper without the two frameworks ever meeting.
*/
#include "RheaKokkos.hpp"
#include "rhea_test_cells.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char* argv[]){
    if(argc < 3){
        std::fprintf(stderr, "usage: %s <model.rhea> <out.bin> [ncells] [team_size]\n", argv[0]);
        return -1;
    }
    const int ncells    = (argc > 3) ? atoi(argv[3]) : 4096;
    const int team_size = (argc > 4) ? atoi(argv[4]) : 0;   // 0 = the default of 32

    Kokkos::initialize(argc, argv);
    {
    using Space = Kokkos::DefaultExecutionSpace;
    using mem   = Space::memory_space;

    RheaModelKokkos<Space> model(argv[1], team_size);
    std::printf("#  kokkos on %s, %d cells, per-cell scratch %d bytes\n",
                Space::name(), ncells, model.scratch_bytes());

    const std::vector<float> inputs = rhea_test_cells(ncells);
    const int nper = model.tables().nnodes*4;

    Kokkos::View<float*, mem> d_in ("in", (size_t)ncells*nper), d_out("out", (size_t)ncells*nper);
    Kokkos::View<float*, mem> d_gr ("gr", ncells), d_st("st", ncells);
    auto h_in = Kokkos::create_mirror_view(d_in);
    for(size_t i=0; i<h_in.extent(0); i++) h_in(i) = inputs[i];
    Kokkos::deep_copy(d_in, h_in);

    Kokkos::fence();
    Kokkos::Timer timer;
    model.predict_all(Kokkos::View<const float*, mem>(d_in), d_out, d_gr, d_st);
    Kokkos::fence();
    const double elapsed = timer.seconds();
    // labelled with the backend on purpose: a Serial run is one CPU thread and is not a
    // throughput number for anything
    std::printf("#  predict_all on %s: %.2f ms (%.2f us/cell, team size %d)\n",
                Space::name(), elapsed*1e3, elapsed*1e6/ncells, model.last_team_size());

    auto h_out = Kokkos::create_mirror_view(d_out); Kokkos::deep_copy(h_out, d_out);
    auto h_gr  = Kokkos::create_mirror_view(d_gr);  Kokkos::deep_copy(h_gr,  d_gr);
    auto h_st  = Kokkos::create_mirror_view(d_st);  Kokkos::deep_copy(h_st,  d_st);
    rhea_write_predictions(argv[2], ncells, inputs.data(), h_out.data(), h_gr.data(), h_st.data());
    }
    Kokkos::finalize();
    return 0;
}
