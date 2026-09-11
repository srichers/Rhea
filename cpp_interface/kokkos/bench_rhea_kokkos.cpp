/*
Author: Sherwood Richers; Claude-AI
NOTE: This file was AI-generated and not human vetted. We keep it in the repository as a useful probe of performance.

License: GPLv3 (see LICENSE file)

Sweeps batch size, team size and scratch level for the Kokkos evaluator. The per-cell
working set is ~5.5 kB, which is the whole difficulty: it is too large for shared memory
at any useful team size, and putting it in global memory costs the inner loops dearly.
This measures which side of that trade wins, and at what batch the GPU is even full.

No LibTorch here, so it builds with nvcc_wrapper alone.
*/
#include "RheaKokkos.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdlib>

static std::vector<float> make_cells(int ncells){
    std::vector<float> F4((size_t)ncells*24);
    uint32_t rng = 12345u;
    auto u = [&]()->float { rng = rng*1664525u + 1013904223u;
                            return (float)((rng>>8)&0xFFFFFF)/(float)0x1000000; };
    for(int c=0; c<ncells; c++) for(int a=0; a<2; a++) for(int fl=0; fl<3; fl++){
        const float n = 1e33f*(0.2f + u());
        float d[3], nrm = 0.0f;
        for(int x=0; x<3; x++){ d[x] = u()-0.5f; nrm += d[x]*d[x]; }
        nrm = sqrtf(nrm);
        const float ff = 0.9f*u();
        for(int x=0; x<3; x++) F4[((size_t)(c*2+a)*3+fl)*4+x] = n*ff*d[x]/nrm;
        F4[((size_t)(c*2+a)*3+fl)*4+3] = n;
    }
    return F4;
}

int main(int argc, char* argv[]){
    if(argc < 2){
        std::fprintf(stderr,"usage: %s <model.rhea> [ncells] [team_size]\n"
                            "  with no ncells, sweeps the whole grid; with both, runs one\n"
                            "  configuration once, which is what a profiler wants\n", argv[0]);
        return -1;
    }
    const int one_ncells = (argc > 2) ? atoi(argv[2]) : 0;
    const int one_team   = (argc > 3) ? atoi(argv[3]) : 0;
    Kokkos::initialize(argc, argv);
    {
    using Space = Kokkos::DefaultExecutionSpace;
    using mem   = Space::memory_space;

    const int sweep_sizes[] = {4096, 16384, 42496, 169984};
    const int sweep_teams[] = {1, 4, 8, 16, 32, 64, 128, 256};
    const int one_size[]    = {one_ncells};
    const int one_t[]       = {one_team};
    const int  nsize  = one_ncells ? 1 : 4;
    const int  nteam  = one_ncells ? 1 : 8;
    const int* sizes  = one_ncells ? one_size : sweep_sizes;
    const int* teams  = one_ncells ? one_t    : sweep_teams;
    const int  nreps  = one_ncells ? 1 : 3;

    std::printf("# backend %s\n", Space::name());
    std::printf("# %8s %6s %6s %10s %10s\n", "ncells","want","got","ms","us/cell");

    {
      for(int ti=0; ti<nteam; ti++){ const int want = teams[ti];
        for(int si=0; si<nsize; si++){ const int ncells = sizes[si];
            std::vector<float> host = make_cells(ncells);
            Kokkos::View<float*, mem> d_in("in",(size_t)ncells*24), d_out("out",(size_t)ncells*24),
                                      d_gr("gr",ncells), d_st("st",ncells);
            auto h = Kokkos::create_mirror_view(d_in);
            for(size_t i=0;i<h.extent(0);i++) h(i)=host[i];
            Kokkos::deep_copy(d_in,h);

            RheaModelKokkos<Space>* m = nullptr;
            try { m = new RheaModelKokkos<Space>(argv[1], want); }
            catch(...) { std::printf("# %8d %6d   (model load failed)\n", ncells, want); continue; }

            double best = 1e30;
            try {
                for(int rep=0; rep<nreps; rep++){
                    Kokkos::fence();
                    Kokkos::Timer t;
                    m->predict_all(Kokkos::View<const float*,mem>(d_in), d_out, d_gr, d_st);
                    Kokkos::fence();
                    const double e = t.seconds();
                    if(e < best) best = e;
                }
            } catch(const std::exception& ex){
                std::printf("  %8d %6d %6s %10s %10s   %s\n",
                            ncells, want, "-", "-", "-", ex.what());
                std::fflush(stdout);
                delete m;
                continue;
            }
            std::printf("  %8d %6d %6d %10.2f %10.3f\n",
                        ncells, want, m->last_team_size(), best*1e3, best*1e6/ncells);
            std::fflush(stdout);
            delete m;
        }
    }
    }
    }
    Kokkos::finalize();
    return 0;
}
