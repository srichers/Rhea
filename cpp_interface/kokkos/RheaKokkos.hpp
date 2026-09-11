/*
Authors: Sherwood Richers; Claude AI

License: GPLv3 (see LICENSE file)

Kokkos driver for the standalone evaluator in rhea_model.hpp. One kernel launch runs
Box3D, all seven blocks and the conservation projections for every cell, keeping each
cell's state in scratch from start to finish. The LibTorch path issues roughly 1500
kernels for the same work and round-trips every intermediate through device memory.

This is the entry point for a Kokkos consumer: including it pulls in rhea_loader.hpp and
rhea_model.hpp, and nothing in this directory depends on LibTorch or e3nn.

Kokkos_Core.hpp must be included before rhea_model.hpp so that RHEA_FN resolves to
KOKKOS_INLINE_FUNCTION; including this header first is what guarantees that.
*/
#ifndef RHEA_KOKKOS_HPP
#define RHEA_KOKKOS_HPP

#include <Kokkos_Core.hpp>
#include "rhea_loader.hpp"

// A named functor rather than a lambda, so the policy can be asked for the largest team
// the backend will accept: Serial caps it at one, CUDA does not.
template<class ExecSpace>
struct RheaPredictFunctor {
    using memory_space = typename ExecSpace::memory_space;
    using member_type  = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

    RheaTables                                T;
    Kokkos::View<const float*, memory_space>  F4_in;
    Kokkos::View<float*, memory_space>        F4_out, growthrate, stability;
    int ncells, nper, team_size;

    KOKKOS_INLINE_FUNCTION void operator()(const member_type& team) const {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, team_size), [&](const int t){
            const int c = team.league_rank()*team_size + t;
            if(c >= ncells) return;
            rhea_predict_cell(T, &F4_in(c*nper), &F4_out(c*nper),
                              &growthrate(c), &stability(c));
        });
    }
};

template<class ExecSpace = Kokkos::DefaultExecutionSpace>
class RheaModelKokkos {
  public:
    using memory_space = typename ExecSpace::memory_space;
    using policy_type  = Kokkos::TeamPolicy<ExecSpace>;
    using member_type  = typename policy_type::member_type;
    template<class T> using View = Kokkos::View<T*, memory_space>;

    // team_size 0 means 32, clamped to what the backend accepts. On CUDA the team size is
    // the block size, so this decides occupancy and is worth 29x between the worst and
    // best settings: at team 1 a warp has one active lane and only 32 blocks fit per SM.
    // It plateaus at 32 and stays flat above - see the sweep in README_athenak.md. Aurora
    // has a different sub-group width, so the plateau may not start in the same place.
    RheaModelKokkos(const std::string& filename, int team_size = 0)
        : team_size_(team_size)
    {
        RheaModelHost host(filename);
        T_ = host.tables();

        w3j_     = mirror("rhea_w3j",     host.w3j);
        tp_ins_  = mirror("rhea_tp_ins",  host.tp_ins);
        tps_     = mirror("rhea_tps",     host.tps);
        lin_ins_ = mirror("rhea_lin_ins", host.lin_ins);
        lins_    = mirror("rhea_lins",    host.lins);
        blocks_  = mirror("rhea_blocks",  host.blocks);
        w_       = mirror("rhea_w",       host.w);
        pts_     = mirror("rhea_pts",     host.pts);
        wts_     = mirror("rhea_wts",     host.wts);

        T_.w3j     = w3j_.data();
        T_.tp_ins  = tp_ins_.data();
        T_.tps     = tps_.data();
        T_.lin_ins = lin_ins_.data();
        T_.lins    = lins_.data();
        T_.blocks  = blocks_.data();
        T_.w       = w_.data();
        T_.pts     = pts_.data();
        T_.wts     = wts_.data();
    }

    const RheaTables& tables() const { return T_; }
    int last_team_size() const { return last_team_size_; }
    int scratch_bytes() const { return (int)(T_.scratch_floats*sizeof(float)); }

    // F4_in and F4_out are [ncells, 2, NF, 4] flattened, xyzt last, in raw number-density
    // units. growthrate comes back in number-density units - the caller multiplies by
    // sqrt(2)*G_F/hbar to recover 1/s, because doing it here overflows float32.
    void predict_all(const View<const float>& F4_in,
                     const View<float>&       F4_out,
                     const View<float>&       growthrate,
                     const View<float>&       stability) const
    {
        const int ncells = (int)growthrate.extent(0);
        if(ncells == 0) return;

        RheaPredictFunctor<ExecSpace> f{T_, F4_in, F4_out, growthrate, stability,
                                        ncells, T_.nnodes*4, 1};

        // no Kokkos scratch any more: the per-cell workspace is a local array inside
        // rhea_predict_cell, so the compiler interleaves it across the warp
        policy_type probe(1, 1);
        const int ts_max = probe.team_size_max(f, Kokkos::ParallelForTag());
        if(ts_max < 1) throw std::runtime_error("Kokkos rejects every team size on " +
                                                std::string(ExecSpace::name()));
        const int want   = (team_size_ > 0) ? team_size_ : 32;
        f.team_size       = (want < ts_max) ? want : ts_max;
        last_team_size_   = f.team_size;

        policy_type policy((ncells + f.team_size - 1)/f.team_size, f.team_size);
        Kokkos::parallel_for("rhea_predict_all", policy, f);
    }

  private:
    template<class T>
    View<T> mirror(const char* name, const std::vector<T>& v){
        View<T> d(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(name)), v.size());
        auto h = Kokkos::create_mirror_view(d);
        for(size_t i=0; i<v.size(); i++) h(i) = v[i];
        Kokkos::deep_copy(d, h);
        return d;
    }

    RheaTables      T_;
    int             team_size_;
    mutable int     last_team_size_ = 0;
    View<RheaW3J>    w3j_;
    View<RheaTPIns>  tp_ins_;
    View<RheaTP>     tps_;
    View<RheaLinIns> lin_ins_;
    View<RheaLin>    lins_;
    View<RheaBlock>  blocks_;
    View<float>      w_, pts_, wts_;
};

#endif
