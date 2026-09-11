/*
Authors: Sherwood Richers

License: GPLv3 (see LICENSE file)

Standalone evaluator for a Rhea model exported by export_rhea.py in this directory. It
needs neither LibTorch nor e3nn: the irreps lists, tensor product instruction tables,
Wigner 3j tables and weights all come out of the .rhea file, so a checkpoint of a
different width or depth is a file swap rather than a code change.

The whole network for one cell runs inside rhea_predict_cell, keeping that cell's ~200
floats of state in the caller's scratch buffer from the first block to the last. That is
the point of the exercise: the LibTorch path spends its time round-tripping those same
intermediates through device memory across ~1500 kernel launches.

Kokkos is not required. RHEA_FN is empty unless Kokkos_Macros.hpp has been included
first, which lets the evaluator be compiled and tested with a plain host compiler.

The tensor product instruction tables, path weight conventions and irreps layout
implemented here follow e3nn (MIT, (c) 2020 The Regents of the University of California
through Lawrence Berkeley National Laboratory, EPFL, Free University of Berlin, and
Kostiantyn Lapchevskyi). No e3nn source is reproduced; see THIRD_PARTY_LICENSES at the
repository root for what is borrowed and for the MIT notice.
*/
#ifndef RHEA_MODEL_HPP
#define RHEA_MODEL_HPP

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef KOKKOS_INLINE_FUNCTION
#define RHEA_FN KOKKOS_INLINE_FUNCTION
#else
#define RHEA_FN inline
#endif

// std::isfinite is not reliably callable from device code across CUDA versions, and the
// unqualified name is only in scope there. __CUDA_ARCH__ is defined in the device pass
// only, which is exactly the distinction needed.
#ifdef __CUDA_ARCH__
#define RHEA_ISFINITE(x) (::isfinite(x))
#define RHEA_ISNAN(x)    (::isnan(x))
#else
#define RHEA_ISFINITE(x) (std::isfinite(x))
#define RHEA_ISNAN(x)    (std::isnan(x))
#endif

#define RHEA_PI 3.14159265358979323846f

// Per-cell scratch is an ordinary local array, not Kokkos scratch. Kokkos hands each
// thread its own *contiguous* block, so lane n and lane n+1 touch addresses thousands of
// bytes apart and every access in the tensor product inner loop is an uncoalesced global
// transaction; a local array is interleaved across the warp by the compiler. Both caps
// are checked against the model at load time.
#ifndef RHEA_MAX_SCRATCH
#define RHEA_MAX_SCRATCH 1088
#endif

// largest 2l+1 the evaluator will meet
#ifndef RHEA_MAX_IRREP_DIM
#define RHEA_MAX_IRREP_DIM 9
#endif

//===================#
// on-device tables  #
//===================#
struct RheaW3J     { int i, j, k; float v; };

// one tensor product path: out[w,k] += pw * sum_{u,v} W[u,v,w] sum_{ij} w3j[i,j,k] x1[u,i] x2[v,j]
struct RheaTPIns {
    int   mode, has_w, woff;             // mode 0 = uvw, 1 = uuu
    int   u, v, w;                       // path shape, padded to three
    float pw;
    int   w3j_off, w3j_nnz;
    int   in1_off, in1_d;                // slice of x1: mul u (or the uuu multiplicity), 2l+1 = in1_d
    int   in2_off, in2_d;
    int   out_off, out_d;
};
struct RheaTP      { int ins_off, n_ins, w_off, dim_out; };

// one linear path: out[w,i] += pw * sum_u W[u,w] x[u,i]
struct RheaLinIns  { int woff, u, w; float pw; int in_off, in_d, out_off, out_d; };
struct RheaLin     { int ins_off, n_ins, w_off, dim_out; };

struct RheaBlock {
    int   kind;                          // 0 = PETP_Linear, 1 = PETP_Quadratic
    float alpha;
    float eps2;
    int   tp_norm, tp_normmul;           // quadratic only
    int   branch[4];                     // TP indices (quadratic) or Linear indices (linear)
    int   is_gate;
    int   dim_scalars, dim_gates, dim_gated;
    int   act_scalar, act_gate;
    float scale_scalar, scale_gate;
    int   tp_mul;                        // the gate's elementwise tensor product
    int   has_skip, lin_skip;
    int   dim_in, dim_tp, dim_out;
};

// every pointer the evaluator needs, so the same code runs on host and device
struct RheaTables {
    const RheaW3J*    w3j;
    const RheaTPIns*  tp_ins;
    const RheaTP*     tps;
    const RheaLinIns* lin_ins;
    const RheaLin*    lins;
    const RheaBlock*  blocks;
    const float*      w;                 // every weight, concatenated
    const float*      pts;               // lebedev directions, [3][npts]
    const float*      wts;               // lebedev weights, [npts]
    int npts, NF, nnodes, dim_in;
    int conserve_lepton_number, average_heavies;
    int stack_off[4];                    // block ranges: shared, growthrate, F4
    int dim_max, dim_tp_max;
    int f4_perm[4];                      // F4 head output slot -> xyzt component
    int scratch_floats;                  // what rhea_predict_cell needs, in floats
};

//=============#
// activations #
//=============#
// exp(-x) overflows for large negative x, so branch on the sign - the numpy reference
// uses the same stable form and the two must agree in the tails
RHEA_FN float rhea_sigmoid(float x){
    const float e = expf(-fabsf(x));
    return (x >= 0.0f) ? 1.0f/(1.0f + e) : e/(1.0f + e);
}
RHEA_FN float rhea_act(int id, float x){
    switch(id){
        case 1: return x * rhea_sigmoid(x);                                   // silu
        case 2: return rhea_sigmoid(x);                                       // sigmoid
        case 3: return tanhf(x);
        case 4: return (x > 0.0f) ? x : 0.0f;                                 // relu
        case 5: return 0.5f*x*(1.0f + erff(x*0.70710678118654752f));          // gelu
        default: return x;                                                    // identity
    }
}

//=================#
// core operations #
//=================#
// out must be zeroed by the caller.
RHEA_FN void rhea_eval_tp(const RheaTables& T, int itp,
                          const float* x1, const float* x2, float* out){
    const RheaTP& tp = T.tps[itp];
    for(int n=0; n<tp.n_ins; n++){
        const RheaTPIns& ins = T.tp_ins[tp.ins_off + n];

        if(ins.mode == 1){ // uuu: one multiplicity index shared by both inputs and the output
            for(int u=0; u<ins.u; u++){
                const float* a = x1 + ins.in1_off + u*ins.in1_d;
                const float* b = x2 + ins.in2_off + u*ins.in2_d;
                float*       o = out + ins.out_off + u*ins.out_d;
                const float  s = ins.pw * (ins.has_w ? T.w[tp.w_off + ins.woff + u] : 1.0f);
                for(int z=0; z<ins.w3j_nnz; z++){
                    const RheaW3J& t = T.w3j[ins.w3j_off + z];
                    o[t.k] += s * t.v * a[t.i] * b[t.j];
                }
            }
            continue;
        }

        // uvw. The (i,j)->k contraction for one (u,v) pair is at most 2l+1 numbers, so it
        // is formed in registers and distributed to every output multiplicity straight
        // away. Materializing t[u,v,k] for the whole instruction first saves no arithmetic
        // and costs ~320 floats of per-thread state, which is what bounds occupancy: the
        // working set has to stay small enough that resident threads x footprint fits in
        // L2. See the GV100 section of README_athenak.md.
        const int D = ins.out_d;
        const float* W = T.w + tp.w_off + ins.woff;
        for(int u=0; u<ins.u; u++){
            const float* a = x1 + ins.in1_off + u*ins.in1_d;
            for(int v=0; v<ins.v; v++){
                const float* b = x2 + ins.in2_off + v*ins.in2_d;
                float t[RHEA_MAX_IRREP_DIM];
                for(int k=0; k<D; k++) t[k] = 0.0f;
                for(int z=0; z<ins.w3j_nnz; z++){
                    const RheaW3J& c = T.w3j[ins.w3j_off + z];
                    t[c.k] += c.v * a[c.i] * b[c.j];
                }
                const float* Wuv = W + (u*ins.v + v)*ins.w;
                for(int w=0; w<ins.w; w++){
                    const float Wt = ins.pw * Wuv[w];
                    float* o = out + ins.out_off + w*D;
                    for(int k=0; k<D; k++) o[k] += Wt * t[k];
                }
            }
        }
    }
}

// out must be zeroed by the caller
RHEA_FN void rhea_eval_linear(const RheaTables& T, int ilin, const float* x, float* out){
    const RheaLin& lin = T.lins[ilin];
    for(int n=0; n<lin.n_ins; n++){
        const RheaLinIns& ins = T.lin_ins[lin.ins_off + n];
        const float* W = T.w + lin.w_off + ins.woff;
        for(int w=0; w<ins.w; w++){
            float* o = out + ins.out_off + w*ins.out_d;
            for(int u=0; u<ins.u; u++){
                const float Wt = ins.pw * W[u*ins.w + w];
                const float* a = x + ins.in_off + u*ins.in_d;
                for(int i=0; i<ins.in_d; i++) o[i] += Wt * a[i];
            }
        }
    }
}

// NormActivation: tanh(|x|)/|x| per irrep, with the norm clamped before the sqrt so the
// division stays finite where an irrep is exactly zero
RHEA_FN void rhea_norm_activation(const RheaTables& T, const RheaBlock& b,
                                  const float* x, float* out, float* nrm){
    const RheaTP& nt = T.tps[b.tp_norm];
    for(int i=0; i<nt.dim_out; i++) nrm[i] = 0.0f;
    rhea_eval_tp(T, b.tp_norm, x, x, nrm);
    for(int i=0; i<nt.dim_out; i++){
        const float n2 = (nrm[i] < b.eps2) ? b.eps2 : nrm[i];
        const float n  = sqrtf(n2);
        nrm[i] = tanhf(n)/n;
    }
    for(int i=0; i<b.dim_in; i++) out[i] = 0.0f;
    rhea_eval_tp(T, b.tp_normmul, nrm, x, out);
}

// Gate emits activated scalars followed by the gated irreps; a block whose output has no
// non-scalar irreps has a plain Activation instead and nothing to gate.
RHEA_FN void rhea_eval_gate(const RheaTables& T, const RheaBlock& b,
                            const float* y, float* out, float* gts){
    if(!b.is_gate){
        for(int i=0; i<b.dim_out; i++) out[i] = b.scale_scalar * rhea_act(b.act_scalar, y[i]);
        return;
    }
    const int ns = b.dim_scalars, ng = b.dim_gates;
    for(int i=0; i<ns; i++) out[i] = b.scale_scalar * rhea_act(b.act_scalar, y[i]);
    for(int i=0; i<ng; i++) gts[i] = b.scale_gate   * rhea_act(b.act_gate,   y[ns+i]);
    for(int i=ns; i<b.dim_out; i++) out[i] = 0.0f;
    // Gate calls mul(gated, gates), so the gated irreps are the first argument
    rhea_eval_tp(T, b.tp_mul, y + ns + ng, gts, out + ns);
}

//=====================#
// blocks and the stack #
//=====================#
// scratch layout, in floats. Everything a cell needs from the first block to the last.
struct RheaScratch {
    float *cur, *nxt, *stash, *S_f, *S_n, *S_all, *xb, *ga, *nrm, *y, *gts, *box;
};
RHEA_FN RheaScratch rhea_scratch(const RheaTables& T, float* p){
    RheaScratch s;
    float* const base = p;
    s.cur = p; p += T.nnodes*T.dim_max;
    s.nxt = p; p += T.nnodes*T.dim_max;
    s.stash=p; p += T.nnodes*T.dim_max;
    s.S_f = p; p += 2*T.dim_max;
    s.S_n = p; p += T.NF*T.dim_max;
    s.S_all= p; p += T.dim_max;
    s.xb  = p; p += T.dim_max;
    s.ga  = p; p += T.dim_max;
    s.nrm = p; p += T.dim_max;
    s.y   = p; p += T.dim_tp_max;
    s.gts = p; p += T.dim_tp_max;
    // box3d finishes before the network touches any of the above, and hands its results
    // back in the caller's locals, so its workspace overlaps them rather than adding to
    // the footprint. Do not reorder rhea_predict_cell around this.
    s.box = base;
    return s;
}

// input dimensions: [nu/nubar, flavor, features], flattened as node = a*NF + f
RHEA_FN void rhea_block(const RheaTables& T, int iblk, RheaScratch& s){
    const RheaBlock& b = T.blocks[iblk];
    const int NF = T.NF, D = b.dim_in;

    // the three partial sums PE_inputs needs, formed once per block rather than per node
    for(int i=0; i<D; i++) s.S_all[i] = 0.0f;
    for(int a=0; a<2; a++) for(int i=0; i<D; i++) s.S_f[a*T.dim_max+i] = 0.0f;
    for(int f=0; f<NF; f++) for(int i=0; i<D; i++) s.S_n[f*T.dim_max+i] = 0.0f;
    for(int a=0; a<2; a++) for(int f=0; f<NF; f++){
        const float* x = s.cur + (a*NF+f)*T.dim_max;
        for(int i=0; i<D; i++){
            s.S_f[a*T.dim_max+i] += x[i];
            s.S_n[f*T.dim_max+i] += x[i];
            s.S_all[i]           += x[i];
        }
    }

    const float inv_f = 1.0f/(float)(NF-1), inv_n = 1.0f, inv_a = 1.0f/(float)(NF-1);
    for(int a=0; a<2; a++) for(int f=0; f<NF; f++){
        const int z = a*NF+f;
        const float* x = s.cur + z*T.dim_max;
        float* y = s.y;
        for(int i=0; i<b.dim_tp; i++) y[i] = 0.0f;

        for(int br=0; br<4; br++){
            // x_all uses the *undivided* flavor and nunubar messages, so it reduces to
            // S_all - S_f - S_n + x before its own normalization
            for(int i=0; i<D; i++){
                const float S_f = s.S_f[a*T.dim_max+i], S_n = s.S_n[f*T.dim_max+i];
                switch(br){
                    case 0: s.xb[i] = x[i];                                    break;
                    case 1: s.xb[i] = (S_f - x[i]) * inv_f;                    break;
                    case 2: s.xb[i] = (S_n - x[i]) * inv_n;                    break;
                    default:s.xb[i] = (s.S_all[i] - S_f - S_n + x[i]) * inv_a; break;
                }
            }
            if(b.kind == 1){
                rhea_norm_activation(T, b, s.xb, s.ga, s.nrm);
                rhea_eval_tp(T, b.branch[br], s.xb, s.ga, y);
            } else {
                rhea_eval_linear(T, b.branch[br], s.xb, y);
            }
        }

        float* out = s.nxt + z*T.dim_max;
        rhea_eval_gate(T, b, y, s.ga, s.gts);   // gated result lands in s.ga

        // skip(x) + alpha*gate(...). The two paths are added componentwise with whatever
        // irrep order each produced, which is what ml_neuralnet does.
        if(b.has_skip){
            for(int i=0; i<b.dim_out; i++) out[i] = 0.0f;
            rhea_eval_linear(T, b.lin_skip, x, out);
        } else {
            for(int i=0; i<b.dim_out; i++) out[i] = x[i];
        }
        for(int i=0; i<b.dim_out; i++) out[i] += b.alpha * s.ga[i];
    }

    for(int z=0; z<T.nnodes; z++)
        for(int i=0; i<b.dim_out; i++) s.cur[z*T.dim_max+i] = s.nxt[z*T.dim_max+i];
}

// y_F4 [nnodes,4] and y_growthrate [nnodes,1] from the joint input already in s.cur
RHEA_FN void rhea_forward(const RheaTables& T, RheaScratch& s, float* y_F4, float* y_gr){
    for(int i=T.stack_off[0]; i<T.stack_off[1]; i++) rhea_block(T, i, s);

    // both heads read the trunk output, so keep a copy before the first head overwrites it
    const int Dtrunk = T.blocks[T.stack_off[1]].dim_in;
    for(int z=0; z<T.nnodes; z++)
        for(int i=0; i<Dtrunk; i++) s.stash[z*T.dim_max+i] = s.cur[z*T.dim_max+i];

    for(int i=T.stack_off[1]; i<T.stack_off[2]; i++) rhea_block(T, i, s);
    const int Dg = T.blocks[T.stack_off[2]-1].dim_out;
    for(int z=0; z<T.nnodes; z++) for(int i=0; i<Dg; i++) y_gr[z*Dg+i] = s.cur[z*T.dim_max+i];

    for(int z=0; z<T.nnodes; z++)
        for(int i=0; i<Dtrunk; i++) s.cur[z*T.dim_max+i] = s.stash[z*T.dim_max+i];
    for(int i=T.stack_off[2]; i<T.stack_off[3]; i++) rhea_block(T, i, s);
    const int Df = T.blocks[T.stack_off[3]-1].dim_out;
    for(int z=0; z<T.nnodes; z++) for(int i=0; i<Df; i++) y_F4[z*Df+i] = s.cur[z*T.dim_max+i];
}


//=======#
// box3d #
//=======#
// Transliteration of model_training/box3d.py. It returns the *change* to F4, not the
// mixed moments, and it never raises: a point it cannot interpret comes back as nan, so
// one bad cell in a simulation cannot spoil the batch.
RHEA_FN float rhea_get_Z(float ff){
    const float f2 = ff*ff, f4 = f2*f2, f6 = f4*f2, f8 = f4*f4;
    const float p = 1.0f - (2.0f*(1.0f-ff)*(1.0f+1.01524f*ff))
                  / (3.0f - 1.00651f*f2 - 0.962251f*f4 + 1.47353f*f6 - 0.48953f*f8);
    return (2.0f*ff)/(1.0f-p);
}
RHEA_FN float rhea_distrib(float n, float Z, float mu){
    // Z/sinh(Z)*exp(Z*mu) written so the exponent is never positive, with a small-Z form
    // that avoids dividing by a vanishing denominator
    const float g = (Z > 1e-3f) ? (2.0f*Z/(-expm1f(-2.0f*Z))) * expf(Z*(mu-1.0f))
                                : mu*Z + 1.0f;
    return g * (n/(4.0f*RHEA_PI));
}

// F4 [2,NF,4] normalized so the total density is one; dF4 [2,NF,4]; rate [3] in pair
// order (01,02,12). box needs T.nnodes*T.npts + 3*T.npts floats.
RHEA_FN void rhea_box3d(const RheaTables& T, const float* F4, float* dF4, float* rate,
                        float* box){
    const int NF = T.NF, NQ = T.npts, NN = T.nnodes;
    float* g    = box;                 // [2,NF,NQ]
    float* Psur = box + NN*NQ;         // [3,NQ]

    bool bad = false;
    float Ntot = 0.0f;
    for(int i=0; i<NN*4; i++) if(!RHEA_ISFINITE(F4[i])) bad = true;
    for(int a=0; a<2; a++) for(int f=0; f<NF; f++) Ntot += F4[(a*NF+f)*4+3];
    // torch.isclose defaults to rtol=1e-5 alongside the atol=1e-5 box3d.py passes
    if(!(fabsf(Ntot - 1.0f) <= 1e-5f + 1e-5f)) bad = true;

    for(int a=0; a<2; a++) for(int f=0; f<NF; f++){
        const float* v = F4 + (a*NF+f)*4;
        const float  n = v[3];

        // scale by the largest component before squaring so small fluxes do not underflow
        float Fscale = 0.0f;
        for(int c=0; c<3; c++) Fscale = fmaxf(Fscale, fabsf(v[c]));
        const float sc = (Fscale > 0.0f) ? Fscale : 1.0f;
        float s2 = 0.0f;
        for(int c=0; c<3; c++){ const float t = v[c]/sc; s2 += t*t; }
        const float normF = sc*sqrtf(s2);
        if(n < normF) bad = true;

        const float den = (normF > 0.0f) ? normF : 1.0f;
        const bool  empty = (n == 0.0f && normF == 0.0f);
        float ff = normF / (empty ? 1.0f : n);
        // the quadrature cannot resolve a beam narrower than ~1/Z, so cap what the
        // closure sees; only Z is affected
        ff = fminf(ff, 0.98f);
        const float Z = rhea_get_Z(ff);

        for(int q=0; q<NQ; q++){
            float mu = 0.0f;
            for(int c=0; c<3; c++) mu += (v[c]/den) * T.pts[c*NQ+q];
            g[(a*NF+f)*NQ+q] = rhea_distrib(n, Z, mu);
        }
    }

    // each flavor pair is an independent two-flavor problem on the crossing of its
    // lepton number distributions
    const int pi[3] = {0,0,1}, pj[3] = {1,2,2};
    float crosses[3], invN = 0.0f;
    int   ncrossing = 0;
    for(int p=0; p<3; p++){
        float Iplus = 0.0f, Iminus = 0.0f;
        for(int q=0; q<NQ; q++){
            const float G = (g[         pi[p] *NQ+q] - g[(NF+pi[p])*NQ+q])
                          - (g[         pj[p] *NQ+q] - g[(NF+pj[p])*NQ+q]);
            Iplus  += fmaxf( G, 0.0f)*T.wts[q];
            Iminus += fmaxf(-G, 0.0f)*T.wts[q];
        }
        const bool  swap  = (Iplus < Iminus);
        const float hi    = swap ? Iminus : Iplus;
        const float lo    = swap ? Iplus  : Iminus;
        const float ratio = lo/((hi > 0.0f) ? hi : 1.0f);
        const bool  noELN = (Iplus <= 0.0f && Iminus <= 0.0f);
        for(int q=0; q<NQ; q++){
            const float G = (g[         pi[p] *NQ+q] - g[(NF+pi[p])*NQ+q])
                          - (g[         pj[p] *NQ+q] - g[(NF+pj[p])*NQ+q]);
            const float Hplus = (G > 0.0f) ? 1.0f : 0.0f, Hminus = (G < 0.0f) ? 1.0f : 0.0f;
            float P = (1.0f/3.0f)*(swap ? Hplus : Hminus)
                    + (1.0f - (2.0f/3.0f)*ratio)*(swap ? Hminus : Hplus);
            if(noELN) P = 1.0f;
            if(!(P >= 0.0f && P <= 1.0f)) bad = true;
            Psur[p*NQ+q] = P;
        }
        crosses[p] = (Iplus > 0.0f && Iminus > 0.0f) ? 1.0f : 0.0f;
        ncrossing += (int)crosses[p];
        rate[p]    = sqrtf(Iplus*Iminus);
    }
    invN = 1.0f/((ncrossing > 0) ? (float)ncrossing : 1.0f);

    // M[i,j] = (1-Psur_ij)/ncrossing off the diagonal, so M is symmetric and doubly
    // stochastic and the flavor trace is conserved at every direction. Apply (M-I) and
    // integrate against the quadrature in one pass rather than materializing dg.
    for(int i=0; i<NN*4; i++) dF4[i] = 0.0f;
    for(int a=0; a<2; a++) for(int q=0; q<NQ; q++){
        const float m01 = invN*crosses[0]*(1.0f-Psur[0*NQ+q]);
        const float m02 = invN*crosses[1]*(1.0f-Psur[1*NQ+q]);
        const float m12 = invN*crosses[2]*(1.0f-Psur[2*NQ+q]);
        const float g0 = g[(a*NF+0)*NQ+q], g1 = g[(a*NF+1)*NQ+q], g2 = g[(a*NF+2)*NQ+q];
        const float dg[3] = { m01*(g1-g0) + m02*(g2-g0),
                              m01*(g0-g1) + m12*(g2-g1),
                              m02*(g0-g2) + m12*(g1-g2) };
        for(int f=0; f<NF; f++){
            float* o = dF4 + (a*NF+f)*4;
            const float dw = dg[f]*T.wts[q];
            for(int c=0; c<3; c++) o[c] += dw*T.pts[c*NQ+q];
            o[3] += dw;
        }
    }

    for(int i=0; i<NN*4; i++) if(!RHEA_ISFINITE(dF4[i])) bad = true;
    for(int z=0; z<NN; z++) if(F4[z*4+3] + dF4[z*4+3] < 0.0f) bad = true;
    if(bad){
        const float nan = NAN;
        for(int i=0; i<NN*4; i++) dF4[i] = nan;
        for(int p=0; p<3; p++)    rate[p] = nan;
    }
}

//=============#
// predict_all #
//=============#
// F4_in and F4_out are [2,NF,4] in raw number-density units, xyzt last. growthrate comes
// back in number-density units: the caller multiplies by sqrt(2)*G_F/hbar to recover
// 1/s, because doing it here overflows float32.
RHEA_FN void rhea_predict_cell(const RheaTables& T, const float* F4_in, float* F4_out,
                               float* growthrate, float* stability){
    const int NF = T.NF, NN = T.nnodes;
    float scratch[RHEA_MAX_SCRATCH];
    RheaScratch s = rhea_scratch(T, scratch);

    float ntot = 0.0f;
    for(int z=0; z<NN; z++) ntot += F4_in[z*4+3];
    float F4n[64], dF4[64], rate[3];
    for(int i=0; i<NN*4; i++) F4n[i] = F4_in[i]/ntot;

    rhea_box3d(T, F4n, dF4, rate, s.box);

    // the fastest growing pair sets the growth rate; every pair is tested, so a crossing
    // between the heavies is seen
    const float gr_box = fmaxf(rate[0], fmaxf(rate[1], rate[2]));
    // each flavor's two incident pairwise rates, sorted, as extra scalar inputs
    const int ia[3] = {0,0,1}, ib[3] = {1,2,2};
    for(int a=0; a<2; a++) for(int f=0; f<NF; f++){
        const int z = a*NF+f;
        float* x = s.cur + z*T.dim_max;
        for(int c=0; c<4; c++){ x[c] = F4n[z*4+c]; x[4+c] = dF4[z*4+c]; }
        x[8] = fminf(rate[ia[f]], rate[ib[f]]);
        x[9] = fmaxf(rate[ia[f]], rate[ib[f]]);
    }

    float y_F4[64], y_gr[8];
    rhea_forward(T, s, y_F4, y_gr);

    float gr_net = 0.0f;
    for(int z=0; z<NN; z++) gr_net += y_gr[z];
    gr_net /= (float)NN;

    // the F4 head emits its irreps in Gate's order, which need not be [x,y,z,t]
    for(int z=0; z<NN; z++){
        for(int c=0; c<4; c++) F4_out[z*4+c] = F4n[z*4+c] + dF4[z*4+c];
        for(int i=0; i<4; i++) F4_out[z*4 + T.f4_perm[i]] += y_F4[z*4+i];
    }
    float gr = gr_box + gr_net;

    if(T.average_heavies){
        for(int a=0; a<2; a++) for(int c=0; c<4; c++){
            float m = 0.0f;
            for(int f=1; f<NF; f++) m += F4_out[(a*NF+f)*4+c];
            m /= (float)(NF-1);
            for(int f=1; f<NF; f++) F4_out[(a*NF+f)*4+c] = m;
        }
    }

    // the flavor-traced number is conserved
    for(int a=0; a<2; a++) for(int c=0; c<4; c++){
        float so = 0.0f, si = 0.0f;
        for(int f=0; f<NF; f++){ so += F4_out[(a*NF+f)*4+c]; si += F4n[(a*NF+f)*4+c]; }
        const float excess = (so - si)/(float)NF;
        for(int f=0; f<NF; f++) F4_out[(a*NF+f)*4+c] -= excess;
    }

    // ELN is conserved. xyzt index 3 is the number density; correcting a flux component
    // instead would not be rotationally equivariant.
    if(T.conserve_lepton_number){
        for(int f=0; f<NF; f++){
            const float in  = F4n    [f*4+3] - F4n    [(NF+f)*4+3];
            const float out = F4_out [f*4+3] - F4_out [(NF+f)*4+3];
            const float ex  = out - in;
            F4_out[      f *4+3] -= ex/2.0f;
            F4_out[(NF+f)*4+3]   += ex/2.0f;
        }
    }

    for(int i=0; i<NN*4; i++) F4_out[i] *= ntot;
    *growthrate = gr * ntot;
    *stability  = RHEA_ISNAN(gr_box) ? NAN : ((gr_box <= 0.0f) ? 1.0f : 0.0f);
}

#endif
