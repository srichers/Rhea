/*
Author: Sherwood Richers

License: GPLv3 (see LICENSE file)

Deterministic test cells, and the file both predictors write. Shared so that the LibTorch
and Kokkos consumers can be compared without ever being linked into the same binary -
LibTorch's headers do not survive nvcc, and keeping the two apart is what removes the need
for a shim between them.

Each predictor writes the inputs it used alongside its outputs, so compare_predictions.py
can confirm the two really saw the same cells before comparing anything else.

Depends on nothing but the standard library, and compiles under g++ and nvcc alike.
*/
#ifndef RHEA_TEST_CELLS_HPP
#define RHEA_TEST_CELLS_HPP

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// input dimensions: [cell, nu/nubar, flavor, xyzt], xyzt last, raw number-density units
inline std::vector<float> rhea_test_cells(int ncells){
    std::vector<float> F4((size_t)ncells*24, 0.0f);
    auto at = [&](int c, int a, int f, int x)->float& { return F4[((size_t)(c*2 + a)*3 + f)*4 + x]; };

    // cell 0 is the NSM1 point this test has always used
    at(0,0,0,3) = 1.4220e+33f;  at(0,1,0,3) = 1.9146e+33f;
    at(0,0,1,3) = 4.7209e+32f;  at(0,1,1,3) = 4.7209e+32f;
    at(0,0,2,3) = 4.7209e+32f;  at(0,1,2,3) = 4.7209e+32f;
    at(0,0,0,0) = 0.0974e+33f;  at(0,0,0,1) = 0.0421e+33f;  at(0,0,0,2) = -0.1343e+33f;
    at(0,1,0,0) = 0.0723e+33f;  at(0,1,0,1) = 0.0313e+33f;  at(0,1,0,2) = -0.3446e+33f;
    at(0,0,1,0) = 0.0216e+32f;  at(0,0,1,1) = 0.0093e+32f;  at(0,0,1,2) = -0.0217e+32f;
    at(0,1,1,0) = 0.0216e+32f;  at(0,1,1,1) = 0.0093e+32f;  at(0,1,1,2) = -0.0217e+32f;
    at(0,0,2,0) = 0.0216e+32f;  at(0,0,2,1) = 0.0093e+32f;  at(0,0,2,2) = -0.0217e+32f;
    at(0,1,2,0) = 0.0216e+32f;  at(0,1,2,1) = 0.0093e+32f;  at(0,1,2,2) = -0.0217e+32f;

    // the rest sweep magnitude and flux factor; the last three are deliberately
    // unphysical (flux larger than density) so the nan path is exercised on both sides
    uint32_t rng = 12345u;
    auto uniform = [&]()->float {
        rng = rng*1664525u + 1013904223u;
        return (float)((rng >> 8) & 0xFFFFFF) / (float)0x1000000;
    };
    for(int c=1; c<ncells; c++){
        const float scale = powf(10.0f, 30.0f + 4.0f*uniform());
        const float ff    = (c >= ncells-3) ? 1.4f : 0.9f*uniform();
        for(int a=0; a<2; a++) for(int f=0; f<3; f++){
            const float n = scale*(0.2f + uniform());
            at(c,a,f,3) = n;
            float d[3], norm = 0.0f;
            for(int x=0; x<3; x++){ d[x] = uniform() - 0.5f; norm += d[x]*d[x]; }
            norm = sqrtf(norm);
            for(int x=0; x<3; x++) at(c,a,f,x) = n*ff*d[x]/norm;
        }
    }
    return F4;
}

// magic, ncells, then F4_in, F4_out, growthrate, stability as little-endian float32
inline void rhea_write_predictions(const std::string& filename, int ncells,
                                   const float* F4_in, const float* F4_out,
                                   const float* growthrate, const float* stability){
    FILE* f = fopen(filename.c_str(), "wb");
    if(!f) throw std::runtime_error("cannot write " + filename);
    fwrite("RHEAPRED", 1, 8, f);
    fwrite(&ncells, 4, 1, f);
    fwrite(F4_in,      4, (size_t)ncells*24, f);
    fwrite(F4_out,     4, (size_t)ncells*24, f);
    fwrite(growthrate, 4, (size_t)ncells,    f);
    fwrite(stability,  4, (size_t)ncells,    f);
    fclose(f);
    printf("#  wrote %s (%d cells)\n", filename.c_str(), ncells);
}

#endif
