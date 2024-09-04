/*
Author: Sherwood Richers

License: GPLv3 (see LICENSE file)

This header file can be included in a C++ program to more easily interface with a trained ML model.
 */

#include <torch/script.h>
#include <iostream>
#include <string>

using namespace torch::indexing;

template<int NF>
class FFISubgridModel{
  public:

  // constants defining the number of inputs/outputs
  // NX = NF*(1 + 2*NF) if no fdotu input, with additional 2*NF if fdotu input
  // NY is (2*NF)*(2*NF)
  const bool do_fdotu = true; // assume to be true for now
  const int NX_nofdotu = NF*(1 + 2*NF);
  const int NX_fdotu = NX_nofdotu + 2*NF;
  const int NX = do_fdotu ? NX_fdotu : NX_nofdotu;
  const int Ny = (2*NF)*(2*NF); // number of elements in Y tensor
  
  // the actual pytorch model
  torch::jit::script::Module model;

  //==================//
  // Four dot-product //
  //==================//
  // input dimensions: [# grid cells, xyzt]
  auto dot4_Minkowski(const auto& v1, const auto& v2){
    // time component is negative
    auto result = -v1.index({Slice(),3}) * v2.index({Slice(),3});

    // spatial components are positive
    for(int i=0; i<3; i++){
      result += v1.index({Slice(),i}) * v2.index({Slice(),i});
    }

    return result;
  }

  //===================================//
  // Load the serialized pytorch model //
  //===================================//
  FFISubgridModel(std::string filename, auto device){
    // Deserialize the ScriptModule from a file using torch::jit::load().
    model = torch::jit::load(filename.c_str());
    model.to(device);

    // set the model to evaluation mode
    model.eval();
  }

  // ensure that the four-vectors are time-like and have positive density
  // F4_final has shape [sim, xyzt, nu/nubar, flavor]
  torch::Tensor restrict_to_physical(const torch::Tensor& F4_final){
    torch::Tensor avgF4 = torch::sum(F4_final, {3}).index({Slice(), Slice(), Slice(), None}) / NF; // [:,:,:,None]

    // enforce that all four-vectors are time-like
    // choose the branch that leads to the most positive alpha
    torch::Tensor a = dot4_Minkowski(avgF4, avgF4); // [sim, nu/nubar, flavor];
    torch::Tensor b = 2.*dot4_Minkowski(avgF4, F4_final);
    torch::Tensor c = dot4_Minkowski(F4_final, F4_final);
    torch::Tensor radical = b*b - 4*a*c;
    //assert(torch::all(radical>=-1e-6).item<bool>());
    radical = torch::maximum(radical, torch::zeros_like(radical));
    torch::Tensor alpha = (-b + torch::sign(a)*torch::sqrt(radical)) / (2*a);

    // fix the case where a is zero
    torch::where(torch::abs(a/b)<1e-6, (-c/b), alpha);

    // find the nu/nubar and flavor indices where alpha is maximized
    torch::Tensor maxalpha = torch::amax(alpha, {1,2}); // [sim]
    maxalpha += 1e-6;
    maxalpha = torch::maximum(maxalpha, torch::zeros_like(maxalpha));

    // modify the four-vectors to be time-like
    torch::Tensor result = (F4_final + maxalpha.index({Slice(),None,None,None})*avgF4) / (maxalpha.index({Slice(),None,None,None}) + 1);

    return result;
  }

};
