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
  torch::Tensor dot4_Minkowski(const torch::Tensor& v1, const torch::Tensor& v2){
    // time component is negative
    torch::Tensor result = -v1.index({Slice(),3}) * v2.index({Slice(),3});

    // spatial components are positive
    for(int i=0; i<3; i++){
      result += v1.index({Slice(),i}) * v2.index({Slice(),i});
    }

    return result;
  }

  //==========================================//
  // Create X tensor from four-vector objects //
  //==========================================//
  // F4 dimensions: [# grid cells, 4, 2, NF]
  // X dimensions: [# grid cells, NX]
  // assume Minkowski metric
  // Assuming NF==3, the X tensor is
  // 0: Fe.Fe
  // 1: Fe.Fmu
  // 2: Fe.Ftau
  // 3: Fe.Febar
  // 4: Fe.Fmubar
  // 5: Fe.Ftaubar
  // 6: Fmu.Fmu
  // 7: Fmu.Ftau
  // 8: Fmu.Febar
  // 9: Fmu.Fmubar
  // 10: Fmu.Ftaubar
  // 11: Ftau.Ftau
  // 12: Ftau.Febar
  // 13: Ftau.Fmubar
  // 14: Ftau.Ftaubar
  // 15: Febar.Febar
  // 16: Febar.Fmubar
  // 17: Febar.Ftaubar
  // 18: Fmubar.Fmubar
  // 19: Fmubar.Ftaubar
  // 20: Ftaubar.Ftaubar
  // IF do_fdotu:
  // 21: Fe.u
  // 22: Fmu.u
  // 23: Ftau.u
  // 24: Febar.u
  // 25: Fmubar.u
  // 26: Ftaubar.u
  torch::Tensor X_from_F4_Minkowski(const torch::Tensor F4){
    int nsims = F4.size(0);

    // calculate the total number density based on the t component of the four-vector
    // [sim]
    torch::Tensor ndens_total = torch::sum(F4.index({Slice(), 3, Slice(), Slice()}), {1,2});

    // normalize F4 by the total number density
    // [sim, xyzt, 2*NF]
    torch::Tensor F4_normalized = F4.reshape({nsims,4,2*NF}) / ndens_total.reshape({nsims,1,1});

    // create the X tensor
    torch::Tensor X = torch::zeros({nsims, NX}, torch::dtype(torch::kFloat32));
    int index = 0;

    // put the dot product of each species with each other species into the X tensor
    for(int a=0; a<2*NF; a++){
      torch::Tensor F1 = F4_normalized.index({Slice(), Slice(), a});
      for(int b=a; b<2*NF; b++){
        torch::Tensor F2 = F4_normalized.index({Slice(), Slice(), b});
        X.index_put_({Slice(), index}, dot4_Minkowski(F1,F2));
        index++;
      }
    }

    if(do_fdotu){
      // the fluid velocity just has a 1 in the t component
      torch::Tensor u = torch::zeros({nsims,4});
      u.index_put_({Slice(), 3}, 1.0);

      // put the dot product of each species with the fluid velocity into the X tensor
      for(int a=0; a<2*NF; a++){
        torch::Tensor F1 = F4_normalized.index({Slice(), Slice(), a});
        torch::Tensor u = F4_normalized.index({Slice(), Slice(), 3});
        X.index_put_({Slice(), index}, dot4_Minkowski(F1,u));
        index++;
      }
    }

    assert(index == NX);
    return X;
  }


  //===================================//
  // Load the serialized pytorch model //
  //===================================//
  FFISubgridModel(std::string filename){
    // Deserialize the ScriptModule from a file using torch::jit::load().
    model = torch::jit::load(filename.c_str());

    // set the model to evaluation mode
    model.eval();
  }

  // function that takes in a list of X (built from dot products) and outputs the prediction for the transformation matrix
  // inputs are arrays if four-vector sets.
  // the dimensions of X are [# grid cells, NX]
  // the dimensions of y are [# grid cells, 2,NF,2,NF]
  torch::Tensor predict_y(const torch::Tensor& X){
    return model.forward({X}).toTensor();
  }

  // convert a 3-flavor y tensor to a 2-flavor y tensor
  // y3 has shape [sim,2,NF,2,NF]
  // y2 has shape [sim,2,2,2]
  torch::Tensor convert_y_to_2F(const torch::Tensor& y){
    int nsims = y.size(0);
    torch::Tensor y2 = torch::zeros({nsims,2,2,2,2});
    y2.index_put_({Slice(), Slice(), 0, Slice(), 0},                 (y.index({Slice(), Slice(), 0            , Slice(), 0            })       ));
    y2.index_put_({Slice(), Slice(), 0, Slice(), 1}, 0.5 * torch::sum(y.index({Slice(), Slice(), 0            , Slice(), Slice(1,None)}), {  3}));
    y2.index_put_({Slice(), Slice(), 1, Slice(), 1}, 0.5 * torch::sum(y.index({Slice(), Slice(), Slice(1,None), Slice(), Slice(1,None)}), {2,4}));
    y2.index_put_({Slice(), Slice(), 1, Slice(), 0},       torch::sum(y.index({Slice(), Slice(), Slice(1,None), Slice(), 0            }), {2  }));
    return y2;
  }

  // function to convert an input F4 and y into an output F4
  // mirrors the python function F4_from_y
  // F4_initial must have shape [sim, xyzt, nu/nubar, flavor]
  // y must have shape [sim,2,NF,2,NF]
  // F4_final has shape [sim, xyzt, nu/nubar, flavor]
  torch::Tensor F4_from_y(const torch::Tensor& F4_initial, const torch::Tensor& y){
    // tensor product with y
    // n indicates the simulation index
    // m indicates the spacetime index
    // i and j indicate nu/nubar
    // a and b indicate flavor
    torch::Tensor F4_final = torch::einsum("niajb,nmjb->nmia", {y, F4_initial});

    // make sure the result is physical
    F4_final = restrict_to_physical(F4_final);

    return F4_final;
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
    assert(torch::all(radical>=-1e-6).item<bool>());
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
