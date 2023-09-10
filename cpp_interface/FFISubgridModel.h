#include <torch/script.h>
#include <iostream>
#include <string>

using namespace torch::indexing;

template<int NF>
class FFISubgridModel{
  public:

  // constants defining the number of inputs/outputs
  const int NX = NF*(1 + 2*NF); // number of elements in X tensor
  const int Ny = (2*NF)*(2*NF); // number of elements in Y tensor

  // the actual pytorch model
  torch::jit::script::Module model;

  //==================//
  // Four dot-product //
  //==================//
  // input dimensions: [# grid cells, xyzt]
  torch::Tensor dot4_Minkowski(const torch::Tensor& v1, const torch::Tensor& v2){
    // time component is positive
    torch::Tensor result = -v1.index({Slice(),3}) * v2.index({Slice(),3});

    // spatial components are negative
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
    for(int a=0; a<2*NF; a++){
      torch::Tensor F1 = F4_normalized.index({Slice(), Slice(), a});
      for(int b=a; b<2*NF; b++){
        torch::Tensor F2 = F4_normalized.index({Slice(), Slice(), b});
        X.index_put_({Slice(), index}, dot4_Minkowski(F1,F2));
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
    int nsims = X.size(0);
    torch::Tensor y = model.forward({X}).toTensor();
    return y;
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

    return F4_final;
  }

};
