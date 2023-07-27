#include <torch/script.h>
#include <iostream>
#include <string>

using namespace torch::indexing;

template<int NF>
class FFISubgridModel{
  public:

  // constants defining the number of inputs/outputs
  const int NX = NF*(1 + 2*NF) + 2*NF; // number of elements in X tensor
  const int Ny = 4*NF*(NF-1); // number of elements in Y tensor
  const int Nalpha = 4*NF*(NF-1); // number of elements in alpha tensor

  // the actual pytorch model
  torch::jit::script::Module model;

  //==================//
  // Four dot-product //
  //==================//
  // input dimensions: [# grid cells, xyzt]
  torch::Tensor dot4(const torch::Tensor& v1, const torch::Tensor& v2){
    // time component is positive
    torch::Tensor result = v1.index({Slice(),3}) * v2.index({Slice(),3});

    // spatial components are negative
    for(int i=0; i<3; i++){
      result -= v1.index({Slice(),i}) * v2.index({Slice(),i});
    }

    return result;
  }

  //==========================================//
  // Create X tensor from four-vector objects //
  //==========================================//
  // F4 dimensions: [# grid cells, 4, 2, NF]
  // X dimensions: [# grid cells, NX]
  // u dimensions: [# grid cells, 4]
  torch::Tensor X_from_F4(const torch::Tensor F4, const torch::Tensor u){
    int nsims = F4.size(0);

    // copy the input tensor so the values don't change
    torch::Tensor F4_flat = F4.detach().clone().reshape({nsims, 4, 2*NF}); // [# grid cells, xyzt, species]

    // calculate the total number density
    torch::Tensor ndens_total = torch::zeros({nsims});
    for(int a=0; a<2*NF; a++){
      torch::Tensor F1 = F4_flat.index({Slice(), Slice(), a}); // [# grid cells, xyzt]
      ndens_total += dot4(F1,u); // [# grid cells]
    }

    // normalize F4_flat by the total number density
    F4_flat /= ndens_total.reshape({nsims,1,1});

    // create the X tensor
    torch::Tensor X = torch::zeros({nsims, NX}, torch::dtype(torch::kFloat32));
    int index = 0;
    for(int a=0; a<2*NF; a++){
      torch::Tensor F1 = F4_flat.index({Slice(), Slice(), a});
      for(int b=a; b<2*NF; b++){
        torch::Tensor F2 = F4_flat.index({Slice(), Slice(), b});

        X.index_put_({Slice(), index}, dot4(F1,F2));
        index += 1;
      }

      // add the dot product with u
      // subtract mean value to zero-center the data
      X.index_put_({Slice(), index}, dot4(F1,u) - 1./(2*NF));
      index += 1;
    }
    assert(index==NX);
    return X;
  }


  //===================================//
  // Load the serialized pytorch model //
  //===================================//
  FFISubgridModel(std::string filename){
    try {
      // Deserialize the ScriptModule from a file using torch::jit::load().
      model = torch::jit::load(filename.c_str());
    }
    catch (const c10::Error& e) {
      std::cerr << "error loading the model\n";
      exit(1);
    }

    // for now, tell the ML model to run on the CPU
    model.to(torch::kCPU);

    // set the model to evaluation mode
    model.eval();
  }

  // function that takes in a list of F4 vectors and outputs the prediction for the transformation matrix
  // inputs are arrays if four-vector sets.
  // the dimensions of F4_initial are [# grid cells, xyzt, 2, NF]
  // the dimensions of u are [# grid cells, xyzt]
  // the dimensions of the output are [# grid cells, Ny]
  torch::Tensor predict_y(const torch::Tensor& F4_initial, const torch::Tensor& u){
    torch::Tensor X = X_from_F4(F4_initial, u);

    torch::Tensor F4_final = model.forward({X}).toTensor();

    return F4_final;
  }

  // function to convert an input F4 and y into an output F4
  // mirrors the python function F4_from_y
  torch::Tensor F4_from_y(const torch::Tensor& F4_initial, const torch::Tensor& y){
    int nsims = F4_initial.size(0);
    torch::Tensor alpha = y.index({Slice(), Slice(0,Nalpha)}).reshape({nsims, 2*NF, 2*(NF-1)});
    torch::Tensor F4i_flat = F4_initial.reshape({nsims, 4, 2*NF}); // [simulationIndex, xyzt, species]
    torch::Tensor F4_final = torch::zeros({nsims, 4, 2, NF});
    F4_final.index_put_({Slice(), Slice(), Slice(), Slice(0,NF-1)}, torch::matmul(F4i_flat, alpha).reshape({nsims,4,2,NF-1}));

    // Set the final flavor such that the flavor trace is conserved
    auto tmp = torch::sum(F4_initial, 3) - torch::sum(F4_final.index({Slice(), Slice(), Slice(), Slice(0,NF-1)}), 3);
    F4_final.index_put_({Slice(), Slice(), Slice(), Slice(NF-1,NF)}, tmp.reshape({nsims,4,2,1}));

    // set the antineutrino number densities to conserve lepton number
    F4_final.index_put_({Slice(), 3, 1, Slice()}, F4_initial.index({Slice(), 3, 1, Slice()}) + (F4_final.index({Slice(), 3, 0, Slice()}) - F4_initial.index({Slice(), 3, 0, Slice()})));

    return F4_final;
  }

  // function that takes in a list of F4 vectors and outputs the prediction of F4_final
  // inputs are arrays if four-vector sets.
  // the dimensions of F4_initial are [# grid cells, xyzt, 2, NF]
  // the dimensions of u are [# grid cells, xyzt]
  // the dimensions of the output are [# grid cells, xyzt, 2, NF]
  torch::Tensor predict_F4(const torch::Tensor& F4_initial, const torch::Tensor& u){
    torch::Tensor y = predict_y(F4_initial, u);

    torch::Tensor F4_final = F4_from_y(F4_initial, y);

    return F4_final;
  }

};