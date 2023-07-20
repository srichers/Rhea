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

  // the actual pytorch model
  torch::jit::script::Module model;

  //==================//
  // Four dot-product //
  //==================//
  // input dimensions: [# grid cells, 4]
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
    int index = 0;
    int nsims = F4.size(0);

    torch::Tensor X = torch::zeros({nsims, NX}, torch::dtype(torch::kFloat32));
    torch::Tensor F4_flat = F4.reshape({nsims, 4, 2*NF}); // [simulationIndex, xyzt, species]
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
    std::cout << "Model "<< filename <<" loaded fine\n";

    // for now, tell the ML model to run on the CPU
    model.to(torch::kCPU);

    // set the model to evaluation mode
    model.eval();
  }

  // function that takes in a list of F4 vectors and outputs the prediction
  // input and output are arrays if four-vector sets.
  // they should be indexed according to [grid cell index, spacetime component, matter/antimatter, flavor]
  // they should have size [# grid cells, 4, 2, NF]
  torch::Tensor predict(const torch::Tensor& F4_in, const torch::Tensor& u){
    torch::Tensor X = X_from_F4(F4_in, u);
    std::cout << X << std::endl;

    auto F4_out = model.forward({X}).toTensor();

    return F4_out;
  }

};