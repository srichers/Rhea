#include <torch/script.h>
#include <iostream>
#include <string>

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
  // input dimensions: [4]
  static double dot4(const std::vector<double>& v1, const std::vector<double>& v2){
    double result = v1[3]*v2[3] - v1[0]*v2[0] - v1[1]*v2[1] - v1[2]*v2[2];
    return result;
  }

  //==========================================//
  // Create X tensor from four-vector objects //
  //==========================================//
  // F4 dimensions: [4, 2, NF]
  // X dimensions: [NX]
  /*static torch::jit::IValue X_from_F4(const torch::jit::IValue F4){
    int index = 0;
    int nsims = F4.toTensor().size(0);
  }*/

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
  torch::Tensor predict(const torch::Tensor& F4_in){
    torch::Tensor X = torch::zeros({1, NX}, torch::dtype(torch::kFloat32));
    std::cout << X << std::endl;

    auto F4_out = model.forward({X}).toTensor();

    return F4_out;
  }

};