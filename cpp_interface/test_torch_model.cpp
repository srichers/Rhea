#include <torch/script.h>
#include <iostream>

const int NF=3; // number of flavors
const int NX = NF*(1 + 2*NF) + 2*NF; // number of elements in X tensor
const int Ny = 4*NF*(NF-1); // number of elements in Y tensor

//==================//
// Four dot-product //
//==================//
// input dimensions: [4]
double dot4(std::vector<double>& v1, std::vector<double>& v2){
  double result = v1[3]*v2[3] - v1[0]*v2[0] - v1[1]*v2[1] - v1[2]*v2[2];
  return result;
}

//==========================================//
// Create X tensor from four-vector objects //
//==========================================//
// F4 dimensions: [4, 2, NF]
// X dimensions: [NX]
torch::jit::IValue X_from_F4(torch::jit::IValue F4){
  int index = 0;
  int nsims = F4.toTensor().size(0);

}

//======//
// main //
//======//
int main(int argc, const char* argv[]){
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  //===================================//
  // Load the serialized pytorch model //
  //===================================//
  torch::jit::script::Module model;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    model = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
  std::cout << "Model "<< argv[1]<<" loaded fine\n";

  // for now, tell the ML model to run on the CPU
  model.to(torch::kCPU);

  // set the model to evaluation mode
  model.eval();

  //==================================================//
  // Create a sample tensor to pass through the model //
  //==================================================//
  torch::Tensor F4 = torch::zeros({1, NX}, torch::dtype(torch::kFloat32));
  std::cout << F4 << std::endl;

  auto output = model.forward({F4}).toTensor();
  std::cout << output << std::endl;

  // save output to file
  std::ofstream outfile;
  outfile.open("output.txt");
  outfile << output << std::endl;
  outfile.close();

  return 0;
}
