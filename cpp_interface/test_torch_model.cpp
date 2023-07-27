#include "FFISubgridModel.h"

//======//
// main //
//======//
int main(int argc, const char* argv[]){
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  FFISubgridModel<3> model{std::string(argv[1])};

  //==================================================//
  // Create a sample tensor to pass through the model //
  //==================================================//
  torch::Tensor F4_in = torch::zeros({1,4,2,3});
  F4_in.index_put_({torch::indexing::Slice(), 3, 0, 0}, 1.0);
  F4_in.index_put_({torch::indexing::Slice(), 3, 1, 0}, 1.0);

  torch::Tensor u = torch::zeros({1,4});
  u.index_put_({torch::indexing::Slice(),3}, 1.0);

  auto output = model.predict_F4(F4_in, u);
  std::cout << output << std::endl;

  return 0;
}
