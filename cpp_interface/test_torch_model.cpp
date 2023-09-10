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
  // dimensions are [ngridzones, xyzt, nu/nubar, NF]
  const int ngridzones = 1;
  torch::Tensor F4_in = torch::zeros({ngridzones,4,2,3});
  F4_in.index_put_({Slice(), 3, 0, 0}, 3.0);
  F4_in.index_put_({Slice(), 3, 1, 0}, 3.0);

  // put the input through the model 10 times
  auto F4_out = F4_in;
  torch::Tensor X, y;
  for(int i=0; i<10; i++){
    X = model.X_from_F4_Minkowski(F4_out);
    y = model.predict_y(X);
    F4_out = model.F4_from_y(F4_out, y);
  }

  // the expected result is an even mixture of all flavors
  torch::Tensor F4_expected = torch::zeros({ngridzones,4,2,3});
  F4_expected.index_put_({Slice(), 3, Slice(), Slice()}, 1.0);

  // check that the results are correct
  // by asserting that all elements are equal to 1 with an absolute and relative tolerance of 1e-2
  std::cout << F4_out << std::endl;
  assert(torch::allclose(F4_out, F4_expected, 3e-2, 3e-2));
  
  
  return 0;
}
