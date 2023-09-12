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
  F4_in.index_put_({Slice(), 3, 0, 0},  1.0  );
  F4_in.index_put_({Slice(), 3, 1, 0},  1.0  );
  F4_in.index_put_({Slice(), 2, 0, 0},  1./3.);
  F4_in.index_put_({Slice(), 2, 1, 0}, -1./3.);
  std::cout << std::endl;
  std::cout << "F4_in" << std::endl;
  std::cout << F4_in.index({Slice(),3,Slice(),Slice()}) << std::endl;

  // put the input through the model 10 times
  auto F4_out = F4_in;
  torch::Tensor X, y;
  //for(int i=0; i<10; i++){
    X = model.X_from_F4_Minkowski(F4_out);
    y = model.predict_y(X);
    F4_out = model.F4_from_y(F4_out, y);
  //}

  // the expected result is an even mixture of all flavors
  torch::Tensor F4_expected = torch::zeros({ngridzones,4,2,3});
  F4_expected.index_put_({Slice(), 3, Slice(), Slice()}, 1./3.);

  // check that the results are correct
  // by asserting that all elements are equal to 1 with an absolute and relative tolerance of 1e-2
  std::cout << std::endl;
  std::cout << "F4_out" << std::endl;
  std::cout << F4_out.index({Slice(),3,Slice(),Slice()}) << std::endl;
  std::cout << std::endl;
  std::cout << "y" << std::endl;
  std::cout << y.index({Slice(), 0,0,Slice(),Slice()}) << std::endl;
  std::cout << std::endl << "==========================" << std::endl;
  //assert(torch::allclose(F4_out, F4_expected, 3e-2, 3e-2));
  
  //====================================//
  // Test the two-flavor transformation //
  //====================================//
  torch::Tensor F4_in_2F = torch::zeros({ngridzones,4,2,2});
  F4_in_2F.index_put_({Slice(), 3, 0, 0}, 1.0);
  F4_in_2F.index_put_({Slice(), 3, 1, 0}, 1.0);
  std::cout << std::endl;
  std::cout << "F4_in_2F" << std::endl;
  std::cout << F4_in_2F.index({Slice(),3,Slice(),Slice()}) << std::endl;

  // the expected result is an even mixture of all flavors
  torch::Tensor F4_expected_2F = torch::zeros({ngridzones,4,2,2});
  F4_expected_2F.index_put_({Slice(), 3, Slice(), 0}, 1./3.);
  F4_expected_2F.index_put_({Slice(), 3, Slice(), 1}, 2./3.);

  torch::Tensor y2F = model.convert_y_to_2F(y);
  torch::Tensor F4_out_2F = model.F4_from_y(F4_in_2F, y2F);

  // check that the results are correct
  // by asserting that all elements are equal to 1 with an absolute and relative tolerance of 1e-2
  std::cout << std::endl;
  std::cout << "F4_out_2F" << std::endl;
  std::cout << F4_out_2F.index({Slice(),3,Slice(),Slice()}) << std::endl;
  std::cout << std::endl;
  std::cout << "y2F" << std::endl;
  std::cout << y2F.index({Slice(), 0,0,Slice(),Slice()}) << std::endl;
  //assert(torch::allclose(F4_out_2F, F4_expected_2F, 3e-2, 3e-2));
  

  return 0;
}
