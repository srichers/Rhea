/*
Author: Sherwood Richers

License: GPLv3 (see LICENSE file)

Example program that runs a test data point through a trained ML model.
 */

#include "FFISubgridModel.h"
#include <torch/torch.h>

//======//
// main //
//======//
int main(int argc, const char* argv[]){
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  auto device = torch::kCUDA; //torch::kCPU; //

  FFISubgridModel<3> model(std::string(argv[1]), device);

  std::cout << "Available methods in the module:" << std::endl;
  for (const auto& method : model.model.get_methods()) {
    std::cout << method.name() << std::endl;
  }
  
  auto options =
    torch::TensorOptions()
    .device(device)
    .requires_grad(false);
  
  //==================================================//
  // Create a sample tensor to pass through the model //
  //==================================================//
  // dimensions are [ngridzones, xyzt, nu/nubar, NF]
  const int ngridzones = 1;
  torch::Tensor F4_in = torch::zeros({ngridzones,4,2,3}, options);
  std::cout << "tensor device: " << F4_in.device() << std::endl;
  std::cout << std::endl;
  // Fiducial
  //F4_in.index_put_({Slice(), 3, 0, 0},  1.0  );
  //F4_in.index_put_({Slice(), 3, 1, 0},  1.0  );
  //F4_in.index_put_({Slice(), 2, 0, 0},  1./3.);
  //F4_in.index_put_({Slice(), 2, 1, 0}, -1./3.);
  F4_in.index_put_({Slice(), 3, 0, 0},  1.4220e+33  ); //Nee
  F4_in.index_put_({Slice(), 3, 1, 0},  1.9146e+33  ); //Neebar
  F4_in.index_put_({Slice(), 3, 0, 1},  4.9113e+32  ); //Nmumu
  F4_in.index_put_({Slice(), 3, 1, 1},  4.9113e+32  ); //Nmumubar
  F4_in.index_put_({Slice(), 3, 0, 2},  4.9113e+32  ); //Ntautau
  F4_in.index_put_({Slice(), 3, 1, 2},  4.9113e+32  ); //Ntautaubar

  F4_in.index_put_({Slice(), 0, 0, 0},  1.3850e+32  ); //Fee_x
  F4_in.index_put_({Slice(), 0, 1, 0},  1.3843e+32  ); //Feebar_x
  F4_in.index_put_({Slice(), 0, 0, 1},  -1.0608e+31 ); //Fmumu_x
  F4_in.index_put_({Slice(), 0, 1, 1},  -1.0608e+31 ); //Fmumubar_x
  F4_in.index_put_({Slice(), 0, 0, 2},  -1.0608e+31 ); //Ftautau_x
  F4_in.index_put_({Slice(), 0, 1, 2},  -1.0608e+31 ); //Ftautaubar_x
  
  F4_in.index_put_({Slice(), 1, 0, 0},  5.9866e+31  ); //Fee_y
  F4_in.index_put_({Slice(), 1, 1, 0},  5.9927e+31  ); //Feebar_y
  F4_in.index_put_({Slice(), 1, 0, 1},  3.6491e+31  ); //Fmumu_y
  F4_in.index_put_({Slice(), 1, 1, 1},  3.6491e+31  ); //Fmumubar_y
  F4_in.index_put_({Slice(), 1, 0, 2},  3.6491e+31  ); //Ftautau_y
  F4_in.index_put_({Slice(), 1, 1, 2},  3.6491e+31  ); //Ftautaubar_y
  
  F4_in.index_put_({Slice(), 2, 0, 0},  -1.9097e+32  ); //Fee_z
  F4_in.index_put_({Slice(), 2, 1, 0},  -6.5977e+32  ); //Feebar_z
  F4_in.index_put_({Slice(), 2, 0, 1},  -2.6295e+32  ); //Fmumu_z
  F4_in.index_put_({Slice(), 2, 1, 1},  -2.6295e+32  ); //Fmumubar_z
  F4_in.index_put_({Slice(), 2, 0, 2},  -2.6295e+32  ); //Ftautau_z
  F4_in.index_put_({Slice(), 2, 1, 2},  -2.6295e+32  ); //Ftautaubar_z*/

  std::cout << std::endl;
  std::cout << "Input number densities" << std::endl;
  std::cout << F4_in.index({0,3,Slice(),Slice()}) << std::endl;

  // fetch output
  torch::IValue output = model.model.get_method("predict_all")({F4_in});

  // Convert to tuple
  auto output_tuple = output.toTuple()->elements();
  
  // Extract individual tensors
  torch::Tensor F4_out        = output_tuple[0].toTensor();
  torch::Tensor logGrowthRate = output_tuple[1].toTensor();
  torch::Tensor stability     = output_tuple[2].toTensor();

  std::cout << std::endl;
  std::cout << "Output number densities" << std::endl;
  std::cout << F4_out.index({0,3,Slice(),Slice()}) << std::endl;
  std::cout << "Stability: " << stability << std::endl;
  std::cout << "logGrowthRate: " << torch::exp(logGrowthRate) << std::endl;
  std::cout << std::endl << "==========================" << std::endl;
  
  return 0;
}
