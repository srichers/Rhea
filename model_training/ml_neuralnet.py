'''
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This file contains the structure of the neural network, including the means of transforming between angular moments and the actual inputs/outputs of the ML model
'''

import torch
import numpy as np
from torch import nn
from ml_tools import restrict_F4_to_physical, ntotal
import e3nn.o3
import e3nn.nn
import box3d

# generate the set of permutation equivariant inputs for each node
def PE_inputs(x):
        # compute the inputs to each node
        # subtract self to get the messages from neighbors orthogonal to the self node
        x_self = x
        x_flavor  = x.sum(dim=2, keepdim=True) - x_self
        x_nunubar = x.sum(dim=1, keepdim=True) - x_self
        x_all = x.sum(dim=(1,2), keepdim=True) - x_self - x_flavor - x_nunubar

        # normalize by the number of nodes contributing. Supposedly helps keep the scale of the activations reasonable.
        x_flavor  = x_flavor / (x.shape[2] - 1)
        x_nunubar = x_nunubar / (x.shape[1] - 1)
        x_all     = x_all / ((x.shape[1] - 1) * (x.shape[2] - 1))

        return x_self, x_flavor, x_nunubar, x_all

# permutation equivariant tensor product between irreps and their norms
class PETP_Norm(nn.Module):
    def __init__(self, irreps_in, irreps_out):
        super().__init__()
        self.norm = e3nn.o3.Norm(irreps_in)
        self.tp_self    = e3nn.o3.FullyConnectedTensorProduct(irreps_in, self.norm.irreps_out, irreps_out)
        self.tp_flavor  = e3nn.o3.FullyConnectedTensorProduct(irreps_in, self.norm.irreps_out, irreps_out)
        self.tp_nunubar = e3nn.o3.FullyConnectedTensorProduct(irreps_in, self.norm.irreps_out, irreps_out)
        self.tp_all     = e3nn.o3.FullyConnectedTensorProduct(irreps_in, self.norm.irreps_out, irreps_out)

    # x has shape [nsamples, nunubar, flavor, features]
    def forward(self, x):
        # compute the inputs to each node
        x_self, x_flavor, x_nunubar, x_all = PE_inputs(x)

        # compute invariant scalars from all irreps
        s_self = self.norm(x_self)
        s_flavor = self.norm(x_flavor)
        s_nunubar = self.norm(x_nunubar)
        s_all = self.norm(x_all)

        # apply the tensor products to the original features, using the invariant scalars as gates
        y_self = self.tp_self(x_self, s_self)
        y_flavor = self.tp_flavor(x_flavor, s_flavor)
        y_nunubar = self.tp_nunubar(x_nunubar, s_nunubar)
        y_all = self.tp_all(x_all, s_all)

        # sum the contributions
        y = y_self + y_flavor + y_nunubar + y_all
        return y

# permutation equivariant tensor product between irreps and themselves (quadratic in the features)
class PETP_Quadratic(nn.Module):
    def __init__(self, irreps_in, irreps_out):
        super().__init__()
        self.tp_self    = e3nn.o3.FullyConnectedTensorProduct(irreps_in, irreps_in, irreps_out)
        self.tp_flavor  = e3nn.o3.FullyConnectedTensorProduct(irreps_in, irreps_in, irreps_out)
        self.tp_nunubar = e3nn.o3.FullyConnectedTensorProduct(irreps_in, irreps_in, irreps_out)
        self.tp_all     = e3nn.o3.FullyConnectedTensorProduct(irreps_in, irreps_in, irreps_out)

    # x has shape [nsamples, nunubar, flavor, features]
    def forward(self, x):
        # compute the inputs to each node
        x_self, x_flavor, x_nunubar, x_all = PE_inputs(x)

        # compute tensor products
        y_self = self.tp_self(x_self, x_self)
        y_flavor = self.tp_flavor(x_flavor, x_flavor)
        y_nunubar = self.tp_nunubar(x_nunubar, x_nunubar)
        y_all = self.tp_all(x_all, x_all)

        # sum the contributions
        y = y_self + y_flavor + y_nunubar + y_all
        return y

class PE_GatedBlock(nn.Module):
    def __init__(self, irreps_in, irreps_out, act_scalars, act_gates, tensor_product_class):
        super().__init__()
        # determine the irreps that need to go into gate
        self.irreps_out = irreps_out
        irreps_scalars = irreps_out.filter(lambda mul_ir: mul_ir.ir.l == 0)
        irreps_nonscalars = irreps_out.filter(lambda mul_ir: mul_ir.ir.l > 0)
        irreps_gates = e3nn.o3.Irreps(f"{irreps_nonscalars.num_irreps}x0e")
        irreps_with_gates = irreps_scalars + irreps_gates + irreps_nonscalars

        # define the tensor product that produces the scalars, gates, and nonscalars all at once
        self.tp = tensor_product_class(irreps_in, irreps_with_gates)

        # gate the scalars, and use additional gate scalars to gate the nonscalars
        self.gate = e3nn.nn.Gate(
            irreps_scalars = irreps_scalars,
            act_scalars = [act_scalars] * len(irreps_scalars),
            irreps_gates = irreps_gates,
            act_gates = [act_gates] * len(irreps_gates),
            irreps_gated = irreps_nonscalars,
        )

    def forward(self, x):
        # get the full output (scalars + gates + nonscalars) from one linear
        y = self.tp(x)

        # apply the gate. 
        y = self.gate(y)

        return y
    
class PE_ResidualGatedBlock(PE_GatedBlock):
    def __init__(self, irreps_in, irreps_out, act_scalars, act_gates, tensor_product_class):
        super().__init__(irreps_in, irreps_out, act_scalars, act_gates, tensor_product_class)
        assert irreps_in == irreps_out, "For residual connection, input and output irreps must be the same"

    def forward(self, x):
        # get the full output (scalars + gates + nonscalars) from one linear
        y = self.tp(x)

        # apply the gate. 
        y = self.gate(y)

        return x + y
    
# define the NN model
class NeuralNetwork(nn.Module):
    def __init__(self, parms):
        super().__init__()

        # make sure the irreps are scalar first to make sure it is consistent with the output of Gate
        assert parms["irreps_hidden"] == parms["irreps_hidden"].sort().irreps, "Hidden irreps must have scalars first"

        # store input arguments
        self.NF = parms["NF"]

        # store loss weight parameters for tasks
        self.log_task_weights = nn.ParameterDict({
            name: nn.Parameter(
                torch.tensor(-np.log(parms[f"task_weight_{name}"]), dtype=torch.float32),
                requires_grad=parms["do_learn_task_weights"],
            )
            for name in ["growthrate", "F4", "unphysical"]
        })

        # The input irreps for each node are just the 4 components of F4_in followed by 4 components of F4_box3d
        self.irreps_in  = e3nn.o3.Irreps("1x1o + 1x0e + 1x1o + 1x0e")
        
        # one y matrix for ndens, one for flux
        # add one extra to predict growth rate
        self.average_heavies_in_final_state = parms["average_heavies_in_final_state"]
        self.conserve_lepton_number = parms["conserve_lepton_number"]

        # append a full layer including linear, activation, and batchnorm/dropout if desired
        def append_full_layer(modules, in_irreps, out_irreps):

            # use PE layer if irreps_in==irreps_out
            if in_irreps == out_irreps:
                gate_class = PE_ResidualGatedBlock
            else:
                gate_class = PE_GatedBlock
            
            # select the type of tensor product to use in the gate
            if parms["tensor_product_class"] == "norm":
                tensor_product_class = PETP_Norm
            elif parms["tensor_product_class"] == "quadratic":
                tensor_product_class = PETP_Quadratic
            else:                    
                assert False, f"Unknown tensor product class {parms['tensor_product_class']}"

            modules.append(gate_class(
                in_irreps,
                out_irreps,
                parms["scalar_activation"   ],
                parms["nonscalar_activation"],
                tensor_product_class
            ))


        # set up shared layers
        def build_shared(modules_shared):
            assert(parms["nhidden_shared"] >= 1), "Number of shared hidden layers must be positive"
            append_full_layer(modules_shared, self.irreps_in, parms["irreps_hidden"])
            for _ in range(parms["nhidden_shared"]-1):
                append_full_layer(modules_shared, parms["irreps_hidden"], parms["irreps_hidden"])

        # set up task-specific layers
        def build_task(modules_task, nhidden_task, irreps_final):
            assert nhidden_task >= 1, "Number of hidden layers must be positive"
            for _ in range(nhidden_task):
                append_full_layer(modules_task, parms["irreps_hidden"], parms["irreps_hidden"])

            # append linear layer at the end to get the correct output irreps for the task
            # couldn't get the gated block to work with a final layer with no nonscalar irreps
            modules_task.append(e3nn.o3.Linear(parms["irreps_hidden"], irreps_final))

        # put together the layers of the neural network
        modules_shared = []
        modules_growthrate = []
        modules_F4 = []
        build_shared(modules_shared)
        build_task(modules_growthrate, parms["nhidden_growthrate"], e3nn.o3.Irreps("1x0e"       ))
        build_task(modules_F4,         parms["nhidden_F4"],         e3nn.o3.Irreps("1x1o + 1x0e"))

        # turn the list of modules into a sequential model
        self.TP_activation_stack_shared     = nn.Sequential(*modules_shared)
        self.TP_activation_stack_growthrate = nn.Sequential(*modules_growthrate)
        self.TP_activation_stack_F4         = nn.Sequential(*modules_F4)

        # register lebedev quadrature tensors as buffers so they are scriptable and move with the model
        self.register_buffer('lebedev_pts', box3d.pts)
        self.register_buffer('lebedev_weights', box3d.weights)

        # initialize the weights
        torch.manual_seed(parms["random_seed"])
        np.random.seed(parms["random_seed"])

        # print the model structure
        print(self)

    # convert the 3-flavor matrix into an effective 2-flavor matrix
    # input and output are indexed as [sim, nu/nubar, flavor, xyzt]
    # This assumes that the x flavors will represent the SUM of mu and tau flavors.
    @torch.jit.export
    def convert_y_to_2flavor(self, y):
        y2F = torch.zeros((y.shape[0],2,2,4), device=y.device)

        y2F[:,:,0,:] = y[:,:,0,:]
        y2F[:,:,1,:] = torch.sum(y[:,:,1:,:], dim=(2))

        return y2F

    # Push the inputs through the neural network
    # output is indexed as [sim, nu/nubar(out), flavor(out), nu/nubar(in), flavor(in)]
    def forward(self,x):

        # evaluate the shared portion of the network
        y_shared = self.TP_activation_stack_shared(x)

        # evaluate each task
        y_growthrate = self.TP_activation_stack_growthrate(y_shared)
        y_F4         = self.TP_activation_stack_F4(y_shared)

        #assert torch.all(torch.isfinite(y_stability))
        #assert torch.all(torch.isfinite(y_growthrate))
        #assert torch.all(torch.isfinite(y_F4))

        return y_F4, y_growthrate

    def _prepare_input(self, F4_in):
        nsims = F4_in.shape[0]
        F4_in = F4_in.view((nsims, 2, self.NF, 4))
        ntot = ntotal(F4_in)
        F4_in_norm = F4_in / ntot[:, None, None, None]
        return nsims, F4_in_norm, ntot

    def _predict_box3d_normalized(self, F4_in_norm):
        return box3d.mixBox3D_lebedev(F4_in_norm, self.lebedev_pts, self.lebedev_weights)

    def _apply_physical_constraints(self, F4_in_norm, F4_out_norm):
        F4_out = F4_out_norm.clone()

        if self.average_heavies_in_final_state:
            F4_out[:, :, 1:, :] = F4_out[:, :, 1:, :].mean(dim=2, keepdim=True)

        F4_in_flavortrace = torch.sum(F4_in_norm, dim=2, keepdim=True)
        F4_out_flavortrace = torch.sum(F4_out, dim=2, keepdim=True)
        F4_out_excess = F4_out_flavortrace - F4_in_flavortrace
        F4_out = F4_out - F4_out_excess / self.NF

        if self.conserve_lepton_number:
            ELN_in = F4_in_norm[:, 0, :, 3] - F4_in_norm[:, 1, :, 3]
            ELN_out = F4_out[:, 0, :, 3] - F4_out[:, 1, :, 3]
            ELN_excess = ELN_out - ELN_in
            F4_out[:, 0, :, 3] -= ELN_excess / 2.0
            F4_out[:, 1, :, 3] += ELN_excess / 2.0

        return F4_out

    def _finalize_prediction(self, F4_in_norm, F4_box3d, growthrate_box3d, F4_residual, growthrate_residual, ntot, apply_constraints: bool):
        nsims = F4_in_norm.shape[0]

        F4_out = F4_box3d + F4_residual.reshape((nsims, 2, self.NF, 4))
        if apply_constraints:
            F4_out = self._apply_physical_constraints(F4_in_norm, F4_out)

        growthrate = growthrate_box3d + growthrate_residual.reshape((nsims,))
        stability = (growthrate_box3d <= 0).float()

        return F4_out * ntot[:, None, None, None], growthrate * ntot, stability

    def _predict_baseline_variant(self, F4_in, apply_constraints: bool):
        _, F4_in_norm, ntot = self._prepare_input(F4_in)
        F4_box3d, growthrate_box3d = self._predict_box3d_normalized(F4_in_norm)
        zero_F4 = torch.zeros_like(F4_box3d)
        zero_growth = torch.zeros_like(growthrate_box3d)
        return self._finalize_prediction(
            F4_in_norm,
            F4_box3d,
            growthrate_box3d,
            zero_F4,
            zero_growth,
            ntot,
            apply_constraints,
        )

    @torch.jit.export
    def predict_box3d(self, F4_in):
        return self._predict_baseline_variant(F4_in, False)

    @torch.jit.export
    def predict_control(self, F4_in):
        return self._predict_baseline_variant(F4_in, True)

    # X is just F4_initial [nsamples, 2, NF, xyzt]
    @torch.jit.export
    def predict_all(self, F4_in):
        nsims, F4_in_norm, ntot = self._prepare_input(F4_in)
        F4_box3d, growthrate_box3d = self._predict_box3d_normalized(F4_in_norm)

        # Combine F4_in and F4_box3D into a joint input of shape [nsamples, nu/nubar, flavor, xyzt(in)/xyzt(box3d)]
        F4_joint = torch.cat([F4_in_norm, F4_box3d], dim=-1)

        # propagate through the network
        y_F4, y_growthrate = self.forward(F4_joint)

        # pool over features to get permutation-invariant output
        # Averaging over nodes in the graph
        y_growthrate = torch.mean(y_growthrate, dim=(1,2)).reshape((nsims,))

        return self._finalize_prediction(
            F4_in_norm,
            F4_box3d,
            growthrate_box3d,
            y_F4,
            y_growthrate,
            ntot,
            True,
        )
