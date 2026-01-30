'''
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This file contains the structure of the neural network, including the means of transforming between angular moments and the actual inputs/outputs of the ML model
'''

import torch
import numpy as np
from torch import nn
from ml_tools import restrict_F4_to_physical, ntotal
from ml_tools import dot4, restrict_F4_to_physical, ntotal
import ml_constants as const
import e3nn.o3
import e3nn.nn

# define a permutation-equivariant message passing layer
# 0 means flavor edge, 1 means neutrino/antineutrino edge
class PermutationEquivariantLinear(nn.Module):
    def __init__(self, irreps_in, irreps_out, do_batchnorm=False):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.batchnorm = e3nn.nn.BatchNorm(irreps_in) if do_batchnorm else None
        self.lin_self    = e3nn.o3.Linear(irreps_in, irreps_out)
        self.lin_flavor  = e3nn.o3.Linear(irreps_in, irreps_out)
        self.lin_nunubar = e3nn.o3.Linear(irreps_in, irreps_out)
        self.lin_all     = e3nn.o3.Linear(irreps_in, irreps_out)

    # x has shape [nsamples, nunubar, flavor, features]
    def forward(self, x):

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

        # optional per-irrep normalization before the linear maps
        if self.batchnorm is not None:
            x_self = self.batchnorm(x_self)
            x_flavor = self.batchnorm(x_flavor)
            x_nunubar = self.batchnorm(x_nunubar)
            x_all = self.batchnorm(x_all)

        # compute the outputs from each part
        y_self = self.lin_self(x_self)
        y_flavor = self.lin_flavor(x_flavor)
        y_nunubar = self.lin_nunubar(x_nunubar)
        y_all = self.lin_all(x_all)

        # sum the contributions, using the residual connection when possible
        y = y_self + y_flavor + y_nunubar + y_all
        return y

class PermutationEquivariantScalarTensorProduct(nn.Module):
    def __init__(self, irreps_in, irreps_out, do_batchnorm=False):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.batchnorm = e3nn.nn.BatchNorm(irreps_in) if do_batchnorm else None
        self.norm = e3nn.o3.Norm(irreps_in)
        self.tp_self    = e3nn.o3.FullyConnectedTensorProduct(irreps_in, self.norm.irreps_out, irreps_out)
        self.tp_flavor  = e3nn.o3.FullyConnectedTensorProduct(irreps_in, self.norm.irreps_out, irreps_out)
        self.tp_nunubar = e3nn.o3.FullyConnectedTensorProduct(irreps_in, self.norm.irreps_out, irreps_out)
        self.tp_all     = e3nn.o3.FullyConnectedTensorProduct(irreps_in, self.norm.irreps_out, irreps_out)

    # x has shape [nsamples, nunubar, flavor, features]
    def forward(self, x):
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

        # optional per-irrep normalization before the tensor product
        if self.batchnorm is not None:
            x_self = self.batchnorm(x_self)
            x_flavor = self.batchnorm(x_flavor)
            x_nunubar = self.batchnorm(x_nunubar)
            x_all = self.batchnorm(x_all)

        # compute invariant scalars and apply scalar-gated tensor products
        s_self = self.norm(x_self)
        s_flavor = self.norm(x_flavor)
        s_nunubar = self.norm(x_nunubar)
        s_all = self.norm(x_all)
        y_self = self.tp_self(x_self, s_self)
        y_flavor = self.tp_flavor(x_flavor, s_flavor)
        y_nunubar = self.tp_nunubar(x_nunubar, s_nunubar)
        y_all = self.tp_all(x_all, s_all)

        # sum the contributions, using the residual connection when possible
        y = y_self + y_flavor + y_nunubar + y_all
        return y

class PermutationEquivariantGatedBlock(nn.Module):
    def __init__(self, irreps_in, irreps_out, act_scalars, act_gates, do_batchnorm=False):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        irreps_scalars = irreps_out.filter(lambda mul_ir: mul_ir.ir.l == 0)
        irreps_nonscalars = irreps_out.filter(lambda mul_ir: mul_ir.ir.l > 0)
        irreps_gates = e3nn.o3.Irreps(f"{irreps_nonscalars.num_irreps}x0e")
        irreps_with_gates = irreps_scalars + irreps_gates + irreps_nonscalars

        self.lin = PermutationEquivariantScalarTensorProduct(
            irreps_in,
            irreps_with_gates,
            do_batchnorm=do_batchnorm,
        )

        self.gate = e3nn.nn.Gate(
            irreps_scalars = irreps_scalars,
            act_scalars = [act_scalars] * len(irreps_scalars),
            irreps_gates = irreps_gates,
            act_gates = [act_gates] * len(irreps_gates),
            irreps_gated = irreps_nonscalars,
        )

    def forward(self, x):
        # get the full output (scalars + gates + nonscalars) from one linear
        y = self.lin(x)

        # apply the gate. If input and output irreps are the same, add residual connection
        y = self.gate(y)
        #TODO implement residual connection - the below fails during scripting
        #if self.irreps_in == self.irreps_out:
        #    y = x + y
        return y
    
# define the NN model
class NeuralNetwork(nn.Module):
    def __init__(self, parms):
        super().__init__()

        # make sure the irreps are scalar first to make sure it is consistent with the output of Gate
        assert parms["irreps_hidden"] == parms["irreps_hidden"].sort().irreps, "Hidden irreps must have scalars first"

        # store input arguments
        self.NF = parms["NF"]
        # allow legacy configs that used do_layernorm
        do_batchnorm = parms.get("do_batchnorm", parms.get("do_layernorm", False))
        # optionally scale predicted growthrate back to physical units
        # default False because growthrate labels are already normalized in the dataset
        self.scale_growthrate_by_ntot = parms.get("scale_growthrate_by_ntot", False)

        # store loss weight parameters for tasks
        if parms["do_learn_task_weights"]:
            self.log_task_weights = nn.ParameterDict({
                name: nn.Parameter(torch.tensor(-np.log(parms[f"task_weight_{name}"]), dtype=torch.float32))
                for name in ["stability", "growthrate", "F4", "unphysical"]
            })
        else:
            self.log_task_weights = {
                name: torch.tensor(-np.log(parms[f"task_weight_{name}"]), dtype=torch.float32)
                for name in ["stability", "growthrate", "F4", "unphysical"]
            }

        # The input irreps for each node are just the 4 components of F4
        self.irreps_in  = e3nn.o3.Irreps("1x1o + 1x0e")
        
        # one y matrix for ndens, one for flux
        # add one extra to predict growth rate
        self.average_heavies_in_final_state = parms["average_heavies_in_final_state"]
        self.conserve_lepton_number = parms["conserve_lepton_number"]

        # append a full layer including linear, activation, and batchnorm/dropout if desired
        def append_full_layer(modules, in_irreps, out_irreps):

            # for gated layers, add separate gate scalars to the main output
            modules.append(PermutationEquivariantGatedBlock(
                in_irreps,
                out_irreps,
                parms["scalar_activation"   ],
                parms["nonscalar_activation"],
                do_batchnorm=do_batchnorm,
            ))

            if parms["dropout_probability"] > 0:
                modules.append(e3nn.nn.Dropout(out_irreps, p=parms["dropout_probability"]))
        
        # set up shared layers
        def build_shared(modules_shared):
            if parms["nhidden_shared"] > 0:
                append_full_layer(modules_shared, self.irreps_in, parms["irreps_hidden"])
                for i in range(parms["nhidden_shared"]):
                    append_full_layer(modules_shared, parms["irreps_hidden"], parms["irreps_hidden"])
            else:
                modules_shared.append(nn.Identity())

        # set up task-specific layers
        def build_task(modules_task, nhidden_task, irreps_final):
            # the width at the beginning of the task is NX if no shared layers, otherwise it's the shared width
            irreps_start = self.irreps_in if parms["nhidden_shared"] == 0 else parms["irreps_hidden"]

            # if no hidden layers, just do linear from start to final
            if nhidden_task == 0:
                modules_task.append(PermutationEquivariantLinear(irreps_start, irreps_final))
            # otherwise, build the full set of hidden layers
            else:
                append_full_layer(modules_task, irreps_start, parms["irreps_hidden"])
                for _ in range(nhidden_task-1):
                    append_full_layer(modules_task, parms["irreps_hidden"], parms["irreps_hidden"])
                modules_task.append(PermutationEquivariantLinear(parms["irreps_hidden"], irreps_final))

        # put together the layers of the neural network
        modules_shared = []
        modules_stability = []
        modules_growthrate = []
        modules_F4 = []
        build_shared(modules_shared)
        build_task(modules_stability,  parms["nhidden_stability"],  e3nn.o3.Irreps("1x0e"       ))
        build_task(modules_growthrate, parms["nhidden_growthrate"], e3nn.o3.Irreps("1x0e"       ))
        build_task(modules_F4,         parms["nhidden_F4"],         e3nn.o3.Irreps("1x1o + 1x0e"))

        # turn the list of modules into a sequential model
        self.linear_activation_stack_shared     = nn.Sequential(*modules_shared)
        self.linear_activation_stack_stability  = nn.Sequential(*modules_stability)
        self.linear_activation_stack_growthrate = nn.Sequential(*modules_growthrate)
        self.linear_activation_stack_F4         = nn.Sequential(*modules_F4)

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
        y_shared = self.linear_activation_stack_shared(x)

        # evaluate each task
        y_stability  = self.linear_activation_stack_stability(y_shared)
        y_growthrate = self.linear_activation_stack_growthrate(y_shared)
        y_F4         = self.linear_activation_stack_F4(y_shared)

        #assert torch.all(torch.isfinite(y_stability))
        #assert torch.all(torch.isfinite(y_growthrate))
        #assert torch.all(torch.isfinite(y_F4))

        return y_F4, y_growthrate, y_stability

    # X is just F4_initial [nsamples, 2, NF, xyzt]
    @torch.jit.export
    def predict_all(self, F4_in):

        # get the total density
        nsims = F4_in.shape[0]
        F4_in = F4_in.view((nsims,2,self.NF,4))
        ntot = ntotal(F4_in) # [nsims]

        # normalize the inputs by the total density
        F4_in = F4_in / ntot[:,None,None,None]

        # propagate through the network
        y_F4, y_growthrate, y_stability = self.forward(F4_in)

        # reshape into a matrix
        F4_out = F4_in + y_F4.reshape((nsims,2,self.NF,4))

        # enforce symmetry in the heavies
        if self.average_heavies_in_final_state:
            F4_out[:,:,1:,:] = F4_out.clone().detach()[:,:,1:,:].mean(dim=3, keepdim=True)

        # ensure the flavor-traced number is conserved
        F4_in = F4_in.view((nsims,2,self.NF,4))
        F4_in_flavortrace = torch.sum(F4_in, dim=2, keepdim=True)
        F4_out_flavortrace = torch.sum(F4_out, dim=2, keepdim=True)
        F4_out_excess = F4_out_flavortrace - F4_in_flavortrace
        F4_out[:,:,:,:] -= F4_out_excess / self.NF

        # ensure that ELN is conserved
        if self.conserve_lepton_number:
            ELN_in  = F4_in[:,0,:,0]  - F4_in[:,1,:,0]
            ELN_out = F4_out[:,0,:,0] - F4_out[:,1,:,0]
            ELN_excess = ELN_out - ELN_in
            F4_out[:,0,:,0] -= ELN_excess / 2.0
            F4_out[:,1,:,0] += ELN_excess / 2.0

        # pool over features to get permutation-invariant output
        # Averaging over nodes in the graph
        y_stability = torch.mean(y_stability, dim=(1,2))
        y_growthrate = torch.mean(y_growthrate, dim=(1,2))

        # squeeze stability logit (NOT passed through sigmoid yet)
        stability = torch.squeeze(y_stability)

        # apply total density scaling to log growth rate
        growthrate = torch.squeeze(y_growthrate)

        # rescale F4_out to the original total density
        F4_out = F4_out * ntot[:,None,None,None]
        if self.scale_growthrate_by_ntot:
            growthrate = growthrate * ntot*const.ndens_to_invsec

        #assert torch.all(torch.isfinite(F4_out))
        #assert torch.all(torch.isfinite(growthrate))
        #assert torch.all(torch.isfinite(stability))

        return F4_out, growthrate, stability
