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
from ml_e3nn import PermutationEquivariantLinear, PermutationEquivariantScalarTensorProduct, PermutationEquivariantGatedBlock
import ml_constants as const
import e3nn.o3
import e3nn.nn
    
# define the NN model
class NeuralNetwork(nn.Module):
    def __init__(self, parms):
        super().__init__()

        # make sure the irreps are scalar first to make sure it is consistent with the output of Gate
        assert parms["irreps_hidden"] == parms["irreps_hidden"].sort().irreps, "Hidden irreps must have scalars first"

        # store input arguments
        self.NF = parms["NF"]
        do_batchnorm = parms.get("do_batchnorm", False)

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
        # always return growthrate in physical units (1/s)
        growthrate = growthrate * ntot*const.ndens_to_invsec

        #assert torch.all(torch.isfinite(F4_out))
        #assert torch.all(torch.isfinite(growthrate))
        #assert torch.all(torch.isfinite(stability))

        return F4_out, growthrate, stability
