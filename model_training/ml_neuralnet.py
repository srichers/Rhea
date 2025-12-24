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
from torchview import draw_graph

# define a permutation-equivariant message passing layer
# 0 means flavor edge, 1 means neutrino/antineutrino edge
class PermutationEquivariantLinear(nn.Module):
    def __init__(self, width_in, width_out):
        super().__init__()
        self.lin_self    = nn.Linear(width_in, width_out)
        self.lin_flavor  = nn.Linear(width_in, width_out, bias=False)
        self.lin_nunubar = nn.Linear(width_in, width_out, bias=False)
        self.lin_all     = nn.Linear(width_in, width_out, bias=False)

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

        # compute the outputs from each part
        y_self = self.lin_self(x_self)
        y_flavor = self.lin_flavor(x_flavor)
        y_nunubar = self.lin_nunubar(x_nunubar)
        y_all = self.lin_all(x_all)

        # sum the contributions, using the residual connection when possible
        y = y_self + y_flavor + y_nunubar + y_all
        if x.shape[-1] == y_self.shape[-1]:
            y = y + x

        return y
    
# define the NN model
class NeuralNetwork(nn.Module):
    def __init__(self, parms):
        super().__init__()

        # store input arguments
        self.NF = parms["NF"]

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

        # construct number of X and y values
        # one X for each pair of species, and one for each product with u
        self.NX = 4
        self.NY = 4

        # one y matrix for ndens, one for flux
        # add one extra to predict growth rate
        self.average_heavies_in_final_state = parms["average_heavies_in_final_state"]
        self.conserve_lepton_number = parms["conserve_lepton_number"]

        # append a full layer including linear, activation, and layernorm/dropout if desired
        def append_full_layer(modules, in_dim, out_dim):
            modules.append(PermutationEquivariantLinear(in_dim, out_dim))
            if parms["do_layernorm"]:
                modules.append(nn.LayerNorm(out_dim))
            modules.append(parms["activation"]())
            if parms["dropout_probability"] > 0:
                modules.append(nn.Dropout(parms["dropout_probability"]))

        # put together the layers of the neural network
        modules_shared = []
        modules_stability = []
        modules_growthrate = []
        modules_F4 = []
        
        # set up shared layers
        if parms["nhidden_shared"] > 0:
            append_full_layer(modules_shared, self.NX, parms["width_hidden"])
            for i in range(parms["nhidden_shared"]):
                append_full_layer(modules_shared, parms["width_hidden"], parms["width_hidden"])
        else:
            modules_shared.append(nn.Identity())

        def build_task(modules_task, nhidden_task, width_final):
            # the width at the beginning of the task is NX if no shared layers, otherwise it's the shared width
            width_start = self.NX if parms["nhidden_shared"] == 0 else parms["width_hidden"]

            # if no hidden layers, just do linear from start to final
            if nhidden_task == 0:
                modules_task.append(PermutationEquivariantLinear(width_start, width_final))
            # otherwise, build the full set of hidden layers
            else:
                append_full_layer(modules_task, width_start, parms["width_hidden"])
                for _ in range(nhidden_task-1):
                    append_full_layer(modules_task, parms["width_hidden"], parms["width_hidden"])
                modules_task.append(PermutationEquivariantLinear(parms["width_hidden"], width_final))

        build_task(modules_stability,  parms["nhidden_stability"],  1      )
        build_task(modules_growthrate, parms["nhidden_growthrate"], 1      )
        build_task(modules_F4,         parms["nhidden_F4"],         self.NY)

        # turn the list of modules into a sequential model
        self.linear_activation_stack_shared     = nn.Sequential(*modules_shared)
        self.linear_activation_stack_stability  = nn.Sequential(*modules_stability)
        self.linear_activation_stack_growthrate = nn.Sequential(*modules_growthrate)
        self.linear_activation_stack_F4         = nn.Sequential(*modules_F4)

        # initialize the weights
        torch.manual_seed(parms["random_seed"])
        np.random.seed(parms["random_seed"])
        self.linear_activation_stack_shared.apply(self._init_weights)
        self.linear_activation_stack_stability.apply(self._init_weights)
        self.linear_activation_stack_growthrate.apply(self._init_weights)
        self.linear_activation_stack_F4.apply(self._init_weights)

        # print the model structure
        print(self)
        graph = draw_graph(self, torch.zeros((1, 2, self.NF, 4)))
        graph.visual_graph.render("nn_model_structure", format="pdf", cleanup=True)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    
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
        #assert torch.all(torch.isfinite(x))

        # evaluate the shared portion of the network
        y_shared = self.linear_activation_stack_shared(x)
        #assert torch.all(torch.isfinite(y_shared))
        # evaluate each task
        y_stability  = self.linear_activation_stack_stability(y_shared)
        y_growthrate = self.linear_activation_stack_growthrate(y_shared)
        y_F4         = self.linear_activation_stack_F4(y_shared)

        #assert torch.all(torch.isfinite(y_stability))
        #assert torch.all(torch.isfinite(y_growthrate))
        #assert torch.all(torch.isfinite(y_F4)) # HERE

        return y_F4, y_growthrate, y_stability

    # X is just F4_initial [nsamples, 2, NF, xyzt]
    @torch.jit.export
    def predict_all(self, F4_in):
        #assert torch.all(torch.isfinite(F4_in))
        # get the total density
        nsims = F4_in.shape[0]
        F4_in = F4_in.view((nsims,2,self.NF,4))

        # propagate through the network
        y_F4, y_growthrate, y_stability = self.forward(F4_in)

        #assert torch.all(torch.isfinite(y_F4))
        #print("y_F4 min/max:", y_F4.min().item(), y_F4.max().item())

        # reshape into a matrix
        #print("y_F4 min/max before reshape:", y_F4.min().item(), y_F4.max().item())
        F4_out = F4_in + y_F4.reshape((nsims,2,self.NF,4))
        #print("F4_out min/max:", F4_out.min().item(), F4_out.max().item())

        # enforce symmetry in the heavies
        #if self.average_heavies_in_final_state:
        #    F4_out[:,:,1:,:] = F4_out.clone().detach()[:,:,1:,:].mean(dim=3, keepdim=True)
        ntot_new = ntotal(F4_out)
        ntot_y = ntotal(y_F4.reshape((nsims,2,self.NF,4)))

        #assert torch.all(torch.isfinite(F4_out))

        # ensure the flavor-traced number is conserved
        #F4_in = F4_in.view((nsims,2,self.NF,4))
        #F4_in_flavortrace = torch.sum(F4_in, dim=2, keepdim=True)
        #F4_out_flavortrace = torch.sum(F4_out, dim=2, keepdim=True)
        #F4_out_excess = F4_out_flavortrace - F4_in_flavortrace
        #F4_out[:,:,:,:] -= F4_out_excess / self.NF
        ###print("ntot_y min/max/mean:", ntot_y.min().item(), ntot_y.max().item(), ntot_y.mean().item())
        #print("ntot_old min/max:", ntot.min().item(), ntot.max().item())
        #print("ntot_new min/max:", ntot_new.min().item(), ntot_new.max().item())
        #print("F4_out min/max:", F4_out.min().item(), F4_out.max().item())
        #assert torch.all(torch.isfinite(F4_out))

        # ensure that ELN is conserved
        #if self.conserve_lepton_number:
        #    ELN_in  = F4_in[:,0,:,0]  - F4_in[:,1,:,0]
        #    ELN_out = F4_out[:,0,:,0] - F4_out[:,1,:,0]
        #    ELN_excess = ELN_out - ELN_in
        #    F4_out[:,0,:,0] -= ELN_excess / 2.0
        #    F4_out[:,1,:,0] += ELN_excess / 2.0

        # pool over features to get permutation-invariant output
        y_stability = torch.mean(y_stability, dim=(1,2))
        y_growthrate = torch.mean(y_growthrate, dim=(1,2))

        # apply sigmoid to stability logits
        stability = torch.squeeze(y_stability)

        # apply total density scaling to log growth rate
        growthrate = torch.squeeze(y_growthrate)

        #assert torch.all(torch.isfinite(F4_out))
        #assert torch.all(torch.isfinite(growthrate))
        #assert torch.all(torch.isfinite(stability))

        return F4_out, growthrate, stability
