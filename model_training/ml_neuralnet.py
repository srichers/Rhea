'''
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This file contains the structure of the neural network, including the means of transforming between angular moments and the actual inputs/outputs of the ML model
'''

import torch
import numpy as np
from torch import nn
from ml_tools import dot4, restrict_F4_to_physical, ntotal
from torchview import draw_graph


# constants used to get growth rate to order unity
hbar = 1.05457266e-27 # erg s
c = 2.99792458e10 # cm/s
eV = 1.60218e-12 # erg
GeV = 1e9 * eV
GF = 1.1663787e-5 / GeV**2 * (hbar*c)**3 # erg cm^3
ndens_to_invsec = GF/hbar
print("ndens_to_invsec")
print(ndens_to_invsec)

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
                for name in ["stability", "growthrate", "ndens", "fluxmag", "direction", "unphysical"]
            })
        else:
            self.log_task_weights = {
                name: torch.tensor(-np.log(parms[f"task_weight_{name}"]), dtype=torch.float32)
                for name in ["stability", "growthrate", "ndens", "fluxmag", "direction", "unphysical"]
            }

        # construct number of X and y values
        # one X for each pair of species, and one for each product with u
        self.NX = 4*2*self.NF
        self.NY = 4*2*self.NF

        # one y matrix for ndens, one for flux
        # add one extra to predict growth rate
        self.average_heavies_in_final_state = parms["average_heavies_in_final_state"]
        self.conserve_lepton_number = parms["conserve_lepton_number"]

        # append a full layer including linear, activation, and batchnorm
        def append_full_layer(modules, in_dim, out_dim):
            modules.append(nn.Linear(in_dim, out_dim))
            if parms["do_batchnorm"]:
                modules.append(nn.BatchNorm1d(out_dim))
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
            append_full_layer(modules_shared, self.NX, parms["width_shared"])
            for i in range(parms["nhidden_shared"]):
                append_full_layer(modules_shared, parms["width_shared"], parms["width_shared"])
        else:
            modules_shared.append(nn.Identity())

        def build_task(modules_task, nhidden_task, width_task, width_final):
            # the width at the beginning of the task is NX if no shared layers, otherwise it's the shared width
            width_start = self.NX if parms["nhidden_shared"] == 0 else parms["width_shared"]

            # if no hidden layers, just do linear from start to final
            if nhidden_task == 0:
                modules_task.append(nn.Linear(width_start, width_final))
            # otherwise, build the full set of hidden layers
            else:
                append_full_layer(modules_task, width_start, width_task)
                for _ in range(nhidden_task-1):
                    append_full_layer(modules_task, width_task, width_task)
                modules_task.append(nn.Linear(width_task, width_final))
        
        build_task(modules_stability,  parms["nhidden_stability"],  parms["width_stability"],  1      )
        build_task(modules_growthrate, parms["nhidden_growthrate"], parms["width_growthrate"], 1      )
        build_task(modules_F4,         parms["nhidden_F4"],         parms["width_F4"],         self.NY)

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
        graph = draw_graph(self, torch.zeros((1, self.NX)))
        graph.visual_graph.render("nn_model_structure", format="pdf", cleanup=True)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    
    # convert the 3-flavor matrix into an effective 2-flavor matrix
    # input and output are indexed as [sim, xyzt, nu/nubar, flavor]
    # This assumes that the x flavors will represent the SUM of mu and tau flavors.
    @torch.jit.export
    def convert_y_to_2flavor(self, y):
        y2F = torch.zeros((y.shape[0],4,2,2), device=y.device)

        y2F[:,:,:,0] = y[:,:,:,0]
        y2F[:,:,:,1] = torch.sum(y[:,:,:,1:], dim=(3))

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
        
        # enforce symmetry in the heavies
        if self.average_heavies_in_final_state:
            y_F4[:,:,:,1:] = y_F4.clone().detach()[:,:,:,1:].mean(dim=3, keepdim=True)

        return y_F4, y_growthrate, y_stability

    # X is just F4_initial [nsamples, xyzt, 2, NF]
    @torch.jit.export
    def predict_all(self, F4_in):
        # get the total density
        nsims = F4_in.shape[0]
        F4_in = F4_in.view((nsims,4,2,3))
        ntot = ntotal(F4_in)

        # reshape and normalize
        F4_in = F4_in.view((nsims,self.NX))
        X = F4_in / ntot[:,None]

        # propagate through the network
        y_F4, y_growthrate, y_stability = self.forward(X)

        # reshape into a matrix
        F4_out = y_F4.reshape((nsims,4,2,self.NF)) * ntot[:,None,None,None]

        # ensure the flavor-traced number is conserved
        F4_in = F4_in.view((nsims,4,2,3))
        F4_in_flavortrace = torch.sum(F4_in, dim=3, keepdim=True)
        F4_out_flavortrace = torch.sum(F4_out, dim=3, keepdim=True)
        F4_out_excess = F4_out_flavortrace - F4_in_flavortrace
        F4_out[:,:,:,:] -= F4_out_excess / 3.0

        # ensure that ELN is conserved
        if self.conserve_lepton_number:
            ELN_in  = F4_in[:,0,0,:]  - F4_in[:,0,1,:]
            ELN_out = F4_out[:,0,0,:] - F4_out[:,0,1,:]
            ELN_excess = ELN_out - ELN_in
            F4_out[:,0,0,:] -= ELN_excess / 2.0
            F4_out[:,0,1,:] += ELN_excess / 2.0

        # apply sigmoid to stability logits
        stability = torch.squeeze(torch.sigmoid(y_stability))

        # apply total density scaling to log growth rate
        # expects F4 to be in units of cm^-3
        ndens_to_invsec = 1.3615452913035457e-22 # must be declared a literal here or pytorch complains when exporting the model
        growthrate = (torch.squeeze(y_growthrate)) * ntot*ndens_to_invsec # torch.exp

        return F4_out, growthrate, stability
