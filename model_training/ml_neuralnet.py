'''
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This file contains the structure of the neural network, including the means of transforming between angular moments and the actual inputs/outputs of the ML model
'''

import torch
import numpy as np
from torch import nn
from ml_tools import restrict_F4_to_physical, ntotal
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing

# constants used to get growth rate to order unity
hbar = 1.05457266e-27 # erg s
c = 2.99792458e10 # cm/s
eV = 1.60218e-12 # erg
GeV = 1e9 * eV
GF = 1.1663787e-5 / GeV**2 * (hbar*c)**3 # erg cm^3
ndens_to_invsec = GF/hbar
print("ndens_to_invsec")
print(ndens_to_invsec)

def make_batch(x):
    """
    x: [nsamples, 4, 2, 3]
    returns:
        x_nodes:           [nsamples*6, 4]
        edge_index_batch:  [2, nsamples*num_edges]
        edge_type_batch:   [nsamples*num_edges]
    """
    nsamples = x.size(0)

    # Define edges in graph to connect a flavor to each other flavor and also to its anti-flavor
    # 0,1,2,3,4,5 = nue,numu,nutau,anue,anumu,anutau
    edge_index = torch.tensor([
        # same-group edges
        [0,0,1,1,2,2, 3,3,4,4,5,5, 0,1,2, 3,4,5],
        [1,2,0,2,0,1, 4,5,3,5,3,4, 3,4,5, 0,1,2]
    ], dtype=torch.long).to(x.device)

    # edge types: 0 = flavor, 1 = neutrino/antineutrino
    edge_type = torch.tensor( [0]*12 + [1]*6, dtype=torch.long ).to(x.device)

    # move feature dim last, then flatten nodes
    # [nsamples, 2, 3, 4] → [nsamples*6, 4]
    x_nodes = x.permute(0, 2, 3, 1).reshape(nsamples * 6, 4)

    # repeat edge structure with offsets
    num_edges = edge_index.size(1)

    edge_index_batch = edge_index.repeat(1, nsamples)
    offsets = (torch.arange(nsamples, device=x.device) * 6)\
              .repeat_interleave(num_edges)
    edge_index_batch = edge_index_batch + offsets

    edge_type_batch = edge_type.repeat(nsamples)

    return x_nodes, edge_index_batch, edge_type_batch

# define a permutation-equivariant message passing layer
# 0 means flavor edge, 1 means neutrino/antineutrino edge
class PermutationEquivariantLinear(MessagePassing):
    def __init__(self, width_in, width_out):
        super().__init__(aggr='add')
        self.lin_self  = nn.Linear(width_in, width_out)
        self.lin_edges = nn.ModuleList([
            nn.Linear(width_in, width_out)
            for _ in range(2)])

    def forward(self, x, edge_index_batch, edge_type_batch):
        out = self.lin_self(x)
        out += self.propagate(edge_index_batch, x=x, edge_type=edge_type_batch)
        return out

    def message(self, x_j, edge_type):
        # prepare output tensor with size defined by output features
        out = torch.empty(
            x_j.size(0),
            self.lin_self.out_features,
            device=x_j.device,
            dtype=x_j.dtype,
        )

        # loop through the edge types and use corresponding linear for each edge type
        for t in range(len(self.lin_edges)):
            mask = (edge_type == t)
            if mask.any():
                out[mask] = self.lin_edges[t](x_j[mask])

        return out


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
        self.NX = 4
        self.NY = 4

        # one y matrix for ndens, one for flux
        # add one extra to predict growth rate
        self.average_heavies_in_final_state = parms["average_heavies_in_final_state"]
        self.conserve_lepton_number = parms["conserve_lepton_number"]

        # append a full layer including linear, activation, and batchnorm
        def append_full_layer(modules, in_dim, out_dim):
            modules.append(PermutationEquivariantLinear(in_dim, out_dim))
            if parms["do_layernorm"]:
                modules.append(nn.LayerNorm(out_dim))
            modules.append(parms["activation"]())
            if parms["dropout_probability"] > 0:
                modules.append(nn.Dropout(parms["dropout_probability"]))

        # put together the layers of the neural network
        self.modules_shared = nn.ModuleList()
        self.modules_stability = nn.ModuleList()
        self.modules_growthrate = nn.ModuleList()
        self.modules_F4 = nn.ModuleList()
        
        # set up shared layers
        if parms["nhidden_shared"] > 0:
            append_full_layer(self.modules_shared, self.NX, parms["width_shared"])
            for i in range(parms["nhidden_shared"]):
                append_full_layer(self.modules_shared, parms["width_shared"], parms["width_shared"])
        else:
            self.modules_shared.append(nn.Identity())

        def build_task(modules_task, nhidden_task, width_task, width_final):
            # the width at the beginning of the task is NX if no shared layers, otherwise it's the shared width
            width_start = self.NX if parms["nhidden_shared"] == 0 else parms["width_shared"]

            # if no hidden layers, just do linear from start to final
            if nhidden_task == 0:
                modules_task.append(PermutationEquivariantLinear(width_start, width_final))
            # otherwise, build the full set of hidden layers
            else:
                append_full_layer(modules_task, width_start, width_task)
                for _ in range(nhidden_task-1):
                    append_full_layer(modules_task, width_task, width_task)
                modules_task.append(PermutationEquivariantLinear(width_task, width_final))

        build_task(self.modules_stability,  parms["nhidden_stability"],  parms["width_stability"],  1      )
        build_task(self.modules_growthrate, parms["nhidden_growthrate"], parms["width_growthrate"], 1      )
        build_task(self.modules_F4,         parms["nhidden_F4"],         parms["width_F4"],         self.NY)

        # initialize the weights
        torch.manual_seed(parms["random_seed"])
        np.random.seed(parms["random_seed"])
        for module in self.modules_shared:
            module.apply(self._init_weights)
        for module in self.modules_stability:
            module.apply(self._init_weights)
        for module in self.modules_growthrate:
            module.apply(self._init_weights)
        for module in self.modules_F4:
            module.apply(self._init_weights)

        # print the model structure
        print(self)

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
        # create a batch for the graph network
        x_nodes, edge_index_batch, edge_type_batch = make_batch(x)

        def module_stack(modules, x):
            for layer in modules:
                if isinstance(layer, PermutationEquivariantLinear):
                    x = layer(x, edge_index_batch, edge_type_batch)
                else:
                    x = layer(x)
            return x

        # evaluate the shared portion of the network, then each task
        x_nodes     = module_stack(self.modules_shared    , x_nodes)
        xStability  = module_stack(self.modules_stability , x_nodes)
        xGrowthrate = module_stack(self.modules_growthrate, x_nodes)
        xF4         = module_stack(self.modules_F4        , x_nodes)

        return xF4, xGrowthrate, xStability

    # X is just F4_initial [nsamples, xyzt, 2, NF]
    @torch.jit.export
    def predict_all(self, F4_in):
        # get the total density
        nsims = F4_in.shape[0]
        F4_in = F4_in.view((nsims,4,2,3))
        ntot = ntotal(F4_in)

        # reshape and normalize
        X = F4_in / ntot[:,None,None,None]

        # propagate through the network
        y_F4, y_growthrate, y_stability = self.forward(X)

        # reshape into a matrix
        F4_out = y_F4.reshape((nsims,4,2,self.NF)) * ntot[:,None,None,None]

        # enforce symmetry in the heavies
        if self.average_heavies_in_final_state:
            F4_out[:,:,:,1:] = F4_out.clone().detach()[:,:,:,1:].mean(dim=3, keepdim=True)

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

        # pool the scalars to make permutation-invariants
        y_stability = y_stability.view((nsims,6)).mean(dim=1)
        y_growthrate = y_growthrate.view((nsims,6)).mean(dim=1)

        # apply sigmoid to stability logits
        stability = torch.squeeze(torch.sigmoid(y_stability))

        # apply total density scaling to log growth rate
        # expects F4 to be in units of cm^-3
        ndens_to_invsec = 1.3615452913035457e-22 # must be declared a literal here or pytorch complains when exporting the model
        growthrate = (torch.squeeze(y_growthrate)) * ntot*ndens_to_invsec # torch.exp

        return F4_out, growthrate, stability
