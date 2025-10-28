'''
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This file contains the structure of the neural network, including the means of transforming between angular moments and the actual inputs/outputs of the ML model
'''

import torch
import numpy as np
from torch import nn
from ml_tools import dot4, restrict_F4_to_physical, ntotal

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

        # construct number of X and y values
        # one X for each pair of species, and one for each product with u
        self.do_fdotu = parms["do_fdotu"]
        self.NX = self.NF * (1 + 2*self.NF)
        if self.do_fdotu:
            self.NX += 2*self.NF

        # one y matrix for ndens, one for flux
        # add one extra to predict growth rate
        self.average_heavies_in_final_state = parms["average_heavies_in_final_state"]
        self.conserve_lepton_number = parms["conserve_lepton_number"]
        self.Ny = (2*parms["NF"])**2

        # append a full layer including linear, activation, and batchnorm
        def append_full_layer(modules, in_dim, out_dim):
            modules.append(nn.Linear(in_dim, out_dim))
            if parms["do_batchnorm"]:
                modules.append(nn.BatchNorm1d(out_dim))
            modules.append(parms["activation"]())
            if parms["dropout_probability"] > 0:
                modules.append(nn.Dropout(parms["dropout_probability"]))
            return modules

        # put together the layers of the neural network
        modules_shared = []
        modules_stability = []
        modules_growthrate = []
        modules_density = []
        modules_flux = []
        
        # set up shared layers
        modules_shared = append_full_layer(modules_shared, self.NX, parms["width_shared"])
        for i in range(parms["nhidden_shared"]):
            modules_shared = append_full_layer(modules_shared, parms["width_shared"], parms["width_shared"])

        # set up stability layers. Note that output is a logit, NOT a number between 0 and 1. The loss should be BCEWithLogitsLoss
        modules_stability = append_full_layer(modules_stability, parms["width_shared"], parms["width_stability"])
        for i in range(parms["nhidden_stability"]):
            modules_stability = append_full_layer(modules_stability, parms["width_stability"], parms["width_stability"])
        modules_stability.append(nn.Linear(parms["width_stability"], 1))

        # set up growthrate layers
        modules_growthrate = append_full_layer(modules_growthrate, parms["width_shared"], parms["width_growthrate"])
        for i in range(parms["nhidden_growthrate"]):
            modules_growthrate = append_full_layer(modules_growthrate, parms["width_growthrate"], parms["width_growthrate"])
        modules_growthrate.append(nn.Linear(parms["width_growthrate"], 1))

        # set up the density layers
        modules_density = append_full_layer(modules_density, parms["width_shared"], parms["width_density"])
        for i in range(parms["nhidden_density"]):
            modules_density = append_full_layer(modules_density, parms["width_density"], parms["width_density"])
        modules_density.append(nn.Linear(parms["width_density"], self.Ny))

        # set up the flux layers
        modules_flux = append_full_layer(modules_flux, parms["width_shared"], parms["width_flux"])
        for i in range(parms["nhidden_flux"]):
            modules_flux = append_full_layer(modules_flux, parms["width_flux"], parms["width_flux"])
        modules_flux.append(nn.Linear(parms["width_flux"], self.Ny))

        # turn the list of modules into a sequential model
        self.linear_activation_stack_shared     = nn.Sequential(*modules_shared)
        self.linear_activation_stack_stability  = nn.Sequential(*modules_stability)
        self.linear_activation_stack_growthrate = nn.Sequential(*modules_growthrate)
        self.linear_activation_stack_density    = nn.Sequential(*modules_density)
        self.linear_activation_stack_flux       = nn.Sequential(*modules_flux)
        
        # initialize the weights
        torch.manual_seed(parms["random_seed"])
        np.random.seed(parms["random_seed"])
        self.linear_activation_stack_shared.apply(self._init_weights)
        self.linear_activation_stack_stability.apply(self._init_weights)
        self.linear_activation_stack_growthrate.apply(self._init_weights)
        self.linear_activation_stack_density.apply(self._init_weights)
        self.linear_activation_stack_flux.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    
    @torch.jit.export
    def X_from_F4(self, F4):
        """Given the a list of four fluxes, calculate the inputs to the neural network out of dot products of four fluxes with each other and the four velocity. The four-velocity is assumed to be timelike in an orthonormal tetrad.

        Args:
            F4 (torch.Tensor): Four-flux tensor. Indexed as [sim, xyzt, nu/nubar, flavor]

        Returns:
            torch.Tensor: Neural network input tensor. Indexed as [sim, iX]
        """
        index = 0
        nsims = F4.shape[0]
        X = torch.zeros((nsims, self.NX), device=F4.device)
        F4_flat = F4.reshape((nsims, 4, 2*self.NF)) # [simulationIndex, xyzt, species]

        # calculate the total number density based on the t component of the four-vector
        # [sim]
        N = torch.sum(F4_flat[:,3,:], dim=1)

        # normalize F4 by the total number density
        # [sim, xyzt, 2*NF]
        F4_flat = F4_flat / N[:,None,None]

        # add the dot products of each species with each other species
        for a in range(2*self.NF):
            for b in range(a,2*self.NF):
                F1 = F4_flat[:,:,a]
                F2 = F4_flat[:,:,b]

                X[:,index] = dot4(F1,F2)
                index += 1

        # add the u dot products
        if self.do_fdotu:
            u = torch.zeros((4), device=F4.device)
            u[3] = 1
            for a in range(2*self.NF):
                X[:,index] = dot4(F4_flat[:,:,a], u[None,:])
                index += 1
        
        assert(index==self.NX)
        return X


    # convert the 3-flavor matrix into an effective 2-flavor matrix
    # input and output are indexed as [sim, nu/nubar(out), flavor(out), nu/nubar(in), flavor(in)]
    # This assumes that the x flavors will represent the SUM of mu and tau flavors.
    @torch.jit.export
    def convert_y_to_2flavor(self, y):
        y2F = torch.zeros((y.shape[0],2,2,2,2), device=y.device)

        y2F[:,:,0,:,0] =                 y[:, :, 0 , :, 0 ]
        y2F[:,:,0,:,1] = 0.5 * torch.sum(y[:, :, 0 , :, 1:], dim=(  3))
        y2F[:,:,1,:,1] = 0.5 * torch.sum(y[:, :, 1:, :, 1:], dim=(2,4))
        y2F[:,:,1,:,0] =       torch.sum(y[:, :, 1:, :, 0 ], dim=(2  ))

        return y2F

    # Push the inputs through the neural network
    # output is indexed as [sim, nu/nubar(out), flavor(out), nu/nubar(in), flavor(in)]
    def forward(self,x):
        # evaluate the shared portion of the network
        y_shared = self.linear_activation_stack_shared(x)

        # evaluate each task
        y_stability  = self.linear_activation_stack_stability(y_shared)
        y_growthrate = self.linear_activation_stack_growthrate(y_shared)
        y_dens       = self.linear_activation_stack_density(y_shared)
        y_flux       = self.linear_activation_stack_flux(y_shared)

        # reshape into a matrix
        y_dens = y_dens.reshape(x.shape[0], 2,self.NF,2,self.NF)
        y_flux = y_flux.reshape(x.shape[0], 2,self.NF,2,self.NF)
        
        # enforce conservation of particle number
        delta_nunubar = torch.eye(2, device=x.device)[None,:,None,:,None]
        y_dens_flavorsum = torch.sum(y_dens,dim=2)[:,:,None,:,:]
        y_flux_flavorsum = torch.sum(y_flux,dim=2)[:,:,None,:,:]
        y_dens = y_dens + (delta_nunubar - y_dens_flavorsum) / self.NF
        y_flux = y_flux + (delta_nunubar - y_flux_flavorsum) / self.NF

        # enforce symmetry in the heavies
        if self.average_heavies_in_final_state:
            y_dens[:,:,1:,:,:] = y_dens.clone().detach()[:,:,1:,:,:].mean(dim=2, keepdim=True)
            y_flux[:,:,1:,:,:] = y_flux.clone().detach()[:,:,1:,:,:].mean(dim=2, keepdim=True)

        # enforce eln conservation through y matrix
        if self.conserve_lepton_number == "ymatrix":
            delta_flavor = torch.eye(self.NF, device=x.device)[None,None,:,None,:]
            ELN_excess = y_dens[:,0] - y_dens[:,1] - (delta_nunubar[:,0]-delta_nunubar[:,1]) * delta_flavor[:,0]
            y_dens[:,0] -= ELN_excess/2.
            y_dens[:,1] += ELN_excess/2.

        return y_dens, y_flux, y_growthrate, y_stability
  
    # Use the initial F4 and the ML output to calculate the final F4
    # F4_initial must have shape [sim, xyzt, nu/nubar, flavor]
    # y must have shape [sim,2,NF,2,NF]
    # F4_final has shape [sim, xyzt, nu/nubar, flavor]
    # Treat all but the last flavor as output, since it is determined by conservation
    @torch.jit.export
    def F4_from_y(self, F4_initial, y_dens, y_flux):
        # tensor product with y
        # n indicates the simulation index
        # m indicates the spacetime index
        # i and j indicate nu/nubar
        # a and b indicate flavor
        F4_final = torch.einsum("niajb,nmjb->nmia", y_flux, F4_initial)
        F4_final[:,3,:,:] = torch.einsum("niajb,njb->nia", y_dens, F4_initial[:,3,:,:])

        # enforce eln
        if self.conserve_lepton_number == "direct":
            Jt_initial = F4_initial[:,3,0,:] - F4_initial[:,3,1,:]
            Jt_final   =   F4_final[:,3,0,:] -   F4_final[:,3,1,:]
            delta_Jt = Jt_final - Jt_initial
            F4_final[:,3,0,:] = F4_final[:,3,0,:] - 0.5*delta_Jt
            F4_final[:,3,1,:] = F4_final[:,3,1,:] + 0.5*delta_Jt

        return F4_final

    @torch.jit.export
    def predict_all(self, F4_initial):
        # get X values from input
        X = self.X_from_F4(F4_initial)

        # propagate through the network
        y_dens, y_flux, y_growthrate, y_stability = self.forward(X)

        # apply sigmoid to stability logits
        stability = torch.squeeze(torch.sigmoid(y_stability))

        # convert transformation matrix to flux values
        F4_final = self.F4_from_y(F4_initial, y_dens, y_flux)

        # apply total density scaling to log growth rate
        # expects F4 to be in units of cm^-3
        ndens_to_invsec = 1.3615452913035457e-22 # must be declared a literal here or pytorch complains when exporting the model
        growthrate = torch.exp(torch.squeeze(y_growthrate)) * ntotal(F4_initial)*ndens_to_invsec

        return F4_final, growthrate, stability

    
    @torch.jit.export
    def predict_WhiskyTHC(self, F4_initial):
        # get X values from input
        X = self.X_from_F4(F4_initial)

        # propagate through the network
        y_dens, y_flux, y_growthrate, y_stability = self.forward(X)

        # apply total density scaling to log growth rate
        # expects F4 to be in units of cm^-3
        ndens_to_invsec = 1.3615452913035457e-22 # must be declared a literal here or pytorch complains when exporting the model
        growthrate = torch.exp(torch.squeeze(y_growthrate)) * ntotal(F4_initial)*ndens_to_invsec

        # apply sigmoid to stability logits
        stability = torch.squeeze(torch.sigmoid(y_stability))
        growthrate[torch.where(stability>0.5)] = 0
        
        return [y_dens, y_flux, growthrate]
