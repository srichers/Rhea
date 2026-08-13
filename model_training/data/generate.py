'''
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This file contains a variety of functions that generate randomized distributions
'''

import torch

def generate_random_F4(NF, n_generate, average_heavies_in_final_state, zero_weight, max_fluxfac):
    assert(n_generate >= 0)
    F4i = torch.zeros((n_generate, 4, 2, NF))

    # choose a random number density
    Ndens = torch.rand(n_generate, 2, NF)
    Ndens[torch.where(Ndens==0)] = 1
    F4i[:,3,:,:] = Ndens

    # choose the flux to be in a random direction
    costheta = 2*(torch.rand(n_generate, 2, NF) - 0.5)
    phi = 2*torch.pi*torch.rand(n_generate, 2, NF)
    sintheta = torch.sqrt(1-costheta**2)
    F4i[:,0,:,:] = sintheta * torch.cos(phi)
    F4i[:,1,:,:] = sintheta * torch.sin(phi)
    F4i[:,2,:,:] = costheta
    F4i[:,3,:,:] = 1

    # choose a random flux factor
    fluxfac = torch.rand(n_generate, 2, NF)*max_fluxfac
    fluxfac = fluxfac**zero_weight

    # multiply the spatial flux by the flux factor times the density.
    F4i[:,0:3,:,:] = F4i[:,0:3,:,:] * fluxfac[:,None,:,:]
    
    # scale by the number density
    F4i = F4i * Ndens[:,None,:,:]

    return F4i

