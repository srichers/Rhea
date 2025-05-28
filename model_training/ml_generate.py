'''
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This file contains a variety of functions that generate randomized distributions
'''

import torch

# create list of simulations known to be stable when the flux factor is zero
# input has dimensions  [nsims  , xyzt, nu/nubar, flavor]
# output has dimensions [nsims*2, xyzt, nu/nubar, flavor]
# array of fluxes that are stable to the FFI
def generate_stable_F4_zerofluxfac(parms):
    F4i = torch.zeros((parms["n_generate"], 4, 2, parms["NF"]), device=parms["device"])
    
    # keep all of the fluxes zero
    # densities randomly from 0 to 1
    F4i[:,3,:,:] = torch.rand(parms["n_generate"], 2, parms["NF"], device=parms["device"])

    # normalize
    ntot = torch.sum(F4i[:,3,:,:], dim=(1,2))
    F4i = F4i / ntot[:,None,None,None]

    # average if necessary
    if parms["average_heavies_in_final_state"]:
        F4i[:,:,:,1:] = torch.mean(F4i[:,:,:,1:], dim=3, keepdims=True)
    
    return F4i

def generate_stable_F4_oneflavor(parms):
    # choose the flux to be in a random direction
    costheta = 2*(torch.rand(parms["n_generate"], device=parms["device"]) - 0.5)
    phi = 2*torch.pi*torch.rand(parms["n_generate"], device=parms["device"])
    Frand = torch.zeros(parms["n_generate"], 3, device=parms["device"])
    Frand[:,0] = torch.sqrt(1-costheta**2) * torch.cos(phi)
    Frand[:,1] = torch.sqrt(1-costheta**2) * torch.sin(phi)
    Frand[:,2] = costheta

    # choose a random flux factor
    fluxfac = torch.rand(parms["n_generate"], device=parms["device"])

    # set Frand to be the flux factor times the density. Assume that the density is 1
    Frand = Frand * fluxfac[:,None]

    # set F4i with the new flux values
    F4i = torch.zeros((parms["n_generate"]*2*parms["NF"], 4, 2, parms["NF"]), device=parms["device"])
    for i in range(2):
        for j in range(parms["NF"]):
            # set start and end indices for current flavor and nu/nubar
            start = i*parms["n_generate"] + j*parms["n_generate"]*2
            end   = start + parms["n_generate"]
            F4i[start:end,0:3,i,j] = Frand
            F4i[start:end,  3,i,j] = 1
    
    # average if necessary
    if parms["average_heavies_in_final_state"]:
        F4i[:,:,:,1:] = torch.mean(F4i[:,:,:,1:], dim=3, keepdims=True)

    return F4i

def generate_random_F4(parms):
    assert(parms["n_generate"] >= 0)
    F4i = torch.zeros((parms["n_generate"], 4, 2, parms["NF"]), device=parms["device"])

    # choose a random number density
    Ndens = torch.rand(parms["n_generate"], 2, parms["NF"], device=parms["device"])
    Ndens[torch.where(Ndens==0)] = 1
    F4i[:,3,:,:] = Ndens

    # choose the flux to be in a random direction
    costheta = 2*(torch.rand(parms["n_generate"], 2, parms["NF"], device=parms["device"]) - 0.5)
    phi = 2*torch.pi*torch.rand(parms["n_generate"], 2, parms["NF"], device=parms["device"])
    sintheta = torch.sqrt(1-costheta**2)
    F4i[:,0,:,:] = sintheta * torch.cos(phi)
    F4i[:,1,:,:] = sintheta * torch.sin(phi)
    F4i[:,2,:,:] = costheta
    F4i[:,3,:,:] = 1

    # choose a random flux factor
    fluxfac = torch.rand(parms["n_generate"], 2, parms["NF"], device=parms["device"])*parms["generate_max_fluxfac"]
    fluxfac = fluxfac**parms["generate_zero_weight"]

    # multiply the spatial flux by the flux factor times the density.
    F4i[:,0:3,:,:] = F4i[:,0:3,:,:] * fluxfac[:,None,:,:]
    
    # scale by the number density
    F4i = F4i * Ndens[:,None,:,:]

    # normalize so the total number density is 1
    ntot = torch.sum(F4i[:,3,:,:], dim=(1,2))
    F4i = F4i / ntot[:,None,None,None]

    # average if necessary
    if parms["average_heavies_in_final_state"]:
        F4i[:,:,:,1:] = torch.mean(F4i[:,:,:,1:], dim=3, keepdims=True)

    return F4i
