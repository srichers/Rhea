'''
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This file contains a variety of functions that generate randomized distributions
'''

import torch
import sys
sys.path.append("..")
import ml_maxentropy

# create list of simulations known to be stable when the flux factor is zero
# input has dimensions  [nsims  , xyzt, nu/nubar, flavor]
# output has dimensions [nsims*2, xyzt, nu/nubar, flavor]
# array of fluxes that are stable to the FFI
def generate_stable_F4_zerofluxfac(NF, n_generate, average_heavies_in_final_state):
    F4i = torch.zeros((n_generate, 4, 2, NF))
    
    # keep all of the fluxes zero
    # densities randomly from 0 to 1
    F4i[:,3,:,:] = torch.rand(n_generate, 2, NF)

    # normalize
    ntot = torch.sum(F4i[:,3,:,:], dim=(1,2))
    F4i = F4i / ntot[:,None,None,None]

    # average if necessary
    if average_heavies_in_final_state:
        F4i[:,:,:,1:] = torch.mean(F4i[:,:,:,1:], dim=3, keepdims=True)
    
    return F4i

def generate_stable_F4_oneflavor(NF, n_generate, average_heavies_in_final_state):
    # choose the flux to be in a random direction
    costheta = 2*(torch.rand(n_generate) - 0.5)
    phi = 2*torch.pi*torch.rand(n_generate)
    Frand = torch.zeros(n_generate, 3)
    Frand[:,0] = torch.sqrt(1-costheta**2) * torch.cos(phi)
    Frand[:,1] = torch.sqrt(1-costheta**2) * torch.sin(phi)
    Frand[:,2] = costheta

    # choose a random flux factor
    fluxfac = torch.rand(n_generate)

    # set Frand to be the flux factor times the density. Assume that the density is 1
    Frand = Frand * fluxfac[:,None]

    # set F4i with the new flux values
    F4i = torch.zeros((n_generate*2*NF, 4, 2, NF))
    for i in range(2):
        for j in range(NF):
            # set start and end indices for current flavor and nu/nubar
            start = i*n_generate + j*n_generate*2
            end   = start + n_generate
            F4i[start:end,0:3,i,j] = Frand
            F4i[start:end,  3,i,j] = 1
    
    # average if necessary
    if average_heavies_in_final_state:
        F4i[:,:,:,1:] = torch.mean(F4i[:,:,:,1:], dim=3, keepdims=True)

    return F4i

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

    # normalize so the total number density is 1
    ntot = torch.sum(F4i[:,3,:,:], dim=(1,2))
    F4i = F4i / ntot[:,None,None,None]

    # average if necessary
    if average_heavies_in_final_state:
        F4i[:,:,:,1:] = torch.mean(F4i[:,:,:,1:], dim=3, keepdims=True)

    return F4i


if __name__ == "__main__":
    NF = 3
    result = generate_stable_F4_zerofluxfac(NF, 100, False).numpy()
    print("generate_stable_F4_zerofluxfac output: ",result.shape)
    assert(all(ml_maxentropy.has_crossing(result, 3, 64)==False))
    
    result = generate_stable_F4_oneflavor(NF, 100, False).numpy()
    print("generate_stable_F4_oneflavor output: ",result.shape)
    assert(all(ml_maxentropy.has_crossing(result, 3, 64)==False))

    result = generate_random_F4(NF, 100, False, 10, 0.95).numpy()
    print("generate_random_F4 output:", result.shape)
    #assert(all(ml_maxentropy.has_crossing(result, 3, 64)==False)) # these are NOT all stable
