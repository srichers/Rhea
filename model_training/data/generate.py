'''
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This file contains a variety of functions that generate randomized distributions
'''

import torch
import sys
sys.path.append("..")
import maxentropy
import h5py

# create list of simulations known to be stable when the flux factor is zero
# input has dimensions  [nsims  , xyzt, nu/nubar, flavor]
# output has dimensions [nsims*2, xyzt, nu/nubar, flavor]
# array of fluxes that are stable to the FFI
def generate_stable_F4_zerofluxfac(NF, n_generate, average_heavies_in_final_state):
    F4i = torch.zeros((n_generate, 4, 2, NF))
    
    # keep all of the fluxes zero
    # densities randomly from 0 to 1
    F4i[:,3,:,:] = torch.rand(n_generate, 2, NF)

    return F4i

def generate_stable_F4_oneflavor(NF, n_generate, average_heavies_in_final_state):
    n_generate_each_species = n_generate//(2*NF)
    
    # set F4i with the new flux values
    F4i = torch.zeros((n_generate_each_species*2*NF, 4, 2, NF))


    for s in range(2):
        for f in range(NF):
            # set electron neutrino number densities to 1
            F4i[:,3,s,f] = 1
            
            # choose the flux to be in a random direction
            costheta = 2*(torch.rand(n_generate_each_species) - 0.5)
            phi = 2*torch.pi*torch.rand(n_generate_each_species)
            Frand = torch.zeros(n_generate_each_species, 3)
            Frand[:,0] = torch.sqrt(1-costheta**2) * torch.cos(phi)
            Frand[:,1] = torch.sqrt(1-costheta**2) * torch.sin(phi)
            Frand[:,2] = costheta
            
            # choose a random flux factor
            fluxfac = torch.rand(n_generate_each_species)
            
            # set Frand to be the flux factor times the density. Assume that the density is 1
            Frand = Frand * fluxfac[:,None]
            
            # set the fluxes of F4i
            start = n_generate_each_species * (f + NF*s)
            stop = start + n_generate_each_species
            F4i[start:stop,0:3,s,f] = Frand
    
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

    return F4i

def write_stable_dataset(outfilename, F4i, stable):
    f = h5py.File(outfilename,"w")
    f["F4_initial(1|ccm)"] = F4i
    f["stable"] = stable
    f.close()

if __name__ == "__main__":
    NF = 3
    n_generate = 100

    result = generate_stable_F4_zerofluxfac(NF, n_generate, False).numpy()
    print("generate_stable_F4_zerofluxfac output: ",result.shape)
    assert(all(maxentropy.has_crossing(result, 3, 64)==False))
    write_stable_dataset("stable_zerofluxfac_database.h5",result, torch.ones(n_generate))
    
    result = generate_stable_F4_oneflavor(NF, n_generate, False).numpy()
    print("generate_stable_F4_oneflavor output: ",result.shape)
    assert(all(maxentropy.has_crossing(result, 3, 64)==False))
    write_stable_datset("stable_oneflavor_database.h5", result, torch.ones(n_generate))

    result = generate_random_F4(NF, n_generate, False, 10, 0.95).numpy()
    hascrossing = torch.tensor(maxentropy.has_crossing(result, 3, 200))
    print("generate_stable_F4_oneflavor output: ",result.shape)
    print("nstable:",torch.sum(hascrossing))
    write_stable_dataset("stable_random_database.h5", result, 1-hascrossing)
    
