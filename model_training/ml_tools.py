'''
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This file contains a variety of utilities used in the model training process.
'''

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from itertools import permutations

# input dimensions: [sim, xyzt, ...]
# output dimensions: [sim, ...]
def dot4(v1,v2):
    return -v1[:,3]*v2[:,3] + torch.sum(v1[:,:3]*v2[:,:3], dim=1)

# input dimensions: [sim, xyzt, nu/nubar, flavor]
# output dimensions: [sim]
def ntotal(F4):
    ntotal = torch.sum(F4[:,3], dim=(1,2)) # [sim]
    return ntotal

def flux_factor(F4):
    return torch.sqrt(torch.sum(F4[:,:3]**2, dim=1)) / F4[:,3]

def check_conservation(F4_initial_list, F4_final_list, tolerance = 1e-3):
    ntot = ntotal(F4_initial_list)

    # N-Nbar must be preserved for SI-only Hamiltonian
    # Check that this actually happened in the simulations
    N_initial = F4_initial_list[:,3,:,:] # [sim, nu/nubar, flavor]
    N_final   =   F4_final_list[:,3,:,:]
    N_Nbar_initial = N_initial[:,0,:] - N_initial[:,1,:]
    N_Nbar_final   = N_final[  :,0,:] - N_final[  :,1,:] # [sim, flavor]
    N_Nbar_difference = (N_Nbar_initial - N_Nbar_final) # [sim, flavor]
    N_Nbar_error = torch.max(torch.abs(N_Nbar_difference)/ntot[:,None]) # [sim, flavor]
    print("    N_Nbar_error = ", N_Nbar_error.item())
    #assert(N_Nbar_error < tolerance)

    # check for number conservation
    F4_flavorsum_initial = torch.sum(F4_initial_list, dim=(3)) # [sim, xyzt, nu/nubar]
    F4_flavorsum_final   = torch.sum(F4_final_list,   dim=(3))
    F4_flavorsum_error = torch.max(torch.abs(F4_flavorsum_initial - F4_flavorsum_final)/ntot[:,None,None])
    print("    F4_flavorsum_difference = ", F4_flavorsum_error.item())
    assert(F4_flavorsum_error < tolerance)

# assume input of size [nsims, xyzt, nu/nubar, flavor]
def permute_F4(F4, mperm, fperm):
    result = F4
    result = result[:,:,mperm,:]
    result = result[:,:,:,fperm]
    return result

# create list of equivalent simulations
# input has dimensions  [nsims              , xyzt, nu/nubar, flavor]
# output has dimensions [nsims*npermutations, xyzt, nu/nubar, flavor]
# each simulation should occupy a block of size nperms
def augment_permutation(F4_list):
    nsims = F4_list.shape[0]
    NF = F4_list.shape[-1]

    # set the number of permutations used for data augmentation
    matter_permutations = list(permutations(range(2)))
    flavor_permutations = list(permutations(range(NF)))
    nmperm = len(matter_permutations)
    nfperm = len(flavor_permutations)
    nperms = nmperm * nfperm

    F4_augmented = torch.zeros((nsims*nperms, 4, 2, NF), device=F4_list.device).float()
    
    for mi in range(len(matter_permutations)):
        for fi in range(len(flavor_permutations)):
            iperm = fi + mi*nfperm
            mperm = matter_permutations[mi]
            fperm = flavor_permutations[fi]

            # put the permuted F4 vectors into the augmented arrays
            F4_augmented[iperm::nperms] = permute_F4(F4_list, mperm, fperm)
            
    return F4_augmented

def restrict_F4_to_physical(F4_final):
    NF = F4_final.shape[-1]
    avgF4 = torch.sum(F4_final, dim=3)[:,:,:,None] / NF

    # enforce that all four-vectors are time-like
    # choose the branch that leads to the most positive alpha
    a = dot4(avgF4, avgF4) # [sim, nu/nubar, flavor]
    b = 2.*dot4(avgF4, F4_final)
    c = dot4(F4_final, F4_final)
    radical = b**2 - 4*a*c
    assert(torch.all(radical>=-1e-6))
    radical = torch.maximum(radical, torch.zeros_like(radical))
    alpha = (-b + torch.sign(a)*torch.sqrt(radical)) / (2*a)

    # fix the case where a is zero
    badlocs = torch.where(torch.abs(a/b)<1e-6)
    if len(badlocs[0]) > 0:
        alpha[badlocs] = (-c/b)[badlocs]

    # find the nu/nubar and flavor indices where alpha is maximized
    maxalpha = torch.amax(alpha, dim=(1,2)) # [sim]
    maxalpha = torch.maximum(maxalpha, torch.zeros_like(maxalpha))
    maxalpha[torch.where(maxalpha>0)] += 1e-6

    # modify the four-vectors to be time-like
    F4_final = (F4_final + maxalpha[:,None,None,None]*avgF4) / (maxalpha[:,None,None,None] + 1)

    return F4_final

def get_ndens_fluxmag_fhat(F4):
    ntot = ntotal(F4)
    ndens = F4[:, 3,:,:]/ntot[:,None,None]
    flux  = F4[:,:3,:,:]/ntot[:,None,None,None]
    fluxmag = torch.sqrt(torch.sum(flux**2, dim=1))
    Fhat = flux/fluxmag[:,None,:,:]
    return ndens, fluxmag, Fhat

def save_model(model, outfilename, device, F4i_test):
    with torch.no_grad():
        print(F4i_test.shape)
        model.eval()
        model.to(device)
        #X = model.X_from_F4(F4i_test.to(device))
        #traced_model = torch.jit.trace(model, X)
        #torch.jit.save(traced_model, outfilename+"_"+device+".ptc")
        scripted_model = torch.jit.script(model)
        torch.jit.save(scripted_model, outfilename+"_"+device+".pt")
        print("Saving to",outfilename+"_"+device+".pt")
