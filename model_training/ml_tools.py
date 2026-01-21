'''
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This file contains a variety of utilities used in the model training process.
'''

import numpy as np
import torch
import e3nn
import copy

# set e3nn to trace without scripting. Scripting e3nn seems to make it fail
e3nn.set_optimization_defaults(jit_mode="eager")

# input dimensions: [sim, ..., xyzt]
# output dimensions: [sim, ...]
def dot4(v1,v2):
    return -v1[...,3]*v2[...,3] + torch.sum(v1[...,:3]*v2[...,:3], dim=-1)

# input dimensions: [sim, nu/nubar, flavor, xyzt]
# output dimensions: [sim]
def ntotal(F4):
    ntotal = torch.sum(F4[:,:,:,3], dim=(1,2)) # [sim]
    return ntotal

# input dimensions: [sim, nu/nubar, flavor, xyzt]
# output dimensions: [sim, nunubar, flavor]
def flux_factor(F4):
    return torch.sqrt(torch.sum(F4[:,:,:,:3]**2, dim=1)) / F4[:,:,:,3]

def check_conservation(F4_initial_list, F4_final_list, tolerance = 1e-3):
    ntot = ntotal(F4_initial_list)

    # N-Nbar must be preserved for SI-only Hamiltonian
    # Check that this actually happened in the simulations
    N_initial = F4_initial_list[:,:,:,3] # [sim, nu/nubar, flavor]
    N_final   =   F4_final_list[:,:,:,3]
    N_Nbar_initial = N_initial[:,0,:] - N_initial[:,1,:]
    N_Nbar_final   = N_final[  :,0,:] - N_final[  :,1,:] # [sim, flavor]
    N_Nbar_difference = (N_Nbar_initial - N_Nbar_final) # [sim, flavor]
    N_Nbar_error = torch.max(torch.abs(N_Nbar_difference)/ntot[:,None]) # [sim, flavor]
    print("#    N_Nbar_error = ", N_Nbar_error.item())
    #assert(N_Nbar_error < tolerance)

    # check for number conservation
    F4_flavorsum_initial = torch.sum(F4_initial_list, dim=(2)) # [sim, nu/nubar, xyzt]
    F4_flavorsum_final   = torch.sum(F4_final_list,   dim=(2))
    F4_flavorsum_error = torch.max(torch.abs(F4_flavorsum_initial - F4_flavorsum_final)/ntot[:,None,None])
    print("#    F4_flavorsum_difference = ", F4_flavorsum_error.item())
    assert(F4_flavorsum_error < tolerance)

# F4 has dimensions [sim, nu/nubar, flavor, xyzt]
def restrict_F4_to_physical(F4_final):
    NF = F4_final.shape[-1]
    avgF4 = torch.sum(F4_final, dim=2, keepdim=True) / NF

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

def save_model(model, outfilename, device, F4i_test):
    with torch.no_grad():
        # run around tracing/scripting issues by creating a copy of the model and turning off gradients
        model_to_trace = copy.deepcopy(model)
        model_to_trace.eval()
        model_to_trace.to(device)
        for param in model_to_trace.parameters():
            param.requires_grad_(False)

        # evaluate the trace
        example = F4i_test[:1].to(device)
        def forward_fn(x):
            return model_to_trace(x)
        traced_fn = torch.jit.trace(forward_fn, example, strict=False)
        torch.jit.save(traced_fn, outfilename+"_"+device+".pt")
        print("Saving to",outfilename+"_"+device+".pt")
