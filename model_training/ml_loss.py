'''
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This is the file contains the functions that define the various loss sources.
'''

import torch

# weight is an optional per-point weight that sums to one. weight==None weights every
# point equally, which is identical to the plain mean over all elements.
# reshape rather than flatten(1), because growthrate has only the point dimension.

# with mean, this is equivalent to torch.sum(diff**2) / F4f_pred.numel()
def comparison_loss_fn(y_pred, y_true, weight=None):
    if weight == None:
        return torch.nn.MSELoss(reduction='mean')(y_pred, y_true)
    else:
        error2 = ((y_pred - y_true)**2).reshape(y_pred.shape[0],-1) # [sim, everything else]
        return torch.sum(weight * torch.mean(error2, dim=1))

def unphysical_loss_fn(F4f_pred, F4f_true, weight=None):
    assert(F4f_true == None)

    # enforce that number density cannot be less than zero
    negative_density_error = torch.min(F4f_pred[:,:,:,3], torch.zeros_like(F4f_pred[:,:,:,3])) # [sim, nu/nubar, flavor]

    # enforce that flux factors cannot be larger than 1
    flux_mag2 = torch.sum(F4f_pred[:,:,:,0:3]**2, dim=3) # [sim, nu/nubar, flavor]
    ndens2 = F4f_pred[:,:,:,3]**2 # [sim, nu/nubar, flavor]
    fluxfac_error = torch.max(flux_mag2 - ndens2, torch.zeros_like(ndens2)) # [sim, nu/nubar, flavor]

    if weight == None:
        negative_density_loss = torch.mean(negative_density_error**2)
        fluxfac_loss          = torch.mean(fluxfac_error)
    else:
        negative_density_loss = torch.sum(weight * torch.mean((negative_density_error**2).reshape(F4f_pred.shape[0],-1), dim=1))
        fluxfac_loss          = torch.sum(weight * torch.mean( fluxfac_error             .reshape(F4f_pred.shape[0],-1), dim=1))

    # total conservation loss
    return negative_density_loss + fluxfac_loss

def max_error(F4f_pred, F4f_true):
    if F4f_true == None:
        return 0
    else:
        return torch.max(torch.abs(F4f_pred - F4f_true)).item()
    
