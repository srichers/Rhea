'''
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This is the file contains the functions that define the various loss sources.
'''

import torch

# Mean over everything but the point index, then a weighted sum over points. weight is
# an optional per-point weight that sums to one; weight==None weights every point
# equally, which is identical to the plain mean over all elements.
# reshape rather than flatten(1), because growthrate has only the point dimension.
def weighted_mean(error, weight):
    if weight == None:
        return torch.mean(error)
    else:
        return torch.sum(weight * torch.mean(error.reshape(error.shape[0],-1), dim=1)) # [sim, everything else]

#===================================================================#
# violations of the physical bounds that the final state must obey. #
# input dimensions: [sim, nu/nubar, flavor, xyzt]                   #
#===================================================================#
# the amount by which each number density falls below zero. Never positive.
def negative_density_error(F4f_pred):
    return torch.min(F4f_pred[:,:,:,3], torch.zeros_like(F4f_pred[:,:,:,3])) # [sim, nu/nubar, flavor]

# the amount by which each squared flux magnitude exceeds the squared number density,
# i.e. by which the flux factor exceeds one. Never negative.
def fluxfac_error(F4f_pred):
    flux_mag2 = torch.sum(F4f_pred[:,:,:,0:3]**2, dim=3) # [sim, nu/nubar, flavor]
    ndens2 = F4f_pred[:,:,:,3]**2 # [sim, nu/nubar, flavor]
    return torch.max(flux_mag2 - ndens2, torch.zeros_like(ndens2)) # [sim, nu/nubar, flavor]

#================#
# loss functions #
#================#
# with mean, this is equivalent to torch.sum(diff**2) / F4f_pred.numel()
def comparison_loss_fn(y_pred, y_true, weight=None):
    return weighted_mean((y_pred - y_true)**2, weight)

# enforce that number density cannot be less than zero. The second argument is ignored, and
# is kept only so that every loss function can be called the same way.
def negative_density_loss_fn(F4f_pred, _, weight=None):
    return weighted_mean(negative_density_error(F4f_pred)**2, weight)

# enforce that flux factors cannot be larger than 1
def fluxfac_loss_fn(F4f_pred, _, weight=None):
    return weighted_mean(fluxfac_error(F4f_pred), weight)

#==============================================================================#
# largest error anywhere in the batch, to accompany the mean losses in loss.dat #
#==============================================================================#
def max_error(F4f_pred, F4f_true):
    return torch.max(torch.abs(F4f_pred - F4f_true)).item()

# the unphysical losses have no true value to compare against, so they report the
# largest violation of the bound rather than the largest deviation from a known answer
def negative_density_max(F4f_pred, _):
    return torch.max(torch.abs(negative_density_error(F4f_pred))).item()

def fluxfac_max(F4f_pred, _):
    return torch.max(fluxfac_error(F4f_pred)).item()
