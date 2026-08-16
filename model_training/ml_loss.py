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

# the amount by which each flux magnitude exceeds its number density, i.e. by which the
# flux factor exceeds one. Never negative. Same units as negative_density_error, so the
# two task weights are commensurate. The epsilon keeps the sqrt gradient finite at zero
# flux; d|f|/df is bounded by one no matter how small it is.
# the density is clamped at zero so that this measures only the flux factor violation.
def fluxfac_error(F4f_pred):
    flux_mag = torch.sqrt(torch.sum(F4f_pred[:,:,:,0:3]**2, dim=3) + torch.finfo(F4f_pred.dtype).tiny) # [sim, nu/nubar, flavor]
    ndens = torch.max(F4f_pred[:,:,:,3], torch.zeros_like(F4f_pred[:,:,:,3])) # [sim, nu/nubar, flavor]
    return torch.max(flux_mag - ndens, torch.zeros_like(ndens)) # [sim, nu/nubar, flavor]

#================#
# loss functions #
#================#
# with mean, this is equivalent to torch.sum(diff**2) / F4f_pred.numel()
def comparison_loss_fn(y_pred, y_true, weight=None):
    return weighted_mean((y_pred - y_true)**2, weight)

# enforce that number density cannot be less than zero. The second argument is ignored, and
# is kept only so that every loss function can be called the same way.
# both unphysical penalties are linear in the violation so that the gradient does not vanish
# at the bound.
def negative_density_loss_fn(F4f_pred, _, weight=None):
    return weighted_mean(torch.abs(negative_density_error(F4f_pred)), weight)

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
