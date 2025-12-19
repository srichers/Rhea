'''
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This is the file contains the functions that define the various loss sources.
'''

import torch

# with mean, this is equivalent to torch.sum(diff**2) / F4f_pred.numel()
def comparison_loss_fn(F4_pred, F4_true):
    return torch.nn.MSELoss(reduction='mean')(F4_pred, F4_true)

def unphysical_loss_fn(F4f_pred, F4f_true):
    assert(F4f_true == None)

    # enforce that number density cannot be less than zero
    negative_density_error = torch.min(F4f_pred[:,3,:,:], torch.zeros_like(F4f_pred[:,3,:,:])) # [sim, nu/nubar, flavor]
    negative_density_loss = torch.mean(negative_density_error**2)

    # enforce that flux factors cannot be larger than 1
    flux_mag2 = torch.sum(F4f_pred[:,0:3,:,:]**2, dim=1) # [sim, nu/nubar, flavor]
    ndens2 = F4f_pred[:,3,:,:]**2 # [sim, nu/nubar, flavor]
    fluxfac_error = torch.max(flux_mag2 - ndens2, torch.zeros_like(ndens2)) # [sim, nu/nubar, flavor]
    fluxfac_loss = torch.mean(fluxfac_error)

    # total conservation loss
    return negative_density_loss + fluxfac_loss

def direction_loss_fn(Fhat_pred, Fhat_true):
    fdotf = torch.sum(Fhat_pred * Fhat_true, dim=1)
    return torch.mean(torch.ones_like(fdotf) - fdotf)

def stability_loss_fn(stability_pred, stability_true):
    return torch.nn.BCEWithLogitsLoss(reduction='mean')(stability_pred, stability_true)

def max_error(F4f_pred, F4f_true):
    if F4f_true == None:
        return 0
    else:
        return torch.max(torch.abs(F4f_pred - F4f_true))

def pcgrad(task_losses, params, eps=1e-12):
    assert params
    assert len(task_losses) > 0

    task_grads = []
    for loss in task_losses:
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]
        task_grads.append(grads)

    projected_grads = []
    for i, grads_i in enumerate(task_grads):
        grads_proj = [g.clone() for g in grads_i]
        for j, grads_j in enumerate(task_grads):
            if i == j:
                continue
            dot = sum((gi * gj).sum() for gi, gj in zip(grads_proj, grads_j))
            if dot.item() < 0:
                denom = sum((gj * gj).sum() for gj in grads_j)
                if denom.item() > 0:
                    scale = dot / (denom + eps)
                    grads_proj = [gi - scale * gj for gi, gj in zip(grads_proj, grads_j)]
        projected_grads.append(grads_proj)

    merged_grads = []
    for k in range(len(params)):
        merged_grads.append(sum(grads[k] for grads in projected_grads))
    return merged_grads
    
