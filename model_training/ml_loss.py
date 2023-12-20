import torch

def comparison_loss_fn(model, F4_pred, F4_true):
    return torch.nn.MSELoss(reduction='mean')(F4_pred, F4_true)

def unphysical_loss_fn(model, F4f_pred, F4f_true):
    assert(F4f_true == None)

    # enforce that number density cannot be less than zero
    negative_density_error = torch.min(F4f_pred[:,3,:,:], torch.zeros_like(F4f_pred[:,3,:,:])) # [sim, nu/nubar, flavor]
    negative_density_loss = torch.mean(negative_density_error**2)

    # enforce that flux factors cannot be larger than 1
    flux_mag2 = torch.sum(F4f_pred[:,0:3,:,:]**2, axis=1) # [sim, nu/nubar, flavor]
    ndens2 = F4f_pred[:,3,:,:]**2 # [sim, nu/nubar, flavor]
    fluxfac_error = torch.max(flux_mag2 - ndens2, torch.zeros_like(ndens2)) # [sim, nu/nubar, flavor]
    fluxfac_loss = torch.mean(fluxfac_error)

    # total conservation loss
    return negative_density_loss + fluxfac_loss

# enforce conservation of particle number
# inputs are indexed as [sim, xyzt, nu/nubar, flavor]
def particle_number_loss_fn(model, F4f, F4i):
    PNi = torch.sum(F4i[:,3,:,:], axis=2)
    PNf = torch.sum(F4f[:,3,:,:], axis=2)
    return torch.nn.MSELoss(reduction='mean')(PNi, PNf)
