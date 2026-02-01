'''
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This is the file contains the functions that define the various loss sources.
'''

import torch

class LossTracker:
    # with mean, this is equivalent to torch.sum(diff**2) / F4f_pred.numel()
    def comparison_loss_fn(self, y_pred, y_true):
        return torch.nn.MSELoss(reduction='mean')(y_pred, y_true)

    def unphysical_loss_fn(self, F4f_pred, F4f_true):
        assert(F4f_true == None)

        # enforce that number density cannot be less than zero
        negative_density_error = torch.min(F4f_pred[:,:,:,3], torch.zeros_like(F4f_pred[:,:,:,3])) # [sim, nu/nubar, flavor]
        negative_density_loss = torch.mean(negative_density_error**2)

        # enforce that flux factors cannot be larger than 1
        flux_mag2 = torch.sum(F4f_pred[:,:,:,0:3]**2, dim=3) # [sim, nu/nubar, flavor]
        ndens2 = F4f_pred[:,:,:,3]**2 # [sim, nu/nubar, flavor]
        fluxfac_error = torch.max(flux_mag2 - ndens2, torch.zeros_like(ndens2)) # [sim, nu/nubar, flavor]
        fluxfac_loss = torch.mean(fluxfac_error)

        # total conservation loss
        return negative_density_loss + fluxfac_loss

    def stability_loss_fn(self, logit, stability_true):
        #print("logit min/max:", logit.min().item(), logit.max().item())

        # clamp the logits to improve numerical stability
        logit = logit.clamp(min=-10, max=10)

        # determine the relative class weights for stable/unstable points
        N_neg = torch.sum(1.0 - stability_true).item()
        N_pos = torch.sum(stability_true).item()
        
        if N_pos>0 and N_neg>0:
            pos_weight = torch.tensor([N_neg / N_pos], device=logit.device)
            result = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)(logit, stability_true)
        else:
            result = torch.nn.BCEWithLogitsLoss(reduction='mean')(logit, stability_true)

        return result

    def max_error(self, F4f_pred, F4f_true):
        if F4f_true == None:
            return 0
        else:
            return torch.max(torch.abs(F4f_pred - F4f_true))
    
