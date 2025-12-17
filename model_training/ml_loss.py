"""
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This is the file contains the functions that define the various loss sources.
"""

import torch
import torch.nn.functional as F


# with mean, this is equivalent to torch.sum(diff**2) / F4f_pred.numel()
def comparison_loss_fn(F4_pred, F4_true):
    return F.mse_loss(F4_pred, F4_true, reduction="mean")


def unphysical_loss_fn(F4f_pred, F4f_true):
    assert F4f_true == None

    # enforce that number density cannot be less than zero
    ndens = F4f_pred[:, 3, :, :]  # [sim, nu/nubar, flavor]
    negative_density_error = torch.clamp(ndens, max=0.0)
    negative_density_loss = torch.mean(negative_density_error**2)

    # enforce that flux factors cannot be larger than 1
    flux_mag2 = torch.sum(F4f_pred[:, 0:3, :, :] ** 2, dim=1)  # [sim, nu/nubar, flavor]
    ndens2 = F4f_pred[:, 3, :, :] ** 2  # [sim, nu/nubar, flavor]
    fluxfac_error = torch.clamp(flux_mag2 - ndens2, min=0.0)  # [sim, nu/nubar, flavor]
    fluxfac_loss = torch.mean(fluxfac_error)

    # total conservation loss
    return negative_density_loss + fluxfac_loss


def direction_loss_fn(Fhat_pred, Fhat_true):
    fdotf = torch.sum(Fhat_pred * Fhat_true, dim=1)
    return torch.mean(1.0 - fdotf)


def stability_loss_fn(stability_pred, stability_true):
    return F.binary_cross_entropy_with_logits(
        stability_pred, stability_true, reduction="mean"
    )


def max_error(F4f_pred, F4f_true):
    if F4f_true == None:
        return 0.0
    return torch.max(torch.abs(F4f_pred - F4f_true)).detach().item()


loss_dict = {}


def contribute_loss(pred, true, traintest, key, loss_fn):
    loss = loss_fn(pred, true)
    loss_dict[key + "_" + traintest + "_loss"] = loss.detach().item()
    loss_dict[key + "_" + traintest + "_max"] = max_error(pred, true)
    return loss
