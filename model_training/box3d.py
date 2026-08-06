import torch
from ml_tools import check_conservation

from scipy.integrate import lebedev_rule
pts, weights = lebedev_rule(17)
pts = torch.tensor(pts, dtype=torch.float32, requires_grad=False)
weights = torch.tensor(weights, dtype=torch.float32, requires_grad=False)

# fluxfac has shape [npoints, nunubar, flavor]
def get_Z(fluxfac):
    p = 1- (2*(1-fluxfac)*(1+1.01524*fluxfac))/(3 - 1.00651*fluxfac**2 - 0.962251*fluxfac**4 + 1.47353*fluxfac**6 - 0.48953*fluxfac**8)
    # Expression by Jedynak https://www.sciencedirect.com/science/article/pii/S0377025717302525
    
    return (2*fluxfac)/(1-p)

# n,Z have shape [npoints, nunubar, flavor]
# mu, g have shape [npoints, nunubar, flavor, quadrature]
def distrib(n,Z,mu):
    # Z/sinh(Z)*exp(Z*mu) rewritten so the exponent is never positive. The
    # identity is exact for all Z>0, so there is no large-Z branch to overflow.
    g = (2*Z/(-torch.expm1(-2*Z))) * torch.exp(Z*(mu-1))

    # for small Z use form of expression that avoids division by small numbers
    g = torch.where(Z > 1e-3, g, mu*Z + 1.0)

    return g * (n/(4*torch.pi))


# assume that F4 has shape [npoints, nunubar, flavor, xyzt]
def mixBox3D_lebedev(F4, pts, weights):

    # Operate on a detached clone so Box3D computations don't modify
    # tensors required for gradient computation elsewhere in the graph.
    # This module is not intended to contribute gradients.
    F4 = F4.detach().clone()

    # points with non-finite input cannot be interpreted at all [point]
    unphysical = torch.logical_not(torch.isfinite(F4).flatten(start_dim=1).all(dim=1))

    # get total number of neutrinos
    Ntot = torch.sum(F4[:,:,:,3:], dim=(1,2), keepdim=True)

    # the caller is expected to have normalized so that Ntot is 1
    Ntot_bad = torch.logical_not(torch.isclose(Ntot, torch.ones_like(Ntot), atol=1e-5))
    unphysical = torch.logical_or(unphysical, Ntot_bad.flatten(start_dim=1).any(dim=1))

    # average the heavy flavors
    F4[:,:,1:,:] = torch.mean(F4[:,:,1:,:], dim=2, keepdim=True)

    # get the Z value for each point, nunubar, and flavor
    # only use one heavy flavor since they are averaged
    n = F4[:,:,:2,3:]
    F = F4[:,:,:2,0:3]

    # scale by the largest component before squaring so small fluxes do not
    # underflow float32, which would collapse normF to exactly zero
    Fscale = torch.amax(torch.abs(F), dim=3, keepdim=True)
    Fscale = torch.where(Fscale > 0, Fscale, torch.ones_like(Fscale))
    normF  = Fscale * torch.sqrt(torch.sum((F/Fscale)**2, dim=3, keepdim=True))

    # normF is zero only if F is, so the floor leaves Fhat exactly zero there
    Fhat = F / torch.where(normF > 0, normF, torch.ones_like(normF))

    # Nonpositive density, and flux the quadrature cannot represent, are
    # unphysical. The limit is below the causal value of 1 because the beam
    # width goes like 1/Z: the total density is off by 3% at a flux factor of
    # 0.98, 54% at 0.995 and 666% at 0.999. It must stay inline rather than
    # become a module constant, which torch.jit.script cannot close over. [point]
    fluxfac = normF/n
    bad     = torch.logical_or(n <= 0, fluxfac >= 0.98)
    unphysical = torch.logical_or(unphysical, bad.flatten(start_dim=1).any(dim=1))

    Z = get_Z(fluxfac)

    # evaluate the distribution function at each quadrature point
    mu = torch.matmul(Fhat, pts) # [point, nunubar, flavor, quadrature]
    g = distrib(n,Z,mu) # [point, nunubar, flavor, quadrature]

    # get eln-xln distribution [point, quadrature]
    gELN = (g[:,0,0,:] - g[:,1,0,:]) - (g[:,0,1,:] - g[:,1,1,:])

    # get the integrals of gELN over the quadrature points, weighted by the weights
    # [point]
    Iplus  = torch.sum(   gELN .clamp(min=0)*weights, dim=1)
    Iminus = torch.sum((-gELN).clamp(min=0)*weights, dim=1)

    # set Psur depending on the relative size of Iplus and Iminus [point, quadrature]
    # Written without index gathers, which would force a device sync.
    swap = (Iplus < Iminus)
    hi   = torch.where(swap, Iminus, Iplus)
    lo   = torch.where(swap, Iplus,  Iminus)
    ratio = (lo/torch.where(hi > 0, hi, torch.ones_like(hi))).unsqueeze(1)
    Hplus  = (gELN > 0).to(gELN.dtype)
    Hminus = (gELN < 0).to(gELN.dtype)
    swapq = swap.unsqueeze(1)
    Psur = (1/3)*torch.where(swapq, Hplus, Hminus) + (1-2/3*ratio)*torch.where(swapq, Hminus, Hplus)

    # gELN identically zero gives Iplus==Iminus==0 and hence 0/0 above. There
    # is no crossing, so the distribution is stable and nothing converts.
    noELN = torch.logical_and(Iplus <= 0, Iminus <= 0)
    Psur  = torch.where(noELN[:,None], torch.ones_like(Psur), Psur)

    unphysical = torch.logical_or(unphysical, torch.logical_not((Psur >= 0) & (Psur <= 1)).any(dim=1))

    # compute the mixed distribution function at each quadrature point, using Psur to mix the contributions from the different flavors
    g_e = g[:,:,0,:]
    g_x = g[:,:,1,:]
    Psurq = Psur[:,None,:]
    g_t = torch.stack([ Psurq*g_e     + (1-Psurq)*g_x,
                       ((1+Psurq)*g_x + (1-Psurq)*g_e)/2], dim=2)

    # integrate the mixed distribution function over the quadrature points, weighted by the weights, to get the mixed moments
    # the basis columns are the three direction cosines followed by 1, so a
    # single matmul produces all four components at once [quadrature, xyzt]
    ptsT  = pts.transpose(0,1)
    basis = torch.cat([ptsT, torch.ones_like(ptsT[:,0:1])], dim=1) * weights[:,None]
    F4mix = torch.zeros_like(F4)
    F4mix[:,:,:2,:] = torch.matmul(g_t, basis)

    # Identical mu = tau = x flavors
    F4mix[:,:,2,:] = F4mix[:,:,1,:]

    # check that ELN-xln is still conserved
    #check_conservation(F4, F4mix)

    # negative or non-finite densities are not a usable prediction [point]
    unphysical = torch.logical_or(unphysical, (F4mix[:,:,:,3] < 0).flatten(start_dim=1).any(dim=1))
    unphysical = torch.logical_or(unphysical, torch.logical_not(torch.isfinite(F4mix)).flatten(start_dim=1).any(dim=1))

    # return nan rather than raising, so that a single bad point does not kill
    # the whole batch. Callers running inside a simulation handle the nans.
    growthrate = torch.sqrt(Iplus*Iminus)
    F4mix      = torch.where(unphysical[:,None,None,None], torch.full_like(F4mix,      float("nan")), F4mix)
    growthrate = torch.where(unphysical,                   torch.full_like(growthrate, float("nan")), growthrate)

    return F4mix, growthrate



if __name__ == "__main__":
    # fluxes must be smaller than the density to stay subluminal, and the
    # caller normalizes the total density to 1
    F4 = torch.rand(10,2,3,4)
    F4[:,:,:,:3] = (F4[:,:,:,:3] - 0.5) * F4[:,:,:,3:]
    F4 = F4 / torch.sum(F4[:,:,:,3], dim=(1,2))[:,None,None,None]
    F4mix, I = mixBox3D_lebedev(F4, pts, weights)
    print(F4mix)
    print(I)