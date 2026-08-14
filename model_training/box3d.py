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

# Box3D applied to a single pair of flavors, which is an ordinary two-flavor problem
# on the crossing of their lepton number distributions.
# ELN has shape [npoints, flavor, quadrature]
# returns the survival probability [npoints, quadrature], whether the pair crosses as
# 0 or 1 [npoints], and the growth rate [npoints]
# (the index type hints are required by torch.jit.script, as in predict_all)
def pair_box3d(ELN,i: int,j: int,weights):
    G      = ELN[:,i,:] - ELN[:,j,:]
    Iplus  = torch.sum(   G .clamp(min=0)*weights, dim=1)
    Iminus = torch.sum((-G).clamp(min=0)*weights, dim=1)

    # set Psur depending on the relative size of Iplus and Iminus [point, quadrature]
    # Written without index gathers, which would force a device sync.
    swap = (Iplus < Iminus)
    hi   = torch.where(swap, Iminus, Iplus)
    lo   = torch.where(swap, Iplus,  Iminus)
    ratio = (lo/torch.where(hi > 0, hi, torch.ones_like(hi))).unsqueeze(1)
    Hplus  = (G > 0).to(G.dtype)
    Hminus = (G < 0).to(G.dtype)
    swapq = swap.unsqueeze(1)
    Psur = (1/3)*torch.where(swapq, Hplus, Hminus) + (1-2/3*ratio)*torch.where(swapq, Hminus, Hplus)

    # G identically zero gives Iplus==Iminus==0 and hence 0/0 above. There is no
    # crossing in this pair, so nothing converts between these two flavors.
    noELN = torch.logical_and(Iplus <= 0, Iminus <= 0)
    Psur  = torch.where(noELN[:,None], torch.ones_like(Psur), Psur)

    return Psur, torch.logical_and(Iplus > 0, Iminus > 0).to(ELN.dtype), torch.sqrt(Iplus*Iminus)


# assume that F4 has shape [npoints, nunubar, flavor, xyzt]
# returns the *change* to F4, not the mixed moments themselves
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

    # get the Z value for each point, nunubar, and flavor
    # every flavor is kept separate so that crossings between the heavies are visible
    n = F4[:,:,:,3:]
    F = F4[:,:,:,0:3]

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

    # lepton number for each flavor separately [point, flavor, quadrature]
    ELN = g[:,0,:,:] - g[:,1,:,:]

    # Box3D on each flavor pair. Each pair is an independent two-flavor problem whose
    # survival probability conserves that pair's lepton numbers exactly.
    Psur01, crosses01, rate01 = pair_box3d(ELN,0,1,weights)
    Psur02, crosses02, rate02 = pair_box3d(ELN,0,2,weights)
    Psur12, crosses12, rate12 = pair_box3d(ELN,1,2,weights)

    # The fastest growing pair sets the growth rate. Every pair is tested, so a
    # crossing between the heavies is seen. This is invariant under any permutation
    # of the three flavors.
    growthrate = torch.maximum(rate01, torch.maximum(rate02, rate12))

    unphysical = torch.logical_or(unphysical, torch.logical_not((Psur01 >= 0) & (Psur01 <= 1)).any(dim=1))
    unphysical = torch.logical_or(unphysical, torch.logical_not((Psur02 >= 0) & (Psur02 <= 1)).any(dim=1))
    unphysical = torch.logical_or(unphysical, torch.logical_not((Psur12 >= 0) & (Psur12 <= 1)).any(dim=1))

    # Average the pairwise solutions over the pairs that cross, giving the mixing
    # matrix M[i,j] = (1-Psur_ij)/ncrossing off the diagonal and M[i,i] = 1 - sum_j
    # M[i,j] on it. A pair that does not cross contributes exactly zero, and no
    # crossing at all leaves M the identity. M is symmetric and doubly stochastic, so
    # the flavor trace is conserved at every direction. [point, nunubar, quadrature]
    ncrossing = crosses01 + crosses02 + crosses12
    invN = 1.0/torch.where(ncrossing > 0, ncrossing, torch.ones_like(ncrossing))
    m01 = (invN*crosses01)[:,None,None] * (1-Psur01)[:,None,:]
    m02 = (invN*crosses02)[:,None,None] * (1-Psur02)[:,None,:]
    m12 = (invN*crosses12)[:,None,None] * (1-Psur12)[:,None,:]

    # Apply (M-I) rather than M, so that what comes out is the *change* to the
    # distribution and the caller adds it to its own F4. 
    # [point, nunubar, flavor, quadrature]
    dg = torch.stack([m01*(g[:,:,1,:]-g[:,:,0,:]) + m02*(g[:,:,2,:]-g[:,:,0,:]),
                      m01*(g[:,:,0,:]-g[:,:,1,:]) + m12*(g[:,:,2,:]-g[:,:,1,:]),
                      m02*(g[:,:,0,:]-g[:,:,2,:]) + m12*(g[:,:,1,:]-g[:,:,2,:])], dim=2)

    # integrate the change over the quadrature points, weighted by the weights, to get the change in the moments
    # the basis columns are the three direction cosines followed by 1, so a
    # single matmul produces all four components at once [quadrature, xyzt]
    ptsT  = pts.transpose(0,1)
    basis = torch.cat([ptsT, torch.ones_like(ptsT[:,0:1])], dim=1) * weights[:,None]
    dF4 = torch.matmul(dg, basis)

    # check that ELN-xln is still conserved
    #check_conservation(F4, F4+dF4)

    # negative or non-finite densities are not a usable prediction [point]
    unphysical = torch.logical_or(unphysical, ((F4[:,:,:,3]+dF4[:,:,:,3]) < 0).flatten(start_dim=1).any(dim=1))
    unphysical = torch.logical_or(unphysical, torch.logical_not(torch.isfinite(dF4)).flatten(start_dim=1).any(dim=1))

    # return nan rather than raising, so that a single bad point does not kill
    # the whole batch. Callers running inside a simulation handle the nans.
    dF4        = torch.where(unphysical[:,None,None,None], torch.full_like(dF4,        float("nan")), dF4)
    growthrate = torch.where(unphysical,                   torch.full_like(growthrate, float("nan")), growthrate)

    return dF4, growthrate



if __name__ == "__main__":
    # fluxes must be smaller than the density to stay subluminal, and the
    # caller normalizes the total density to 1
    F4 = torch.rand(10,2,3,4)
    F4[:,:,:,:3] = (F4[:,:,:,:3] - 0.5) * F4[:,:,:,3:]
    F4 = F4 / torch.sum(F4[:,:,:,3], dim=(1,2))[:,None,None,None]
    F4mix, I = mixBox3D_lebedev(F4, pts, weights)
    print(F4mix)
    print(I)