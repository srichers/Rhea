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
    g = Z / torch.sinh(Z) * torch.exp(Z*mu)

    # for large Z use form of expression that avoids large exponentials
    g = torch.where(Z < 100, g, torch.exp(Z * (mu-1)) )

    # for small Z use form of expression that avoids division by small numbers
    g = torch.where(Z > 1e-3, g, mu*Z + 1.0)

    return g * (n/(4*torch.pi))


# assume that F4 has shape [npoints, nunubar, flavor, xyzt]
def mixBox3D_lebedev(F4, pts, weights):

    # Operate on a detached clone so Box3D computations don't modify
    # tensors required for gradient computation elsewhere in the graph.
    # This module is not intended to contribute gradients.
    F4 = F4.detach().clone()

    # get total number of neutrinos
    Ntot = torch.sum(F4[:,:,:,3:], dim=(1,2), keepdim=True)

    # make sure Ntot is still close to 1
    assert torch.allclose(Ntot, torch.ones_like(Ntot), atol=1e-5), "Total number of neutrinos is not close to 1"

    # average the heavy flavors
    F4[:,:,1:,:] = torch.mean(F4[:,:,1:,:], dim=2, keepdim=True)

    # get the Z value for each point, nunubar, and flavor
    # only use one heavy flavor since they are averaged
    n = F4[:,:,:2,3:]
    F = F4[:,:,:2,0:3]
    normF = torch.sqrt(torch.sum(F*F, dim=3, keepdim=True))
    Fhat = torch.nan_to_num(F/normF, nan=0)
    Z = get_Z(normF/n)

    # evaluate the distribution function at each quadrature point
    mu = torch.sum(pts[None,None,None,:,:] * Fhat[:,:,:,:,None], dim=3) # [point, nunubar, flavor, quadrature]
    g = distrib(n,Z,mu) # [point, nunubar, flavor, quadrature]

    # get eln-xln distribution [point, quadrature]
    gELN = (g[:,0,0,:] - g[:,1,0,:]) - (g[:,0,1,:] - g[:,1,1,:])

    # get the integrals of gELN over the quadrature points, weighted by the weights
    # [point]
    zero = torch.tensor(0., device=gELN.device)
    Iplus  =  torch.sum(gELN*weights * torch.heaviside( gELN,zero), dim=1)
    Iminus = -torch.sum(gELN*weights * torch.heaviside(-gELN,zero), dim=1)

    # set Psur depending on the relative size of Iplus and Iminus [point, quadrature]
    Psur = torch.zeros_like(gELN)
    locs1 = torch.where(Iplus <  Iminus)[0]
    locs2 = torch.where(Iplus >= Iminus)[0]
    IpIm = (Iplus[locs1]/Iminus[locs1]).unsqueeze(1)
    ImIp = (Iminus[locs2]/Iplus[locs2]).unsqueeze(1)
    Psur[locs1,:] = (1/3)*torch.heaviside( gELN[locs1],zero) + (1-2/3*IpIm)*torch.heaviside(-gELN[locs1],zero)
    Psur[locs2,:] = (1/3)*torch.heaviside(-gELN[locs2],zero) + (1-2/3*ImIp)*torch.heaviside( gELN[locs2],zero)

    assert torch.all((Psur >= 0) & (Psur <= 1)), "Psur is not between 0 and 1"

    # compute the mixed distribution function at each quadrature point, using Psur to mix the contributions from the different flavors
    g_t = torch.zeros_like(g)
    g_t[:,0,0,:] =  Psur*g[:,0,0,:] + (1-Psur)*g[:,0,1,:]
    g_t[:,1,0,:] =  Psur*g[:,1,0,:] + (1-Psur)*g[:,1,1,:]
    g_t[:,0,1,:] = ((1+Psur)*g[:,0,1,:] + (1-Psur)*g[:,0,0,:])/2
    g_t[:,1,1,:] = ((1+Psur)*g[:,1,1,:] + (1-Psur)*g[:,1,0,:])/2

    # integrate the mixed distribution function over the quadrature points, weighted by the weights, to get the mixed moments
    F4mix = torch.zeros_like(F4)
    F4mix[:,:,:2,3] = torch.sum(g_t * weights[None,None,None,:], dim=3)
    for i in range(3):
        F4mix[:,:,:2,i] = torch.sum(g_t * weights[None,None,None,:] * pts[None,None,None,i], dim=3)

    # Identical mu = tau = x flavors
    F4mix[:,:,2,:] = F4mix[:,:,1,:]

    # check that ELN-xln is still conserved
    #check_conservation(F4, F4mix)

    # check that all densities are positive
    assert torch.all(F4mix[:,:,:,3] >= 0), "Negative densities in mixed distribution function"

    
    return F4mix, torch.sqrt(Iplus*Iminus)



if __name__ == "__main__":
    F4 = torch.rand(10,2,3,4)
    F4mix, I = mixBox3D_lebedev(F4, pts, weights)
    print(F4mix)
    print(I)