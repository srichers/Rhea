import torch

# create list of simulations known to be stable when the flux factor is zero
# input has dimensions  [nsims  , xyzt, nu/nubar, flavor]
# output has dimensions [nsims*2, xyzt, nu/nubar, flavor]
# array of fluxes that are stable to the FFI
def generate_stable_F4_zerofluxfac(n_trivial_stable, NF, device):
    F4i = torch.zeros((n_trivial_stable, 4, 2, NF), device=device)
    
    # keep all of the fluxes zero
    # densities randomly from 0 to 1
    F4i[:,3,:,:] = torch.rand(n_trivial_stable, 2, NF, device=device)

    # normalize
    ntot = torch.sum(F4i[:,3,:,:], axis=(1,2))
    F4i /= ntot[:,None,None,None]
    
    return F4i

def generate_stable_F4_oneflavor(n_trivial_stable,NF, device):
    # choose the flux to be in a random direction
    costheta = 2*(torch.rand(n_trivial_stable, device=device) - 0.5)
    phi = 2*torch.pi*torch.rand(n_trivial_stable, device=device)
    Frand = torch.zeros(n_trivial_stable, 3, device=device)
    Frand[:,0] = torch.sqrt(1-costheta**2) * torch.cos(phi)
    Frand[:,1] = torch.sqrt(1-costheta**2) * torch.sin(phi)
    Frand[:,2] = costheta

    # choose a random flux factor
    fluxfac = torch.rand(n_trivial_stable, device=device)

    # set Frand to be the flux factor times the density. Assume that the density is 1
    Frand = Frand * fluxfac[:,None]

    # set F4i with the new flux values
    F4i = torch.zeros((n_trivial_stable*2*NF, 4, 2, NF), device=device)
    for i in range(2):
        for j in range(NF):
            # set start and end indices for current flavor and nu/nubar
            start = i*n_trivial_stable + j*n_trivial_stable*2
            end   = start + n_trivial_stable
            F4i[start:end,0:3,i,j] = Frand
            F4i[start:end,  3,i,j] = 1
    
    return F4i

def generate_random_F4(n_generate, NF, device, zero_weight=1, max_fluxfac=1):
    assert(n_generate >= 0)
    F4i = torch.zeros((n_generate, 4, 2, NF), device=device)

    # choose a random number density
    Ndens = torch.rand(n_generate, 2, NF, device=device)
    Ndens[torch.where(Ndens==0)] = 1
    F4i[:,3,:,:] = Ndens

    # choose the flux to be in a random direction
    costheta = 2*(torch.rand(n_generate, 2, NF, device=device) - 0.5)
    phi = 2*torch.pi*torch.rand(n_generate, 2, NF, device=device)
    sintheta = torch.sqrt(1-costheta**2)
    F4i[:,0,:,:] = sintheta * torch.cos(phi)
    F4i[:,1,:,:] = sintheta * torch.sin(phi)
    F4i[:,2,:,:] = costheta
    F4i[:,3,:,:] = 1

    # choose a random flux factor
    fluxfac = torch.rand(n_generate, 2, NF, device=device)*max_fluxfac
    fluxfac = fluxfac**zero_weight

    # multiply the spatial flux by the flux factor times the density.
    F4i[:,0:3,:,:] = F4i[:,0:3,:,:] * fluxfac[:,None,:,:]
    
    # scale by the number density
    F4i *= Ndens[:,None,:,:]

    # normalize so the total number density is 1
    ntot = torch.sum(F4i[:,3,:,:], axis=(1,2))
    F4i = F4i / ntot[:,None,None,None]

    return F4i
