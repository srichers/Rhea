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
    ntot = torch.sum(F4i[:,3,:,:])
    F4i /= ntot
    
    return F4i

def generate_stable_F4_oneflavor(n_trivial_stable,NF, device):
    F4i = torch.zeros((n_trivial_stable, 4, 2, NF), device=device)
    
    # keep all but one flavor zero
    F4i[:,:,0, 0] = torch.rand(n_trivial_stable, 4, device=device) - 0.5

    # Ensure that the number densities are positive
    F4i[:,3,:,:] = torch.abs(F4i[:,3,:,:])

    # Ensure that the flux factors are less than 1
    fluxfac = torch.sqrt(torch.sum(F4i[:,0:3,:,:]**2, axis=1)) / F4i[:,3,:,:]
    fluxfac[torch.where(fluxfac!=fluxfac)] = 0
    F4i[:,0:3,:,:] = F4i[:,0:3,:,:] / torch.maximum(fluxfac[:,None,:,:], torch.ones_like(fluxfac[:,None,:,:], device=device))

    # normalize
    ntot = torch.sum(F4i[:,3,:,:])
    F4i /= ntot
    
    return F4i

def generate_random_F4(n_unphysical_check, NF, device):
    # generate initial flux factor list
    # initialize with random numbers between -0.5 and 0.5
    F4i = torch.rand(n_unphysical_check, 4, 2, NF, device=device) - 0.5
    # Ensure that the number densities are positive
    F4i[:,3,:,:] = torch.abs(F4i[:,3,:,:])
    # Ensure that the flux factors are less than 1
    fluxfac = torch.sqrt(torch.sum(F4i[:,0:3,:,:]**2, axis=1)) / F4i[:,3,:,:]
    F4i[:,0:3,:,:] = F4i[:,0:3,:,:] / torch.maximum(fluxfac[:,None,:,:], torch.ones_like(fluxfac[:,None,:,:], device=device))
    return F4i
