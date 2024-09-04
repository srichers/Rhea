'''
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This is the file contains functions that describe maximum entropy distributions, and interpolate those functions onto a sphere.
'''

import torch
import numpy as np
import scipy

###################
# MINERBO CLOSURE #
###################
# compute inverse Langevin function for closure
# Levermore 1984 Equation 20
def function(Z,fluxfac):
    return (1./np.tanh(Z) - 1./Z) - fluxfac
def dfunctiondZ(Z,fluxfac):
    return 1./Z**2 - 1./np.sinh(Z)**2
def get_Z(fluxfac):
    badlocs = np.where(np.abs(fluxfac)<1e-4)
    nsims = len(fluxfac)
    initial_guess = 1
    Z = np.array([scipy.optimize.fsolve(function, 
                                        initial_guess,
                                        fprime=dfunctiondZ,
                                        args=(fluxfac[i]))[0]\
                   for i in range(nsims)])
    residual = np.max(np.abs(function(Z,fluxfac)) )
    # when near isotropic, use taylor expansion
    # f \approx Z/3 - Z^3/45 + O(Z^5)
    Z[badlocs] = 3.*fluxfac[badlocs]
    return Z, residual

# compute the value of the distribution at a given costheta
# n is the total number density
# all inputs have dimensions [sim, mu]
def distribution(n, Z, costheta):
    assert(np.all(Z >= 0))
    badlocs = np.where(Z<1e-4)
    prefactor = Z/np.sinh(Z)
    prefactor[badlocs] = 1. - Z[badlocs]**2/6 # taylor expansion
    return n/(2.*np.pi) * prefactor * np.exp(Z*costheta)

# generate an array of theta,phi pairs that uniformily cover the surface of a sphere
# based on DOI: 10.1080/10586458.2003.10504492 section 3.3 but specifying n_j=0 instead of n
# output has dimensions [3,ndirections]
def uniform_sphere(nphi_at_equator):
    assert(nphi_at_equator > 0)

    dtheta = np.pi * np.sqrt(3) / nphi_at_equator

    xyz = []
    theta = 0
    phi0 = 0
    while(theta < np.pi/2):
        nphi = nphi_at_equator if theta==0 else int(round(nphi_at_equator * np.cos(theta)))
        dphi = 2*np.pi/nphi
        if(nphi==1): theta = np.pi/2
        
        for iphi in range(nphi):
            phi = phi0 = iphi*dphi
            x = np.cos(theta) * np.cos(phi)
            y = np.cos(theta) * np.sin(phi)
            z = np.sin(theta)
            xyz.append(np.array([x,y,z]))
            if(theta>0): xyz.append(np.array([-x,-y,-z]))
        theta += dtheta
        phi0 += 0.5 * dphi # offset by half step so adjacent latitudes are not always aligned in longitude

    return np.array(xyz).transpose()



# check whether a collection of number fluxes is stable according to the maximum entropy condition
# generate many directions along which to test whether the distributions cross
# F4i has dimensions [sim, xyzt, nu/nubar, flavor]
def has_crossing(F4i, parms):
    assert(parms["NF"]==3)
    nsims = F4i.shape[0]

    # evaluate the flux factor for each species
    Fmag = np.sqrt(np.sum(F4i[:,0:3,:,:]**2, axis=1))
    Fhat = F4i[:,0:3,:,:] / Fmag[:,None,:,:] # [sim, xyz, nu/nubar, flavor]
    fluxfac = Fmag / F4i[:,3,:,:]

    # avoid nans by setting fluxfac to zero when F4i is zero
    badlocs = np.where(Fmag/np.sum(F4i[:,3,:,:],axis=(1,2))[:,None,None] < 1e-6)
    if len(badlocs)>0:
        Fhat[:,0][badlocs] = 0
        Fhat[:,1][badlocs] = 0
        Fhat[:,2][badlocs] = 1
        fluxfac[badlocs] = 0
    assert(np.all(Fhat==Fhat))

    # avoid nans by setting flux factor of 1 within machine precision to 1
    assert(np.all(fluxfac<=1+1e-6))
    fluxfac = np.minimum(fluxfac, 1)
    assert(np.all(fluxfac>=0))

    # get Z for each species
    Z = np.zeros((nsims,2,parms["NF"]))
    for i in range(2):
        for j in range(parms["NF"]):
            Z[:,i,j], residual = get_Z(fluxfac[:,i,j])

    # generate a bunch of directions along which to test whether the distributions cross
    # result has dimensions [3,ndirections]
    xyz = uniform_sphere(parms["ME_stability_n_equatorial"])
    ndirections = xyz.shape[1]

    # Evaluate costheta relative to Fhat for each species
    costheta = np.zeros((nsims,2,parms["NF"],ndirections))
    for i in range(2):
        for j in range(parms["NF"]):
            costheta[:,i,j,:] = np.sum(xyz[None,:,:] * Fhat[:,:,i,j,None], axis=1)
    costheta[badlocs] = 0
    assert(np.all(np.abs(costheta)<1+1e-6))
    costheta[np.where(costheta>1)] =  1
    costheta[np.where(costheta<-1)] = -1

    # Evaluate the distribution for each species
    f = np.zeros((nsims,2,parms["NF"],ndirections))
    for i in range(2):
        for j in range(parms["NF"]):
            f[:,i,j,:] = distribution(F4i[:,3,i,j,None], Z[:,i,j,None], costheta[:,i,j,:])

    # lepton number is difference between neutrinos and antineutrinos
    # [nsims, NF, ndirections]
    lepton_number = f[:,0,:,:] - f[:,1,:,:]

    # calculate the G quantities
    NG = (parms["NF"]**2-parms["NF"])//2
    G = np.zeros((nsims,NG,ndirections))
    iG = 0
    for i in range(parms["NF"]):
        for j in range(i+1,parms["NF"]):
            G[:,iG,:] = lepton_number[:,i,:] - lepton_number[:,j,:]
            iG += 1
    assert(iG==NG)

    # check whether each G crosses zero
    crosses_zero = np.zeros((nsims,NG))
    for i in range(NG):
        minval = np.min(G[:,i,:], axis=1)
        maxval = np.max(G[:,i,:], axis=1)
        crosses_zero[:,i] = minval*maxval<0
    crosses_zero = np.any(crosses_zero, axis=1).astype(float) # [nsims]

    # add extra unit dimension to compatibility with loss functions
    return crosses_zero[:,None]


if __name__ == "__main__":
    # run test case
    F4i = torch.zeros((1,4,2,3))
    F4i[0,3,0,0] = 1
    F4i[0,3,1,0] = 2
    print("should be false:", has_crossing(F4i.detach().numpy(), 3, 64))

    F4i = torch.zeros((1,4,2,3))
    F4i[0,3,0,0] = 1
    F4i[0,3,1,0] = 1
    F4i[0,0,0,0] =  0.5
    F4i[0,0,1,0] = -0.5
    print("should be true:", has_crossing(F4i.detach().numpy(), 3, 64))


