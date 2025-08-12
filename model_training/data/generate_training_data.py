import torch
import numpy as np
import scipy
import sys
sys.path.append("..")
from ml_tools import *
from ml_maxentropy import *
from ml_generate import *
from ml_read_data import *

nphi_at_equatorial = 64

def main():
    
    ################################################
    ############# Generate the data ################
    ################################################
    print("Generating data for stable_zerofluxfac..")
    F4_zerofluxfac = generate_stable_F4_zerofluxfac(parms)
    #This is the output label for stable vs unstable
    #Is the point unstable?
    #Yes => unstable = 1
    #No => unstable = 0
    unstable_zerofluxfac = torch.zeros((parms["n_generate"], 1), device=parms["device"])
    unstable_zerofluxfac[:, None] = 0 #Since all the generated points are stable
    #print(F4_zerofluxfac.shape)
    #print(F4_zerofluxfac[0,:,:,:])
    #print(unstable_zerofluxfac.shape)
    #print(unstable_zerofluxfac[0])

    print("Generating data for stable_oneflavor..")
    F4_oneflavor = generate_stable_F4_oneflavor(parms)
    unstable_oneflavor = torch.zeros((parms["n_generate"]*2*NF, 1), device=parms["device"])
    unstable_oneflavor[:, None] = 0 #Since all the generated points are stable
    
    print("Generating random data..")
    F4_random = generate_random_F4(parms)
    unstable_random = has_crossing(F4_random.detach().numpy(), parms)
    
    print("Generating NSM stable data..")
    F4_NSM_stable = read_NSM_stable_data(parms)
    unstable_NSM_stable = torch.zeros((F4_NSM_stable.shape[0], 1), device=parms["device"])
    unstable_NSM_stable[:, None] = 0 #Since all the generated points are stable

    ################################################
    ############# Convert from F4 to X #############
    ################################################
    X_zerofluxfac = model.X_from_F4(F4_zerofluxfac)
    print("X_zerofluxfac.shape", X_zerofluxfac.shape)
    print("X_zerofluxfac[0,:]", X_zerofluxfac[0,:])
    print("unstable_zerofluxfac.shape", unstable_zerofluxfac.shape)
    print("unstable_zerofluxfac[0]", unstable_zerofluxfac[0])

    X_oneflavor = model.X_from_F4(F4_oneflavor)
    print("X_oneflavor.shape", X_oneflavor.shape)
    print("X_oneflavor[0,:]", X_oneflavor[0,:])
    print("unstable_oneflavor.shape", unstable_oneflavor.shape)
    print("unstable_oneflavor[0]", unstable_oneflavor[0])

    X_random = model.X_from_F4(F4_random)
    print("X_random.shape", X_random.shape)
    print("X_random[0,:]", X_random[0,:])
    print("unstable_random.shape", unstable_random.shape)
    print("unstable_random[0]", unstable_random[0])

    X_NSM_stable = model.X_from_F4(F4_NSM_stable)
    print("X_NSM_stable.shape", X_NSM_stable.shape)
    print("X_NSM_stable[0,:]", X_NSM_stable[0,:])
    print("unstable_NSM_stable.shape", unstable_NSM_stable.shape)
    print("unstable_NSM_stable[0]", unstable_NSM_stable[0])

    ################################################
    ############# Save the data ####################
    ################################################
    np.savez('train_data_stable_zerofluxfac.npz', X_zerofluxfac=X_zerofluxfac, unstable_zerofluxfac=unstable_zerofluxfac)
    np.savez('train_data_stable_oneflavor.npz', X_oneflavor=X_oneflavor, unstable_oneflavor=unstable_oneflavor)
    np.savez('train_data_random.npz', X_random=X_random, unstable_random=unstable_random)
    np.savez('train_data_NSM_stable.npz', X_NSM_stable=X_NSM_stable, unstable_NSM_stable=unstable_NSM_stable)
    print("*******************TRAINING DATA GENERATED AND SAVED IN NPZ FILES*****************************\n")

if __name__ == "__main__":
    main()
