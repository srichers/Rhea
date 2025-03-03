import torch
import numpy as np
import scipy
import sys

sys.path.append('../model_training')
#Import functions defined in Rhea/model_training
from ml_tools import *
from ml_maxentropy import *
from ml_generate import *

#########################################################################################################
###########################################  X_from_F4    ###############################################
#########################################################################################################
#Adapted here from Rhea/model_training/ml_neuralnet.py (class NeuralNetwork)
#Given the a list of four fluxes, calculate the inputs to the neural network out of dot products of four fluxes with each other and the four velocity. The four-velocity is assumed to be timelike in an orthonormal tetrad.
#Args: F4 (torch.Tensor): Four-flux tensor. Indexed as [sim, xyzt, nu/nubar, flavor]
#Returns: torch.Tensor: Neural network input tensor. Indexed as [sim, iX]
def X_from_F4(parms, F4):
    index = 0
    nsims = F4.shape[0]
    NX = parms["NF"] * (1 + 2*parms["NF"])
    if parms["do_fdotu"]:
            NX += 2*parms["NF"]
    print("NX = ", NX)
    X = torch.zeros((nsims, NX), device=F4.device)
    F4_flat = F4.reshape((nsims, 4, 2*parms["NF"])) # [simulationIndex, xyzt, species]

    # calculate the total number density based on the t component of the four-vector
    # [sim]
    N = torch.sum(F4_flat[:,3,:], dim=1)

    # normalize F4 by the total number density
    # [sim, xyzt, 2*NF]
    F4_flat = F4_flat / N[:,None,None]

    # add the dot products of each species with each other species
    for a in range(2*parms["NF"]):
        for b in range(a,2*parms["NF"]):
            F1 = F4_flat[:,:,a]
            F2 = F4_flat[:,:,b]

            X[:,index] = dot4(F1,F2)
            index += 1

    # add the u dot products
    if parms["do_fdotu"]:
        u = torch.zeros((4), device=F4.device)
        u[3] = 1
        for a in range(2*parms["NF"]):
            X[:,index] = dot4(F4_flat[:,:,a], u[None,:])
            index += 1
    
    assert(index==NX)
    return X


################################################
############# Create a list of options #########
################################################
parms = {}

#TODO: Modify this to generate for data
#The number of points to generate
parms["n_generate"] = 100 #200000

#The number of flavors: should be 3
parms["NF"]= 3

#Use a GPU if available #

#parms["device"] = "cuda" if torch.cuda.is_available() else "cpu"
#print("Using",parms["device"],"device")
#print(torch.cuda.get_device_name(0))
parms["device"] = "cpu"

parms["average_heavies_in_final_state"] = True
parms["do_fdotu"]= True
parms["generate_max_fluxfac"] = 0.95
parms["ME_stability_zero_weight"] = 10
parms["ME_stability_n_equatorial"] = 32


################################################
############# Run the test case ############### 
################################################
#Test cases for has_crossing function
# run test case
F4i = torch.zeros((1,4,2,3))
F4i[0,3,0,0] = 1
F4i[0,3,1,0] = 2
#print("should be false:", has_crossing(F4i.detach().numpy(), 3, 64))
print("should be false:", int(has_crossing(F4i.detach().numpy(), parms)))
assert(int(has_crossing(F4i.detach().numpy(), parms)) == 0)

F4i = torch.zeros((1,4,2,3))
F4i[0,3,0,0] = 1
F4i[0,3,1,0] = 1
F4i[0,0,0,0] =  0.5
F4i[0,0,1,0] = -0.5
#print("should be true:", has_crossing(F4i.detach().numpy(), 3, 64))
print("should be true:", int(has_crossing(F4i.detach().numpy(), parms)))
assert(int(has_crossing(F4i.detach().numpy(), parms)) == 1)



################################################
############# Generate the data ################
################################################
print("Generating data for stable_zerofluxfac..")
F4_zerofluxfac = generate_stable_F4_zerofluxfac(parms)
#This is the output label for stable vs unstable
#Is the point unstable?
#Yes => unstable = 1
#No => unstable = 0
unstable_zerofluxfac = torch.zeros((parms["n_generate"]), device=parms["device"])
unstable_zerofluxfac[:] = 0 #Since all the generated points are stable
#print(F4_zerofluxfac.shape)
#print(F4_zerofluxfac[0,:,:,:])
#print(unstable_zerofluxfac.shape)
#print(unstable_zerofluxfac[0])

print("Generating data for stable_oneflavor..")
F4_oneflavor = generate_stable_F4_oneflavor(parms)
#This is the output label for stable vs unstable
#Is the point unstable?
#Yes => unstable = 1
#No => unstable = 0
unstable_oneflavor = torch.zeros((parms["n_generate"]*2*parms["NF"]), device=parms["device"])
unstable_oneflavor[:] = 0 #Since all the generated points are stable
#print(F4_oneflavor.shape)
#print(F4_oneflavor[0,:,:,:])
#print(unstable_oneflavor.shape)
#print(unstable_oneflavor[0])

print("Generating random data..")
F4_random = generate_random_F4(parms)
#print(F4_random.shape)
#print(F4_random[0,:,:,:])
#If has_crossing returns true, unstable = 1 (i.e this point is unstable)
#If has_crossing returns false, unstable = 0 (i.e this point is stable)
unstable_random = has_crossing(F4_random.detach().numpy(), parms)
#print("unstable_random.shape", unstable_random.shape)
#print("unstable_random", unstable_random)


################################################
############# Convert from F4 to X #############
################################################
X_zerofluxfac = X_from_F4(parms, F4_zerofluxfac)
print("X_zerofluxfac.shape", X_zerofluxfac.shape)
print("X_zerofluxfac[0,:]", X_zerofluxfac[0,:])
print("unstable_zerofluxfac.shape", unstable_zerofluxfac.shape)
print("unstable_zerofluxfac[0]", unstable_zerofluxfac[0])

X_oneflavor = X_from_F4(parms, F4_oneflavor)
print("X_oneflavor.shape", X_oneflavor.shape)
print("X_oneflavor[0,:]", X_oneflavor[0,:])
print("unstable_oneflavor.shape", unstable_oneflavor.shape)
print("unstable_oneflavor[0]", unstable_oneflavor[0])

X_random = X_from_F4(parms, F4_random)
print("X_random.shape", X_random.shape)
print("X_random[0,:]", X_random[0,:])
print("unstable_random.shape", unstable_random.shape)
print("unstable_random[0]", unstable_random[0])


################################################
############# Save the data ####################
################################################
np.savez('train_data_stable_zerofluxfac.npz', X_zerofluxfac=X_zerofluxfac, unstable_zerofluxfac=unstable_zerofluxfac)
np.savez('train_data_stable_oneflavor.npz', X_oneflavor=X_oneflavor, unstable_oneflavor=unstable_oneflavor)
np.savez('train_data_random.npz', X_random=X_random, unstable_random=unstable_random)