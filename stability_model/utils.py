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
def X_from_F4(NF: int, do_fdotu: bool, F4: torch.Tensor):
    index = 0
    nsims = F4.shape[0]
    NX = NF * (1 + 2*NF)
    if do_fdotu:
            NX += 2*NF
    print("NX = ", NX)
    X = torch.zeros((nsims, NX), device=F4.device)
    F4_flat = F4.reshape((nsims, 4, 2*NF)) # [simulationIndex, xyzt, species]

    # calculate the total number density based on the t component of the four-vector
    # [sim]
    N = torch.sum(F4_flat[:,3,:], dim=1)

    # normalize F4 by the total number density
    # [sim, xyzt, 2*NF]
    F4_flat = F4_flat / N[:,None,None]

    # add the dot products of each species with each other species
    for a in range(2*NF):
        for b in range(a,2*NF):
            F1 = F4_flat[:,:,a]
            F2 = F4_flat[:,:,b]

            X[:,index] = dot4(F1,F2)
            index += 1

    # add the u dot products
    if do_fdotu:
        u = torch.zeros((4), device=F4.device)
        u[3] = 1
        for a in range(2*NF):
            X[:,index] = dot4(F4_flat[:,:,a], u[None,:])
            index += 1
    
    assert(index==NX)
    return X

