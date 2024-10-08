'''
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This is the file contains functions that read data from the training datasets.
'''

import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import ml_tools as ml
from ml_loss import *
from ml_neuralnet import *
from ml_optimizer import *
from ml_plot import *
from ml_trainmodel import *
import pickle

def read_test_train_data(parms):
    #===============================================#
    # read in the database from the previous script #
    #===============================================#
    print()
    print("#############################")
    print("# PREPARING TEST/TRAIN DATA #")
    print("#############################")
    F4_initial_list = []
    F4_final_list = []
    for d in parms["database_list"]:
        print("Reading:",d)
        f_in = h5py.File(d,"r")
        F4_initial_list.append(np.array(f_in["F4_initial(1|ccm)"])) # [simulationIndex, xyzt, nu/nubar, flavor]
        F4_final_list.append(  np.array(f_in["F4_final(1|ccm)"  ]))
        assert(parms["NF"] == int(np.array(f_in["nf"])) )
        f_in.close()
        print("    ",len(F4_initial_list[-1]),"points in",d)
    F4_initial_list = torch.tensor(np.concatenate(F4_initial_list), device=parms["device"]).float()
    F4_final_list   = torch.tensor(np.concatenate(F4_final_list  ), device=parms["device"]).float()

    # fix slightly negative energy densities
    ntot = ml.ntotal(F4_initial_list)
    ndens = F4_initial_list[:,3,:,:]
    badlocs = torch.where(ndens < 0)
    assert(torch.all(ndens/ntot[:,None,None] > -1e10))
    for i in range(4):
        F4_initial_list[:,i,:,:][badlocs] = 0

    # normalize the data so the number densities add up to 1
    F4_initial_list = F4_initial_list / ntot[:,None,None,None]
    F4_final_list   = F4_final_list   / ntot[:,None,None,None]

    # make sure the data are good
    check_conservation(F4_initial_list, F4_final_list)

    # split into training and testing sets
    F4i_train, F4i_test, F4f_train, F4f_test = train_test_split(F4_initial_list, F4_final_list, test_size=parms["test_size"], random_state=42)

    if parms["do_augment_permutation"]:
        F4i_train = ml.augment_permutation(F4i_train)
        F4f_train = ml.augment_permutation(F4f_train)
        F4i_test  = ml.augment_permutation(F4i_test )
        F4f_test  = ml.augment_permutation(F4f_test )

    # average heavies if necessary
    if parms["average_heavies_in_final_state"]:
        assert(parms["do_augment_permutation"]==False)
        assert(torch.allclose( torch.mean(F4i_train[:,:,:,1:], dim=3), F4i_train[:,:,:,1] ))
        assert(torch.allclose( torch.mean( F4i_test[:,:,:,1:], dim=3), F4i_test[:,:,:,1] ))
        F4f_train[:,:,:,1:] = torch.mean(F4f_train[:,:,:,1:], dim=3, keepdim=True)
        F4f_test[:,:,:,1:] =  torch.mean( F4f_test[:,:,:,1:], dim=3, keepdim=True)
    
    # pickle the train and test datasets
    with open("train_test_datasets.pkl","wb") as f:
        pickle.dump([F4i_train, F4i_test, F4f_train, F4f_test],f)

    return F4i_train, F4i_test, F4f_train, F4f_test

#=================================================#
# read in the stable points from the NSM snapshot #
#=================================================#
def read_NSM_stable_data(parms):
    print()
    print("#############################")
    print("# READING NSM STABLE POINTS #")
    print("#############################")
    F4_NSM_stable_collected = np.zeros((0, 4, 2, parms["NF"]))
    # note that x represents the SUM of mu, tau, anti-mu, anti-tau and must be divided by 4 to get the individual flavors
    # take only the y-z slice to limit the size of the data.
    xslice = 100
    for filename in parms["NSM_stable_filename"]:
        f_in = h5py.File(filename,"r")
        discriminant = np.array(f_in["crossing_discriminant"])[xslice,:,:]
        # n has shape [Nx,Ny,Nz]]
        ne = np.array(f_in["n_e(1|ccm)"])[xslice,:,:]
        na = np.array(f_in["n_a(1|ccm)"])[xslice,:,:]
        nx = np.array(f_in["n_x(1|ccm)"])[xslice,:,:]
        # f has shape [3, Nx,Ny,Nz]
        fe = np.array(f_in["fn_e(1|ccm)"])[:,xslice,:,:]
        fa = np.array(f_in["fn_a(1|ccm)"])[:,xslice,:,:]
        fx = np.array(f_in["fn_x(1|ccm)"])[:,xslice,:,:]
        f_in.close()
        
        stable_locs = np.where(discriminant<=0)
        nlocs = len(stable_locs[0])
        print(nlocs,"points")
        F4_NSM_stable = np.zeros((nlocs,4,2,parms["NF"]))
        F4_NSM_stable[:,3,0,0  ] = ne[stable_locs]
        F4_NSM_stable[:,3,1,0  ] = na[stable_locs]
        F4_NSM_stable[:,3,:,1:3] = nx[stable_locs][:,None,None] / 4.
        for i in range(3):
            F4_NSM_stable[:,i,0,0  ] = fe[i][stable_locs]
            F4_NSM_stable[:,i,1,0  ] = fa[i][stable_locs]
            F4_NSM_stable[:,i,:,1:3] = fx[i][stable_locs][:,None,None] / 4.
        F4_NSM_stable_collected = np.concatenate((F4_NSM_stable_collected, F4_NSM_stable), axis=0)
        
    # convert into a tensor
    F4_NSM_stable = torch.tensor(F4_NSM_stable).float()
    # normalize the data so the number densities add up to 1
    ntot = ml.ntotal(F4_NSM_stable)
    F4_NSM_stable = F4_NSM_stable / ntot[:,None,None,None]
    
    # don't need the final values because they are the same as the initial
    if parms["do_augment_permutation"]:
        F4_NSM_stable = ml.augment_permutation(F4_NSM_stable)
            
    # move the array to the device
    F4_NSM_stable = torch.Tensor(F4_NSM_stable).to(parms["device"])

    # average heavies if necessary
    if parms["average_heavies_in_final_state"]:
        assert(parms["do_augment_permutation"]==False)
        assert(torch.allclose( torch.mean(F4_NSM_stable[:,:,:,1:], dim=3), F4_NSM_stable[:,:,:,1] ))
    
    return F4_NSM_stable
