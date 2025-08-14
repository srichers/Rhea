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
import sys
sys.path.append("data")
import generate

def read_asymptotic_data(parms):
    #===============================================#
    # read in the database from the previous script #
    #===============================================#
    print()
    print("#############################")
    print("# PREPARING TEST/TRAIN DATA #")
    print("#############################")
    F4_initial_list = []
    F4_final_list = []
    growthrate_list = []
    for d in parms["database_list"]:
        print("Reading:",d)
        f_in = h5py.File(d,"r")
        F4_initial_list.append(np.array(f_in["F4_initial(1|ccm)"])) # [simulationIndex, xyzt, nu/nubar, flavor]
        F4_final_list.append(  np.array(f_in["F4_final(1|ccm)"  ]))
        growthrate_list.append(np.array(f_in["growthRate(1|s)"  ]))
        assert(parms["NF"] == int(np.array(f_in["nf"])) )
        f_in.close()
        print("    ",len(F4_initial_list[-1]),"points in",d)
    F4_initial_list = torch.tensor(np.concatenate(F4_initial_list), device=parms["device"]).float()
    F4_final_list   = torch.tensor(np.concatenate(F4_final_list  ), device=parms["device"]).float()
    growthrate_list = torch.tensor(np.concatenate(growthrate_list), device=parms["device"]).float()

    # fix slightly negative energy densities
    ntot = ml.ntotal(F4_initial_list)
    ndens = F4_initial_list[:,3,:,:]
    badlocs = torch.where(ndens < 0)
    assert(torch.all(ndens/ntot[:,None,None] > -1e10))
    for i in range(4):
        F4_initial_list[:,i,:,:][badlocs] = 0

    # normalize the data so the number densities add up to 1
    # This prevents some cases from outweighing others
    F4_initial_list = F4_initial_list / ntot[:,None,None,None]
    F4_final_list   = F4_final_list   / ntot[:,None,None,None]
    growthrate_list = growthrate_list / ntot[:]

    # make sure the data are good
    ml.check_conservation(F4_initial_list, F4_final_list)
    assert(torch.all(growthrate_list > 0))

    # convert growthrates into logarithms of growthrates
    logGrowthRate_list = torch.log(growthrate_list)

    # split into training and testing sets
    F4i_train, F4i_test, F4f_train, F4f_test, logGrowthRate_train, logGrowthRate_test = train_test_split(F4_initial_list, F4_final_list, logGrowthRate_list, test_size=parms["test_size"], random_state=parms["random_seed"])

    if parms["do_augment_permutation"]:
        F4i_train = ml.augment_permutation(F4i_train)
        F4f_train = ml.augment_permutation(F4f_train)
        F4i_test  = ml.augment_permutation(F4i_test )
        F4f_test  = ml.augment_permutation(F4f_test )
        growthrate_train = ml.augment_permutation(growthrate_train)
        growthrate_test = ml.augment_permutation(growthrate_test)

    # average heavies if necessary
    if parms["average_heavies_in_final_state"]:
        assert(parms["do_augment_permutation"]==False)
        assert(torch.allclose( torch.mean(F4i_train[:,:,:,1:], dim=3), F4i_train[:,:,:,1] ))
        assert(torch.allclose( torch.mean( F4i_test[:,:,:,1:], dim=3), F4i_test[:,:,:,1] ))
        F4f_train[:,:,:,1:] = torch.mean(F4f_train[:,:,:,1:], dim=3, keepdim=True)
        F4f_test[:,:,:,1:] =  torch.mean( F4f_test[:,:,:,1:], dim=3, keepdim=True)

    print("Train:",F4i_train.shape)
    print("Test:",F4i_test.shape)
    
    return F4i_train, F4i_test, F4f_train, F4f_test, logGrowthRate_train, logGrowthRate_test

#=================================================#
# read in the stable points from the NSM snapshot #
#=================================================#
def read_stable_data(parms):
    F4_collected = np.zeros((0, 4, 2, parms["NF"]))
    stable_collected = np.zeros((0))

    for filename in parms["stable_database_list"]:
        f_in = h5py.File(filename,"r")
        F4_collected = np.concatenate((F4_collected, f_in["F4_initial(1|ccm)"][...]), axis=0)
        stable_collected = np.concatenate((stable_collected, f_in["stable"][...]))
        f_in.close()

    # convert into a tensor
    F4_collected = torch.tensor(F4_collected, device=parms["device"]).float()
    stable_collected = torch.tensor(stable_collected, device=parms["device"]).float()

    # split into training and testing sets
    F4_train, F4_test, stable_train, stable_test = train_test_split(F4_collected, stable_collected, test_size=parms["test_size"], random_state=parms["random_seed"])

    # don't need the final values because they are the same as the initial
    if parms["do_augment_permutation"]:
        F4_train = ml.augment_permutation(F4_train)
        F4_test = ml.augment_permutation(F4_test)
        stable_train = ml.augment_permutation(stable_train)
        stable_test = ml.augment_permutation(stable_test)

    return F4_train, F4_test, stable_train, stable_test

if __name__ == "__main__":
    parms = {
        "NF" : 3,
        "database_list" : ["/mnt/scratch/srichers/software/Rhea/model_training/data/asymptotic_M1-NuLib-7ms.h5"],
        "stable_database_list" : ["/mnt/scratch/srichers/software/Rhea/model_training/data/stable_oneflavor.h5"],
        "test_size" : 0.1,
        "random_seed" : 42,
        "do_augment_permutation" : False,
        "average_heavies_in_final_state" : False,
        "device" : 'cpu'
    }
    
    print("reading asymptotic dataset")
    F4i_train, F4i_test, F4f_train, F4f_test, logGrowthRate_train, logGrowthRate_test = read_asymptotic_data(parms)
    print(F4i_train.shape, F4i_test.shape, F4f_train.shape, F4f_test.shape, logGrowthRate_train.shape, logGrowthRate_test.shape)

    print("reading stable dataset")
    F4_train, F4_test, stable_train, stable_test = read_stable_data(parms)
    print(F4_train.shape, F4_test.shape, stable_train.shape, stable_test.shape)

