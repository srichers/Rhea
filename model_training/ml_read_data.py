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
from torch.utils.data import TensorDataset
sys.path.append("data")

def read_asymptotic_data(parms):
    #===============================================#
    # read in the database from the previous script #
    #===============================================#
    print("# PREPARING TEST/TRAIN DATA #")

    dataset_train_list = []
    dataset_test_list = []
    rng = torch.Generator().manual_seed(parms["random_seed"])
    for dind,d in enumerate(parms["database_list"]):
        print()
        print(dind,d)
        # read from file
        with h5py.File(d,"r") as f_in:
            # File contains [simulationIndex, xyzt, nu/nubar, flavor]
            # We want [simulationIndex, nu/nubar, flavor, xyzt]
            F4_initial = torch.Tensor(f_in["F4_initial(1|ccm)"][...]).permute(0,2,3,1) 
            F4_final   = torch.Tensor(f_in["F4_final(1|ccm)"  ][...]).permute(0,2,3,1)
            growthrate = torch.Tensor(f_in["growthRate(1|s)"  ][...])
            assert(parms["NF"] == int(np.array(f_in["nf"])) )
            assert(torch.all(torch.isfinite(F4_initial)))
            assert(torch.all(torch.isfinite(F4_final)))
            assert(torch.all(torch.isfinite(growthrate)))
        print("#   ",len(F4_initial),"points in",d)

        # downsample to less data
        if parms["samples_per_database"]>0:
            print("#    Downsampling to",parms["samples_per_database"],"samples")
            random_indices = torch.randperm(len(growthrate), generator=rng)[:parms["samples_per_database"]]
            F4_initial = F4_initial[random_indices,...]
            F4_final   = F4_final  [random_indices,...]
            growthrate = growthrate[random_indices,...]

        print("#   ",len(F4_initial),"points in resampled dataset")

        # compute stats (all physical units)
        ntot_initial = ml.ntotal(F4_initial)
        ntot_final = ml.ntotal(F4_final)
        print("#    ntot_initial min/max (physical):", ntot_initial.min().item(), ntot_initial.max().item())
        print("#    ntot_final min/max (physical):", ntot_final.min().item(), ntot_final.max().item())
        print("#    growthrate min/max (physical 1/s):", growthrate.min().item(), growthrate.max().item())

        assert torch.all(ntot_initial > 0)
        assert torch.all(ntot_final > 0)
        assert torch.all(torch.isfinite(growthrate))
        assert torch.all(growthrate > 0)

        # fix slightly negative energy densities
        ndens = F4_initial[:,:,:,3]
        badlocs = torch.where(ndens < 0)
        assert(torch.all(ndens > -1e10))
        for i in range(4):
            F4_initial[:,:,:,i][badlocs] = 0

        # make sure the data are good
        ml.check_conservation(F4_initial, F4_final)
        assert(torch.all(growthrate > 0))

        # average heavies if necessary
        if parms["average_heavies_in_final_state"]:
            assert(torch.allclose( torch.mean(F4_initial[:,:,1:,:], dim=2), F4_initial[:,:,1,:] ))
            F4_final[:,:,1:,:] = torch.mean(F4_final[:,:,1:,:], dim=2, keepdim=True)

        # add dataset to the lists
        if dind==0:
            dataset_test_list.append( TensorDataset(F4_initial, F4_final, growthrate) )
        else:
            dataset_train_list.append(TensorDataset(F4_initial, F4_final, growthrate) )

    print()
    print("# Asymptotic Train:",[len(d) for d in dataset_train_list])
    print("# Asymptotic Test:",[len(d) for d in dataset_test_list])
    
    return dataset_train_list, dataset_test_list

#=================================================#
# read in the stable points from the NSM snapshot #
#=================================================#
def read_stable_data(parms):

    dataset_train_list = []
    dataset_test_list = []
    
    rng = torch.Generator().manual_seed(parms["random_seed"])
    for i,filename in enumerate(parms["stable_database_list"]):
        print()
        print(i,filename)
        with h5py.File(filename,"r") as f_in:
            # File contains [simulationIndex, xyzt, nu/nubar, flavor]
            # We want [simulationIndex, nu/nubar, flavor, xyzt]
            F4 = torch.squeeze(torch.Tensor(f_in["F4_initial(1|ccm)"][...])).permute(0,2,3,1)
            stable = torch.squeeze(torch.Tensor(f_in["stable"][...]))
            assert(torch.all(torch.isfinite(F4)==torch.isfinite(F4)))
            assert(torch.all(stable==stable))

        # print number of points
        print("#   ",len(stable),"points in",filename)
        print("#   ",torch.sum(stable).item(),"points are stable.")

        # downsample to less data
        if parms["samples_per_database"]>0:
            print("#    Downsampling to",parms["samples_per_database"],"samples")
            random_indices = torch.randperm(len(stable), generator=rng)[:parms["samples_per_database"]]
            F4 = F4[random_indices,...]
            stable = stable[random_indices]
            print("#   ",torch.sum(stable).item(),"points are stable.")
        
        # basic sanity checks (physical units)
        ntot = ml.ntotal(F4)
        assert(torch.all(ntot > 0))
        assert(torch.all(torch.isfinite(ntot)))

        if i==0:
            dataset_test_list.append(TensorDataset(F4, stable))
        else:
            dataset_train_list.append(TensorDataset(F4, stable))

    print("# Stability Train:",[len(d) for d in dataset_train_list])
    print("# Stability Test:",[len(d) for d in dataset_test_list])

    return dataset_train_list, dataset_test_list

if __name__ == "__main__":
    parms = {
        "NF" : 3,
        "database_list" : ["/mnt/scratch/srichers/software/Rhea/model_training/data/asymptotic_M1-NuLib-7ms.h5"],
        "stable_database_list" : ["/mnt/scratch/srichers/software/Rhea/model_training/data/stable_oneflavor.h5"],
        "test_size" : 0.1,
        "random_seed" : 42,
        "average_heavies_in_final_state" : False,
        "device" : 'cpu'
    }
    
    print("#  reading asymptotic dataset")
    F4i_train, F4i_test, F4f_train, F4f_test, growthrate_train, growthrate_test = read_asymptotic_data(parms)
    print(F4i_train.shape, F4i_test.shape, F4f_train.shape, F4f_test.shape, growthrate_train.shape, growthrate_test.shape)

    print("#  reading stable dataset")
    F4_train, F4_test, stable_train, stable_test = read_stable_data(parms)
    print("# ",F4_train.shape, F4_test.shape, stable_train.shape, stable_test.shape)
