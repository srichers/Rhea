'''
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This is the file contains functions that read data from the training datasets.
'''

import h5py
import numpy as np
import torch
import ml_tools as ml
import sys
import ml_constants as constants
from torch.utils.data import TensorDataset
sys.path.append("data")

# Read a list of asymptotic databases into a list of TensorDatasets. The train,
# validation, and test data are separate databases created by data/split_database.py,
# which owns the flux factor cap and the amount of data in each file. This function
# performs no splitting or subsampling of its own - it loads exactly what it is given.
def read_asymptotic_database_list(parms, label):
    dataset_list = []
    for d in parms[label+"_database_list"]:
        print()
        print("# ",label,d)
        # read from file
        with h5py.File(d,"r") as f_in:
            # File contains [simulationIndex, xyzt, nu/nubar, flavor]
            # We want [simulationIndex, nu/nubar, flavor, xyzt]
            F4_initial = torch.Tensor(f_in["F4_initial(1|ccm)"][...]).permute(0,2,3,1)
            F4_final   = torch.Tensor(f_in["F4_final(1|ccm)"  ][...]).permute(0,2,3,1)
            growthrate = torch.Tensor(f_in["growthRate(1|s)"  ][...]) / constants.ndens_to_invsec
            assert(parms["NF"] == int(np.array(f_in["nf"])) )
            assert(torch.all(torch.isfinite(F4_initial)))
            assert(torch.all(torch.isfinite(F4_final)))
            assert(torch.all(torch.isfinite(growthrate)))
        print("#   ",len(F4_initial),"points in",d)

        # compute stats (all physical units)
        ntot_initial = ml.ntotal(F4_initial)
        ntot_final = ml.ntotal(F4_final)
        print("#    ntot_initial min/max:", ntot_initial.min().item(), ntot_initial.max().item())
        print("#    ntot_final   min/max:", ntot_final.min().item(), ntot_final.max().item())
        print("#    growthrate   min/max:", growthrate.min().item(), growthrate.max().item())

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

        dataset_list.append( TensorDataset(F4_initial, F4_final, growthrate) )

    return dataset_list

def read_asymptotic_data(parms):
    #===============================================#
    # read in the database from the previous script #
    #===============================================#
    print("# PREPARING TEST/TRAIN DATA #")

    # the training loop indexes the first test database and reports a validation
    # loss every epoch, so none of the three lists may be empty
    assert(len(parms["train_database_list"])      > 0)
    assert(len(parms["validation_database_list"]) > 0)
    assert(len(parms["test_database_list"])       > 0)

    dataset_train_list      = read_asymptotic_database_list(parms, "train")
    dataset_validation_list = read_asymptotic_database_list(parms, "validation")
    dataset_test_list       = read_asymptotic_database_list(parms, "test")

    print()
    print("# Asymptotic Train:",[len(d) for d in dataset_train_list])
    print("# Asymptotic Validation:",[len(d) for d in dataset_validation_list])
    print("# Asymptotic Test:",[len(d) for d in dataset_test_list])

    return dataset_train_list, dataset_validation_list, dataset_test_list

if __name__ == "__main__":
    parms = {
        "NF" : 3,
        "train_database_list"      : ["data/dummy_asymptotic_chunk3-0_thin1_maxfluxfac0.9.h5"],
        "validation_database_list" : ["data/dummy_asymptotic_chunk3-1_thin1_maxfluxfac0.9.h5"],
        "test_database_list"       : ["data/dummy_asymptotic_chunk3-2_thin1_maxfluxfac0.9.h5"],
        "random_seed" : 42,
        "average_heavies_in_final_state" : False,
        "device" : 'cpu'
    }

    print("#  reading asymptotic dataset")
    train, validation, test = read_asymptotic_data(parms)
    print("# ",[len(d) for d in train], [len(d) for d in validation], [len(d) for d in test])
