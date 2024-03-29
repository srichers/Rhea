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

def read_test_train_data(NF, basedir, directory_list, test_size, device, do_augment_permutation):
    #===============================================#
    # read in the database from the previous script #
    #===============================================#
    print()
    print("#############################")
    print("# PREPARING TEST/TRAIN DATA #")
    print("#############################")
    F4_initial_list = []
    F4_final_list = []
    for d in directory_list:
        f_in = h5py.File(basedir+"/input_data/"+d+"/many_sims_database.h5","r")
        F4_initial_list.append(np.array(f_in["F4_initial(1|ccm)"])) # [simulationIndex, xyzt, nu/nubar, flavor]
        F4_final_list.append(  np.array(f_in["F4_final(1|ccm)"  ]))
        assert(NF == int(np.array(f_in["nf"])) )
        f_in.close()
        print(len(F4_initial_list[-1]),"points in",d)
    F4_initial_list = torch.tensor(np.concatenate(F4_initial_list), device=device).float()
    F4_final_list   = torch.tensor(np.concatenate(F4_final_list  ), device=device).float()

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
    F4i_train, F4i_test, F4f_train, F4f_test = train_test_split(F4_initial_list, F4_final_list, test_size=test_size, random_state=42)

    if do_augment_permutation:
        F4i_train = ml.augment_permutation(F4i_train)
        F4f_train = ml.augment_permutation(F4f_train)
        F4i_test  = ml.augment_permutation(F4i_test )
        F4f_test  = ml.augment_permutation(F4f_test )

    # pickle the train and test datasets
    with open("train_test_datasets.pkl","wb") as f:
        pickle.dump([F4i_train, F4i_test, F4f_train, F4f_test],f)

    return F4i_train, F4i_test, F4f_train, F4f_test

#=================================================#
# read in the stable points from the NSM snapshot #
#=================================================#
def read_NSM_stable_data(NF, basedir, device, do_augment_permutation):
    print()
    print("#############################")
    print("# READING NSM STABLE POINTS #")
    print("#############################")
    # note that x represents the SUM of mu, tau, anti-mu, anti-tau and must be divided by 4 to get the individual flavors
    # take only the y-z slice to limit the size of the data.
    f_in = h5py.File(basedir+"/input_data/model_rl0_orthonormal.h5","r")
    discriminant = np.array(f_in["crossing_discriminant"])[100,:,:]
    # n has shape [Nx,Ny,Nz]]
    ne = np.array(f_in["n_e(1|ccm)"])[0,:,:]
    na = np.array(f_in["n_a(1|ccm)"])[0,:,:]
    nx = np.array(f_in["n_x(1|ccm)"])[0,:,:]
    # f has shape [3, Nx,Ny,Nz]
    fe = np.array(f_in["fn_e(1|ccm)"])[:,0,:,:]
    fa = np.array(f_in["fn_a(1|ccm)"])[:,0,:,:]
    fx = np.array(f_in["fn_x(1|ccm)"])[:,0,:,:]
    f_in.close()
    stable_locs = np.where(discriminant<=0)
    nlocs = len(stable_locs[0])
    print(nlocs,"points")
    F4_NSM_stable = np.zeros((nlocs,4,2,NF))
    F4_NSM_stable[:,3,0,0  ] = ne[stable_locs]
    F4_NSM_stable[:,3,1,0  ] = na[stable_locs]
    F4_NSM_stable[:,3,:,1:3] = nx[stable_locs][:,None,None] / 4.
    for i in range(3):
        F4_NSM_stable[:,i,0,0  ] = fe[i][stable_locs]
        F4_NSM_stable[:,i,1,0  ] = fa[i][stable_locs]
        F4_NSM_stable[:,i,:,1:3] = fx[i][stable_locs][:,None,None] / 4.
    # convert into a tensor
    F4_NSM_stable = torch.tensor(F4_NSM_stable).float()
    # normalize the data so the number densities add up to 1
    ntot = ml.ntotal(F4_NSM_stable)
    F4_NSM_stable = F4_NSM_stable / ntot[:,None,None,None]
    # split into training and testing sets
    # don't need the final values because they are the same as the initial
    if do_augment_permutation:
        F4_NSM_stable = ml.augment_permutation(F4_NSM_stable)

    # move the array to the device
    F4_NSM_stable = torch.Tensor(F4_NSM_stable).to(device)

    return F4_NSM_stable
