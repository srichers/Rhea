# credit to https://thinkingneuron.com/using-artificial-neural-networks-for-regression-in-python/
# Convert the flavor transformation data to one with reduced dimensionality to make it easier to train on
# Run from the directory containin the joint dataset
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from ml_loss import *
from ml_neuralnet import *
from ml_optimizer import *
from ml_plot import *
from ml_trainmodel import *
from ml_read_data import *
from ml_tools import *
import pickle
import glob
import os
from torchmetrics.classification import BinaryPrecisionRecallCurve
from torcheval.metrics import BinaryF1Score, BinaryNormalizedEntropy, BinaryConfusionMatrix

parms = {}
parms["NSM_stable_filename"] = "/mnt/scratch/NSM_ML/spec_data/M1-NuLib/M1VolumeData/model_rl0_orthonormal.h5"
parms["NSM_unstable_filename"] = "/mnt/scratch/NSM_ML/Emu_merger_grid2/many_sims_database.h5"

def get_dataset_size_list(search_string):
    filename_list = glob.glob(search_string)
    print(filename_list)
    dataset_size_list = []
    for filename in filename_list:
        dataset_size_list.append(int(filename.split("_")[-1].split(".")[0]))
    return sorted(dataset_size_list)
dataset_size_list = get_dataset_size_list("model_[0-9]*.pkl")

parms["n_generate"] = 1000
parms["test_size"] = 0.1
parms["generate_max_fluxfac"] = 0.95
parms["zero_weight"] = 10
parms["n_equatorial"] = 64
parms["average_heavies_in_final_state"] = True

# data augmentation options
parms["do_augment_permutation"]=False

# the number of flavors should be 3
parms["NF"] = 3

#========================#
# use a GPU if available #
#========================#
parms["device"] = "cuda" if torch.cuda.is_available() else "cpu"
print("Using",parms["device"],"device")

#===============#
# read the data #
#===============#
# unpickle the data
with open("train_test_datasets.pkl","rb") as f:
    F4i_train, F4i_test, F4f_train, F4f_test = pickle.load(f)
F4_NSM_stable = read_NSM_stable_data(parms)

# [simulationIndex, xyzt, nu/nubar, flavor]
if parms["average_heavies_in_final_state"]:
    assert(parms["do_augment_permutation"]==False)
    assert(torch.allclose( torch.mean(F4i_train[:,:,:,1:], dim=3), F4i_train[:,:,:,1] ))
    assert(torch.allclose( torch.mean( F4i_test[:,:,:,1:], dim=3), F4i_test[:,:,:,1] ))
    F4f_train[:,:,:,1:] = torch.mean(F4f_train[:,:,:,1:], dim=3)[:,:,:,None]
    F4f_test[:,:,:,1:] =  torch.mean( F4f_test[:,:,:,1:], dim=3)[:,:,:,None]    


# adjust entries of -1 to instead have the correct size of the dataset
for i in range(len(dataset_size_list)):
    if dataset_size_list[i] == -1:
        dataset_size_list[i] = F4i_train.shape[0]

#F4i_NSMunstable, _, F4f_NSMunstable, _ = read_test_train_data(NF, [NSM_unstable_filename], test_size=10, device=device, do_augment_permutation=False)
# verify that all of the F4_NSM_stable data is stable
#print("verifying stability")
#unstable = has_crossing(F4_NSM_stable.cpu().detach().numpy(), NF, n_equatorial).squeeze()
#assert(np.all(unstable==False))

#=======================#
# instantiate the model #
#=======================#
print()
print("#############################")
print("# SETTING UP NEURAL NETWORK #")
print("#############################")
def get_plotter_model_arrays(search_string, dataset_size_list):
    # set up an array of models, optimizers, and plotters for different dataset sizes
    model_array = []
    plotter_array = []
    for dataset_size in dataset_size_list:
        filename = search_string+str(dataset_size)+".pkl"
        print("Loading",filename)
        with open(filename, "rb") as f:
            model, optimizer, plotter = pickle.load(f)

        plotter_array.append(plotter)
        model_array.append(model)
    print(model_array[-1])
    return plotter_array, model_array

plotter_array, model_array_asymptotic = get_plotter_model_arrays("model_",dataset_size_list)

# use the largest dataset size for the rest of these metrics
p_asymptotic = plotter_array[-1]
model = model_array_asymptotic[-1]
 
# set model to evaluation mode
model.eval()

#===================================#
# Test one point from training data #
#===================================#
print()
print("##########################################")
print("# Testing a point from the TRAINING data #")
print("##########################################")
print()
print("N initial")
before = F4i_train[0:1,:,:,:]
print(before[0,3])

print()
print("N final (actual)")
print(F4f_train[0,3])

print()
print("N predicted")
after = model.predict_F4(before,"eval")
print(after[0,3])

print()
print("loss = ",comparison_loss_fn(after, F4f_train[0:1,:,:,:]).item())
print("error = ",torch.max(torch.abs(after - F4f_train[0:1,:,:,:])).item())
check_conservation(before,after)

#===================================#
# Test one point from test data #
#===================================#
print()
print("######################################")
print("# Testing a point from the TEST data #")
print("######################################")
print()
print("N initial")
before = F4i_test[0:1,:,:,:]
print(before[0,3])

print()
print("N final (actual)")
print(F4f_test[0,3])

print()
print("N predicted")
after = model.predict_F4(before,"eval")
print(after[0,3])

print()
print("loss = ",comparison_loss_fn(after, F4f_train[0:1,:,:,:]).item()) # 
print("error = ",torch.max(torch.abs(after - F4f_train[0:1,:,:,:])).item())

#=====================================#
# create test ("Fiducial" simulation) #
#=====================================#
print()
print("#############################")
print("# Testing the FIDUCIAL case #")
print("#############################")
F4_test = np.zeros((4,2,parms["NF"])) # [xyzt, nu/nubar, flavor]
F4_test[3, 0, 0] =  1
F4_test[3, 1, 0] =  1
F4_test[2, 0, 0] =  1/3
F4_test[2, 1, 0] = -1/3
before = torch.Tensor(F4_test[None,:,:,:]).to(parms["device"])
after = model.predict_F4(before,"eval")

print()
print("N initial")
print(before[0,3])

print()
print("N predicted")
after = model.predict_F4(before,"eval")
print(after[0,3])

print()
print("N re-predicted")
after = model.predict_F4(after,"eval")
print(after[0,3])

print()
print("2 Flavor")
before_2F = before[:,:,:,0:2]
X = model.X_from_F4(before)
y = model.forward(X)
y2F = model.convert_y_to_2flavor(y)
after_3F = model.F4_from_y(before   , y  )
after_2F = model.F4_from_y(before_2F, y2F)
print("3F")
print(after_3F[0,3])
print("2F")
print(after_2F[0,3])
check_conservation(before,after)

print()
print("#########################")
print("# Testing the NSM1 case #")
print("#########################")
F4_test = np.zeros((4,2,parms["NF"])) # [xyzt, nu/nubar, flavor]

F4_test[3, 0, 0] =  14.22e32
F4_test[0, 0, 0] =  0.0974 * F4_test[3, 0, 0]
F4_test[1, 0, 0] =  0.0421 * F4_test[3, 0, 0]
F4_test[2, 0, 0] =  -0.1343 * F4_test[3, 0, 0]

F4_test[3, 1, 0] =  19.15e32
F4_test[0, 1, 0] = 0.0723 * F4_test[3, 1, 0]
F4_test[1, 1, 0] = 0.0313 * F4_test[3, 1, 0]
F4_test[2, 1, 0] = -0.3446 * F4_test[3, 1, 0]

F4_test[3, :, 1:] = 19.65e32/4.
F4_test[0, :, 1:] = -0.0216 * F4_test[3, 0, 1]
F4_test[1, :, 1:] = 0.0743 * F4_test[3, 0, 1]
F4_test[2, :, 1:] = -0.5354 * F4_test[3, 0, 1]
before = torch.Tensor(F4_test[None,:,:,:]).to(parms["device"])
after = model.predict_F4(before,"eval")

print()
print("N initial")
print(before[0,3])

print()
print("N predicted")
after = model.predict_F4(before,"eval")
print(after[0,3])

print()
print("N re-predicted")
after = model.predict_F4(after,"eval")
print(after[0,3])

print()
print("Emu expected:")
print("[[[ 7.64375814e-02  3.97575192e-02  1.23841745e-02]")
print("[ 5.72762474e-02  3.22184364e-02  7.65071452e-03]]")
print("")
print(" [[ 4.36780086e-02  6.05246388e-02  6.77405394e-02]")
print("  [ 3.18509891e-02  5.72619715e-02  6.56920392e-02]]")
print("")
print(" [[-3.33201733e-01 -2.13291599e-01 -3.68908362e-01]")
print("  [-4.83817623e-01 -2.85258423e-01 -4.03983122e-01]]")
print("")
print(" [[ 9.95460068e+32  7.85471511e+32  6.23293031e+32]")
print("  [ 1.48812960e+33  7.85471511e+32  6.23293016e+32]]]")

#=====================================#
# create test ("Zero FF" simulation) #
#=====================================#
print()
print("####################################")
print("# Testing the Zero Flux Factor case#")
print("####################################")
F4_test = np.zeros((4,2,parms["NF"])) # [xyzt, nu/nubar, flavor]
F4_test[3, 0, 0] =  1
F4_test[3, 1, 0] =  .5
F4_test[2, 0, 0] =  0
F4_test[2, 1, 0] =  0
before = torch.Tensor(F4_test[None,:,:,:]).to(parms["device"])
after = model.predict_F4(before,"eval")

print()
print("N initial")
print(before[0,3])

print("N predicted")
after = model.predict_F4(before,"eval")
print(after[0,3])


print("N re-predicted")
after = model.predict_F4(after,"eval")
print(after[0,3])

print()
check_conservation(before,after)

#==================#
# plot the results #
#==================#
print()
print("########################")
print("# Plotting the results #")
print("########################")
npoints = 11
nreps = 20
p_asymptotic.init_plot_options()
plot_nue_nuebar(model, npoints, nreps)
p_asymptotic.plot_error("train_test_error_asymptotic.pdf", ymin=1e-5)

# plot the loss as a function of dataset size using the array of plotters
def plot_dataset_size(plotter_array, dataset_size_list, quantity, outfilename):
    train_loss = np.array([p.data[quantity].train_loss[-1] for p in plotter_array])
    test_loss  = np.array([p.data[quantity].test_loss[-1]  for p in plotter_array])
    xvals = np.array(dataset_size_list)

    # plot the loss as a function of dataset size
    plt.clf()
    fig,ax=plt.subplots(1,1)
    ax.tick_params(axis='both',which="both", direction="in",top=True,right=True)
    ax.minorticks_on()
    plt.plot(xvals, np.sqrt(train_loss), label="train")
    plt.plot(xvals, np.sqrt(test_loss),  label="test")
    plt.legend(frameon=False)
    plt.xlabel("Dataset size")
    plt.ylabel("Max Component Error")
    plt.savefig(outfilename+"_"+quantity+".pdf",bbox_inches="tight")

plot_dataset_size(plotter_array, dataset_size_list, "ndens","dataset_size_ndens")
plot_dataset_size(plotter_array, dataset_size_list, "fluxmag","dataset_size_fluxfac")
plot_dataset_size(plotter_array, dataset_size_list, "direction","dataset_size_direction")

n_generate = 10000
F4i_0ff = generate_stable_F4_zerofluxfac(n_generate, parms["NF"], parms["device"])
F4i_1f = generate_stable_F4_oneflavor(n_generate, parms["NF"], parms["device"])
F4i_unphysical = generate_random_F4(n_generate, parms["NF"], parms["device"], zero_weight=10, max_fluxfac=0.95)

# set up datasets of stable distributions based on the max entropy stability condition
unstable_random = has_crossing(F4i_unphysical.detach().cpu().numpy(), parms["NF"], parms["n_equatorial"]).squeeze()
F4_random_stable = F4i_unphysical[unstable_random==False]
F4_random_stable = augment_permutation(F4_random_stable)
F4_random_stable = F4_random_stable.float().to(parms["device"])
print("random Stable:",np.sum(unstable_random==False))
print("random Unstable:",np.sum(unstable_random==True))

dirlist = ["base","corrected"]
for d in dirlist:
    print()
    print(d)
    if not os.path.exists(d):
        os.mkdir(d)

    do_restrict_to_physical = True if d=="corrected" else False
    
    # plot the error histogram for the test data
    error_histogram(model, F4i_train,     F4f_train,     100, 0, 0.1, do_restrict_to_physical,d+"/histogram_train")
    error_histogram(model, F4i_test,      F4f_test,      100, 0, 0.1, do_restrict_to_physical,d+"/histogram_test")
    error_histogram(model, F4_NSM_stable, F4_NSM_stable, 100, 0, 0.1, do_restrict_to_physical,d+"/histogram_NSM_stable")
    error_histogram(model, F4i_0ff,       F4i_0ff,       100, 0, 0.1, do_restrict_to_physical,d+"/histogram_0ff")
    error_histogram(model, F4i_1f,        F4i_1f,        100, 0, 0.1, do_restrict_to_physical,d+"/histogram_1f")
    error_histogram(model, F4_random_stable, F4_random_stable, 100, 0, 0.1, do_restrict_to_physical,d+"/histogram_random_stable")
    error_histogram(model, F4i_train,     F4i_train,     100, 0, 0.1, do_restrict_to_physical,d+"/histogram_donothing")
    error_histogram(model, F4f_train,     F4f_train,     100, 0, 0.1, do_restrict_to_physical,d+"/histogram_finalstable_train")
    error_histogram(model, F4f_test,      F4f_test,      100, 0, 0.1, do_restrict_to_physical,d+"/histogram_finalstable_test")
    #error_histogram(model, F4i_NSMunstable, F4f_NSMunstable, 100, 0, 0.1, do_restrict_to_physical, d+"/histogram_NSM_unstable")
    
    F4f_pred = model.predict_F4(F4i_unphysical,"eval")
    if do_restrict_to_physical:
        F4f_pred = restrict_F4_to_physical(F4f_pred)

    # enforce that number density cannot be less than zero
    F4f_pred = F4f_pred.cpu().detach().numpy()
    ndens = F4f_pred[:,3,:,:] # [sim, nu/nubar, flavor]
    negative_density_error = np.minimum(ndens, np.zeros_like(ndens)) # [sim, nu/nubar, flavor]
    negative_density_error = np.max(np.abs(negative_density_error), axis=(1,2))
    plot_histogram(negative_density_error, 100, 0, 0.1, d+"/histogram_negative_density")
    print("negative density:",np.min(negative_density_error), np.max(negative_density_error))

    # enforce that flux factors cannot be larger than 1
    fluxfac = np.sqrt(np.sum(F4f_pred[:,0:3,:,:]**2, axis=1) / ndens**2) # [sim, nu/nubar, flavor]
    fluxfac_error = np.maximum(fluxfac, np.ones_like(fluxfac)) - np.ones_like(fluxfac) # [si, conserve_lepton_number, restrict_to_physical)
    fluxfac_error = np.max(np.abs(fluxfac_error), axis=(1,2))
    plot_histogram(fluxfac_error, 100, 0, 0.1, d+"/histogram_fluxfac")
    print("fluxfac:",np.min(fluxfac_error), np.max(fluxfac_error))
