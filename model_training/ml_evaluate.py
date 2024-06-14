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
from ml_tools import flux_factor, restrict_F4_to_physical
import pickle
import glob
import os
from torchmetrics.classification import BinaryPrecisionRecallCurve
from torcheval.metrics import BinaryF1Score, BinaryNormalizedEntropy, BinaryConfusionMatrix

NSM_stable_filename = "/mnt/scratch/NSM_ML/spec_data/M1-NuLib/M1VolumeData/model_rl0_orthonormal.h5"
NSM_unstable_filename = "/mnt/scratch/NSM_ML/Emu_merger_grid2/many_sims_database.h5"

def get_dataset_size_list(search_string):
    filename_list = glob.glob(search_string)
    print(filename_list)
    dataset_size_list = []
    for filename in filename_list:
        dataset_size_list.append(int(filename.split("_")[-1].split(".")[0]))
    return sorted(dataset_size_list)
dataset_size_list_asymptotic = get_dataset_size_list("model_[0-9]*.pkl")

n_generate = 1000
test_size = 0.1
stability_cutoff = 0.5 # number between 0 and 1. If ML output is larger than this, consider unstable.
generate_max_fluxfac = 0.95
zero_weight = 10
n_equatorial = 64

# data augmentation options
do_augment_permutation=True

# the number of flavors should be 3
NF = 3

#========================#
# use a GPU if available #
#========================#
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#===============#
# read the data #
#===============#
# unpickle the data
with open("train_test_datasets.pkl","rb") as f:
    F4i_train, F4i_test, F4f_train, F4f_test = pickle.load(f)
F4_NSM_stable = read_NSM_stable_data(NF, NSM_stable_filename, device, do_augment_permutation)
# adjust entries of -1 to instead have the correct size of the dataset
for i in range(len(dataset_size_list_asymptotic)):
    if dataset_size_list_asymptotic[i] == -1:
        dataset_size_list_asymptotic[i] = F4i_train.shape[0]

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

plotter_array_asymptotic, model_array_asymptotic = get_plotter_model_arrays("model_",dataset_size_list_asymptotic)

# use the largest dataset size for the rest of these metrics
p_asymptotic = plotter_array_asymptotic[-1]
model_asymptotic = model_array_asymptotic[-1]
 
# set model to evaluation mode
model_asymptotic.eval()

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
after = model_asymptotic.predict_F4(before)
print(after[0,3])

print()
print("loss = ",comparison_loss_fn(model_asymptotic, after, F4f_train[0:1,:,:,:]).item())
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
after = model_asymptotic.predict_F4(before)
print(after[0,3])

print()
print("loss = ",comparison_loss_fn(model_asymptotic, after, F4f_train[0:1,:,:,:]).item())
print("error = ",torch.max(torch.abs(after - F4f_train[0:1,:,:,:])).item())

#=====================================#
# create test ("Fiducial" simulation) #
#=====================================#
print()
print("#############################")
print("# Testing the FIDUCIAL case #")
print("#############################")
F4_test = np.zeros((4,2,NF)) # [xyzt, nu/nubar, flavor]
F4_test[3, 0, 0] =  1
F4_test[3, 1, 0] =  1
F4_test[2, 0, 0] =  1/3
F4_test[2, 1, 0] = -1/3
before = torch.Tensor(F4_test[None,:,:,:]).to(device)
after = model_asymptotic.predict_F4(before)

print()
print("N initial")
print(before[0,3])

print()
print("N predicted")
after = model_asymptotic.predict_F4(before)
print(after[0,3])

print()
print("N re-predicted")
after = model_asymptotic.predict_F4(after)
print(after[0,3])

print()
print("2 Flavor")
before_2F = before[:,:,:,0:2]
X = model_asymptotic.X_from_F4(before)
y = model_asymptotic.forward(X)
y2F = model_asymptotic.convert_y_to_2flavor(y)
after_3F = model_asymptotic.F4_from_y(before   , y  )
after_2F = model_asymptotic.F4_from_y(before_2F, y2F)
print("3F")
print(after_3F[0,3])
print("2F")
print(after_2F[0,3])
check_conservation(before,after)

print()
print("#########################")
print("# Testing the NSM1 case #")
print("#########################")
F4_test = np.zeros((4,2,NF)) # [xyzt, nu/nubar, flavor]

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
before = torch.Tensor(F4_test[None,:,:,:]).to(device)
after = model_asymptotic.predict_F4(before)

print()
print("N initial")
print(before[0,3])

print()
print("N predicted")
after = model_asymptotic.predict_F4(before)
print(after[0,3])

print()
print("N re-predicted")
after = model_asymptotic.predict_F4(after)
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
F4_test = np.zeros((4,2,NF)) # [xyzt, nu/nubar, flavor]
F4_test[3, 0, 0] =  1
F4_test[3, 1, 0] =  .5
F4_test[2, 0, 0] =  0
F4_test[2, 1, 0] =  0
before = torch.Tensor(F4_test[None,:,:,:]).to(device)
after = model_asymptotic.predict_F4(before)

print()
print("N initial")
print(before[0,3])

print("N predicted")
after = model_asymptotic.predict_F4(before)
print(after[0,3])


print("N re-predicted")
after = model_asymptotic.predict_F4(after)
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
plot_nue_nuebar(model_asymptotic, npoints, nreps)
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

plot_dataset_size(plotter_array_asymptotic, dataset_size_list_asymptotic, "knownData","dataset_size_asymptotic")

n_generate = 10000
F4i_0ff = generate_stable_F4_zerofluxfac(n_generate, NF, device)
F4i_1f = generate_stable_F4_oneflavor(n_generate, NF, device)
F4i_unphysical = generate_random_F4(n_generate, NF, device, zero_weight=10, max_fluxfac=0.95)

# set up datasets of stable distributions based on the max entropy stability condition
unstable_random = has_crossing(F4i_unphysical.detach().cpu().numpy(), NF, n_equatorial).squeeze()
F4_random_stable = F4i_unphysical[unstable_random==False]
F4_random_stable = augment_permutation(F4_random_stable)
F4_random_stable = F4_random_stable.float().to(device)
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
    error_histogram(model_asymptotic, F4i_train,     F4f_train,     100, 0, 0.1, do_restrict_to_physical,d+"/histogram_train.pdf")
    error_histogram(model_asymptotic, F4i_test,      F4f_test,      100, 0, 0.1, do_restrict_to_physical,d+"/histogram_test.pdf")
    error_histogram(model_asymptotic, F4_NSM_stable, F4_NSM_stable, 100, 0, 0.1, do_restrict_to_physical,d+"/histogram_NSM_stable.pdf")
    error_histogram(model_asymptotic, F4i_0ff,       F4i_0ff,       100, 0, 0.1, do_restrict_to_physical,d+"/histogram_0ff.pdf")
    error_histogram(model_asymptotic, F4i_1f,        F4i_1f,        100, 0, 0.1, do_restrict_to_physical,d+"/histogram_1f.pdf")
    error_histogram(model_asymptotic, F4_random_stable, F4_random_stable, 100, 0, 0.1, do_restrict_to_physical,d+"/histogram_random_stable.pdf")
    error_histogram(model_asymptotic, F4i_train,     F4i_train,     100, 0, 0.1, do_restrict_to_physical,d+"/histogram_donothing.pdf")
    error_histogram(model_asymptotic, F4f_train,     F4f_train,     100, 0, 0.1, do_restrict_to_physical,d+"/histogram_finalstable_train.pdf")
    error_histogram(model_asymptotic, F4f_test,      F4f_test,      100, 0, 0.1, do_restrict_to_physical,d+"/histogram_finalstable_test.pdf")
    #error_histogram(model_asymptotic, F4i_NSMunstable, F4f_NSMunstable, 100, 0, 0.1, do_restrict_to_physical, d+"/histogram_NSM_unstable.pdf")
    
    F4f_pred = model_asymptotic.predict_F4(F4i_unphysical)
    if do_restrict_to_physical:
        F4f_pred = ml_tools.restrict_F4_to_physical(F4f_pred)

    # enforce that number density cannot be less than zero
    F4f_pred = F4f_pred.cpu().detach().numpy()
    ndens = F4f_pred[:,3,:,:] # [sim, nu/nubar, flavor]
    negative_density_error = np.minimum(ndens, np.zeros_like(ndens)) # [sim, nu/nubar, flavor]
    negative_density_error = np.max(np.abs(negative_density_error), axis=(1,2))
    plot_histogram(negative_density_error, 100, 0, 0.1, d+"/histogram_negative_density.pdf")
    print("negative density:",np.min(negative_density_error), np.max(negative_density_error))

    # enforce that flux factors cannot be larger than 1
    fluxfac = np.sqrt(np.sum(F4f_pred[:,0:3,:,:]**2, axis=1) / ndens**2) # [sim, nu/nubar, flavor]
    fluxfac_error = np.maximum(fluxfac, np.ones_like(fluxfac)) - np.ones_like(fluxfac) # [si, conserve_lepton_number, restrict_to_physical)
    fluxfac_error = np.max(np.abs(fluxfac_error), axis=(1,2))
    plot_histogram(fluxfac_error, 100, 0, 0.1, d+"/histogram_fluxfac.pdf")
    print("fluxfac:",np.min(fluxfac_error), np.max(fluxfac_error))
