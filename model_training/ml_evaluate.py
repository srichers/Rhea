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
import copy
from torchmetrics.classification import BinaryPrecisionRecallCurve
from torcheval.metrics import BinaryF1Score, BinaryNormalizedEntropy, BinaryConfusionMatrix

basedir = "/lustre/isaac/scratch/slagergr/ML_FFI"
directory_list = ["manyflavor_twobeam", "manyflavor_twobeam_z", "fluxfac_one","fluxfac_one_z"] # "fluxfac_one_twobeam",
NSM_simulated_filename = "many_sims_database_RUN_lowres_sqrt2_RUN_standard.h5"

def get_dataset_size_list(search_string):
    filename_list = glob.glob(search_string)
    print(filename_list)
    dataset_size_list = []
    for filename in filename_list:
        dataset_size_list.append(int(filename.split("_")[-1].split(".")[0]))
    return sorted(dataset_size_list)
dataset_size_list_asymptotic = get_dataset_size_list("model_[0-9]*.pkl")
dataset_size_list_stability = get_dataset_size_list("model_stability_[0-9]*.pkl")

n_generate = 1000
test_size = 0.1
stability_cutoff = 0.5 # number between 0 and 1. If ML output is larger than this, consider unstable.
generate_max_fluxfac = 0.95
zero_weight = 10
n_equatorial = 64

# data augmentation options
do_augment_permutation=True # this is the most expensive option to make true, and seems to make things worse...
conserve_lepton_number = True
restrict_to_physical = False #True

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
F4i_train, F4i_test, F4f_train, F4f_test, F4_NSM_train, F4_NSM_test = read_data(NF, basedir, directory_list, test_size, device, do_augment_permutation)
# adjust entries of -1 to instead have the correct size of the dataset
for i in range(len(dataset_size_list_asymptotic)):
    if dataset_size_list_asymptotic[i] == -1:
        dataset_size_list_asymptotic[i] = F4i_train.shape[0]

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
plotter_array_stability, model_array_stability = get_plotter_model_arrays("model_stability_",dataset_size_list_stability)

# use the largest dataset size for the rest of these metrics
p_asymptotic = plotter_array_asymptotic[-1]
model_asymptotic = model_array_asymptotic[-1]
p_stability = plotter_array_stability[-1]
model_stability = model_array_stability[-1]
 
# set model to evaluation mode
model_asymptotic.eval()
model_stability.eval()

#=============================#
# make precision recall curve #
#=============================#
print()
print("##########################")
print("# Precision recall curve #")
print("##########################")
F4i_random = generate_random_F4(n_generate, NF, 'cpu', zero_weight=zero_weight, max_fluxfac=generate_max_fluxfac)
unstable_random = torch.tensor(has_crossing(F4i_random.detach().numpy(), NF, n_equatorial), device=device).to(torch.long)
print("Random Stable:",torch.sum(unstable_random==False).item())
print("Random Unstable:",torch.sum(unstable_random==True).item())
F4i_random = F4i_random.to(device)
unstable_pred = model_stability.predict_unstable(F4i_random)
print("Unstable pred:",torch.min(unstable_pred).item(), torch.max(unstable_pred).item())
pr_curve = BinaryPrecisionRecallCurve(thresholds=101).to(device)
pr_curve.update(unstable_pred, unstable_random)
#precision, recall, thresholds = pr_curve(unstable_pred, unstable_random)

plt.clf()
fig,ax=plt.subplots(1,1)
pr_curve.plot(score=True, ax=ax)
#ax.tick_params(axis='both',which="both", direction="in",top=True,right=True)
#ax.minorticks_on()
#plt.plot(thresholds, precision, label="Precision")
#plt.plot(thresholds, recall,  label="Recall")
#plt.legend(frameon=False)
#plt.xlabel("Threshold")
plt.savefig("precision_recall.pdf",bbox_inches="tight")

# empty dimension was introduced for compatibility w/ loss functions. Here it's just clunky
metric = BinaryF1Score(threshold=stability_cutoff)
metric.update(unstable_pred[:,0], unstable_random[:,0])
print()
print("Binary F1 Score:",metric.compute().item())

metric = BinaryNormalizedEntropy()
metric.update(unstable_pred[:,0], unstable_random[:,0].to(torch.float))
print("Normalized cross entropy:",metric.compute().item())

metric = BinaryConfusionMatrix(threshold=stability_cutoff, normalize="all")
metric.update(unstable_pred[:,0], unstable_random[:,0])
confusion_matrix = metric.compute()
print("True positive:",confusion_matrix[0,0])
print("False negative:",confusion_matrix[0,1])
print("False positive:",confusion_matrix[1,0])
print("True negative:",confusion_matrix[1,1])
exit()

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
print("unstable = ",model_stability.predict_unstable(before).item(),"(should be 1)")

print()
print("N final (actual)")
print(F4f_train[0,3])

print()
print("N predicted")
after = model_asymptotic.predict_F4(before, conserve_lepton_number, restrict_to_physical)
print(after[0,3])
print("unstable = ",model_stability.predict_unstable(after).item(),"(should be 0)")

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
print("unstable = ",model_stability.predict_unstable(before).item(),"(should be 1)")

print()
print("N final (actual)")
print(F4f_test[0,3])

print()
print("N predicted")
after = model_asymptotic.predict_F4(before, conserve_lepton_number, restrict_to_physical)
print(after[0,3])
print("unstable = ",model_stability.predict_unstable(after).item(),"(should be 0)")

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
after = model_asymptotic.predict_F4(before, conserve_lepton_number, restrict_to_physical)

print()
print("N initail")
print(before[0,3])
print("unstable = ",model_stability.predict_unstable(before).item(),"(should be 1)")

print()
print("N predicted")
after = model_asymptotic.predict_F4(before, conserve_lepton_number, restrict_to_physical)
print(after[0,3])
print("unstable = ",model_stability.predict_unstable(after).item(),"(should be 0)")

print()
print("N re-predicted")
after = model_asymptotic.predict_F4(after, conserve_lepton_number, restrict_to_physical)
print(after[0,3])
print("unstable = ",model_stability.predict_unstable(after).item(),"(should be 0)")

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
after = model_asymptotic.predict_F4(before, conserve_lepton_number, restrict_to_physical)

print()
print("N initial")
print(before[0,3])
print("unstable = ",model_stability.predict_unstable(before).item(),"(should be 1)")

print("N predicted")
after = model_asymptotic.predict_F4(before, conserve_lepton_number, restrict_to_physical)
print(after[0,3])
print("unstable = ",model_stability.predict_unstable(after).item(),"(should be 0)")


print("N re-predicted")
after = model_asymptotic.predict_F4(after, conserve_lepton_number, restrict_to_physical)
print(after[0,3])
print("unstable = ",model_stability.predict_unstable(after).item(),"(should be 0)")

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
plot_nue_nuebar(model_asymptotic, npoints, nreps, conserve_lepton_number, restrict_to_physical)
p_asymptotic.plot_error(ymin=1e-5)

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
plot_dataset_size(plotter_array_stability, dataset_size_list_stability, "random","dataset_size_stability")

n_generate = 10000
F4i_0ff = generate_stable_F4_zerofluxfac(n_generate, NF, device)
F4i_1f = generate_stable_F4_oneflavor(n_generate, NF, device)
F4i_unphysical = generate_random_F4(n_generate, NF, device, zero_weight=10, max_fluxfac=0.95)

dirlist = ["base","corrected","masked","both"]
for d in dirlist:
    print()
    print(d)
    if not os.path.exists(d):
        os.mkdir(d)
    
    restrict_to_physical = True if (d=="corrected" or d=="both") else False

    # plot the error histogram for the test data
    error_histogram(model_asymptotic, F4i_train, F4f_train, 100, 0, 0.1, d,"/histogram_train.pdf", conserve_lepton_number, restrict_to_physical, model_stability, stability_cutoff)
    error_histogram(model_asymptotic, F4i_test, F4f_test, 100, 0, 0.1, d,"/histogram_test.pdf", conserve_lepton_number, restrict_to_physical, model_stability, stability_cutoff)
    error_histogram(model_asymptotic, F4_NSM_train, F4_NSM_train, 100, 0, 0.1, d,"/histogram_NSM_train.pdf", conserve_lepton_number, restrict_to_physical, model_stability, stability_cutoff)
    error_histogram(model_asymptotic, F4_NSM_test, F4_NSM_test, 100, 0, 0.1, d,"/histogram_NSM_test.pdf", conserve_lepton_number, restrict_to_physical, model_stability, stability_cutoff)
    error_histogram(model_asymptotic, F4i_0ff, F4i_0ff, 100, 0, 0.1, d,"/histogram_0ff.pdf", conserve_lepton_number, restrict_to_physical, model_stability, stability_cutoff)
    error_histogram(model_asymptotic, F4i_1f, F4i_1f, 100, 0, 0.1, d,"/histogram_1f.pdf", conserve_lepton_number, restrict_to_physical, model_stability, stability_cutoff)
    error_histogram(model_asymptotic, F4i_train, F4i_train, 100, 0, 0.1, d,"/histogram_donothing.pdf", conserve_lepton_number, restrict_to_physical, model_stability, stability_cutoff)
    error_histogram(model_asymptotic, F4f_train, F4f_train, 100, 0, 0.1, d,"/histogram_finalstable_train.pdf", conserve_lepton_number, restrict_to_physical, model_stability, stability_cutoff)
    error_histogram(model_asymptotic, F4f_test, F4f_test, 100, 0, 0.1, d,"/histogram_finalstable_test.pdf", conserve_lepton_number, restrict_to_physical, model_stability, stability_cutoff)

    F4f_pred = model_asymptotic.predict_F4(F4i_unphysical,conserve_lepton_number,restrict_to_physical)
    F4f_pred = modify_F4(F4i_unphysical, F4f_pred,d, model_stability, stability_cutoff)

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
