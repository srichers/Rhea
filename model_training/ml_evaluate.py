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
from ml_tools import flux_factor
import pickle

basedir = "/lustre/isaac/scratch/slagergr/ML_FFI"
directory_list = ["manyflavor_twobeam", "manyflavor_twobeam_z", "fluxfac_one","fluxfac_one_z"] # "fluxfac_one_twobeam",
NSM_simulated_filename = "many_sims_database_RUN_lowres_sqrt2_RUN_standard.h5"
dataset_size_list = [10,100,1000,10000,38172] # -1 means use all the data
n_generate = 1000
test_size = 0.1

# data augmentation options
do_augment_permutation=True # this is the most expensive option to make true, and seems to make things worse...
do_augment_final_stable = False # True
do_trivial_stable   = False # True
do_NSM_stable = False # True
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
for i in range(len(dataset_size_list)):
    if dataset_size_list[i] == -1:
        dataset_size_list[i] = F4i_train.shape[0]

#=======================#
# instantiate the model #
#=======================#
print()
print("#############################")
print("# SETTING UP NEURAL NETWORK #")
print("#############################")
# set up an array of models, optimizers, and plotters for different dataset sizes
model_array = []
plotter_array = []

for dataset_size in dataset_size_list:
    filename = "model_"+str(dataset_size)+".pkl"
    print("Loading",filename)
    with open(filename, "rb") as f:
        model, optimizer, plotter = pickle.load(f)

    plotter_array.append(plotter)
    model_array.append(model)

print(model_array[-1])

# use the largest dataset size for the rest of these metrics
p = plotter_array[-1]
model = model_array[-1]
 
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

print("N final (actual)")
print(F4f_train[0,3])

print("N predicted")
after = model.predict_F4(before, conserve_lepton_number, restrict_to_physical)
print(after[0,3])

#print("N re-predicted")
#after = model.predict_F4(after, conserve_lepton_number, restrict_to_physical)
#print(after[0,3])

print()
print("loss = ",comparison_loss_fn(model, after, F4f_train[0:1,:,:,:]).item())
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

print("N final (actual)")
print(F4f_test[0,3])

print("N predicted")
after = model.predict_F4(before, conserve_lepton_number, restrict_to_physical)
print(after[0,3])

#print("N re-predicted")
#after = model.predict_F4(after, conserve_lepton_number, restrict_to_physical)
#print(after[0,3])

print()
print("loss = ",comparison_loss_fn(model, after, F4f_train[0:1,:,:,:]).item())
print("error = ",torch.max(torch.abs(after - F4f_train[0:1,:,:,:])).item())
check_conservation(before,after)

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
after = model.predict_F4(before, conserve_lepton_number, restrict_to_physical)

print()
print("N initail")
print(before[0,3])

print("N predicted")
after = model.predict_F4(before, conserve_lepton_number, restrict_to_physical)
print(after[0,3])

print("N re-predicted")
after = model.predict_F4(after, conserve_lepton_number, restrict_to_physical)
print(after[0,3])

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

print()
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
after = model.predict_F4(before, conserve_lepton_number, restrict_to_physical)

print()
print("N initial")
print(before[0,3])

print("N predicted")
after = model.predict_F4(before, conserve_lepton_number, restrict_to_physical)
print(after[0,3])

print("N re-predicted")
after = model.predict_F4(after, conserve_lepton_number, restrict_to_physical)
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
p.init_plot_options()
plot_nue_nuebar(model, npoints, nreps, conserve_lepton_number, restrict_to_physical)
p.plot_error(ymin=1e-5)

# plot the loss as a function of dataset size using the array of plotters
train_loss = np.array([p.data["knownData"].train_loss[-1] for p in plotter_array])
test_loss  = np.array([p.data["knownData"].test_loss[-1]  for p in plotter_array])
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
plt.savefig("dataset_size.pdf",bbox_inches="tight")

# plot the error histogram for the test data
n_generate = 10000
F4i_0ff = generate_stable_F4_zerofluxfac(n_generate, NF, device)
F4i_1f = generate_stable_F4_oneflavor(n_generate, NF, device)
error_histogram(model, F4i_train, F4f_train, 100, 0, 0.1, "histogram_train.pdf", conserve_lepton_number, restrict_to_physical)
error_histogram(model, F4i_test, F4f_test, 100, 0, 0.1, "histogram_test.pdf", conserve_lepton_number, restrict_to_physical)
error_histogram(model, F4_NSM_train, F4_NSM_train, 100, 0, 0.1, "histogram_NSM_train.pdf", conserve_lepton_number, restrict_to_physical)
error_histogram(model, F4_NSM_test, F4_NSM_test, 100, 0, 0.1, "histogram_NSM_test.pdf", conserve_lepton_number, restrict_to_physical)
error_histogram(model, F4i_0ff, F4i_0ff, 100, 0, 0.1, "histogram_0ff.pdf", conserve_lepton_number, restrict_to_physical)
error_histogram(model, F4i_1f, F4i_1f, 100, 0, 0.1, "histogram_1f.pdf", conserve_lepton_number, restrict_to_physical)
error_histogram(model, F4i_train, F4i_train, 100, 0, 0.1, "histogram_donothing.pdf", conserve_lepton_number, restrict_to_physical)
error_histogram(model, F4f_train, F4f_train, 100, 0, 0.1, "histogram_finalstable_train.pdf", conserve_lepton_number, restrict_to_physical)
error_histogram(model, F4f_test, F4f_test, 100, 0, 0.1, "histogram_finalstable_test.pdf", conserve_lepton_number, restrict_to_physical)

# histogram of how unphysical the results are
F4i_unphysical = generate_random_F4(n_generate, NF, device)
F4f_pred = model.predict_F4(F4i_unphysical,conserve_lepton_number,restrict_to_physical).cpu().detach().numpy()

# normalize F4f_pred by the total number density
Ntot = np.sum(F4f_pred[:,3,:,:], axis=(1,2)) # [sim]

# enforce that number density cannot be less than zero
ndens = F4f_pred[:,3,:,:] # [sim, nu/nubar, flavor]
negative_density_error = np.minimum(ndens/Ntot[:,None,None], np.zeros_like(ndens)) # [sim, nu/nubar, flavor]
negative_density_error = np.max(np.abs(negative_density_error), axis=(1,2))
plot_histogram(negative_density_error, 100, 0, 0.1, "histogram_negative_density.pdf")
print("negative density:",np.min(negative_density_error), np.max(negative_density_error))

# enforce that flux factors cannot be larger than 1
fluxfac = np.sqrt(np.sum(F4f_pred[:,0:3,:,:]**2, axis=1) / ndens**2) # [sim, nu/nubar, flavor]
fluxfac_error = np.maximum(fluxfac, np.ones_like(fluxfac)) - np.ones_like(fluxfac) # [si, conserve_lepton_number, restrict_to_physical)
fluxfac_error = np.max(np.abs(fluxfac_error), axis=(1,2))
plot_histogram(fluxfac_error, 100, 0, 0.1, "histogram_fluxfac.pdf")
print("fluxfac:",np.min(fluxfac_error), np.max(fluxfac_error))
