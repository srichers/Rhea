# credit to https://thinkingneuron.com/using-artificial-neural-networks-for-regression-in-python/
# Convert the flavor transformation data to one with reduced dimensionality to make it easier to train on
# Run from the directory containin the joint dataset
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import ml_tools as ml
from ml_loss import *
from ml_neuralnet import *
from ml_optimizer import *
from ml_plot import *
from ml_trainmodel import *

basedir = "/mnt/scratch/srichers/ML_FFI"
directory_list = ["manyflavor_twobeam", "manyflavor_twobeam_z", "fluxfac_one","fluxfac_one_twobeam","fluxfac_one_z"]
test_size = 0.1
epochs = 500
batch_size = -1
dataset_size_list = [2,10,100,1000,-1]
print_every = 1

# data augmentation options
do_augment_permutation=False # this is the most expensive option to make true, and seems to make things worse...
do_augment_final_stable = True
do_unphysical_check = True
do_particlenumber_conservation_check = True # really doesn't do anything, since it's built into the ML structure
do_trivial_stable   = True
do_NSM_stable = True

# neural network options
conserve_lepton_number=True
nhidden = 3
width = 32
dropout_probability = 0.5
do_fdotu = False
activation = nn.LeakyReLU # nn.LeakyReLU, nn.ReLU

# optimizer options
op = torch.optim.Adam # Adam, SGD, RMSprop
weight_decay = 1e-4
learning_rate = 1e-3

#========================#
# use a GPU if available #
#========================#
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

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
    NF = int(np.array(f_in["nf"]))
    f_in.close()
    print(len(F4_initial_list[-1]),"points in",d)
F4_initial_list = torch.tensor(np.concatenate(F4_initial_list), device=device).float()
F4_final_list   = torch.tensor(np.concatenate(F4_final_list  ), device=device).float()

# normalize the data so the number densities add up to 1
ntot = ml.ntotal(F4_initial_list)
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

# move the arrays over to the gpu
F4i_train = torch.Tensor(F4i_train).to(device)
F4f_train = torch.Tensor(F4f_train).to(device)
F4i_test  = torch.Tensor(F4i_test ).to(device)
F4f_test  = torch.Tensor(F4f_test ).to(device)
print("Train:",F4i_train.shape)
print("Test:",F4i_test.shape)

#=================================================#
# read in the stable points from the NSM snapshot #
print()
print("#############################")
print("# READING NSM STABLE POINTS #")
print("#############################")
F4_NSM_train = None
F4_NSM_test = None
# note that x represents the SUM of mu, tau, anti-mu, anti-tau and must be divided by 4 to get the individual flavors
# take only the y-z slice to limit the size of the data.
if do_NSM_stable:
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
    F4_NSM_train, F4_NSM_test, _, _ = train_test_split(F4_NSM_stable, F4_NSM_stable, test_size=test_size, random_state=42)

    if do_augment_permutation:
        F4_NSM_train = ml.augment_permutation(F4_NSM_train)
        F4_NSM_test  = ml.augment_permutation(F4_NSM_test )

    # move the array to the device
    F4_NSM_train = torch.Tensor(F4_NSM_train).to(device)
    F4_NSM_test  = torch.Tensor(F4_NSM_test ).to(device)

#=======================#
# instantiate the model #
#=======================#
print()
print("#############################")
print("# SETTING UP NEURAL NETWORK #")
print("#############################")
model = NeuralNetwork(NF,
                      do_fdotu,
                      nhidden,
                      width,
                      dropout_probability,
                      activation).to(device)
optimizer = Optimizer(model,
                      op,
                      weight_decay,
                      learning_rate,
                      device,
                      conserve_lepton_number=conserve_lepton_number)
print(model)


#=============================================================#
# check that we can obtain a value for y using pseudoinverses #
#=============================================================#
print()
print("#################################################")
print("# CHECKING PSEUDOINVERSE METHOD FOR OBTAINING Y #")
print("#################################################")
y_list = model.y_from_F4(F4i_train, F4f_train)
test = model.F4_from_y(F4i_train, y_list)
error = torch.max(torch.abs(test-F4f_train)).item()
print("max reconstruction error:",error)
assert(error < 1e-3)

print("######################")
print("# Training the model #")
print("######################")
plotter_array = []
for dataset_size in dataset_size_list:
    print("Training dataset size:",dataset_size)
    model, p = train_model(NF,
                   do_fdotu,
                   nhidden,
                   width,
                   dropout_probability,
                   activation,
                   op,
                   weight_decay,
                   learning_rate,
                   epochs,
                   batch_size,
                   dataset_size,
                   print_every,
                   device,
                   conserve_lepton_number,
                   do_augment_final_stable,
                   do_NSM_stable,
                   do_unphysical_check,
                   do_particlenumber_conservation_check,
                   do_trivial_stable,
                   comparison_loss_fn,
                   unphysical_loss_fn,
                   particle_number_loss_fn,
                   F4i_train,
                   F4f_train,
                   F4i_test,
                   F4f_test,
                   F4_NSM_train,
                   F4_NSM_test)
    plotter_array.append(p)

# use the largest dataset size for the rest of these metrics
p = plotter_array[-1]
 
# save the model to file
print()
print("########################")
print("# Saving model to file #")
print("########################")
outfilename = "model"
def save_model(model, outfilename, device):
    with torch.no_grad():
        print(F4i_test.shape)

        model.to(device)
        X = model.X_from_F4(F4i_test.to(device))
        traced_model = torch.jit.trace(model, X)
        torch.jit.save(traced_model, outfilename+"_"+device+".ptc")
        print("Saving to",outfilename+"_"+device+".ptc")

save_model(model, outfilename, "cpu")
if device=="cuda":
    save_model(model, outfilename, "cuda")

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
after = model.predict_F4(before, conserve_lepton_number=conserve_lepton_number)
print(after[0,3])

print("N re-predicted")
after = model.predict_F4(after, conserve_lepton_number=conserve_lepton_number)
print(after[0,3])

print()
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
after = model.predict_F4(before, conserve_lepton_number=conserve_lepton_number)
print(after[0,3])

print("N re-predicted")
after = model.predict_F4(after, conserve_lepton_number=conserve_lepton_number)
print(after[0,3])

print()
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
after = model.predict_F4(before, conserve_lepton_number=conserve_lepton_number)

print()
print("N initail")
print(before[0,3])

print("N predicted")
after = model.predict_F4(before, conserve_lepton_number=conserve_lepton_number)
print(after[0,3])

print("N re-predicted")
after = model.predict_F4(after, conserve_lepton_number=conserve_lepton_number)
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
after = model.predict_F4(before, conserve_lepton_number=conserve_lepton_number)

print()
print("N initial")
print(before[0,3])

print("N predicted")
after = model.predict_F4(before, conserve_lepton_number=conserve_lepton_number)
print(after[0,3])

print("N re-predicted")
after = model.predict_F4(after, conserve_lepton_number=conserve_lepton_number)
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
p.plot_nue_nuebar(model, npoints, nreps)
p.plot_error()

# plot the loss as a function of dataset size using the array of plotters
train_loss = np.array([p.knownData.train_loss[-1] for p in plotter_array])
test_loss  = np.array([p.knownData.test_loss[-1]  for p in plotter_array])
xvals = np.array(dataset_size_list)
xvals[np.where(xvals==-1)] = F4i_train.shape[0]

plt.clf()
fig,ax=plt.subplots(1,1)
ax.tick_params(axis='both',which="both", direction="in",top=True,right=True)
ax.minorticks_on()
plt.plot(xvals, np.sqrt(train_loss), label="train")
plt.plot(xvals, np.sqrt(test_loss),  label="test")
plt.legend(frameon=False)
plt.xlabel("Dataset size")
plt.ylabel("Error")
plt.savefig("dataset_size.pdf",bbox_inches="tight")

