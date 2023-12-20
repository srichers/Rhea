# credit to https://thinkingneuron.com/using-artificial-neural-networks-for-regression-in-python/
# Convert the flavor transformation data to one with reduced dimensionality to make it easier to train on
# Run from the directory containin the joint dataset
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import matplotlib as mpl
import matplotlib.pyplot as plt
import ml_tools as ml
import torchviz
from ml_loss import *
from ml_neuralnet import *
from ml_optimizer import *
from ml_plot import *

input_filename = "many_sims_database.h5"
directory_list = ["manyflavor_twobeam", "manyflavor_twobeam_z", "fluxfac_one","fluxfac_one_twobeam","fluxfac_one_z"]
test_size = 0.1
epochs = 5000
batch_size = 1000
print_every = 1
outfilename = "model"

# data augmentation options
do_augment_permutation=True
do_augment_final_stable = False
do_unphysical_check = True
do_particlenumber_conservation_check = True
do_trivial_stable   = True

# neural network options
conserve_lepton_number=True
nhidden = 1
width = 32
dropout_probability = 0.5
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
    f_in = h5py.File(d+"/"+input_filename,"r")
    F4_initial_list.append(np.array(f_in["F4_initial(1|ccm)"])) # [simulationIndex, xyzt, nu/nubar, flavor]
    F4_final_list.append(  np.array(f_in["F4_final(1|ccm)"  ]))
    NF = int(np.array(f_in["nf"]))
    f_in.close()
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

#=======================#
# instantiate the model #
#=======================#
print()
print("#############################")
print("# SETTING UP NEURAL NETWORK #")
print("#############################")
model = NeuralNetwork(NF,
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

#===================================#
# Test the neutron star merger data #
#===================================#
print()
print("######################")
print("# PREPARING NSM DATA #")
print("######################")
print("NSM data is useless. Ignoring.")

#=====================================================#
# Load training data into data loader for minibatches #
#=====================================================#
print()
print("###########################################")
print("# LOADING TRAINING DATA INTO DATA LOADERS #")
print("###########################################")
dataset = torch.utils.data.TensorDataset(F4i_train, F4f_train)
batch_size = max(batch_size, len(dataset))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
print("batchsize=",batch_size)

#===============#
# training loop #
#===============#
print()
print("############")
print("# Training #")
print("############")
p = Plotter(epochs)
for t in range(epochs):

    # load in a batch of data from the dataset
    for F4i_batch, F4f_batch in dataloader:

        # reset gradients in the optimizer
        optimizer.optimizer.zero_grad()

        # train on making sure the model prediction is correct
        loss = optimizer.train(model, F4i_batch, F4f_batch, comparison_loss_fn)
        loss.backward()

        if do_augment_final_stable:
            loss = optimizer.train(model, F4f_batch, F4f_batch, comparison_loss_fn)
            loss.backward()

        # train on making sure the model prediction is physical
        if do_unphysical_check:
            F4i_unphysical = generate_random_F4(batch_size, NF, device)
            loss = optimizer.train(model, F4i_unphysical, None, unphysical_loss_fn)
            loss.backward()

        # train on making sure the model prediction is physical
        if do_particlenumber_conservation_check:
            F4i_particlenumber = generate_random_F4(batch_size, NF, device)
            loss = optimizer.train(model, F4i_particlenumber, F4i_particlenumber, particle_number_loss_fn)
            loss.backward()

        # train on making sure known stable distributions dont change
        if do_trivial_stable:
            F4i_0ff = generate_stable_F4_zerofluxfac(batch_size, NF, device)
            loss = optimizer.train(model, F4i_0ff, F4i_0ff, comparison_loss_fn)
            loss.backward()

            F4i_1f = generate_stable_F4_oneflavor(batch_size, NF, device)
            loss = optimizer.train(model, F4i_1f, F4i_1f, comparison_loss_fn)
            loss.backward()

        # update the weights
        optimizer.optimizer.step()

    # Evaluate training errors
    p.knownData.test_loss[t],  p.knownData.test_err[t]  = optimizer.test(model, F4i_test,  F4f_test,  comparison_loss_fn)
    p.knownData.train_loss[t], p.knownData.train_err[t] = optimizer.test(model, F4i_train, F4f_train, comparison_loss_fn)
    if do_augment_final_stable:
        p.knownData_FS.test_loss[t],  p.knownData_FS.test_err[t]  = optimizer.test(model, F4f_test,  F4f_test,  comparison_loss_fn)
        p.knownData_FS.train_loss[t], p.knownData_FS.train_err[t] = optimizer.test(model, F4f_train, F4f_train, comparison_loss_fn)    
    if do_unphysical_check:
        F4i_unphysical = generate_random_F4(batch_size, NF, device)
        p.unphysical.test_loss[t],  p.unphysical.test_err[t]  = optimizer.test(model, F4i_unphysical, None, unphysical_loss_fn)
    if do_particlenumber_conservation_check:
        # don't collect error because we are not testing the final state
        # rather, we are just using the defined loss function to estimate the particle number conservation violation
        F4i_particlenumber = generate_random_F4(batch_size, NF, device)
        p.particlenumber.test_loss[t],  _  = optimizer.test(model, F4i_particlenumber, F4i_particlenumber, particle_number_loss_fn)
    if do_trivial_stable:
        F4i_0ff = generate_stable_F4_zerofluxfac(batch_size, NF, device)
        p.zerofluxfac.test_loss[t],  p.zerofluxfac.test_err[t]  = optimizer.test(model, F4i_0ff, F4i_0ff, comparison_loss_fn)
        F4i_1f = generate_stable_F4_oneflavor(batch_size, NF, device)
        p.oneflavor.test_loss[t],  p.oneflavor.test_err[t]  = optimizer.test(model, F4i_1f, F4i_1f, comparison_loss_fn)
    
    # report max error
    if((t+1)%print_every==0):
        print(f"Epoch {t+1}")
        print("Train max error:", p.knownData.train_err[t])
        print("Test max error:",  p.knownData.test_err[t])
        print("Train loss:",      p.knownData.train_loss[t])
        print("Test loss:",       p.knownData.test_loss[t])
        print()

print("Done!")

# save the model to file
print()
print("########################")
print("# Saving model to file #")
print("########################")
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
print("Fiducial Simulation")
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
print("Fiducial Simulation")
print("N initail")
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


