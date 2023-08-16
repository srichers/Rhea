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
n_unphysical_check = 10000
n_trivial_stable   = 100

# data augmentation options
do_augment_permutation=True
do_augment_final_stable = True

# neural network options
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
F4_initial_list = []
F4_final_list = []
for d in directory_list:
    f_in = h5py.File(d+"/"+input_filename,"r")
    F4_initial_list.append(np.array(f_in["F4_initial(1|ccm)"])) # [simulationIndex, xyzt, nu/nubar, flavor]
    F4_final_list.append(  np.array(f_in["F4_final(1|ccm)"  ]))
    NF = np.array(f_in["nf"])
    f_in.close()
F4_initial_list = torch.tensor(np.concatenate(F4_initial_list), device=device).float()
F4_final_list   = torch.tensor(np.concatenate(F4_final_list  ), device=device).float()

# split into training and testing sets
F4i_train, F4i_test, F4f_train, F4f_test = train_test_split(F4_initial_list, F4_final_list, test_size=test_size, random_state=42)

if do_augment_permutation:
    F4i_train = ml.augment_permutation(F4i_train)
    F4f_train = ml.augment_permutation(F4f_train)
    F4i_test  = ml.augment_permutation(F4i_test )
    F4f_test  = ml.augment_permutation(F4f_test )

u = torch.tensor([0,0,0,1], device=device)[None,:]

# normalize the data so the number densities add up to 1
F4i_train, F4f_train = ml.normalize_data(F4i_train, F4f_train, u)
F4i_test,  F4f_test  = ml.normalize_data(F4i_test,  F4f_test,  u)

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
model = NeuralNetwork(NF,
                      nhidden,
                      width,
                      dropout_probability,
                      activation).to(device)
optimizer = Optimizer(model,
                      op,
                      weight_decay,
                      learning_rate,
                      device)

# some helpful info
print(model)
y_list = model.y_from_F4(F4i_train, F4f_train)
test = model.F4_from_y(F4i_train, y_list)
print("max reconstruction error:",torch.max(torch.abs(test-F4f_train)).item())

#===================================#
# Test the neutron star merger data #
#===================================#
f_in = h5py.File("Emu_many1D/many_sims_database_RUN_standard_3F_RUN_standard_3F.h5","r")
F4_initial_NSM_missingflavor = np.array(f_in["F4_initial_Nsum1"]) # [simulationIndex, xyzt, nu/nubar, flavor]
F4_final_NSM_missingflavor   = np.array(f_in["F4_final_Nsum1"  ])
f_in.close()

# reshape to three flavor array
nsims_NSM = F4_initial_NSM_missingflavor.shape[0]
newshape = (nsims_NSM,4,2,3)
F4_initial_NSM = np.zeros(newshape)
F4_final_NSM = np.zeros(newshape)
F4_initial_NSM[:,:,:,:2] = F4_initial_NSM_missingflavor
F4_final_NSM[:,:,:,:2] = F4_final_NSM_missingflavor
F4_initial_NSM[:,:,:,2] = F4_initial_NSM[:,:,:,1]
F4_final_NSM[:,:,:,2] = F4_final_NSM[:,:,:,1]
F4_initial_NSM = torch.tensor(F4_initial_NSM, device=device).float()
F4_final_NSM = torch.tensor(F4_final_NSM, device=device).float()

# normalize the data so the number densities add up to 1
F4_initial_NSM, F4_final_NSM = ml.normalize_data(F4_initial_NSM, F4_final_NSM, u)
F4_initial_NSM = torch.Tensor(F4_initial_NSM).to(device)
F4_final_NSM   = torch.Tensor(F4_final_NSM  ).to(device)
F4_final_NSM[:,:,1,:] = F4_final_NSM[:,:,0,:] + F4_initial_NSM[:,:,1,:] - F4_initial_NSM[:,:,0,:]

print("NSM:",F4_initial_NSM.shape)
print()

#===============#
# training loop #
#===============#
p = Plotter(epochs)
for t in range(epochs):
    optimizer.optimizer.zero_grad()

    # train on making sure the model prediction is correct
    p.knownData.test_loss[t],  p.knownData.test_err[t]  = optimizer.test(model, F4i_test,  F4f_test,  u, comparison_loss_fn)
    loss = optimizer.train(model, F4i_train, F4f_train, u, comparison_loss_fn)
    loss.backward()
    p.knownData.train_loss[t], p.knownData.train_err[t] = optimizer.test(model, F4i_train, F4f_train, u, comparison_loss_fn)

    if do_augment_final_stable:
        p.knownData_FS.test_loss[t],  p.knownData_FS.test_err[t]  = optimizer.test(model, F4i_test,  F4i_test,  u, comparison_loss_fn)
        loss = optimizer.train(model, F4i_train, F4f_train, u, comparison_loss_fn)
        loss.backward()
        p.knownData_FS.train_loss[t], p.knownData_FS.train_err[t] = optimizer.test(model, F4i_train, F4i_train, u, comparison_loss_fn)
    
    if n_unphysical_check>0:
        # train on making sure the model prediction is physical
        F4i = generate_random_F4(n_unphysical_check, NF, device)
        p.unphysical.test_loss[t],  p.unphysical.test_err[t]  = optimizer.test(model, F4i, None, u, unphysical_loss_fn)
        loss = optimizer.train(model, F4i, None, u, unphysical_loss_fn)
        loss.backward()
        p.unphysical.train_loss[t], p.unphysical.train_err[t] = optimizer.test(model, F4i, None, u, unphysical_loss_fn)

    if n_trivial_stable>0:
        # train on making sure known stable distributions dont change
        F4i = generate_stable_F4_zerofluxfac(n_trivial_stable, NF, device)
        p.zerofluxfac.test_loss[t],  p.zerofluxfac.test_err[t]  = optimizer.test(model, F4i, F4i, u, comparison_loss_fn)
        loss = optimizer.train(model, F4i, F4i, u, comparison_loss_fn)
        loss.backward()
        p.zerofluxfac.train_loss[t], p.zerofluxfac.train_err[t] = optimizer.test(model, F4i, F4i, u, comparison_loss_fn)
        
        F4i = generate_stable_F4_oneflavor(n_trivial_stable, NF, device)
        p.oneflavor.test_loss[t],  p.oneflavor.test_err[t]  = optimizer.test(model, F4i, F4i, u, comparison_loss_fn)
        loss = optimizer.train(model, F4i, F4i, u, comparison_loss_fn)
        loss.backward()
        p.oneflavor.train_loss[t], p.oneflavor.train_err[t] = optimizer.test(model, F4i, F4i, u, comparison_loss_fn)
                
    optimizer.optimizer.step()

    p.NSM.test_loss[t], p.NSM.test_err[t] = optimizer.test(model, F4_initial_NSM, F4_final_NSM, u, comparison_loss_fn)
    
    # report max error
    if((t+1)%(epochs//10)==0):
        print(f"Epoch {t+1}")
        print("Train max error:",        p.knownData.train_err[t])
        print("Test max error:",         p.knownData.test_err[t])
        print("NSM max error:",          p.NSM.test_err[t])
        print()

print("Done!")

# save the model to file
with torch.no_grad():
    print(F4_initial_NSM.shape)
    X = model.X_from_F4(F4_initial_NSM, u)
    print(X.shape)
    traced_model = torch.jit.trace(model, X)
    torch.jit.save(traced_model, "model.ptc")


#===================================#
# Test one point from training data #
#===================================#
print()
print("Training Data")
print("N initial")
before = F4i_train[0:1,:,:,:]
print(before[0,3])

print("N final (actual)")
print(F4f_train[0,3])

print("N predicted")
after = model.predict_F4(before, u)
print(after[0,3])

print("N re-predicted")
after = model.predict_F4(after, u)
print(after[0,3])

#===================================#
# Test one point from test data #
#===================================#
print()
print("Test Data")
print("N initial")
before = F4i_test[0:1,:,:,:]
print(before[0,3])

print("N final (actual)")
print(F4f_test[0,3])

print("N predicted")
after = model.predict_F4(before, u)
print(after[0,3])

print("N re-predicted")
after = model.predict_F4(after, u)
print(after[0,3])

#=====================================#
# create test ("Fiducial" simulation) #
#=====================================#
F4_test = np.zeros((4,2,NF)) # [xyzt, nu/nubar, flavor]
F4_test[3, 0, 0] =  1
F4_test[3, 1, 0] =  1
F4_test[2, 0, 0] =  1/3
F4_test[2, 1, 0] = -1/3
before = torch.Tensor(F4_test[None,:,:,:]).to(device)
after = model.predict_F4(before, u)

print()
print("Fiducial Simulation")
print("N initail")
print(before[0,3])

print("N predicted")
after = model.predict_F4(before, u)
print(after[0,3])

print("N re-predicted")
for i in range(5):
    after = model.predict_F4(after, u)
    print(after[0,3])


#=====================================#
# create test ("Zero FF" simulation) #
#=====================================#
F4_test = np.zeros((4,2,NF)) # [xyzt, nu/nubar, flavor]
F4_test[3, 0, 0] =  1
F4_test[3, 1, 0] =  .5
F4_test[2, 0, 0] =  0
F4_test[2, 1, 0] =  0
before = torch.Tensor(F4_test[None,:,:,:]).to(device)
after = model.predict_F4(before, u)

print()
print("Fiducial Simulation")
print("N initail")
print(before[0,3])

print("N predicted")
after = model.predict_F4(before, u)
print(after[0,3])

print("N re-predicted")
for i in range(5):
    after = model.predict_F4(after, u)
    print(after[0,3])


npoints = 11
nreps = 20
p.plot_nue_nuebar(model, npoints, nreps, u)
p.plot_error()


