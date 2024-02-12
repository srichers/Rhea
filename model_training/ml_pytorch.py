# credit to https://thinkingneuron.com/using-artificial-neural-networks-for-regression-in-python/
# Convert the flavor transformation data to one with reduced dimensionality to make it easier to train on
# Run from the directory containin the joint dataset
import numpy as np
import torch
from torch import nn
from ml_loss import *
from ml_neuralnet import *
from ml_optimizer import *
from ml_plot import *
from ml_trainmodel import *
from ml_maxentropy import *
from ml_read_data import *
import pickle
import torch.optim

directory_list = ["manyflavor_twobeam","manyflavor_twobeam_z", "fluxfac_one","fluxfac_one_z"] # "fluxfac_one_twobeam",
#basedir = "/mnt/scratch/srichers/ML_FFI"
basedir = "/lustre/isaac/scratch/slagergr/ML_FFI"
do_unpickle = False
test_size = 0.1
epochs = 10000
batch_size = -1
dataset_size_list = [-1]#[10,100,1000,10000,-1] # -1 means use all the data
n_generate = 10000
print_every = 10
generate_max_fluxfac = 0.95

# data augmentation options
do_augment_permutation=True # this is the most expensive option to make true, and seems to make things worse...
do_augment_final_stable = True # True
do_unphysical_check = True # True - seems to help prevent crazy results
do_augment_0ff = True
do_augment_1f = True

# neural network options
conserve_lepton_number=True
bound_to_physical = False # causes nans in back propagation
nhidden = 3
width = 1024
dropout_probability = 0.0 #0.1 # 0.5
do_batchnorm = False # False - Seems to make things worse
do_fdotu = True
activation = nn.LeakyReLU # nn.LeakyReLU, nn.ReLU

# optimizer options
op = torch.optim.Adam # Adam, SGD, RMSprop
weight_decay = 0 #1e-5
learning_rate = 1e-3 # 1e-3
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
patience = 100
cooldown = 0
factor = 0.5

# the number of flavors should be 3
NF = 3

#========================#
# use a GPU if available #
#========================#
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
print(torch.cuda.get_device_name(0))

#===============#
# read the data #
#===============#
F4i_train, F4i_test, F4f_train, F4f_test, F4_NSM_train, F4_NSM_test = read_data(NF, basedir, directory_list, test_size, device, do_augment_permutation)

# adjust entries of -1 to instead have the correct size of the dataset
for i in range(len(dataset_size_list)):
    if dataset_size_list[i] == -1:
        dataset_size_list[i] = F4i_train.shape[0]

# test for stability under max entropy condition
#train_crossing = has_crossing(F4i_train.cpu().detach().numpy(), NF, 64)

# count the number of stable simulations
#print("stable:",np.sum(train_crossing==False))
#print("unstable:",np.sum(train_crossing==True))

#=======================#
# instantiate the model #
#=======================#
print()
print("#############################")
print("# SETTING UP NEURAL NETWORK #")
print("#############################")
# set up an array of models, optimizers, and plotters for different dataset sizes
model_array = []
optimizer_array = []
plotter_array = []
scheduler_array = []

for dataset_size in dataset_size_list:
    if do_unpickle:
        with open("model_"+str(dataset_size)+".pkl", "rb") as f:
            model, optimizer, plotter = pickle.load(f)

    else:
        model = AsymptoticNeuralNetwork(NF,
                      nn.Tanh(),
                      do_fdotu,
                      nhidden,
                      width,
                      dropout_probability,
                      activation,
                      do_batchnorm).to(device)
        plotter = Plotter(0,["knownData","unphysical","0ff","1f","finalstable"])

    plotter_array.append(plotter)
    model_array.append(model)
    optimizer_array.append(AsymptoticOptimizer(
        model_array[-1],
        op,
        weight_decay,
        learning_rate,
        device))
    scheduler_array.append(lr_scheduler(optimizer_array[-1].optimizer, patience=patience, cooldown=cooldown, factor=factor))

print(model_array[-1])


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

print()
print("######################")
print("# Training the model #")
print("######################")
for i in range(len(dataset_size_list)):
    model_array[i], optimizer_array[i], scheduler_array[i], plotter_array[i] = train_asymptotic_model(
        model_array[i],
        optimizer_array[i],
        scheduler_array[i],
        plotter_array[i],
        NF,
        epochs,
        batch_size,
        n_generate,
        generate_max_fluxfac,
        dataset_size_list[i],
        print_every,
        device,
        do_unphysical_check,
        do_augment_final_stable,
        do_augment_1f,
        do_augment_0ff,
        conserve_lepton_number,
        bound_to_physical,
        comparison_loss_fn,
        unphysical_loss_fn,
        F4i_train,
        F4f_train,
        F4i_test,
        F4f_test)
    
    # pickle the model, optimizer, and plotter
    with open("model_"+str(dataset_size_list[i])+".pkl", "wb") as f:
        pickle.dump([model_array[i], optimizer_array[i], plotter_array[i]], f)

# use the largest dataset size for the rest of these metrics
p = plotter_array[-1]
model = model_array[-1]
optimizer = optimizer_array[-1]
 
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
