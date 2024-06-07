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
import torch.autograd.profiler as profiler

database_list = [
    #"/mnt/scratch/NSM_ML/ML_models/input_data/manyflavor_twobeam/many_sims_database.h5",
    #"/mnt/scratch/NSM_ML/ML_models/input_data/fluxfac_one/many_sims_database.h5",
    #"/mnt/scratch/NSM_ML/ML_models/input_data/fluxfac_one_twobeam/many_sims_database.h5",
    #"/mnt/scratch/NSM_ML/ML_models/input_data/fluxfac_one_z/many_sims_database.h5",
    #"/mnt/scratch/NSM_ML/ML_models/input_data/manyflavor_twobeam_z/many_sims_database.h5",
    "/mnt/scratch/NSM_ML/ML_models/input_data/maximum_entropy_6beam/many_sims_database.h5",
    "/mnt/scratch/NSM_ML/Emu_merger_grid2/many_sims_database.h5"
]
NSM_stable_filename = "/mnt/scratch/NSM_ML/spec_data/M1-NuLib/M1VolumeData/model_rl0_orthonormal.h5"
do_unpickle = False
test_size = 0.1
epochs = 20000
batch_size = -1
dataset_size_list = [-1] # -1 means use all the data
n_generate = 7500
print_every = 10
generate_max_fluxfac = 0.95
ME_stability_zero_weight = 10
ME_stability_n_equatorial = 64

# data augmentation options
do_augment_permutation=True # this is the most expensive option to make true, and seems to make things worse...
do_augment_final_stable = False # True
do_unphysical_check = True # True - seems to help prevent crazy results
do_augment_0ff = True
do_augment_1f = True
do_augment_random_stable = True
do_augment_NSM_stable = True

# neural network options
nhidden = 3
width = 1024
dropout_probability = 0.1 #0.1 # 0.5
do_batchnorm = False # False - Seems to make things worse
do_fdotu = True
activation = nn.LeakyReLU # nn.LeakyReLU, nn.ReLU

# optimizer options
op = torch.optim.AdamW # Adam, SGD, RMSprop
amsgrad = False
weight_decay = 1e-2 #1e-5
learning_rate = 1e-3 # 1e-3
fused = True
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
patience = 100
cooldown = 100
factor = 0.5
min_lr = 1e-5

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
if do_unpickle:
    with open("train_test_datasets.pkl", "rb") as f:
        F4i_train, F4i_test, F4f_train, F4f_test = pickle.load(f)
else:
    F4i_train, F4i_test, F4f_train, F4f_test = read_test_train_data(NF, database_list, test_size, device, do_augment_permutation)

F4_NSM_stable = read_NSM_stable_data(NF, NSM_stable_filename, device, do_augment_permutation)
F4_NSM_stable_train, F4_NSM_stable_test, _, _ = train_test_split(F4_NSM_stable, F4_NSM_stable, test_size=test_size, random_state=42)

# move the arrays over to the gpu
F4i_train = torch.Tensor(F4i_train).to(device)
F4f_train = torch.Tensor(F4f_train).to(device)
F4i_test  = torch.Tensor(F4i_test ).to(device)
F4f_test  = torch.Tensor(F4f_test ).to(device)
print("Train:",F4i_train.shape)
print("Test:",F4i_test.shape)

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
        plotter = Plotter(0,["knownData","unphysical","0ff","1f","finalstable","randomstable","NSM_stable"])

    plotter_array.append(plotter)
    model_array.append(model)
    optimizer_array.append(AsymptoticOptimizer(
        model_array[-1],
        op(model.parameters(), weight_decay=weight_decay, lr=learning_rate, amsgrad=amsgrad, fused=fused),
        device))
    scheduler_array.append(lr_scheduler(optimizer_array[-1].optimizer, patience=patience, cooldown=cooldown, factor=factor, min_lr=min_lr)) # 

print(model_array[-1])
print("number of parameters:", sum(p.numel() for p in model_array[-1].parameters()))

print()
print("######################")
print("# Training the model #")
print("######################")
#with profiler.profile(with_stack=True, profile_memory=True, record_shapes=True) as prof:
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
        do_augment_random_stable,
        do_augment_NSM_stable,
        ME_stability_zero_weight,
        ME_stability_n_equatorial,
        comparison_loss_fn,
        unphysical_loss_fn,
        F4i_train,
        F4f_train,
        F4i_test,
        F4f_test,
        F4_NSM_stable_train,
        F4_NSM_stable_test)

    # pickle the model, optimizer, and plotter
    with open("model_"+str(dataset_size_list[i])+".pkl", "wb") as f:
        pickle.dump([model_array[i], optimizer_array[i], plotter_array[i]], f)

#print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total'))
        
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
