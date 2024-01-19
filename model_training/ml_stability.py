import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from ml_loss import *
from ml_neuralnet import *
from ml_optimizer import *
from ml_plot import *
from ml_trainmodel import *
import pickle

do_unpickle = False
test_size = 0.1
epochs = 500
dataset_size_list = [10,100,1000] # -1 means use all the data
print_every = 10
n_equatorial = 64
zero_fluxfac_bias = 10

conserve_lepton_number=True
nhidden = 3
width = 256
dropout_probability = 0 #0.1 # 0.5
do_batchnorm = False # False - Seems to make things worse
do_fdotu = True
activation = nn.LeakyReLU # nn.LeakyReLU, nn.ReLU
do_trivial_stable = False

# optimizer options
op = torch.optim.Adam # Adam, SGD, RMSprop
weight_decay = 0
learning_rate = 1e-5 # 1e-3

# the number of flavors should be 3
NF = 3

outfilename = "model_stability"

#========================#
# use a GPU if available #
#========================#
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#===============#
# training data #
#===============#
F4i_test = generate_random_F4(dataset_size_list[-1], NF, device)


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

for dataset_size in dataset_size_list:
    if do_unpickle:
        with open(outfilename+str(dataset_size)+".pkl", "rb") as f:
            model, optimizer, plotter = pickle.load(f)

    else:
        model = NeuralNetwork(NF, 1,
                      None, # don't put sigmoid in model in order to be able to use BCELossWithLogits
                      do_fdotu,
                      nhidden,
                      width,
                      dropout_probability,
                      activation,
                      do_batchnorm).to(device)
        plotter = Plotter(0,["random","heavy","0ff","1f"])

    plotter_array.append(plotter)
    model_array.append(model)
    optimizer_array.append(StabilityOptimizer(
        model_array[-1],
        op,
        weight_decay,
        learning_rate,
        device,
        conserve_lepton_number=conserve_lepton_number))

print(model_array[-1])


print()
print("######################")
print("# Training the model #")
print("######################")
# BCEWithLogistsLoss contains the sigmoid built in. Don't put sigmoid in model):
for i in range(len(dataset_size_list)):
    model_array[i], optimizer_array[i], plotter_array[i] = train_stability_model(
        model_array[i],
        optimizer_array[i],
        plotter_array[i],
        NF,
        epochs,
        dataset_size_list[i],
        print_every,
        device,
        n_equatorial,
        zero_fluxfac_bias,
        nn.BCEWithLogitsLoss()) 
    
    # pickle the model, optimizer, and plotter
    with open(outfilename+str(dataset_size_list[i])+".pkl", "wb") as f:
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
