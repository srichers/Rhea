import torch
import numpy as np
import sys
import json
sys.path.append("../model_training")
from ml_read_data import *

# Load the traced model after training
model = torch.jit.load('nl3_size64_decay1em5_dropout0_ReLu_AdamW/stabilitytModel20000_cpu.pt')
model.eval()  # Set to evaluation mode

# use a GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using",device,"device")
model.to(device)

# Load number of flavors
NF = 3

#############################################################################
# Load parameters saved when training
with open("example1/parms.json", "r") as f:
    parms = json.load(f)
print("Loaded parameters:", parms)
parms["database_list"] = [
        "/mnt/scratch/NSM_ML/Emu_merger_grid/M1-NuLib-old/many_sims_database.h5",
        "/mnt/scratch/NSM_ML/Emu_merger_grid/M1-NuLib/many_sims_database.h5",
        "/mnt/scratch/NSM_ML/Emu_merger_grid/M1-NuLib-7ms/many_sims_database.h5",
        "/mnt/scratch/NSM_ML/Emu_merger_grid/maximum_entropy_32beam_effective2flavor/many_sims_database.h5"
]
parms["test_size"] = 0.99

F4i_train_array, F4i_test_array, F4f_train, F4f_test, logGrowthRate_train_array, logGrowthRate_test_array = read_test_train_data(parms)

print("F4i_train_array.shape = ", F4i_train_array.shape)
print("F4i_test_array.shape = ", F4i_test_array.shape)
print("F4f_train.shape = ", F4f_train.shape)
print("F4f_test.shape = ", F4f_test.shape)
#############################################################################

print()
print("#########################")
print("# Testing the NSM1 case #")
print("#########################")


for n in range(1000):
    # print the number densities predicted by Emu
    print()
    # Inference with real input
    F4_test = torch.Tensor(F4i_test_array[n, :, :, :]).to(device)
    F4_input = torch.Tensor(F4_test[None,:,:,:]).to(device)
    #print("F4_input.shape = ", F4_input.shape)
    print("sample ID = ", n)
    predicted_logGrowthRate = model.predict_logGrowthRate(F4_input)
    print("predicted = ", predicted_logGrowthRate, " expected = ", logGrowthRate_test_array[n].item())  # Prediction, ground truth
    relative_error = np.abs(predicted_logGrowthRate - logGrowthRate_test_array[n].item())/np.abs(logGrowthRate_test_array[n].item())
    print("relative error = {} %\n".format(relative_error*100))
    


#######################################################################################################
sample_size = F4i_test_array.shape[0]
print("sample size = ", sample_size)

total_relative_error = 0
for n in range(sample_size):
    F4_test = torch.Tensor(F4i_test_array[n, :, :, :]).to(device)
    F4_input = torch.Tensor(F4_test[None,:,:,:]).to(device)
    predicted_logGrowthRate = model.predict_logGrowthRate(F4_input)
    relative_error = np.abs(predicted_logGrowthRate - logGrowthRate_test_array[n].item())/np.abs(logGrowthRate_test_array[n].item())
    total_relative_error += relative_error

print("avg error = {} %\n".format(total_relative_error/sample_size*100))

