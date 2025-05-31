import torch
import numpy as np
import sys
import json

# Load the traced model after training
model = torch.jit.load('example1/stabilitytModel80000_cpu.pt')
model.eval()  # Set to evaluation mode

# use a GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using",device,"device")
model.to(device)

# Load number of flavors
NF = 3

print()
print("#########################")
print("# Testing the NSM1 case #")
print("#########################")

# set the initial conditions to the NSM1 distribution
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

# set the tensor to have the correct dimensions (first index is used to evaluate many data points at the same time)
F4_input = torch.Tensor(F4_test[None,:,:,:]).to(device)

# print the number densities of each species before transforming
print()
print("N initial")
print(F4_input[0,3])

# print the number densities predicted by Emu
print()
print("TODO:: Emu expected: ???")
print(F4_input.shape)
# Inference with real input
logGrowthRate = model.predict_logGrowthRate(F4_input)
print("logGrowthRate = ", logGrowthRate)  # Predictions


