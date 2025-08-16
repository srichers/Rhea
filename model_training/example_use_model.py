'''
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This reads a trained model and passes a sample data point through it.
'''


import numpy as np
import torch
import sys

# read the filename
if len(sys.argv) != 2:
    print("Usage: example_use_model.py model.pt")
    exit()
model_filename = sys.argv[1]

# read the model
model = torch.jit.load(model_filename)

# use a GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using",device,"device")
model.to(device)

# assuming 3 flavors
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
before = torch.Tensor(F4_test[None,:,:,:]).to(device)

# print the number densities of each species before transforming
print()
print("N initial")
print(before[0,3])

# print the number densities predicted by the ML model
after, logGrowthRate, stable = model.predict_all(before)
print()
print("Stability prediction:", stable)
print()
print("Growthrate prediction:", torch.exp(logGrowthRate))
print()
print("N predicted")
print(after[0,3])

# print the number densities predicted by Emu
print()
print("Emu expected:")
print(" [[ 9.95460068e+32  7.85471511e+32  6.23293031e+32]")
print("  [ 1.48812960e+33  7.85471511e+32  6.23293016e+32]]")

