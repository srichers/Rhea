import torch
import numpy as np
import sys
import json

# Load the traced model after training
model = torch.jit.load('final_model.pt')
model.eval()  # Set to evaluation mode

# Load the X_from_F4 function
X_from_F4 = torch.jit.load('X_from_F4.pt')

# Load parameters saved when generating the training data
with open("parms.json", "r") as f:
    parms = json.load(f)
print("Loaded parameters:", parms)

# use a GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using",device,"device")
model.to(device)

# Load number of flavors
NF = parms["NF"] 

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

#Convert F4 to X
X_input = X_from_F4(parms["NF"], parms["do_fdotu"], before)

# print the number densities of each species before transforming
print()
print("N initial")
print(before[0,3])

# print the number densities predicted by Emu
print()
print("TODO:: Emu expected: stable/unstable?")

# Inference with real input
output = model(X_input)
unstable_flag = (output >= 0.5).float().item()
print("Model output: unstable_flag = ", unstable_flag)  # Predictions

if unstable_flag > 0.5:
    print("UNSTABLE")
else:
    print("STABLE")

