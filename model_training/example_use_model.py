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
# we want to shuffle to get the index ordering [simulationIndex, nu/nubar, flavor, xyzt]
before = torch.Tensor(F4_test[None,:,:,:]).to(device)
before = before.permute(0,2,3,1) # [sim, nu/nubar, flavor, xyzt]

# print the number densities of each species before transforming
print()
print("N initial")
print(before[0,:,:,3])

# print the number densities predicted by the ML model
after, growthrate, stable = model.predict_all(before)
print()
print("Stability prediction:", stable)
print()
print("Growthrate prediction:", growthrate)
print()

print("N predicted")
print(after[0,:,:,3])

# print the number densities predicted by Emu
print()
print("Emu expected:")
print(" [[ 9.95460068e+32  7.85471511e+32  6.23293031e+32]")
print("  [ 1.48812960e+33  7.85471511e+32  6.23293016e+32]]")

# check rotational equivariance on the spatial components by swapping the x and y components
print()
print("###############################")
print("# Testing rotational equivariance #")
print("###############################")
before_rotated = before.clone()
before_rotated[:,:,:,0] = before[:,:,:,1]
before_rotated[:,:,:,1] = before[:,:,:,0]
after_rotated = after.clone()
after_rotated[:,:,:,0] = after[:,:,:,1]
after_rotated[:,:,:,1] = after[:,:,:,0]
after2, growthrate2, stable2 = model.predict_all(before_rotated)
print()
print("Stability prediction (rotated):", stable2)
print("equivariance_error =", torch.max(torch.abs(stable - stable2)).item())
print()
print("Growthrate prediction (rotated):", growthrate2)
print("equivariance_error =", torch.max(torch.abs(growthrate - growthrate2)).item())
print()
print("N predicted (rotated)")
print(after2[0,:,:,3])
print("equivariance_error =", torch.max(torch.abs(after_rotated - after2)).item())
print()


# check permutation invariance by swapping nu and nubar
print()
print("###############################")
print("# Testing nu/nubar permutation invariance #")
print("###############################")
before_permuted = before.clone()
before_permuted[:,0,:,:] = before[:,1,:,:]
before_permuted[:,1,:,:] = before[:,0,:,:]
after_permuted = after.clone()
after_permuted[:,0,:,:] = after[:,1,:,:]
after_permuted[:,1,:,:] = after[:,0,:,:]
after3, growthrate3, stable3 = model.predict_all(before_permuted)
print()
print("Stability prediction (permuted):", stable3)
print("invariance_error =", torch.max(torch.abs(stable - stable3)).item())
print()
print("Growthrate prediction (permuted):", growthrate3)
print("invariance_error =", torch.max(torch.abs(growthrate - growthrate3)).item())
print()
print("N predicted (permuted)")
print(after3[0,:,:,3])
print("invariance_error =", torch.max(torch.abs(after_permuted - after3)).item())
print()

# check flavor permutation invariance by swapping flavor 0 and flavor 1
print()
print("###############################")
print("# Testing flavor permutation invariance #")
print("###############################")
before_flavor_permuted = before.clone()
before_flavor_permuted[:, :, 0, :] = before[:, :, 1, :]
before_flavor_permuted[:, :, 1, :] = before[:, :, 0, :]
after_flavor_permuted = after.clone()
after_flavor_permuted[:, :, 0, :] = after[:, :, 1, :]
after_flavor_permuted[:, :, 1, :] = after[:, :, 0, :]
after4, growthrate4, stable4 = model.predict_all(before_flavor_permuted)
print()
print("Stability prediction (flavor permuted):", stable4)
print("invariance_error =", torch.max(torch.abs(stable - stable4)).item())
print()
print("Growthrate prediction (flavor permuted):", growthrate4)
print("invariance_error =", torch.max(torch.abs(growthrate - growthrate4)).item())
print()
print("N predicted (flavor permuted)")
print(after4[0,:,:,3])
print("invariance_error =", torch.max(torch.abs(after_flavor_permuted - after4)).item())
print()