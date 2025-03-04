import torch
import numpy as np
import scipy
import json  # Import JSON for saving dictionary
from utils import *


################################################
############# Create a list of options #########
################################################
parms = {}

#TODO: Modify this to generate for data
#The number of points to generate
parms["n_generate"] = 100 #200000

#The number of flavors: should be 3
parms["NF"] = 3

#Use a GPU if available 
#parms["device"] = "cuda" if torch.cuda.is_available() else "cpu"
#print("Using",parms["device"],"device")
#print(torch.cuda.get_device_name(0))
parms["device"] = "cpu"

parms["average_heavies_in_final_state"] = True
parms["do_fdotu"]= True
parms["generate_max_fluxfac"] = 0.95
parms["ME_stability_zero_weight"] = 10
parms["ME_stability_n_equatorial"] = 32


################################################
############# Run the test case ############### 
################################################
#Test cases for has_crossing function
# run test case
F4i = torch.zeros((1,4,2,3))
F4i[0,3,0,0] = 1
F4i[0,3,1,0] = 2
#print("should be false:", has_crossing(F4i.detach().numpy(), 3, 64))
print("should be false:", int(has_crossing(F4i.detach().numpy(), parms)))
assert(int(has_crossing(F4i.detach().numpy(), parms)) == 0)

F4i = torch.zeros((1,4,2,3))
F4i[0,3,0,0] = 1
F4i[0,3,1,0] = 1
F4i[0,0,0,0] =  0.5
F4i[0,0,1,0] = -0.5
#print("should be true:", has_crossing(F4i.detach().numpy(), 3, 64))
print("should be true:", int(has_crossing(F4i.detach().numpy(), parms)))
assert(int(has_crossing(F4i.detach().numpy(), parms)) == 1)



################################################
############# Generate the data ################
################################################
print("Generating data for stable_zerofluxfac..")
F4_zerofluxfac = generate_stable_F4_zerofluxfac(parms)
#This is the output label for stable vs unstable
#Is the point unstable?
#Yes => unstable = 1
#No => unstable = 0
unstable_zerofluxfac = torch.zeros((parms["n_generate"], 1), device=parms["device"])
unstable_zerofluxfac[:, None] = 0 #Since all the generated points are stable
#print(F4_zerofluxfac.shape)
#print(F4_zerofluxfac[0,:,:,:])
#print(unstable_zerofluxfac.shape)
#print(unstable_zerofluxfac[0])

print("Generating data for stable_oneflavor..")
F4_oneflavor = generate_stable_F4_oneflavor(parms)
#This is the output label for stable vs unstable
#Is the point unstable?
#Yes => unstable = 1
#No => unstable = 0
unstable_oneflavor = torch.zeros((parms["n_generate"]*2*parms["NF"], 1), device=parms["device"])
unstable_oneflavor[:, None] = 0 #Since all the generated points are stable
#print(F4_oneflavor.shape)
#print(F4_oneflavor[0,:,:,:])
#print(unstable_oneflavor.shape)
#print(unstable_oneflavor[0])

print("Generating random data..")
F4_random = generate_random_F4(parms)
#print(F4_random.shape)
#print(F4_random[0,:,:,:])
#If has_crossing returns true, unstable = 1 (i.e this point is unstable)
#If has_crossing returns false, unstable = 0 (i.e this point is stable)
unstable_random = has_crossing(F4_random.detach().numpy(), parms)
#print("unstable_random.shape", unstable_random.shape)
#print("unstable_random", unstable_random)


################################################
############# Convert from F4 to X #############
################################################
X_zerofluxfac = X_from_F4(parms["NF"], parms["do_fdotu"], F4_zerofluxfac)
print("X_zerofluxfac.shape", X_zerofluxfac.shape)
print("X_zerofluxfac[0,:]", X_zerofluxfac[0,:])
print("unstable_zerofluxfac.shape", unstable_zerofluxfac.shape)
print("unstable_zerofluxfac[0]", unstable_zerofluxfac[0])

X_oneflavor = X_from_F4(parms["NF"], parms["do_fdotu"], F4_oneflavor)
print("X_oneflavor.shape", X_oneflavor.shape)
print("X_oneflavor[0,:]", X_oneflavor[0,:])
print("unstable_oneflavor.shape", unstable_oneflavor.shape)
print("unstable_oneflavor[0]", unstable_oneflavor[0])

X_random = X_from_F4(parms["NF"], parms["do_fdotu"], F4_random)
print("X_random.shape", X_random.shape)
print("X_random[0,:]", X_random[0,:])
print("unstable_random.shape", unstable_random.shape)
print("unstable_random[0]", unstable_random[0])


################################################
############# Save the data ####################
################################################
np.savez('train_data_stable_zerofluxfac.npz', X_zerofluxfac=X_zerofluxfac, unstable_zerofluxfac=unstable_zerofluxfac)
np.savez('train_data_stable_oneflavor.npz', X_oneflavor=X_oneflavor, unstable_oneflavor=unstable_oneflavor)
np.savez('train_data_random.npz', X_random=X_random, unstable_random=unstable_random)

# Save parameters to a JSON file
with open("parms.json", "w") as f:
    json.dump(parms, f, indent=4)

print("Saved parameters to 'parms.json'.")