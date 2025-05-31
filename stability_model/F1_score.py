import torch
import numpy as np
import sys
import json

# Load the traced model after training
model = torch.jit.load('n50000_decay1em5_dropout0p01_AdamW/stabilitytModel60000_cpu.pt') #TODO: 
model.eval()  # Set to evaluation mode

# use a GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using",device,"device")
model.to(device)

# Load number of flavors
NF = 3

#How many samples to choose
n_for_training = 100000 #TODO: 

#Load the actual data (ground truth)
#Load training data for stable_zerofluxfac
training_data = np.load('train_data_stable_zerofluxfac.npz')
X_zerofluxfac = training_data['X_zerofluxfac'][:n_for_training]
actual_unstable_zerofluxfac = training_data['unstable_zerofluxfac'][:n_for_training]
print("X_zerofluxfac.shape:", X_zerofluxfac.shape)
print(X_zerofluxfac[0,:])
print("actual_unstable_zerofluxfac.shape:", actual_unstable_zerofluxfac.shape)

#Load training data for stable_oneflavor
training_data2 = np.load('train_data_stable_oneflavor.npz')
X_oneflavor = training_data2['X_oneflavor'][:n_for_training]
actual_unstable_oneflavor = training_data2['unstable_oneflavor'][:n_for_training]
print("X_oneflavor.shape:", X_oneflavor.shape)
print(X_oneflavor[0,:])
print("actual_unstable_oneflavor.shape:", actual_unstable_oneflavor.shape)

#Load training data for random
training_data3 = np.load('train_data_random.npz')
X_random = training_data3['X_random'][:n_for_training]
actual_unstable_random = training_data3['unstable_random'][:n_for_training]
actual_unstable_random = actual_unstable_random.astype(np.float32)
print("X_random.shape:", X_random.shape)
print(X_random[0,:])
print("actual_unstable_random.shape:", actual_unstable_random.shape)

#Load training data for NSM stable
training_data4 = np.load('train_data_NSM_stable.npz')
X_NSM_stable = training_data4['X_NSM_stable'][:n_for_training]
actual_unstable_NSM_stable = training_data4['unstable_NSM_stable'][:n_for_training]
print("X_NSM_stable.shape:", X_NSM_stable.shape)
print(X_NSM_stable[0,:])
print("actual_unstable_NSM_stable.shape:", actual_unstable_NSM_stable.shape)

def actual_vs_predicted_unstable_flag(n, X_array, actual_unstable_array, model):
    X_input_n = X_array[n,:]
    #Convert to Tensor and move to GPU
    X_input = torch.from_numpy(X_input_n[None, :]).to(device)
    #print(n, X_input)
    predicted_unstable_probability = model(X_input)
    #print(predicted_unstable_probability.item())
    unstable_flag = -1
    if predicted_unstable_probability.item() > 0.5: #TODO: Let's tweak this threshold and see how the F1 changes. We aim for higher F1 score. We care about removing false positives. 
        unstable_flag = 1 #TODO: Maybe try changing weight decay and dropout, and see how much they improve precision (even zero dropout).
    else:
        unstable_flag = 0
    #print("n = ", n, "Actual flag: ", actual_unstable_array[n].item(), "Predicted flag: ", unstable_flag)
    return unstable_flag, actual_unstable_array[n].item()


'''
Let us assume that unstable_flag = 1 is "positive" and unstable_flag = 0 is "negative"

True Positives (TP): Number of samples correctly predicted as “positive.” => predicted = 1 and actual = 1

False Positives (FP): Number of samples wrongly predicted as “positive.” => predicted = 1 and actual = 0

True Negatives (TN): Number of samples correctly predicted as “negative.” => predicted = 0 and actual = 0

False Negatives (FN): Number of samples wrongly predicted as “negative.” => predicted = 0 and actual = 1
'''

print("\nPrecision, Recall, F1 calculation...")
nsamples = n_for_training #TODO: 

TP = 0
FP = 0
TN = 0
FN = 0
for n in range(nsamples):
    predicted_unstable_flag, actual_unstable_flag = \
        actual_vs_predicted_unstable_flag(n, X_zerofluxfac, actual_unstable_zerofluxfac, model)
    #print("n = ", n, "Actual flag: ", actual_unstable_flag, "Predicted flag: ", predicted_unstable_flag)
    if abs(predicted_unstable_flag - 1) < 1e-6 and abs(actual_unstable_flag - 1) < 1e-6:
        TP += 1
    if abs(predicted_unstable_flag - 0) < 1e-6 and abs(actual_unstable_flag - 0) < 1e-6:
        TN += 1
    if abs(predicted_unstable_flag - 1) < 1e-6 and abs(actual_unstable_flag - 0) < 1e-6:
        FP += 1
    if abs(predicted_unstable_flag - 0) < 1e-6 and abs(actual_unstable_flag - 1) < 1e-6:
        FN += 1


for n in range(nsamples):
    predicted_unstable_flag, actual_unstable_flag = \
        actual_vs_predicted_unstable_flag(n, X_oneflavor, actual_unstable_oneflavor, model)
    #print("n = ", n, "Actual flag: ", actual_unstable_flag, "Predicted flag: ", predicted_unstable_flag)
    if abs(predicted_unstable_flag - 1) < 1e-6 and abs(actual_unstable_flag - 1) < 1e-6:
        TP += 1
    if abs(predicted_unstable_flag - 0) < 1e-6 and abs(actual_unstable_flag - 0) < 1e-6:
        TN += 1
    if abs(predicted_unstable_flag - 1) < 1e-6 and abs(actual_unstable_flag - 0) < 1e-6:
        FP += 1
    if abs(predicted_unstable_flag - 0) < 1e-6 and abs(actual_unstable_flag - 1) < 1e-6:
        FN += 1

for n in range(nsamples):
    predicted_unstable_flag, actual_unstable_flag = \
        actual_vs_predicted_unstable_flag(n, X_random, actual_unstable_random, model)
    #print("n = ", n, "Actual flag: ", actual_unstable_flag, "Predicted flag: ", predicted_unstable_flag)
    if abs(predicted_unstable_flag - 1) < 1e-6 and abs(actual_unstable_flag - 1) < 1e-6:
        TP += 1
    if abs(predicted_unstable_flag - 0) < 1e-6 and abs(actual_unstable_flag - 0) < 1e-6:
        TN += 1
    if abs(predicted_unstable_flag - 1) < 1e-6 and abs(actual_unstable_flag - 0) < 1e-6:
        FP += 1
    if abs(predicted_unstable_flag - 0) < 1e-6 and abs(actual_unstable_flag - 1) < 1e-6:
        FN += 1

for n in range(21360): #Since NSM stable data has only 21360 points
    predicted_unstable_flag, actual_unstable_flag = \
        actual_vs_predicted_unstable_flag(n, X_NSM_stable, actual_unstable_NSM_stable, model)
    #print("n = ", n, "Actual flag: ", actual_unstable_flag, "Predicted flag: ", predicted_unstable_flag)
    if abs(predicted_unstable_flag - 1) < 1e-6 and abs(actual_unstable_flag - 1) < 1e-6:
        TP += 1
    if abs(predicted_unstable_flag - 0) < 1e-6 and abs(actual_unstable_flag - 0) < 1e-6:
        TN += 1
    if abs(predicted_unstable_flag - 1) < 1e-6 and abs(actual_unstable_flag - 0) < 1e-6:
        FP += 1
    if abs(predicted_unstable_flag - 0) < 1e-6 and abs(actual_unstable_flag - 1) < 1e-6:
        FN += 1

print("\nConfusion matrix:")
print(TP, FP)
print(FN, TN)

precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2 * precision * recall / (precision + recall)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1: ", F1)



'''
print("\nDIRECT CALCULATION:")
for n in range(200):
    X_zerofluxfac_input_n = X_zerofluxfac[n,:]
    #Convert to Tensor and move to GPU
    X_zerofluxfac_input = torch.from_numpy(X_zerofluxfac_input_n[None, :]).to(device)
    #print(n, X_zerofluxfac_input)
    predicted_unstable_zerofluxfac = model(X_zerofluxfac_input)
    #print(predicted_unstable_zerofluxfac.item())
    unstable_flag = -1
    if predicted_unstable_zerofluxfac.item() > 0.5:
        unstable_flag = 1
    else:
        unstable_flag = 0
    print("n = ", n, "Actual flag: ", actual_unstable_zerofluxfac[n].item(), "Predicted flag: ", unstable_flag)
'''

'''
#Convert to Tensor and move to GPU
X_zerofluxfac = torch.from_numpy(X_zerofluxfac).to(device) 

predicted_unstable_zerofluxfac = model(X_zerofluxfac)
print()
print(predicted_unstable_zerofluxfac.shape)
print(predicted_unstable_zerofluxfac)
print()
'''
