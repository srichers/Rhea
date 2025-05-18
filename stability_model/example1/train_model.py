import sys
sys.path.append("../")
sys.path.append("../../model_training")
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time 
import json 
from sklearn.model_selection import train_test_split 
from ml_neuralnet import *
from ml_tools import *
from generate_training_data import generate_training_data

###################################################################################################
###################################################################################################

parms = {}

#n_for_training <= n_generate
parms["generate training data"] = True #generate training data? 
parms["n_generate"] = 50000 #200000 #The number of training points to generate
n_for_training = parms["n_generate"] #50000 # Number of samples to use from each training dataset

#The number of flavors: should be 3
parms["NF"] = 3

#Use a GPU if available 
parms["device"] = "cuda" if torch.cuda.is_available() else "cpu"
print("Using",parms["device"],"device")


parms["average_heavies_in_final_state"] = True
parms["conserve_lepton_number"] = "none"
parms["generate_max_fluxfac"] = 0.95
parms["ME_stability_zero_weight"] = 10
parms["ME_stability_n_equatorial"] = 32
parms["NSM_stable_filename"] = ["/mnt/scratch/NSM_ML/spec_data/M1-NuLib/M1VolumeData/model_rl1_orthonormal.h5"]
#parms["NSM_stable_filename"] = ["/mnt/scratch/NSM_ML/spec_data/M1-NuLib-7ms/model_rl1_orthonormal.h5"]
parms["do_augment_permutation"] = False


#Neural network parameters
parms["nhidden"]= 3 #Specify the number of hidden layers in the model
parms["width"]= 64 #Number of nodes in a given layer
parms["dropout_probability"]= 0.1 # Dropout rate for regularization
parms["do_batchnorm"]= False # False - Seems to make things worse #Same as Sherwood
parms["do_fdotu"]= True 
#TODO: Warning -- Only pass "nn.ReLU" and not "nn.ReLU()". We just pass the name here, it is then called inside.
parms["activation"]= nn.ReLU #Sherwood has nn.LeakyReLU  #Activation function for intermediate layers
#TODO: Warning -- Pass "nn.Sigmoid()" and not just "nn.Sigmoid". This is the opposite behavior compared to parms["activation"]
FINAL_LAYER = nn.Sigmoid() #Activation function for the final layer

#Overfit protection parameters
WEIGHT_DECAY = 1e-2 #1e-5  # L2 regularization (weight decay)
USE_EARLY_STOPPING = False #True
PATIENCE = 5000 #100  # Number of epochs to wait for improvement before stopping (Only used if USE_EARLY_STOPPING is True)

#Training parameters
N_epochs = 120000
print_every_epoch = 10 
save_model_every_epoch = 40000

# Save parameters to a JSON file
#TODO: Need to put the lone parameters also within "parms" array
# Save a serializable version
parms_serializable = parms.copy()
parms_serializable["activation"] = parms["activation"].__name__  # e.g., 'ReLU'
with open("parms.json", "w") as f:
    json.dump(parms_serializable, f, indent=4)
print("Saved parameters to 'parms.json'.")

###################################################################################################
###################################################################################################

class StabilityModelNeuralNetwork(NeuralNetwork):
    def __init__(self, parms, final_layer):
        
        self.Ny = 1
        super().__init__(parms, self.Ny, final_layer)

    def forward(self, x):
        y = self.linear_activation_stack(x).reshape(x.shape[0], 1)
        return y
    
    @torch.jit.export
    def predict_unstable_flag(self, F4_initial):
        X_input = self.X_from_F4(F4_initial)
        #output = model(X_input)
        output = self.forward(X_input)
        unstable_probability = output.float().item()
        unstable_flag = (output >= 0.5).float().item()
        return unstable_probability, unstable_flag

# Initialize model, loss function, and optimizer
model = StabilityModelNeuralNetwork(parms, FINAL_LAYER).to(parms["device"])
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss #Can also use AdamW
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=WEIGHT_DECAY) #Add weight decay for L2 regularization 

#Added Learning Rate Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=500, cooldown=500, factor=0.5, min_lr=1e-8, verbose=True)

#Training step 
def train_step(X_batch, y_batch):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)
    return loss

#Validation step
def validate_step(X_batch, y_batch):
    model.eval()
    with torch.no_grad():
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
    return loss.item()

###################################################################################################
###################################################################################################

#TODO: Important step
#Generate training data
if (parms["generate training data"] == True):
    generate_training_data(model, parms)

#Load training data for stable_zerofluxfac
training_data = np.load('../train_data_stable_zerofluxfac.npz')
X_zerofluxfac = training_data['X_zerofluxfac'][:n_for_training]
unstable_zerofluxfac = training_data['unstable_zerofluxfac'][:n_for_training]

#Load training data for stable_oneflavor
training_data2 = np.load('../train_data_stable_oneflavor.npz')
X_oneflavor = training_data2['X_oneflavor'][:n_for_training]
unstable_oneflavor = training_data2['unstable_oneflavor'][:n_for_training]

#Load training data for random
training_data3 = np.load('../train_data_random.npz')
X_random = training_data3['X_random'][:n_for_training]
unstable_random = training_data3['unstable_random'][:n_for_training]
unstable_random = unstable_random.astype(np.float32)

#Load training data for NSM stable
training_data4 = np.load('../train_data_NSM_stable.npz')
X_NSM_stable = training_data4['X_NSM_stable'][:n_for_training]
unstable_NSM_stable = training_data4['unstable_NSM_stable'][:n_for_training]

print("X_zerofluxfac.shape:", X_zerofluxfac.shape)
print("unstable_zerofluxfac.shape:", unstable_zerofluxfac.shape)
print("X_oneflavor.shape:", X_oneflavor.shape)
print("unstable_oneflavor.shape:", unstable_oneflavor.shape)
print("X_random.shape:", X_random.shape)
print("unstable_random.shape:", unstable_random.shape)
print("X_NSM_stable.shape:", X_NSM_stable.shape)
print("unstable_NSM_stable.shape:", unstable_NSM_stable.shape)


# Perform train-test split for each dataset (90% training, 10% testing)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_zerofluxfac, unstable_zerofluxfac, test_size=0.2, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_oneflavor, unstable_oneflavor, test_size=0.2, random_state=42)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X_random, unstable_random, test_size=0.2, random_state=42)
X_train4, X_test4, y_train4, y_test4 = train_test_split(X_NSM_stable, unstable_NSM_stable, test_size=0.2, random_state=42)

#Convert data to PyTorch tensors and move to device, and split into training and testing sets
X_train1 = torch.from_numpy(X_train1).to(parms["device"])
y_train1 = torch.from_numpy(y_train1).to(parms["device"])
X_test1 = torch.from_numpy(X_test1).to(parms["device"])
y_test1 = torch.from_numpy(y_test1).to(parms["device"])

X_train2 = torch.from_numpy(X_train2).to(parms["device"])
y_train2 = torch.from_numpy(y_train2).to(parms["device"])
X_test2 = torch.from_numpy(X_test2).to(parms["device"])
y_test2 = torch.from_numpy(y_test2).to(parms["device"])

X_train3 = torch.from_numpy(X_train3).to(parms["device"])
y_train3 = torch.from_numpy(y_train3).to(parms["device"])
X_test3 = torch.from_numpy(X_test3).to(parms["device"])
y_test3 = torch.from_numpy(y_test3).to(parms["device"])

X_train4 = torch.from_numpy(X_train4).to(parms["device"])
y_train4 = torch.from_numpy(y_train4).to(parms["device"])
X_test4 = torch.from_numpy(X_test4).to(parms["device"])
y_test4 = torch.from_numpy(y_test4).to(parms["device"])


###################################################################################################
###################################################################################################
f_train_test_loss = open("train_test_loss.txt","w+")
f_train_test_loss.write("{}         {}          {}\n".format("#Epoch", "train_loss", "test_loss"))

best_test_loss = float('inf')
patience = PATIENCE  # Number of epochs to wait for improvement before stopping
patience_counter = 0

#Training loop
start_time = time.time()
for i in range(N_epochs):
    # Training step 
    
    #Train each dataset separately
    loss1 = train_step(X_train1, y_train1)
    loss2 = train_step(X_train2, y_train2)
    loss3 = train_step(X_train3, y_train3)
    loss4 = train_step(X_train4, y_train4)
    #We multiply the loss from random by 100 to make it more important
    train_loss = loss1 + loss2 + 100*loss3 + loss4
    train_loss.backward()
    optimizer.step()
    
    #Validation step
    test_loss1 = validate_step(X_test1, y_test1)
    test_loss2 = validate_step(X_test2, y_test2)
    test_loss3 = validate_step(X_test3, y_test3)
    test_loss4 = validate_step(X_test4, y_test4)
    test_loss = test_loss1 + test_loss2 + 100.0*test_loss3 + test_loss4

    #Step scheduler based on test loss
    scheduler.step(test_loss)
    current_lr = optimizer.param_groups[0]['lr']  # Get current learning rate

    #Early stopping logic
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        patience_counter = 0  # Reset patience counter
    else:
        patience_counter += 1  # Increment patience counter
    
    #Print info
    if i % print_every_epoch == 0:  
        elapsed_time = round(time.time() - start_time, 2)
        elapsed_time_hr = round(elapsed_time/3600.0, 3)
        #print("Epoch {}, Train loss = {}, Test loss = {}, lr = {}, Time elapsed = {} sec = {} hr".format(i, train_loss, test_loss, current_lr, round(elapsed_time, 2), elapsed_time_hr))
        print(f"Epoch {i}, Train loss = {train_loss:.6e}, Test loss = {test_loss:.6e}, Learning rate = {current_lr:.8f}, Time elapsed = {elapsed_time:.2f} sec = {elapsed_time_hr:.3f} hr")
        f_train_test_loss.write("{}         {}          {}\n".format(i, train_loss, test_loss))

    # Stop training if validation loss hasn't improved for `patience` epochs
    if patience_counter >= patience and USE_EARLY_STOPPING:
        print(f"Early stopping at epoch {i} as validation loss did not improve for {patience} epochs.")
        break

    if i % save_model_every_epoch == 0:
        outfilename = "stabilitytModel{}".format(i)
        save_model(model, outfilename, "cpu", X_test4)
        if parms["device"]=="cuda":
            save_model(model, outfilename, "cuda", X_test4)
        

# Final validation step after training
final_test_loss1 = validate_step(X_test1, y_test1) 
final_test_loss2 = validate_step(X_test2, y_test2) 
final_test_loss3 = validate_step(X_test3, y_test3) 
final_test_loss4 = validate_step(X_test4, y_test4)
final_test_loss = final_test_loss1 + final_test_loss2 + 100.0*final_test_loss3 + final_test_loss4
print(f"Final Validation Loss: {final_test_loss}")
f_train_test_loss.close()

#Save the final model
outfilename = "finalStabilitytModel{}".format(i)
save_model(model, outfilename, "cpu", X_test4)
if parms["device"]=="cuda":
    save_model(model, outfilename, "cuda", X_test4)


'''
SOME GENERAL COMMENTS:

-> Regularization is a technique used to prevent overfitting in machine learning models. In the context of neural networks, common regularization techniques include:

-> L2 Regularization (Weight Decay): This adds a penalty proportional to the square of the magnitude of the weights to the loss function. It discourages the model from learning large weights, which can lead to overfitting.

-> Dropout: This randomly drops units (along with their connections) from the neural network during training, which helps prevent the network from becoming too reliant on specific neurons.

-> Early Stopping: This stops training when the validation loss stops improving, preventing the model from overfitting to the training data.
'''


