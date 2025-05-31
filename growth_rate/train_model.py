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
from ml_read_data import *
#from generate_training_data import generate_training_data

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
parms["nhidden"]= 3 #Specify the number of hidden layers in the model FIXME:
parms["width"]= 64 #Number of nodes in a given layer  FIXME:
parms["dropout_probability"]= 0.01 # Dropout rate for regularization #FIXME:
parms["do_batchnorm"]= False # False - Seems to make things worse #Same as Sherwood
parms["do_fdotu"]= True 
#TODO: Warning -- Only pass "nn.ReLU" and not "nn.ReLU()". We just pass the name here, it is then called inside.
parms["activation"]= nn.ReLU #nn.LeakyReLU #nn.ReLU #Sherwood has nn.LeakyReLU  #Activation function for intermediate layers
#TODO: Warning -- Pass "nn.Sigmoid()" and not just "nn.Sigmoid". This is the opposite behavior compared to parms["activation"]
#FINAL_LAYER = nn.Sigmoid() #Activation function for the final layer
FINAL_LAYER = None  # No activation for real-valued output

#Overfit protection parameters
WEIGHT_DECAY = 1e-5 #1e-2 #1e-5  # L2 regularization (weight decay) FIXME:
USE_EARLY_STOPPING = False #True
PATIENCE = 5000 #100  # Number of epochs to wait for improvement before stopping (Only used if USE_EARLY_STOPPING is True)

#Training parameters
N_epochs = 21000   
print_every_epoch = 10 
save_model_every_epoch = 5000 #40000

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
    def predict_logGrowthRate(self, F4_initial):
        X_input = self.X_from_F4(F4_initial)
        #output = model(X_input)
        output = self.forward(X_input)
        logGrowthRate = output.float().item()
        return logGrowthRate

# Initialize model, loss function, and optimizer
model = StabilityModelNeuralNetwork(parms, FINAL_LAYER).to(parms["device"])
#criterion = nn.BCELoss()  # Binary Cross-Entropy Loss #Can also use AdamW
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=WEIGHT_DECAY) #Add weight decay for L2 regularization 

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

#Generate data in training format
parms["database_list"] = [
        "/mnt/scratch/NSM_ML/Emu_merger_grid/M1-NuLib-old/many_sims_database.h5",
        "/mnt/scratch/NSM_ML/Emu_merger_grid/M1-NuLib/many_sims_database.h5",
        "/mnt/scratch/NSM_ML/Emu_merger_grid/M1-NuLib-7ms/many_sims_database.h5",
        "/mnt/scratch/NSM_ML/Emu_merger_grid/maximum_entropy_32beam_effective2flavor/many_sims_database.h5"
]


parms["test_size"] = 0.2

F4i_train, F4i_test, F4f_train, F4f_test, logGrowthRate_train, logGrowthRate_test = read_test_train_data(parms)

#Print the values of F4's and logGrowthRate. 
print(F4i_train.shape, logGrowthRate_train.shape)

#Now convert the F4's to X, and then use that for training. 
y_train = logGrowthRate_train
y_test = logGrowthRate_test
X_train = model.X_from_F4(F4i_train)
X_test = model.X_from_F4(F4i_test)
print(X_train.shape, X_test.shape)

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
    loss1 = train_step(X_train, y_train)
    train_loss = loss1
    train_loss.backward()
    optimizer.step()
    
    #Validation step
    test_loss1 = validate_step(X_test, y_test)
    test_loss = test_loss1

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
        save_model(model, outfilename, "cpu", X_test)
        if parms["device"]=="cuda":
            save_model(model, outfilename, "cuda", X_test)
        

# Final validation step after training
final_test_loss1 = validate_step(X_test, y_test)
final_test_loss = final_test_loss1
print(f"Final Validation Loss: {final_test_loss}")
f_train_test_loss.close()

#Save the final model
outfilename = "finalStabilitytModel{}".format(i)
save_model(model, outfilename, "cpu", X_test)
if parms["device"]=="cuda":
    save_model(model, outfilename, "cuda", X_test)
