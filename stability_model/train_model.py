import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time  

# Detect if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#Specify the number of layers in the model
NUM_LAYERS = 3
#Specify the input and hidden layer sizes
INPUT_SIZE = 27
HIDDEN_SIZE = 64

# Define the model
class BinaryClassifier(nn.Module):
    def __init__(self, input_size=27, hidden_size=64, num_layers=3):
        super(BinaryClassifier, self).__init__()
        #nn.ModuleList() is a container that holds multiple layers/modules in a list.
        #Unlike a regular Python list, ModuleList registers the layers as part of the model, ensuring they are correctly included in computations like .to(device), .parameters(), and .train().
        self.layers = nn.ModuleList()
        #This adds a fully connected (linear) layer that transforms input features of size input_size to hidden_size.
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):  #Create variable hidden layers
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
        
        self.layers.append(nn.Linear(hidden_size, 1))
        self.layers.append(nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers:  #Iterate over layers dynamically
            x = layer(x)
        return x


# Initialize model, loss function, and optimizer
model = BinaryClassifier(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device) 
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001) #Maybe can use AdamW

#Regularization -> Need to do if loss for test data starts to go up after a certain number of epochs
#Weight decay, dropout, etc.

# Example of training step (assuming you have data)
def train_step(X_batch, y_batch):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()
    return loss.item()

#Load training data for stable_zerofluxfac
training_data = np.load('train_data_stable_zerofluxfac.npz')
X_zerofluxfac = training_data['X_zerofluxfac']
unstable_zerofluxfac = training_data['unstable_zerofluxfac']

#Load training data for stable_oneflavor
training_data2 = np.load('train_data_stable_oneflavor.npz')
X_oneflavor = training_data2['X_oneflavor']
unstable_oneflavor = training_data2['unstable_oneflavor']

#Load training data for random
training_data3 = np.load('train_data_random.npz')
X_random = training_data3['X_random']
unstable_random = training_data3['unstable_random']

print("X_zerofluxfac.shape:", X_zerofluxfac.shape)
print("unstable_zerofluxfac.shape:", unstable_zerofluxfac.shape)
print("X_oneflavor.shape:", X_oneflavor.shape)
print("unstable_oneflavor.shape:", unstable_oneflavor.shape)
print("X_random.shape:", X_random.shape)
print("unstable_random.shape:", unstable_random.shape)


#Combine the datasets
input_data = np.concatenate((X_zerofluxfac, X_oneflavor, X_random), axis=0)
output_data = np.concatenate((unstable_zerofluxfac, unstable_oneflavor, unstable_random), axis=0).astype(np.float32)

print("input_data.shape:", input_data.shape)
print("output_data.shape:", output_data.shape)

#Convert data to PyTorch tensors and move to device
X_sample = torch.from_numpy(input_data).to(device) #torch.Tensor(X_zerofluxfac)  #  (sims, X)
y_sample = torch.from_numpy(output_data).to(device)  # (sims, unstable parameter value)

#Training loop
N_epochs = 1000
print_every_epoch = 100

start_time = time.time()
for i in range(N_epochs):
    loss = train_step(X_sample, y_sample)
    if i % print_every_epoch == 0:
        elapsed_time = time.time() - start_time
        elapsed_time_hr = round(elapsed_time/3600.0, 3)
        print("Epoch {}, Loss = {}, Time elapsed = {} sec = {} hr".format(i, loss, round(elapsed_time, 2), elapsed_time_hr))

