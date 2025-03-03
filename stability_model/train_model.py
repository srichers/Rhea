import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time  
from sklearn.model_selection import train_test_split  # Import train_test_split

# Detect if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#Specify the number of layers in the model
NUM_LAYERS = 3
#Specify the input and hidden layer sizes
INPUT_SIZE = 27
HIDDEN_SIZE = 64

DROPOUT_RATE = 0.1 #0.5  # Dropout rate for regularization
WEIGHT_DECAY = 1e-2 #1e-5  # L2 regularization (weight decay)
PATIENCE = 1000 #100  # Number of epochs to wait for improvement before stopping

'''
Regularization is a technique used to prevent overfitting in machine learning models. In the context of neural networks, common regularization techniques include:

L2 Regularization (Weight Decay): This adds a penalty proportional to the square of the magnitude of the weights to the loss function. It discourages the model from learning large weights, which can lead to overfitting.

Dropout: This randomly drops units (along with their connections) from the neural network during training, which helps prevent the network from becoming too reliant on specific neurons.

Early Stopping: This stops training when the validation loss stops improving, preventing the model from overfitting to the training data.
'''

# Define the model
class BinaryClassifier(nn.Module):
    def __init__(self, input_size=27, hidden_size=64, num_layers=3, dropout_rate=0.5):
        super(BinaryClassifier, self).__init__()
        #nn.ModuleList() is a container that holds multiple layers/modules in a list.
        #Unlike a regular Python list, ModuleList registers the layers as part of the model, ensuring they are correctly included in computations like .to(device), .parameters(), and .train().
        self.layers = nn.ModuleList()
        #This adds a fully connected (linear) layer that transforms input features of size input_size to hidden_size.
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_rate))  # Add dropout after the first layer
        
        for _ in range(num_layers - 1):  #Create variable hidden layers
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))  # Add dropout after each hidden layer
        
        self.layers.append(nn.Linear(hidden_size, 1))
        self.layers.append(nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers:  #Iterate over layers dynamically
            x = layer(x)
        return x


# Initialize model, loss function, and optimizer
model = BinaryClassifier(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE).to(device) 
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
#Can also use AdamW
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=WEIGHT_DECAY) #Add weight decay for L2 regularization 

#Regularization -> Need to do if loss for test data starts to go up after a certain number of epochs
#Weight decay, dropout, etc.

#Training step 
def train_step(X_batch, y_batch):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()
    return loss.item()

#Validation step
def validate_step(X_batch, y_batch):
    model.eval()
    with torch.no_grad():
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
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

# Perform train-test split (90% training, 10% testing)
X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.1, random_state=42)

#Convert data to PyTorch tensors and move to device, and split into training and testing sets
X_train = torch.from_numpy(X_train).to(device)
y_train = torch.from_numpy(y_train).to(device)
X_test = torch.from_numpy(X_test).to(device)
y_test = torch.from_numpy(y_test).to(device)


#Training loop
N_epochs = 2000
print_every_epoch = 100
best_test_loss = float('inf')
patience = PATIENCE  # Number of epochs to wait for improvement before stopping
patience_counter = 0

start_time = time.time()
for i in range(N_epochs):
    # Training step
    train_loss = train_step(X_train, y_train)

    # Validation step
    test_loss = validate_step(X_test, y_test)

    # Early stopping logic
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        patience_counter = 0  # Reset patience counter
    else:
        patience_counter += 1  # Increment patience counter
    
    #Print info
    if i % print_every_epoch == 0:  
        elapsed_time = time.time() - start_time
        elapsed_time_hr = round(elapsed_time/3600.0, 3)
        print("Epoch {}, Train loss = {}, Test loss = {}, Time elapsed = {} sec = {} hr".format(i, train_loss, test_loss, round(elapsed_time, 2), elapsed_time_hr))

    # Stop training if validation loss hasn't improved for `patience` epochs
    if patience_counter >= patience:
        print(f"Early stopping at epoch {i} as validation loss did not improve for {patience} epochs.")
        break

# Final validation step after training
final_val_loss = validate_step(X_test, y_test)
print(f"Final Validation Loss: {final_val_loss}")