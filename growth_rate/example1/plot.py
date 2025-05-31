import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

#Determine the name of the parent folder
os.system("pwd > temp1.txt")
f = open("temp1.txt", "r")
test_name = f.readline().strip().split("/")[-1]
print("Test name = '{}'".format(test_name))
os.system("rm temp1.txt")


# Read the data file
filename = "train_test_loss.txt"
data = np.loadtxt(filename)

# Extract columns
epochs = data[:, 0]
train_loss = data[:, 1]
test_loss = data[:, 2]

# Plotting
plt.figure(figsize=(8, 6))
plt.semilogy(epochs, train_loss, label='Train Loss', color='blue', linestyle='-', linewidth=2)
plt.semilogy(epochs, test_loss, label='Test Loss', color='orange', linestyle='--', linewidth=2)

# Add labels, title, legend
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('{}'.format(test_name), fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Show plot
plt.tight_layout()
plt.savefig("train_test_loss.png")
#plt.show()

