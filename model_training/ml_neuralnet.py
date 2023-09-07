import torch
from torch import nn
from ml_tools import dot4

# define the NN model
class NeuralNetwork(nn.Module):
    def __init__(self, NF,
                 nhidden = 1,
                 width = 4,
                 dropout_probability = 0.5,
                 activation = nn.ReLU):
        super().__init__()

        # store input arguments
        self.NF = NF

        # construct number of X and y values
        # one X for each pair of species, and one for each product with u
        self.NX = self.NF * (1 + 2*self.NF) + 2*self.NF
        self.Nalpha = 4*self.NF*(self.NF-1)
        self.Ny = self.Nalpha # + 1 # alpha and the stability bit

        # put together the layers of the neural network
        modules = []
        if nhidden==0:
            modules.append(nn.Linear(self.NX, self.Ny))
        else:
            modules.append(nn.Linear(self.NX, width))
            modules.append(activation())
            for i in range(nhidden):
                modules.append(nn.Linear(width,width))
                modules.append(nn.BatchNorm1d(width))
                modules.append(activation())
                modules.append(nn.Dropout(dropout_probability))
            modules.append(nn.Linear(width, self.Ny))
        self.linear_activation_stack = nn.Sequential(*modules)
        
        # initialize the weights
        self.apply(self._init_weights)

    # Push the inputs through the neural network
    def forward(self,x):
        logits = self.linear_activation_stack(x)
        return logits
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    # Create an array of the dot product of each species with itself and each other species
    # Input dimensions expeced to be [sim, xyzt, nu/nubar, flavor]
    def X_from_F4(self, F4, u):
        index = 0
        nsims = F4.shape[0]
        NF = F4.shape[-1]
        X = torch.zeros((nsims, self.NX), device=F4.device)
        F4_flat = F4.reshape((nsims, 4, 2*self.NF)) # [simulationIndex, xyzt, species]
        for a in range(2*self.NF):
            for b in range(a,2*self.NF):
                F1 = F4_flat[:,:,a]
                F2 = F4_flat[:,:,b]

                X[:,index] = dot4(F1,F2)
                index += 1

            # add the dot product with u
            # subtract mean value to zero-center the data
            X[:,index] = -dot4(F1,u) - 1./(2*NF)
            index += 1
        assert(index==self.NX)
        return X

    # get the expected output from the ML model from a pairof initial and final F4 vectors
    # input dimensions expected to be [sim, xyzt, nu/nubar, flavor]
    # Treat all but the last flavor as output, since it is determined by conservation
    # last bit is to indicate instability if >1
    def y_from_F4(self, F4_initial, F4_final):
        # calculate transformation matrix
        nsims = F4_initial.shape[0]
        F4i_flat = F4_initial.reshape((nsims, 4, 2*self.NF)) # [simulationIndex, xyzt, species]
        F4f_flat = F4_final[:,:,:,:self.NF-1].reshape((nsims, 4, 2*(self.NF-1))) # [simulationIndex, xyzt, species]
        alpha = torch.zeros((nsims,2*self.NF,2*(self.NF-1)), device=F4_initial.device)
        alpha = torch.matmul(torch.linalg.pinv(F4i_flat), F4f_flat)

        # store numbers in y
        y = torch.zeros((nsims,self.Ny), device=F4_initial.device)
        y[:,:self.Nalpha] = alpha.reshape((nsims,self.Nalpha))

        return y

    # Use the initial F4 and the ML output to calculate the final F4
    # F4_initial must have shape [sim, xyzt, nu/nubar, flavor]
    # y must have shape [2*NF**2]
    # F4_final has shape [sim, xyzt, nu/nubar, flavor]
    # Treat all but the last flavor as output, since it is determined by conservation
    def F4_from_y(self, F4_initial, y):

        nsims = F4_initial.shape[0]
        alpha = y[:,:self.Nalpha].reshape((nsims, 2*self.NF, 2*(self.NF-1)))
        F4i_flat = F4_initial.reshape((nsims, 4, 2*self.NF)) # [simulationIndex, xyzt, species]
        F4_final = torch.zeros((nsims, 4, 2, self.NF), device=F4_initial.device)
        F4_final[:,:,:,:self.NF-1] = torch.matmul(F4i_flat, alpha).reshape(nsims,4,2,self.NF-1)

        # Set the final flavor such that the flavor trace is conserved
        F4_final[:,:,:, self.NF-1] = torch.sum(F4_initial, axis=3) - torch.sum(F4_final[:,:,:,:self.NF-1], axis=3)

        return F4_final

    def predict_F4(self, F4_initial, u):
        X = self.X_from_F4(F4_initial, u)
        y = self(X)
        F4_final = self.F4_from_y(F4_initial, y)
        return F4_final

