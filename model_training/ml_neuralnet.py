import torch
from torch import nn
from ml_tools import dot4, restrict_F4_to_physical

# define the NN model
class NeuralNetwork(nn.Module):
    def __init__(self, NF, Ny, final_layer,
                 do_fdotu,
                 nhidden,
                 width,
                 dropout_probability,
                 activation,
                 do_batchnorm):
        super().__init__()

        # store input arguments
        self.NF = NF

        # construct number of X and y values
        # one X for each pair of species, and one for each product with u
        self.do_fdotu = do_fdotu
        self.NX = self.NF * (1 + 2*self.NF)
        if do_fdotu:
            self.NX += 2*self.NF
        self.Ny = Ny

        # append a full layer including linear, activation, and batchnorm
        def append_full_layer(modules, in_dim, out_dim):
            modules.append(nn.Linear(in_dim, out_dim))
            if do_batchnorm:
                modules.append(nn.BatchNorm1d(out_dim))
            modules.append(activation())
            if dropout_probability > 0:
                modules.append(nn.Dropout(dropout_probability))
            return modules


        # put together the layers of the neural network
        modules = []
        if nhidden==0:

            # just stupid linear regression
            modules.append(nn.Linear(self.NX, self.Ny))

        else:

            # set up input layer
            modules = append_full_layer(modules, self.NX, width)

            # set up hidden layers
            for i in range(nhidden):
                modules = append_full_layer(modules, width, width)

            # set up final layer
            modules.append(nn.Linear(width, self.Ny))

        # apply tanh to limit outputs to be from -1 to 1 to limit the amount of craziness possible
        if final_layer != None:
            modules.append(final_layer)

        # turn the list of modules into a sequential model
        self.linear_activation_stack = nn.Sequential(*modules)
        
        # initialize the weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    # Push the inputs through the neural network
    def forward(self,x, mode):
        # use the appropriate network structure depending on training vs testing
        if mode=="eval":
            self.eval()
        elif mode=="train":
            self.train()
        else:
            assert(False)

        y = self.linear_activation_stack(x)
        return y

    # Create an array of the dot product of each species with itself and each other species
    # Input dimensions expeced to be [sim, xyzt, nu/nubar, flavor]
    def X_from_F4(self, F4):
        index = 0
        nsims = F4.shape[0]
        X = torch.zeros((nsims, self.NX), device=F4.device)
        F4_flat = F4.reshape((nsims, 4, 2*self.NF)) # [simulationIndex, xyzt, species]

        # calculate the total number density based on the t component of the four-vector
        # [sim]
        N = torch.sum(F4_flat[:,3,:], axis=1)

        # normalize F4 by the total number density
        # [sim, xyzt, 2*NF]
        F4_flat = F4_flat / N[:,None,None]

        # add the dot products of each species with each other species
        for a in range(2*self.NF):
            for b in range(a,2*self.NF):
                F1 = F4_flat[:,:,a]
                F2 = F4_flat[:,:,b]

                X[:,index] = dot4(F1,F2)
                index += 1

        # add the u dot products
        if self.do_fdotu:
            u = torch.zeros((4), device=F4.device)
            u[3] = 1
            for a in range(2*self.NF):
                X[:,index] = dot4(F4_flat[:,:,a], u[None,:])
                index += 1
        
        assert(index==self.NX)
        return X

class AsymptoticNeuralNetwork(NeuralNetwork):
    def __init__(self, NF, final_layer,
                 do_fdotu,
                 nhidden,
                 width,
                 dropout_probability,
                 activation,
                 do_batchnorm):
        
        self.Ny = (2*NF)**2
        super().__init__(NF, self.Ny, final_layer,
                         do_fdotu,
                         nhidden,
                         width,
                         dropout_probability,
                         activation,
                         do_batchnorm)

    # convert the 3-flavor matrix into an effective 2-flavor matrix
    # input and output are indexed as [sim, nu/nubar(out), flavor(out), nu/nubar(in), flavor(in)]
    # This assumes that the x flavors will represent the SUM of mu and tau flavors.
    def convert_y_to_2flavor(self, y):
        y2F = torch.zeros((y.shape[0],2,2,2,2), device=y.device)

        y2F[:,:,0,:,0] =                 y[:, :, 0 , :, 0 ]
        y2F[:,:,0,:,1] = 0.5 * torch.sum(y[:, :, 0 , :, 1:], axis=(  3))
        y2F[:,:,1,:,1] = 0.5 * torch.sum(y[:, :, 1:, :, 1:], axis=(2,4))
        y2F[:,:,1,:,0] =       torch.sum(y[:, :, 1:, :, 0 ], axis=(2  ))

        return y2F

    # Push the inputs through the neural network
    # output is indexed as [sim, nu/nubar(out), flavor(out), nu/nubar(in), flavor(in)]
    def forward(self,x):
        y = self.linear_activation_stack(x).reshape(x.shape[0], 2,self.NF,2,self.NF)

        # enforce conservation of particle number
        deltaij = torch.eye(2, device=x.device)[None,:,None,:,None]
        yflavorsum = torch.sum(y,axis=2)[:,:,None,:,:]
        y = y + (deltaij - yflavorsum) / self.NF

        return y
  
    # Use the initial F4 and the ML output to calculate the final F4
    # F4_initial must have shape [sim, xyzt, nu/nubar, flavor]
    # y must have shape [sim,2,NF,2,NF]
    # F4_final has shape [sim, xyzt, nu/nubar, flavor]
    # Treat all but the last flavor as output, since it is determined by conservation
    def F4_from_y(self, F4_initial, y):
        # tensor product with y
        # n indicates the simulation index
        # m indicates the spacetime index
        # i and j indicate nu/nubar
        # a and b indicate flavor
        F4_final = torch.einsum("niajb,nmjb->nmia", y, F4_initial)

        return F4_final

    def predict_F4(self, F4_initial, mode):
        X = self.X_from_F4(F4_initial)
        y = self.forward(X)
        F4_final = self.F4_from_y(F4_initial, y)

        return F4_final
