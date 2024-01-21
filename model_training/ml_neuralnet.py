import torch
from torch import nn
from ml_tools import dot4

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
    def forward(self,x):
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
    
    # return the y value
    def predict_unstable(self, F4_initial):
        X = self.X_from_F4(F4_initial)
        y = torch.sigmoid(self.forward(X))
        return y


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

    def predict_F4(self, F4_initial, conserve_lepton_number, restrict_to_physical):
        X = self.X_from_F4(F4_initial)
        y = self.forward(X)
        F4_final = self.F4_from_y(F4_initial, y)

        if conserve_lepton_number:
            # set the antineutrino number densities to conserve lepton number
            Jt_initial = F4_initial[:,3,0,:] - F4_initial[:,3,1,:]
            Jt_final   =   F4_final[:,3,0,:] -   F4_final[:,3,1,:]
            delta_Jt = Jt_final - Jt_initial
            F4_final[:,3,0,:] = F4_final[:,3,0,:] - 0.5*delta_Jt
            F4_final[:,3,1,:] = F4_final[:,3,1,:] + 0.5*delta_Jt

        if restrict_to_physical:
            avgF4 = torch.sum(F4_final, axis=3)[:,:,:,None] / self.NF

            # enforce that all four-vectors are time-like
            # choose the branch that leads to the most positive alpha
            a = dot4(avgF4, avgF4) # [sim, nu/nubar, flavor]
            b = 2.*dot4(avgF4, F4_final)
            c = dot4(F4_final, F4_final)
            radical = b**2 - 4*a*c
            assert(torch.all(radical>=-1e-6))
            radical = torch.maximum(radical, torch.zeros_like(radical))
            alpha = (-b + torch.sign(a)*torch.sqrt(radical)) / (2*a)

            # fix the case where a is zero
            badlocs = torch.where(torch.abs(a/b)<1e-6)
            if len(badlocs[0]) > 0:
                alpha[badlocs] = (-c/b)[badlocs]

            # find the nu/nubar and flavor indices where alpha is maximized
            maxalpha = torch.amax(alpha, dim=(1,2)) # [sim]
            maxalpha = torch.maximum(maxalpha, torch.zeros_like(maxalpha))
            maxalpha[torch.where(maxalpha>0)] += 1e-6

            # modify the four-vectors to be time-like
            F4_final = (F4_final + maxalpha[:,None,None,None]*avgF4) / (maxalpha[:,None,None,None] + 1)

        return F4_final

    # get the expected output from the ML model from a pairof initial and final F4 vectors
    # input dimensions expected to be [sim, xyzt, nu/nubar, flavor]
    def y_from_F4(self, F4_initial, F4_final):
        # calculate transformation matrix
        nsims = F4_initial.shape[0]
        F4i_flat = F4_initial.reshape((nsims, 4, 2*self.NF)) # [simulationIndex, xyzt, species]
        F4f_flat = F4_final.reshape(  (nsims, 4, 2*self.NF)) # [simulationIndex, xyzt, species]
        y = torch.matmul(torch.linalg.pinv(F4i_flat), F4f_flat) # [sim, species, species]
        y = torch.transpose(y,1,2).reshape(nsims,2,self.NF,2,self.NF) # [sim, nu/nubar, species, nu/nubar, species]

        return y
