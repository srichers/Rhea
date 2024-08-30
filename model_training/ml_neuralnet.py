import torch
from torch import nn
from ml_tools import dot4, restrict_F4_to_physical

# define the NN model
class NeuralNetwork(nn.Module):
    def __init__(self, parms, Ny, final_layer):
        super().__init__()

        # store input arguments
        self.NF = parms["NF"]

        # construct number of X and y values
        # one X for each pair of species, and one for each product with u
        self.do_fdotu = parms["do_fdotu"]
        self.NX = self.NF * (1 + 2*self.NF)
        if self.do_fdotu:
            self.NX += 2*self.NF
        self.Ny = Ny
        self.average_heavies_in_final_state = parms["average_heavies_in_final_state"]
        self.conserve_lepton_number = parms["conserve_lepton_number"]

        # append a full layer including linear, activation, and batchnorm
        def append_full_layer(modules, in_dim, out_dim):
            modules.append(nn.Linear(in_dim, out_dim))
            if parms["do_batchnorm"]:
                modules.append(nn.BatchNorm1d(out_dim))
            modules.append(parms["activation"]())
            if parms["dropout_probability"] > 0:
                modules.append(nn.Dropout(parms["dropout_probability"]))
            return modules


        # put together the layers of the neural network
        modules = []
        if parms["nhidden"]==0:

            # just stupid linear regression
            modules.append(nn.Linear(self.NX, self.Ny))

        else:

            # set up input layer
            modules = append_full_layer(modules, self.NX, parms["width"])

            # set up hidden layers
            for i in range(parms["nhidden"]):
                modules = append_full_layer(modules, parms["width"], parms["width"])

            # set up final layer
            modules.append(nn.Linear(parms["width"], self.Ny))

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
    
    # Create an array of the dot product of each species with itself and each other species
    # Input dimensions expeced to be [sim, xyzt, nu/nubar, flavor]
    @torch.jit.export
    def X_from_F4(self, F4):
        index = 0
        nsims = F4.shape[0]
        X = torch.zeros((nsims, self.NX), device=F4.device)
        F4_flat = F4.reshape((nsims, 4, 2*self.NF)) # [simulationIndex, xyzt, species]

        # calculate the total number density based on the t component of the four-vector
        # [sim]
        N = torch.sum(F4_flat[:,3,:], dim=1)

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
    def __init__(self, parms, final_layer):
        
        self.Ny = (2*parms["NF"])**2
        super().__init__(parms, self.Ny, final_layer)

    # convert the 3-flavor matrix into an effective 2-flavor matrix
    # input and output are indexed as [sim, nu/nubar(out), flavor(out), nu/nubar(in), flavor(in)]
    # This assumes that the x flavors will represent the SUM of mu and tau flavors.
    @torch.jit.export
    def convert_y_to_2flavor(self, y):
        y2F = torch.zeros((y.shape[0],2,2,2,2), device=y.device)

        y2F[:,:,0,:,0] =                 y[:, :, 0 , :, 0 ]
        y2F[:,:,0,:,1] = 0.5 * torch.sum(y[:, :, 0 , :, 1:], dim=(  3))
        y2F[:,:,1,:,1] = 0.5 * torch.sum(y[:, :, 1:, :, 1:], dim=(2,4))
        y2F[:,:,1,:,0] =       torch.sum(y[:, :, 1:, :, 0 ], dim=(2  ))

        return y2F

    # Push the inputs through the neural network
    # output is indexed as [sim, nu/nubar(out), flavor(out), nu/nubar(in), flavor(in)]
    def forward(self,x):
        y = self.linear_activation_stack(x).reshape(x.shape[0], 2,self.NF,2,self.NF)

        # enforce conservation of particle number
        delta_nunubar = torch.eye(2, device=x.device)[None,:,None,:,None]
        yflavorsum = torch.sum(y,dim=2)[:,:,None,:,:]
        y = y + (delta_nunubar - yflavorsum) / self.NF

        # enforce symmetry in the heavies
        if self.average_heavies_in_final_state:
            y[:,:,1:,:,:] = y[:,:,1:,:,:].mean(dim=2, keepdim=True)

        # enforce eln conservation through y matrix
        if self.conserve_lepton_number == "ymatrix":
            delta_flavor = torch.eye(self.NF, device=x.device)[None,None,:,None,:]
            ELN_excess = y[:,0] - y[:,1] - (delta_nunubar[:,0]-delta_nunubar[:,1]) * delta_flavor[:,0]
            y[:,0] -= ELN_excess/2.
            y[:,1] += ELN_excess/2.

        return y
  
    # Use the initial F4 and the ML output to calculate the final F4
    # F4_initial must have shape [sim, xyzt, nu/nubar, flavor]
    # y must have shape [sim,2,NF,2,NF]
    # F4_final has shape [sim, xyzt, nu/nubar, flavor]
    # Treat all but the last flavor as output, since it is determined by conservation
    @torch.jit.export
    def F4_from_y(self, F4_initial, y):
        # tensor product with y
        # n indicates the simulation index
        # m indicates the spacetime index
        # i and j indicate nu/nubar
        # a and b indicate flavor
        F4_final = torch.einsum("niajb,nmjb->nmia", y, F4_initial)

        # enforce eln
        if self.conserve_lepton_number == "direct":
            Jt_initial = F4_initial[:,3,0,:] - F4_initial[:,3,1,:]
            Jt_final   =   F4_final[:,3,0,:] -   F4_final[:,3,1,:]
            delta_Jt = Jt_final - Jt_initial
            F4_final[:,3,0,:] = F4_final[:,3,0,:] - 0.5*delta_Jt
            F4_final[:,3,1,:] = F4_final[:,3,1,:] + 0.5*delta_Jt

        return F4_final

    @torch.jit.export
    def predict_F4(self, F4_initial):
        X = self.X_from_F4(F4_initial)
        y = self.forward(X)

        F4_final = self.F4_from_y(F4_initial, y)

        return F4_final
