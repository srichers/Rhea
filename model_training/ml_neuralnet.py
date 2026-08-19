'''
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This file contains the structure of the neural network, including the means of transforming between angular moments and the actual inputs/outputs of the ML model
'''

import torch
import numpy as np
from torch import nn
from ml_tools import ntotal
import e3nn.o3
import e3nn.nn
import box3d

# generate the set of permutation equivariant inputs for each node
def PE_inputs(x):
        # compute the inputs to each node
        # subtract self to get the messages from neighbors orthogonal to the self node
        x_self = x
        x_flavor  = x.sum(dim=2, keepdim=True) - x_self
        x_nunubar = x.sum(dim=1, keepdim=True) - x_self
        x_all = x.sum(dim=(1,2), keepdim=True) - x_self - x_flavor - x_nunubar

        # normalize by the number of nodes contributing. Supposedly helps keep the scale of the activations reasonable.
        x_flavor  = x_flavor / (x.shape[2] - 1)
        x_nunubar = x_nunubar / (x.shape[1] - 1)
        x_all     = x_all / ((x.shape[1] - 1) * (x.shape[2] - 1))

        return x_self, x_flavor, x_nunubar, x_all

# permutation equivariant linear map.
# All of this block's nonlinearity comes from the Gate that follows it in
# PE_ResidualGatedBlock; this layer only does the per-branch equivariant linear mixing (and,
# for transition layers, the irreps_in -> irreps_out projection).
class PETP_Linear(nn.Module):
    def __init__(self, irreps_in, irreps_out):
        super().__init__()
        self.lin_self    = e3nn.o3.Linear(irreps_in, irreps_out)
        self.lin_flavor  = e3nn.o3.Linear(irreps_in, irreps_out)
        self.lin_nunubar = e3nn.o3.Linear(irreps_in, irreps_out)
        self.lin_all     = e3nn.o3.Linear(irreps_in, irreps_out)

    # x has shape [nsamples, nunubar, flavor, features]
    def forward(self, x):
        # compute the inputs to each node
        x_self, x_flavor, x_nunubar, x_all = PE_inputs(x)

        # apply the linear maps
        y_self = self.lin_self(x_self)
        y_flavor = self.lin_flavor(x_flavor)
        y_nunubar = self.lin_nunubar(x_nunubar)
        y_all = self.lin_all(x_all)

        # sum the contributions
        y = y_self + y_flavor + y_nunubar + y_all
        return y

# permutation equivariant tensor product between irreps and a bounded rescaling of themselves.
# NormActivation computes x/|x| * tanh(|x|): for large |x| the tanh saturates, so the second argument becomes
# scale-invariant and the product's growth is capped back to degree 1; for small |x| it reduces to x/|x| * |x| = x.
class PETP_Quadratic(nn.Module):
    def __init__(self, irreps_in, irreps_out):
        super().__init__()
        self.norm_act   = e3nn.nn.NormActivation(irreps_in, torch.tanh, normalize=True)
        self.tp_self    = e3nn.o3.FullyConnectedTensorProduct(irreps_in, irreps_in, irreps_out)
        self.tp_flavor  = e3nn.o3.FullyConnectedTensorProduct(irreps_in, irreps_in, irreps_out)
        self.tp_nunubar = e3nn.o3.FullyConnectedTensorProduct(irreps_in, irreps_in, irreps_out)
        self.tp_all     = e3nn.o3.FullyConnectedTensorProduct(irreps_in, irreps_in, irreps_out)

    # x has shape [nsamples, nunubar, flavor, features]
    def forward(self, x):
        # compute the inputs to each node
        x_self, x_flavor, x_nunubar, x_all = PE_inputs(x)

        # compute tensor products against a bounded rescaling of the same tensor
        y_self = self.tp_self(x_self, self.norm_act(x_self))
        y_flavor = self.tp_flavor(x_flavor, self.norm_act(x_flavor))
        y_nunubar = self.tp_nunubar(x_nunubar, self.norm_act(x_nunubar))
        y_all = self.tp_all(x_all, self.norm_act(x_all))

        # sum the contributions
        y = y_self + y_flavor + y_nunubar + y_all
        return y

class PE_ResidualGatedBlock(nn.Module):
    def __init__(self, irreps_in, irreps_out, act_scalars, act_gates, tensor_product_class, dropout_probability=0):
        super().__init__()
        # determine the irreps that need to go into gate
        self.irreps_out = irreps_out
        irreps_scalars = irreps_out.filter(lambda mul_ir: mul_ir.ir.l == 0)
        irreps_nonscalars = irreps_out.filter(lambda mul_ir: mul_ir.ir.l > 0)

        # Gate needs at least one nonscalar irrep to gate - its ElementwiseTensorProduct over
        # an empty pair fails. With none (e.g. growthrate's 1x0e output), there is nothing to
        # gate, so just activate the scalars directly instead.
        if len(irreps_nonscalars) > 0:
            irreps_gates = e3nn.o3.Irreps(f"{irreps_nonscalars.num_irreps}x0e")
            self.tp = tensor_product_class(irreps_in, irreps_scalars + irreps_gates + irreps_nonscalars)
            self.gate = e3nn.nn.Gate(
                irreps_scalars = irreps_scalars,
                act_scalars = [act_scalars] * len(irreps_scalars),
                irreps_gates = irreps_gates,
                act_gates = [act_gates] * len(irreps_gates),
                irreps_gated = irreps_nonscalars,
            )
        else:
            self.tp = tensor_product_class(irreps_in, irreps_scalars)
            self.gate = e3nn.nn.Activation(irreps_scalars, [act_scalars] * len(irreps_scalars))

        # e3nn's Dropout drops whole irreps rather than individual components, and its mask has
        # shape [sim, multiplicity], so it broadcasts over the nu/nubar and flavor axes. Both
        # rotational and flavor-permutation equivariance therefore survive it. torch.nn.Dropout
        # would break both - do not substitute it.
        self.dropout = e3nn.nn.Dropout(irreps_out, p=dropout_probability) if dropout_probability > 0 else nn.Identity()

        # When irreps_in != irreps_out there is no valid x + y to form, so this falls back to a
        # learned linear projection of x (a "projection shortcut", the standard ResNet answer to
        # a shape-changing residual block) instead of dropping the residual connection entirely.
        if irreps_in == irreps_out:
            self.skip = nn.Identity()
        else:
            self.skip = e3nn.o3.Linear(irreps_in, irreps_out)

        # ReZero (Bachlechner et al. 2020): a learnable per-block scale on the residual branch,
        # initialized to zero. Starting alpha at 0 makes every block equal to its skip path at
        # init regardless of depth, so training starts as if the stack had zero extra blocks and
        # each block's contribution grows only as its own alpha moves away from zero. alpha is a
        # single scalar multiplying y uniformly, so it commutes with rotation and flavor
        # permutation and does not disturb equivariance.
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # get the full output (scalars + gates + nonscalars) from one linear
        y = self.tp(x)

        # apply the gate.
        y = self.gate(y)

        # dropout goes on the residual branch only, so the skip path stays clean
        y = self.dropout(y)

        return self.skip(x) + self.alpha * y
    
# define the NN model
class NeuralNetwork(nn.Module):
    def __init__(self, parms):
        super().__init__()

        # The trunk and the two heads each have their own width. 
        # scalars must come first in every branch, to stay consistent with the output of Gate
        for key in ["irreps_shared","irreps_F4","irreps_growthrate"]:
            assert parms[key] == parms[key].sort().irreps, key+" must have scalars first"

        # store input arguments
        self.NF = parms["NF"]

        # The input irreps for each node are just the 4 components of F4_in followed by 4 components of F4_box3d
        self.irreps_in  = e3nn.o3.Irreps("1x1o + 1x0e + 1x1o + 1x0e")
        
        # one y matrix for ndens, one for flux
        # add one extra to predict growth rate
        self.average_heavies_in_final_state = parms["average_heavies_in_final_state"]
        self.conserve_lepton_number = parms["conserve_lepton_number"]

        # append a full layer including linear, activation, and dropout
        def append_full_layer(modules, in_irreps, out_irreps, dropout_probability):

            # select the type of tensor product to use in the gate
            if parms["tensor_product_class"] == "linear":
                tensor_product_class = PETP_Linear
            elif parms["tensor_product_class"] == "quadratic":
                tensor_product_class = PETP_Quadratic
            else:
                assert False, f"Unknown tensor product class {parms['tensor_product_class']}"

            modules.append(PE_ResidualGatedBlock(
                in_irreps,
                out_irreps,
                parms["scalar_activation"   ],
                parms["nonscalar_activation"],
                tensor_product_class,
                dropout_probability
            ))


        # set up shared layers
        def build_shared(modules_shared):
            assert(parms["nhidden_shared"] >= 1), "Number of shared hidden layers must be positive"
            irreps_shared       = parms["irreps_shared"]
            dropout_probability = parms["dropout_shared"]
            append_full_layer(modules_shared, self.irreps_in, irreps_shared, dropout_probability)
            for _ in range(parms["nhidden_shared"]-1):
                append_full_layer(modules_shared, irreps_shared, irreps_shared, dropout_probability)

        # set up task-specific layers. nhidden_task is the exact number of layers: the first
        # layer's input is always irreps_shared (from the trunk) and the last layer's output is
        # always irreps_final (the task's output structure), regardless of nhidden_task - with
        # nhidden_task==1 those are the same layer, and irreps_task goes unused.
        def build_task(modules_task, nhidden_task, irreps_task, irreps_final, dropout_probability):
            assert nhidden_task >= 1, "Number of hidden layers must be positive"
            for i in range(nhidden_task):
                in_irreps  = parms["irreps_shared"] if i == 0 else irreps_task
                out_irreps = irreps_final if i == nhidden_task-1 else irreps_task
                append_full_layer(modules_task, in_irreps, out_irreps, dropout_probability)

        # put together the layers of the neural network
        modules_shared = []
        modules_growthrate = []
        modules_F4 = []
        build_shared(modules_shared)
        build_task(modules_growthrate, parms["nhidden_growthrate"], parms["irreps_growthrate"], e3nn.o3.Irreps("1x0e"       ), parms["dropout_growthrate"])
        build_task(modules_F4,         parms["nhidden_F4"],         parms["irreps_F4"        ], e3nn.o3.Irreps("1x1o + 1x0e"), parms["dropout_F4"        ])

        # turn the list of modules into a sequential model
        self.TP_activation_stack_shared     = nn.Sequential(*modules_shared)
        self.TP_activation_stack_growthrate = nn.Sequential(*modules_growthrate)
        self.TP_activation_stack_F4         = nn.Sequential(*modules_F4)

        # register lebedev quadrature tensors as buffers so they are scriptable and move with the model
        self.register_buffer('lebedev_pts', box3d.pts)
        self.register_buffer('lebedev_weights', box3d.weights)

        # initialize the weights
        torch.manual_seed(parms["random_seed"])
        np.random.seed(parms["random_seed"])

        # print the model structure
        print(self)

    # convert the 3-flavor matrix into an effective 2-flavor matrix
    # input and output are indexed as [sim, nu/nubar, flavor, xyzt]
    # This assumes that the x flavors will represent the SUM of mu and tau flavors.
    @torch.jit.export
    def convert_y_to_2flavor(self, y):
        y2F = torch.zeros((y.shape[0],2,2,4), device=y.device)

        y2F[:,:,0,:] = y[:,:,0,:]
        y2F[:,:,1,:] = torch.sum(y[:,:,1:,:], dim=(2))

        return y2F

    # Push the inputs through the neural network
    # output is indexed as [sim, nu/nubar(out), flavor(out), nu/nubar(in), flavor(in)]
    def forward(self,x):

        # evaluate the shared portion of the network
        y_shared = self.TP_activation_stack_shared(x)

        # evaluate each task
        y_growthrate = self.TP_activation_stack_growthrate(y_shared)
        y_F4         = self.TP_activation_stack_F4(y_shared)

        #assert torch.all(torch.isfinite(y_stability))
        #assert torch.all(torch.isfinite(y_growthrate))
        #assert torch.all(torch.isfinite(y_F4))

        return y_F4, y_growthrate

    # X is just F4_initial [nsamples, 2, NF, xyzt]
    # use_network=False returns the analytic box3d result with no learned correction
    # (the type hint is required by torch.jit.script, which otherwise assumes Tensor)
    @torch.jit.export
    def predict_all(self, F4_in, use_network: bool = True):

        # get the total density
        nsims = F4_in.shape[0]
        F4_in = F4_in.view((nsims,2,self.NF,4))
        ntot = ntotal(F4_in) # [nsims]

        # normalize the inputs by the total density
        F4_in = F4_in / ntot[:,None,None,None]

        # get the Box3D change to F4 (pass lebedev pts/weights registered as buffers)
        dF4_box3d, growthrate_box3d = box3d.mixBox3D_lebedev(F4_in, self.lebedev_pts, self.lebedev_weights)

        if use_network:
            # Combine F4_in and the Box3D change into a joint input of shape [nsamples, nu/nubar, flavor, xyzt(in)/xyzt(box3d)]
            F4_joint = torch.cat([F4_in, dF4_box3d], dim=-1)

            # propagate through the network
            y_F4, y_growthrate = self.forward(F4_joint)

            # pool over features to get permutation-invariant output
            # Averaging over nodes in the graph
            y_growthrate = torch.mean(y_growthrate, dim=(1,2))

            # Box3D and the network both supply a correction to the input
            F4_out     = F4_in + dF4_box3d + y_F4.reshape((nsims,2,self.NF,4))
            growthrate = growthrate_box3d + torch.squeeze(y_growthrate)
        else:
            # clone because the conservation enforcement below is in-place
            F4_out     = (F4_in + dF4_box3d).clone()
            growthrate = growthrate_box3d

        # enforce symmetry in the heavies
        if self.average_heavies_in_final_state:
            F4_out[:,:,1:,:] = F4_out.clone().detach()[:,:,1:,:].mean(dim=2, keepdim=True)

        # ensure the flavor-traced number is conserved
        F4_in = F4_in.view((nsims,2,self.NF,4))
        F4_in_flavortrace = torch.sum(F4_in, dim=2, keepdim=True)
        F4_out_flavortrace = torch.sum(F4_out, dim=2, keepdim=True)
        F4_out_excess = F4_out_flavortrace - F4_in_flavortrace
        F4_out[:,:,:,:] -= F4_out_excess / self.NF

        # ensure that ELN is conserved
        # xyzt index 3 is the number density. Correcting a flux component
        # instead would not be rotationally equivariant.
        if self.conserve_lepton_number:
            ELN_in  = F4_in[:,0,:,3]  - F4_in[:,1,:,3]
            ELN_out = F4_out[:,0,:,3] - F4_out[:,1,:,3]
            ELN_excess = ELN_out - ELN_in
            F4_out[:,0,:,3] -= ELN_excess / 2.0
            F4_out[:,1,:,3] += ELN_excess / 2.0

        # rescale F4_out to the original total density
        F4_out = F4_out * ntot[:,None,None,None]

        # return growthrate in the same units (number density)
        # Multiplying by sqrt(2) G_F not done here because it causes issues with single-precision floats
        growthrate = growthrate * ntot

        #assert torch.all(torch.isfinite(F4_out))
        #assert torch.all(torch.isfinite(growthrate))
        #assert torch.all(torch.isfinite(stability))

        # stability is determined only by box3d, output as float
        stability = (growthrate_box3d <= 0).float()

        return F4_out, growthrate, stability
