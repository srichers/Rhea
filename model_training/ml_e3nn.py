'''
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This file contains e3nn-specific network building blocks.
'''

from torch import nn
import e3nn.o3
import e3nn.nn

# define a permutation-equivariant message passing layer
# 0 means flavor edge, 1 means neutrino/antineutrino edge
class PermutationEquivariantLinear(nn.Module):
    def __init__(self, irreps_in, irreps_out, do_batchnorm=False):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.batchnorm = e3nn.nn.BatchNorm(irreps_in) if do_batchnorm else None
        self.lin_self    = e3nn.o3.Linear(irreps_in, irreps_out)
        self.lin_flavor  = e3nn.o3.Linear(irreps_in, irreps_out)
        self.lin_nunubar = e3nn.o3.Linear(irreps_in, irreps_out)
        self.lin_all     = e3nn.o3.Linear(irreps_in, irreps_out)

    # x has shape [nsamples, nunubar, flavor, features]
    def forward(self, x):

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

        # optional per-irrep normalization before the linear maps
        if self.batchnorm is not None:
            x_self = self.batchnorm(x_self)
            x_flavor = self.batchnorm(x_flavor)
            x_nunubar = self.batchnorm(x_nunubar)
            x_all = self.batchnorm(x_all)

        # compute the outputs from each part
        y_self = self.lin_self(x_self)
        y_flavor = self.lin_flavor(x_flavor)
        y_nunubar = self.lin_nunubar(x_nunubar)
        y_all = self.lin_all(x_all)

        # sum the contributions, using the residual connection when possible
        y = y_self + y_flavor + y_nunubar + y_all
        return y

class PermutationEquivariantScalarTensorProduct(nn.Module):
    def __init__(self, irreps_in, irreps_out, do_batchnorm=False):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.batchnorm = e3nn.nn.BatchNorm(irreps_in) if do_batchnorm else None
        self.norm = e3nn.o3.Norm(irreps_in)
        self.tp_self    = e3nn.o3.FullyConnectedTensorProduct(irreps_in, self.norm.irreps_out, irreps_out)
        self.tp_flavor  = e3nn.o3.FullyConnectedTensorProduct(irreps_in, self.norm.irreps_out, irreps_out)
        self.tp_nunubar = e3nn.o3.FullyConnectedTensorProduct(irreps_in, self.norm.irreps_out, irreps_out)
        self.tp_all     = e3nn.o3.FullyConnectedTensorProduct(irreps_in, self.norm.irreps_out, irreps_out)

    # x has shape [nsamples, nunubar, flavor, features]
    def forward(self, x):
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

        # optional per-irrep normalization before the tensor product
        if self.batchnorm is not None:
            x_self = self.batchnorm(x_self)
            x_flavor = self.batchnorm(x_flavor)
            x_nunubar = self.batchnorm(x_nunubar)
            x_all = self.batchnorm(x_all)

        # compute invariant scalars and apply scalar-gated tensor products
        s_self = self.norm(x_self)
        s_flavor = self.norm(x_flavor)
        s_nunubar = self.norm(x_nunubar)
        s_all = self.norm(x_all)
        y_self = self.tp_self(x_self, s_self)
        y_flavor = self.tp_flavor(x_flavor, s_flavor)
        y_nunubar = self.tp_nunubar(x_nunubar, s_nunubar)
        y_all = self.tp_all(x_all, s_all)

        # sum the contributions, using the residual connection when possible
        y = y_self + y_flavor + y_nunubar + y_all
        return y

class PermutationEquivariantGatedBlock(nn.Module):
    def __init__(self, irreps_in, irreps_out, act_scalars, act_gates, do_batchnorm=False):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        irreps_scalars = irreps_out.filter(lambda mul_ir: mul_ir.ir.l == 0)
        irreps_nonscalars = irreps_out.filter(lambda mul_ir: mul_ir.ir.l > 0)
        irreps_gates = e3nn.o3.Irreps(f"{irreps_nonscalars.num_irreps}x0e")
        irreps_with_gates = irreps_scalars + irreps_gates + irreps_nonscalars

        self.lin = PermutationEquivariantScalarTensorProduct(
            irreps_in,
            irreps_with_gates,
            do_batchnorm=do_batchnorm,
        )

        self.gate = e3nn.nn.Gate(
            irreps_scalars = irreps_scalars,
            act_scalars = [act_scalars] * len(irreps_scalars),
            irreps_gates = irreps_gates,
            act_gates = [act_gates] * len(irreps_gates),
            irreps_gated = irreps_nonscalars,
        )

    def forward(self, x):
        # get the full output (scalars + gates + nonscalars) from one linear
        y = self.lin(x)

        # apply the gate. If input and output irreps are the same, add residual connection
        y = self.gate(y)
        #TODO implement residual connection - the below fails during scripting
        #if self.irreps_in == self.irreps_out:
        #    y = x + y
        return y
