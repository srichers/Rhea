'''
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This is the basic optimizer class used to optimize model parameters.
'''

import torch
from ml_loss import *
from ml_tools import *
from ml_generate import *

class Optimizer():
    def __init__(self, model,
                 op, # Adam, SGD, RMSprop
                 device): 
        self.NF = model.NF
        self.optimizer = op
        self.device = device
