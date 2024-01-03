import torch
from ml_loss import *
from ml_tools import *
from ml_generate import *

class Optimizer():
    def __init__(self, model,
                 op, # Adam, SGD, RMSprop
                 weight_decay,
                 learning_rate,
                 device,
                 conserve_lepton_number
                 ): 
        self.NF = model.NF
        self.optimizer = op(model.parameters(), weight_decay=weight_decay, lr=learning_rate)
        self.device = device
        self.conserve_lepton_number = conserve_lepton_number

        # arrays
        self.train_error = []
        self.test_error = []
        self.train_reconstruction_error = []
        self.test_reconstruction_error = []
        self.train_unphysical_error = []
        self.test_unphysical_error = []
        self.train_stable_error = []
        self.test_stable_error = []

    # function to train the dataset
    def train(self, model, F4i, F4f_true, loss_fn):
        model.train()
        F4f_pred = model.predict_F4(F4i, conserve_lepton_number=self.conserve_lepton_number)
        loss = loss_fn(model, F4f_pred, F4f_true)
        return loss

    # function to test the model performance
    def test(self, model, F4i, F4f_true, loss_fn):
        model.eval()
        with torch.no_grad():
            F4f_pred = model.predict_F4(F4i, conserve_lepton_number=self.conserve_lepton_number)
            loss = loss_fn(model, F4f_pred, F4f_true)
            if F4f_true != None:
                error = torch.max(torch.abs(F4f_pred - F4f_true))
            else:
                error = None
            return loss, error
            
