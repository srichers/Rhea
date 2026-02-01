'''
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This is the file contains the functions that define the various loss sources.
'''

import torch

class LossTracker:
    def __init__(self):
        self.loss_dict = {}

    def start_epoch(self, epoch):
        self.loss_dict = {"epoch": epoch}

    # with mean, this is equivalent to torch.sum(diff**2) / F4f_pred.numel()
    def comparison_loss_fn(self, y_pred, y_true):
        return torch.nn.MSELoss(reduction='mean')(y_pred, y_true)

    def unphysical_loss_fn(self, F4f_pred, F4f_true):
        assert(F4f_true == None)

        # enforce that number density cannot be less than zero
        negative_density_error = torch.min(F4f_pred[:,:,:,3], torch.zeros_like(F4f_pred[:,:,:,3])) # [sim, nu/nubar, flavor]
        negative_density_loss = torch.mean(negative_density_error**2)

        # enforce that flux factors cannot be larger than 1
        flux_mag2 = torch.sum(F4f_pred[:,:,:,0:3]**2, dim=3) # [sim, nu/nubar, flavor]
        ndens2 = F4f_pred[:,:,:,3]**2 # [sim, nu/nubar, flavor]
        fluxfac_error = torch.max(flux_mag2 - ndens2, torch.zeros_like(ndens2)) # [sim, nu/nubar, flavor]
        fluxfac_loss = torch.mean(fluxfac_error)

        # total conservation loss
        return negative_density_loss + fluxfac_loss

    def stability_loss_fn(self, logit, stability_true):
        #print("logit min/max:", logit.min().item(), logit.max().item())

        # clamp the logits to improve numerical stability
        logit = logit.clamp(min=-10, max=10)

        # determine the relative class weights for stable/unstable points
        N_neg = torch.sum(1.0 - stability_true).item()
        N_pos = torch.sum(stability_true).item()
        
        if N_pos>0 and N_neg>0:
            pos_weight = torch.tensor([N_neg / N_pos], device=logit.device)
            result = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)(logit, stability_true)
        else:
            result = torch.nn.BCEWithLogitsLoss(reduction='mean')(logit, stability_true)

        return result

    def max_error(self, F4f_pred, F4f_true):
        if F4f_true == None:
            return 0
        else:
            return torch.max(torch.abs(F4f_pred - F4f_true))

    def _init_metrics(self):
        self.loss_dict["F4_train_loss"] = 0
        self.loss_dict["F4_train_max"] = 0
        self.loss_dict["F4_test_loss"] = 0
        self.loss_dict["F4_test_max"] = 0
        self.loss_dict["growthrate_train_loss"] = 0
        self.loss_dict["growthrate_train_max"] = 0
        self.loss_dict["growthrate_test_loss"] = 0
        self.loss_dict["growthrate_test_max"] = 0
        self.loss_dict["unphysical_train_loss"] = 0
        self.loss_dict["unphysical_train_max"] = 0
        self.loss_dict["unphysical_test_loss"] = 0
        self.loss_dict["unphysical_test_max"] = 0
        self.loss_dict["stability_train_loss"] = 0
        self.loss_dict["stability_train_max"] = 0
        self.loss_dict["stability_test_loss"] = 0
        self.loss_dict["stability_test_max"] = 0

    def _contribute_loss(self, pred, true, traintest, key, loss_fn):
        loss = loss_fn(pred, true)
        self.loss_dict[key+"_"+traintest+"_loss"] += loss.item()
        self.loss_dict[key+"_"+traintest+"_max"]  = max(self.max_error(pred, true), self.loss_dict[key+"_"+traintest+"_max"])
        return loss

    def evaluate(self,
                 parms,
                 model,
                 dataset_asymptotic_train_list,
                 dataset_asymptotic_test_list,
                 dataset_stable_train_list,
                 dataset_stable_test_list,
                 configure_loader):
        model.eval()

        self._init_metrics()

        # Asymptotic losses
        def accumulate_asymptotic_loss(dataset_list, traintest):
            total_loss = torch.tensor(0.0, requires_grad=False)
            for dataset in dataset_list:
                loader_eval = configure_loader(parms, dataset, "eval")
                F4i, F4f_true, growthrate_true = next(iter(loader_eval))
                F4i = F4i.to(parms["device"])
                F4f_true = F4f_true.to(parms["device"])
                growthrate_true = growthrate_true.to(parms["device"])

                F4f_pred, growthrate_pred, _ = model.predict_all(F4i)

                total_loss = total_loss + torch.exp(-model.log_task_weights["F4"]     ) * self._contribute_loss(F4f_pred,
                                                                                                               F4f_true,
                                                                                                               traintest, "F4", self.comparison_loss_fn)
                total_loss = total_loss + torch.exp(-model.log_task_weights["growthrate"]) * self._contribute_loss(growthrate_pred, #torch.log
                                                                                                                  growthrate_true, #torch.log
                                                                                                                  traintest, "growthrate", self.comparison_loss_fn)
                unphysical_loss = torch.exp(-model.log_task_weights["unphysical"]) * self._contribute_loss(F4f_pred,
                                                                                                           None,
                                                                                                           traintest, "unphysical", self.unphysical_loss_fn)
                if parms["do_unphysical_check"]:
                    total_loss = total_loss + unphysical_loss

            return total_loss

        with torch.no_grad():
            train_loss = accumulate_asymptotic_loss(dataset_asymptotic_train_list, "train")
            test_loss  = accumulate_asymptotic_loss(dataset_asymptotic_test_list , "test" )

        # Stability losses
        print()
        def accumulate_stable_loss(dataset_list, traintest):
            total_loss = torch.tensor(0.0, requires_grad=False)
            for dataset in dataset_list:
                loader_eval = configure_loader(parms, dataset, "eval")
                F4i, stable_true = next(iter(loader_eval))
                F4i = F4i.to(parms["device"])
                stable_true = stable_true.to(parms["device"])

                _, _, y_stable_pred = model.predict_all(F4i)

                #print(torch.sum(torch.abs(torch.sigmoid(y_stable_pred)-stable_true)).item()/y_stable_pred.shape[0],"fractional difference in stable points")

                this_loss = torch.exp(-model.log_task_weights["stability"] ) * \
                    self._contribute_loss(y_stable_pred, stable_true, traintest, "stability", self.stability_loss_fn)
                #print("  stability loss contribution:", this_loss.item())
                total_loss = total_loss + this_loss
            return total_loss

        with torch.no_grad():
            train_loss = train_loss + accumulate_stable_loss(dataset_stable_train_list, "train")
            test_loss  = test_loss  + accumulate_stable_loss(dataset_stable_test_list , "test" )

        # track the total loss
        self.loss_dict["train_loss"] = train_loss.item()
        self.loss_dict["test_loss"]  =  test_loss.item()

        # track the task weights
        for name in model.log_task_weights.keys():
            self.loss_dict["weight_"+name] = torch.exp(-model.log_task_weights[name]).item()

        return train_loss

    def training_batch_loss(self,
                            parms,
                            model,
                            F4f_pred_train,
                            growthrate_pred_train,
                            stable_pred_train,
                            F4f_true_train,
                            growthrate_true_train,
                            stable_true_train):
        # accumulate losses. NOTE - I don't use += because pytorch fails if I do. Just don't do it.
        batch_loss = 0.0
        batch_loss = batch_loss + torch.exp(-model.log_task_weights["stability"] ) * self.stability_loss_fn(stable_pred_train, stable_true_train)
        batch_loss = batch_loss + torch.exp(-model.log_task_weights["F4"]     ) * self.comparison_loss_fn(F4f_pred_train, F4f_true_train)
        batch_loss = batch_loss + torch.exp(-model.log_task_weights["growthrate"]) * self.comparison_loss_fn(growthrate_pred_train, growthrate_true_train)
        if parms["do_unphysical_check"]:
            batch_loss = batch_loss + torch.exp(-model.log_task_weights["unphysical"]) * self.unphysical_loss_fn(F4f_pred_train, None)

        # add loss weights to loss
        if parms["do_learn_task_weights"]:
            for name in model.log_task_weights.keys():
                if (not parms["do_unphysical_check"]) and name=="unphysical":
                    continue
                else:
                    batch_loss = batch_loss + torch.sum(model.log_task_weights[name])

        return batch_loss
    
