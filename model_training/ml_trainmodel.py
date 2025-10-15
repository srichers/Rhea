'''
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This is the file contains the main training loop, including accumulation of the loss function from various sources.
'''

import torch
from ml_loss import *
from ml_neuralnet import *
from ml_tools import *
import pickle
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

# create an empty dictionary that will eventually contain all of the loss metrics of an iteration
loss_dict = {}

def configure_loader(parms, dataset_train_list, dataset_test_list):
    assert(len(dataset_train_list) == len(dataset_test_list))

    # create list of weights
    weights = []
    for dataset in dataset_train_list:
        nsamples = len(dataset)
        weights.extend([1.0/nsamples] * nsamples)
    
    # combine the datasets
    dataset_train = ConcatDataset(dataset_train_list)
    dataset_test  = ConcatDataset(dataset_test_list )
    
    # create sampler and data loader for test data
    sampler = WeightedRandomSampler(weights=weights, num_samples=parms["epoch_num_samples"], replacement=True)
    loader = DataLoader(dataset_train, batch_size=parms["batch_size"], sampler=sampler)

    print("Configuring loader with num_samples=",parms["epoch_num_samples"],"and batch_size=",parms["batch_size"],"for a dataset with",len(dataset_train),"samples.")

    return loader, dataset_test


def train_asymptotic_model(parms,
                           model,
                           optimizer,
                           scheduler,
                           dataset_asymptotic_train_list,
                           dataset_asymptotic_test_list,
                           dataset_stable_train_list,
                           dataset_stable_test_list):

    loader_asymptotic, dataset_asymptotic_test = configure_loader(parms, dataset_asymptotic_train_list, dataset_asymptotic_test_list)
    loader_stable    , dataset_stable_test     = configure_loader(parms, dataset_stable_train_list    , dataset_stable_test_list    )

    # separate out test data
    F4i_asymptotic_test  = torch.cat([ds.tensors[0] for ds in dataset_asymptotic_test.datasets], dim=0).to(parms["device"])
    F4f_true_test        = torch.cat([ds.tensors[1] for ds in dataset_asymptotic_test.datasets], dim=0).to(parms["device"])
    growthrate_true_test = torch.cat([ds.tensors[2] for ds in dataset_asymptotic_test.datasets], dim=0).to(parms["device"])
    F4i_stable_test      = torch.cat([ds.tensors[0] for ds in dataset_stable_test.datasets], dim=0).to(parms["device"])
    stable_true_test   = torch.cat([ds.tensors[1] for ds in dataset_stable_test.datasets], dim=0).to(parms["device"])
    ntot_test = ntotal(F4i_asymptotic_test)
    
    def contribute_loss(pred, true, traintest, key, loss_fn):
        loss = loss_fn(pred, true)
        loss_dict[key+"_"+traintest+"_loss"] = train_loss.item()
        loss_dict[key+"_"+traintest+"_max"]  = max_error(pred, true)
        return loss

    # set up file for writing performance metrics
    loss_file = open("loss.dat","w")
    
    #===============#
    # training loop #
    #===============#
    for epoch in range(1,parms["epochs"]+1):
        loss_dict["epoch"] = epoch

        # zero the gradients
        optimizer.zero_grad()
        train_loss = torch.tensor(0.0, requires_grad=True)
        test_loss  = torch.tensor(0.0, requires_grad=False)

        # predict test values
        model.eval()
        F4f_pred_test, growthrate_pred_test, _                = model.predict_all(F4i_asymptotic_test)
        _            , _                   , stable_pred_test = model.predict_all(F4i_stable_test    )

        # loop over batches
        assert(len(loader_asymptotic)==len(loader_stable))
        nbatches = len(loader_asymptotic)
        for i in range(len(loader_asymptotic)):
            # get true values from data loader
            F4i_asymptotic_train, F4f_true_train, growthrate_true_train = next(iter(loader_asymptotic))
            F4i_stable_train, stable_true_train = next(iter(loader_stable))

            # move the minibatch to the device
            F4i_asymptotic_train = F4i_asymptotic_train.to(parms["device"])
            F4f_true_train = F4f_true_train.to(parms["device"])
            growthrate_true_train = growthrate_true_train.to(parms["device"])
            F4i_stable_train = F4i_stable_train.to(parms["device"])
            stable_true_train = stable_true_train.to(parms["device"])

            # get predicted values from the model
            model.train()
            F4f_pred_train, growthrate_pred_train, _                 = model.predict_all(F4i_asymptotic_train)
            _           , _                      , stable_pred_train = model.predict_all(F4i_stable_train    )

            # convert F4 to densities and fluxes to feed to loss functions
            # note the outputs are all normalized to the total number density
            ndens_pred_train, fluxmag_pred_train, Fhat_pred_train = get_ndens_fluxmag_fhat(F4f_pred_train)
            ndens_pred_test , fluxmag_pred_test , Fhat_pred_test  = get_ndens_fluxmag_fhat(F4f_pred_test )
            ndens_true_train, fluxmag_true_train, Fhat_true_train = get_ndens_fluxmag_fhat(F4f_true_train)
            ndens_true_test , fluxmag_true_test , Fhat_true_test  = get_ndens_fluxmag_fhat(F4f_true_test )

            # calculate ELN violation for printout later
            ntot_train = ntotal(F4i_asymptotic_train)
            ELN_initial = F4i_asymptotic_train[:,3,0,:] - F4i_asymptotic_train[:,3,1,:]
            ELN_final   =       F4f_pred_train[:,3,0,:] -       F4f_pred_train[:,3,1,:]
            ELN_violation = torch.max(torch.abs(ELN_final-ELN_initial) / ntot_train[:,None])
            loss_dict["ELN_violation"] = ELN_violation.item()

            # accumulate losses. NOTE - I don't use += because pytorch fails if I do. Just don't do it.
            train_loss = train_loss + contribute_loss(stable_pred_train, stable_true_train, "train", "stability", stability_loss_fn)
            test_loss  = test_loss  + contribute_loss(stable_pred_test , stable_true_test , "test" , "stability", stability_loss_fn)
            
            # train on making sure the model prediction is correct [ndens]
            train_loss = train_loss + contribute_loss(ndens_pred_train, ndens_true_train, "train", "ndens", comparison_loss_fn)
            test_loss  = test_loss  + contribute_loss(ndens_pred_test , ndens_true_test , "test" , "ndens", comparison_loss_fn)
            
            # train on making sure the model prediction is correct [fluxmag]
            train_loss = train_loss + contribute_loss(fluxmag_pred_train, fluxmag_true_train, "train", "fluxmag", comparison_loss_fn)
            test_loss  = test_loss  + contribute_loss(fluxmag_pred_test , fluxmag_true_test , "test" , "fluxmag", comparison_loss_fn)
            
            # train on making sure the model prediction is correct [direction]
            train_loss = train_loss + contribute_loss(Fhat_pred_train, Fhat_true_train, "train", "direction", direction_loss_fn)
            test_loss  = test_loss  + contribute_loss(Fhat_pred_test , Fhat_true_test , "test" , "direction", direction_loss_fn)

            # train on making sure the model prediction is correct [growthrate]
            train_loss = train_loss + 0.01 * contribute_loss(growthrate_pred_train/ntot_train, growthrate_true_train/ntot_train, "train", "growthrate", comparison_loss_fn)
            test_loss  = test_loss  + 0.01 * contribute_loss(growthrate_pred_test /ntot_test , growthrate_true_test /ntot_test , "test" , "growthrate", comparison_loss_fn)
            
            # unphysical. Have experienced heavy over-training in the past if not regenerated every iteration
            if parms["do_unphysical_check"]:
                train_loss = train_loss + 100 * contribute_loss(F4f_pred_train, None, "train", "unphysical", unphysical_loss_fn)
                test_loss  = test_loss  + 100 * contribute_loss(F4f_pred_test , None, "test" , "unphysical", unphysical_loss_fn)
    
        # track the total loss
        loss_dict["train_loss"] = train_loss.item()
        loss_dict["test_loss"]  =  test_loss.item()        

        # have the optimizer take a step
        train_loss.backward()
        optimizer.step()
        loss_dict["learning_rate"] = scheduler.get_last_lr()[0]
        if epoch>parms["warmup_iters"]:
            scheduler.step(train_loss.item())
        else:
            scheduler.step()

        # print headers
        if epoch==1:
            for k,i in zip(loss_dict.keys(), range(len(loss_dict.keys()))):
                loss_file.write(("{:d}:"+k+"\t").format(i+1))
            loss_file.write('\n')

        # print loss values
        for k in loss_dict.keys():
            if k=="epoch":
                loss_file.write("{:<12d}".format(loss_dict[k]))
            else:
                loss_file.write("{:<12.3e}\t".format(loss_dict[k]))
        loss_file.write('\n')

        # output
        if(epoch%parms["output_every"]==0):
            outfilename = "model"+str(epoch)
            save_model(model, outfilename, "cpu", F4i_asymptotic_test)
            print("Saved",outfilename, flush=True)

    return
