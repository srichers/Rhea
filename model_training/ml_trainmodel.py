'''
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This is the file contains the main training loop, including accumulation of the loss function from various sources.
'''

import torch
from ml_loss import *
from ml_neuralnet import *
from ml_plot import *
from ml_tools import *
import pickle
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

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
                           p,
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
    
    def contribute_loss(epoch, p, pred, true, traintest, key, loss_fn):
        loss = loss_fn(pred, true)
        if traintest=="train":
            p.data[key].train_loss[epoch] += train_loss.item()
            p.data[key].train_err[epoch]  += max_error(pred, true)
        if traintest=="test":
            p.data[key].test_loss[epoch] = test_loss.item()
            p.data[key].test_err[epoch]  = max_error(pred, true)        
        return loss

    #===============#
    # training loop #
    #===============#
    for epoch in range(parms["epochs"]):

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

            # accumulate losses. NOTE - I don't use += because pytorch fails if I do. Just don't do it.
            train_loss = train_loss + contribute_loss(epoch, p, stable_pred_train, stable_true_train, "train", "stability", stability_loss_fn)
            test_loss  = test_loss  + contribute_loss(epoch, p, stable_pred_test , stable_true_test , "test" , "stability", stability_loss_fn)
            
            # train on making sure the model prediction is correct [ndens]
            train_loss = train_loss + contribute_loss(epoch, p, ndens_pred_train, ndens_true_train, "train", "ndens", comparison_loss_fn)
            test_loss  = test_loss  + contribute_loss(epoch, p, ndens_pred_test , ndens_true_test , "test" , "ndens", comparison_loss_fn)
            
            # train on making sure the model prediction is correct [fluxmag]
            train_loss = train_loss + contribute_loss(epoch, p, fluxmag_pred_train, fluxmag_true_train, "train", "fluxmag", comparison_loss_fn)
            test_loss  = test_loss  + contribute_loss(epoch, p, fluxmag_pred_test , fluxmag_true_test , "test" , "fluxmag", comparison_loss_fn)
            
            # train on making sure the model prediction is correct [direction]
            train_loss = train_loss + contribute_loss(epoch, p, Fhat_pred_train, Fhat_true_train, "train", "direction", direction_loss_fn)
            test_loss  = test_loss  + contribute_loss(epoch, p, Fhat_pred_test , Fhat_true_test , "test" , "direction", direction_loss_fn)

            # train on making sure the model prediction is correct [growthrate]
            train_loss = train_loss + 0.01 * contribute_loss(epoch, p, growthrate_pred_train/ntot_train, growthrate_true_train/ntot_train, "train", "growthrate", comparison_loss_fn)
            test_loss  = test_loss  + 0.01 * contribute_loss(epoch, p, growthrate_pred_test /ntot_test , growthrate_true_test /ntot_test , "test" , "growthrate", comparison_loss_fn)
            
            # unphysical. Have experienced heavy over-training in the past if not regenerated every iteration
            if parms["do_unphysical_check"]:
                train_loss = train_loss + 100 * contribute_loss(epoch, p, F4f_pred_train, None, "train", "unphysical", unphysical_loss_fn)
                test_loss  = test_loss  + 100 * contribute_loss(epoch, p, F4f_pred_test , None, "test" , "unphysical", unphysical_loss_fn)
    
        # track the total loss
        p.data["loss"].train_loss[epoch] = train_loss.item()
        p.data["loss"].test_loss[epoch]  =  test_loss.item()
        

        # have the optimizer take a step
        train_loss.backward()
        optimizer.step()
        if epoch>=parms["warmup_iters"]:
            scheduler.step(train_loss.item())
        else:
            scheduler.step()

        # report max error
        if((epoch+1)%parms["print_every"]==0):
            print(f"Epoch {epoch+1}")
            print("lr =",scheduler._last_lr)
            print("net loss =", train_loss.item())
            print("ELN violation: ",ELN_violation.item())
            for key in p.data.keys():
                print("{:<15} {:<18} {:<15}".format(key, np.sqrt(p.data[key].train_loss[epoch]),  np.sqrt(p.data[key].test_loss[epoch]) ))
            print("", flush=True)

        if((epoch+1)%parms["output_every"]==0):
            outfilename = "model"+str(epoch+1)
            save_model(model, outfilename, "cpu", F4i_asymptotic_test)

            # pickle the model, optimizer, scheduler, and plotter
            with open("model_epoch"+str(epoch+1)+".pkl", "wb") as f:
                pickle.dump([model, optimizer, scheduler, p], f)

            p.plot_error("train_test_error.pdf", ymin=1e-5)
            

    return model, optimizer, scheduler, p
