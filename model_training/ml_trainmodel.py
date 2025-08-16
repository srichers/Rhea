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

def train_asymptotic_model(parms,
                           model,
                           optimizer,
                           scheduler,
                           p,
                           F4s_train,
                           F4s_test,
                           stable_train,
                           stable_test,
                           F4i_train,
                           F4i_test,
                           F4f_train,
                           F4f_test,
                           logGrowthRate_train,
                           logGrowthRate_test):
    
    # calculate initial number densities
    ntot_train = torch.sum(F4i_train[:,3,:,:], dim=(1,2))
    ntot_test = torch.sum(F4i_test[:,3,:,:], dim=(1,2))
    logntot_train = torch.log(ntot_train)
    logntot_test = torch.log(ntot_test)

    #===============#
    # training loop #
    #===============#
    for t in range(parms["epochs"]):

        # zero the gradients
        optimizer.zero_grad()
        loss = torch.tensor(0.0, requires_grad=True)
        test_loss = torch.tensor(0.0, requires_grad=False)

        def contribute_loss(F4pred_train, F4f_train, F4pred_test, F4f_test, key, loss_fn):
            train_loss = loss_fn(F4pred_train, F4f_train)
            p.data[key].train_loss[t] = train_loss.item()
            p.data[key].train_err[t]  = max_error(F4pred_train, F4f_train)
            
            test_loss = loss_fn(F4pred_test, F4f_test)
            p.data[key].test_loss[t] = test_loss.item()
            p.data[key].test_err[t]  = max_error(F4pred_test, F4f_test)

            return train_loss, test_loss


        # evaluate the training and test datasets
        model.eval()
        F4pred_test, logGrowthRate_pred_test, _  = model.predict_all(F4i_test)
        _, _, stable_pred_test = model.predict_all(F4s_test)
        model.train()
        F4pred_train, logGrowthRate_pred_train, _ = model.predict_all(F4i_train)
        _, _, stable_pred_train = model.predict_all(F4s_train)

        # convert F4 to densities and fluxes to feed to loss functions
        # note the outputs are all normalized to the total number density
        ndens_pred_train, fluxmag_pred_train, Fhat_pred_train = get_ndens_fluxmag_fhat(F4pred_train)
        ndens_pred_test , fluxmag_pred_test , Fhat_pred_test  = get_ndens_fluxmag_fhat(F4pred_test )
        ndens_true_train, fluxmag_true_train, Fhat_true_train = get_ndens_fluxmag_fhat(F4f_train   )
        ndens_true_test , fluxmag_true_test , Fhat_true_test  = get_ndens_fluxmag_fhat(F4f_test    )

        # calculate ELN violation for printout later
        ELN_initial =    F4i_train[:,3,0,:] -    F4i_train[:,3,1,:]
        ELN_final   = F4pred_train[:,3,0,:] - F4pred_train[:,3,1,:]
        ELN_violation = torch.max(torch.abs(ELN_final-ELN_initial) / ntot_train[:,None])
        
        # accumulate losses. NOTE - I don't use += because pytorch fails if I do. Just don't do it.
        loss_stability, test_loss_stability = contribute_loss(stable_pred_train,
                                                              stable_train,
                                                              stable_pred_test,
                                                              stable_test,
                                                              "stability", stability_loss_fn)
        
        # train on making sure the model prediction is correct [ndens]
        loss_ndens, test_loss_ndens = contribute_loss(ndens_pred_train,
                                                      ndens_true_train,
                                                      ndens_pred_test,
                                                      ndens_true_test,
                                                      "ndens", comparison_loss_fn)
        loss = loss + loss_ndens
        test_loss = test_loss + test_loss_ndens

        # train on making sure the model prediction is correct [fluxmag]
        loss_fluxmag, test_loss_fluxmag = contribute_loss(fluxmag_pred_train,
                                                          fluxmag_true_train,
                                                          fluxmag_pred_test,
                                                          fluxmag_true_test,
                                                          "fluxmag", comparison_loss_fn)
        loss = loss + loss_fluxmag
        test_loss = test_loss + test_loss_fluxmag

        # train on making sure the model prediction is correct [direction]
        loss_direction, test_loss_direction = contribute_loss(Fhat_pred_train,
                                                              Fhat_true_train,
                                                              Fhat_pred_test,
                                                              Fhat_true_test,
                                                              "direction", direction_loss_fn)
        loss = loss + loss_direction
        test_loss = test_loss + test_loss_direction

        # train on making sure the model prediction is correct [growthrate]
        loss_growthrate, test_loss_growthrate = contribute_loss(logGrowthRate_pred_train - logntot_train,
                                                                logGrowthRate_train      - logntot_train,
                                                                logGrowthRate_pred_test  - logntot_test,
                                                                logGrowthRate_test       - logntot_test,
                                                                "logGrowthRate", comparison_loss_fn)
        loss = loss + loss_growthrate * 0.01
        test_loss = test_loss + test_loss_growthrate * 0.01
        
        # unphysical. Have experienced heavy over-training in the past if not regenerated every iteration
        loss_unphysical, test_loss_unphysical = contribute_loss(F4pred_train,
                                                                None,
                                                                F4pred_test,
                                                                None,
                                                                "unphysical",
                                                                unphysical_loss_fn)
        if parms["do_unphysical_check"]:
            loss = loss + loss_unphysical * 100
            test_loss = test_loss + test_loss_unphysical * 100

        # track the total loss
        p.data["loss"].train_loss[t] = loss.item()
        p.data["loss"].test_loss[t] = test_loss.item()
        

        # have the optimizer take a step
        scheduler.step(loss.item())
        loss.backward()
        optimizer.step()

        # report max error
        if((t+1)%parms["print_every"]==0):
            print(f"Epoch {t+1}")
            print("lr =",scheduler._last_lr)
            print("net loss =", loss.item())
            print("ELN violation: ",ELN_violation.item())
            for key in p.data.keys():
                print("{:<15} {:<18} {:<15}".format(key, np.sqrt(p.data[key].train_loss[t]),  np.sqrt(p.data[key].test_loss[t]) ))
            print("", flush=True)

        if((t+1)%parms["output_every"]==0):
            outfilename = "model"+str(t+1)
            save_model(model, outfilename, "cpu", F4i_test)

            # pickle the model, optimizer, scheduler, and plotter
            with open("model_epoch"+str(t+1)+".pkl", "wb") as f:
                pickle.dump([model, optimizer, scheduler, p], f)

            p.plot_error("train_test_error.pdf", ymin=1e-5)
            

    return model, optimizer, scheduler, p
