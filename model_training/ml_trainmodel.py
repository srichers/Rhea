'''
Authors: Sherwood Richers, John McGuigan

Copyright: GPLv3 (see LICENSE file)

This is the file contains the main training loop, including accumulation of the loss function from various sources.
'''

import torch
from ml_loss import *
from ml_neuralnet import *
from ml_tools import *
from ml_read_data import *
import torch.autograd.profiler as profiler
import pickle
import os

# create an empty dictionary that will eventually contain all of the loss metrics of an iteration
loss_dict = {}

# Each training step is a single full-batch pass over every training point, so
# concatenate the databases once up front and keep them on the device.
def configure_training_data(parms, dataset_train_list):
    F4i_train        = torch.cat([dataset.tensors[0] for dataset in dataset_train_list]).to(parms["device"])
    F4f_true_train   = torch.cat([dataset.tensors[1] for dataset in dataset_train_list]).to(parms["device"])
    growthrate_train = torch.cat([dataset.tensors[2] for dataset in dataset_train_list]).to(parms["device"])

    # relative weight of each database. None means weight them all equally.
    database_weight_list = parms["database_weight_list"]
    if database_weight_list == None:
        database_weight_list = [1.0 for dataset in dataset_train_list]
    assert(len(database_weight_list) == len(dataset_train_list))
    assert(all([w > 0 for w in database_weight_list]))
    wtot = sum(database_weight_list)

    # Per-point weights, normalized so they sum to one. Each database's share is
    # divided evenly among its points, so the loss is a weighted mean over databases
    # and is independent of both the number of databases and their sizes. With equal
    # weights this is exactly the mean over databases that accumulate_asymptotic_loss
    # reports, and is what WeightedRandomSampler used to target in expectation.
    weight_train = torch.cat([torch.full((len(dataset),), w/(wtot*len(dataset)))
                              for dataset,w in zip(dataset_train_list, database_weight_list)]).to(parms["device"])

    print("#  Training on",len(F4i_train),"points from",len(dataset_train_list),"databases in a single full batch.")

    return F4i_train, F4f_true_train, growthrate_train, weight_train


def train_asymptotic_model(parms,
                           dataset_asymptotic_train_list,
                           dataset_asymptotic_validation_list,
                           dataset_asymptotic_test_list,
                           report_fn=None):

    # print out all parameters for the record
    parmfile = open(os.getcwd()+"/parameters.txt","w")
    for key in parms.keys():
        parmfile.write(key+" = "+str(parms[key])+"\n")
    parmfile.close()
    
    print("#Using",parms["device"],"device")
    if parms["device"] == "cuda":
        print("# ",torch.cuda.get_device_name(0))

    #=======================#
    # instantiate the model #
    #=======================#
    print("#SETTING UP NEURAL NETWORK")
    model = NeuralNetwork(parms).to(parms["device"])
    if parms["op"] == torch.optim.AdamW:
        optimizer = parms["op"](model.parameters(),
                                weight_decay=parms["adamw.weight_decay"],
                                lr=parms["learning_rate"],
                                amsgrad=parms["adamw.amsgrad"],
                                fused=parms["adamw.fused"]
        )
    elif parms["op"] == torch.optim.SGD:
        optimizer = torch.optim.SGD(model.parameters(),lr=parms["learning_rate"])
    else:
        raise ValueError("Unknown optimizer "+str(parms["op"]))

    print("#  number of parameters:", sum(p.numel() for p in model.parameters()))

    #=======================#
    # set up the schedulers #
    #=======================#
    print("#SETTING UP SCHEDULERS")
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                         start_factor=1.0/max(1,parms["warmup_iters"]),
                                                         end_factor=1,
                                                         total_iters=parms["warmup_iters"])
    scheduler_main = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                patience=parms["patience"],
                                                                cooldown=parms["cooldown"],
                                                                factor=parms["factor"],
                                                                min_lr=parms["min_lr"]) #
    schedulers = [scheduler_warmup, scheduler_main]

    #=======================#
    # set up the input data #
    #=======================#
    print("#SETTING UP TRAINING DATA")
    F4i_train, F4f_true_train, growthrate_true_train, weight_train = configure_training_data(parms, dataset_asymptotic_train_list)


    def contribute_loss(pred, true, traintest, key, loss_fn, max_fn):
        loss = loss_fn(pred, true)
        loss_dict[key+"_"+traintest+"_loss"] += loss.item()
        loss_dict[key+"_"+traintest+"_max"]  = max(max_fn(pred, true), loss_dict[key+"_"+traintest+"_max"])
        return loss

    # set up file for writing performance metrics
    loss_file = open(os.getcwd()+"/loss.dat","w")
    
    #===============#
    # training loop #
    #===============#
    print("#STARTING TRAINING LOOP")
    torch.backends.cudnn.benchmark = True # may help with performance
    final_metrics = {}
    for epoch in range(1,parms["epochs"]+1):
        # Set up the loss dictionary for IO. Every key is seeded here so that the
        # column order in loss.dat is defined in this one place, rather than being an
        # artifact of the order in which the values happen to be computed below.
        loss_dict = {}
        loss_dict["epoch"] = epoch
        for key in ["F4","growthrate","negative_density","fluxfac"]:
            for traintest in ["train","validation","test"]:
                loss_dict[key+"_"+traintest+"_loss"] = 0
                loss_dict[key+"_"+traintest+"_max"]  = 0
        for traintest in ["train","validation","test"]:
            loss_dict[traintest+"_loss"] = 0
        for name in model.log_task_weights.keys():
            loss_dict["weight_"+name] = 0
        loss_dict["learning_rate"] = 0

        #===================================#
        # TRAINING STEP ON THE FULL DATASET #
        #===================================#
        model.train()

        # get predicted values from the model
        F4f_pred_train, growthrate_pred_train, stable = model.predict_all(F4i_train)

        # convert F4 to densities and fluxes to feed to loss functions
        # note the outputs are all normalized to the total number density
        ntot_t = ntotal(F4f_true_train)
        ntot_p = ntotal(F4f_pred_train)
        assert torch.all(ntot_t > 0)
        assert torch.all(ntot_p > 0)

        # normalize quantities before computing losses
        F4f_true_norm        = F4f_true_train        / ntot_t[:,None,None,None]
        F4f_pred_norm        = F4f_pred_train        / ntot_p[:,None,None,None]
        growthrate_true_norm = growthrate_true_train / ntot_t
        growthrate_pred_norm = growthrate_pred_train / ntot_p

        # reset the loss and gradients
        optimizer.zero_grad()

        # accumulate losses. NOTE - I don't use += because pytorch fails if I do. Just don't do it.
        batch_loss = 0.0
        batch_loss = batch_loss + torch.exp(-model.log_task_weights["F4"]     ) * comparison_loss_fn(F4f_pred_norm, F4f_true_norm, weight_train)
        batch_loss = batch_loss + torch.exp(-model.log_task_weights["growthrate"]) * comparison_loss_fn(growthrate_pred_norm, growthrate_true_norm, weight_train)
        if parms["do_negative_density_check"]:
            batch_loss = batch_loss + torch.exp(-model.log_task_weights["negative_density"]) * negative_density_loss_fn(F4f_pred_norm, None, weight_train)
        if parms["do_fluxfac_check"]:
            batch_loss = batch_loss + torch.exp(-model.log_task_weights["fluxfac"]) * fluxfac_loss_fn(F4f_pred_norm, None, weight_train)

        # add loss weights to loss
        if parms["do_learn_task_weights"]:
            for name in model.log_task_weights.keys():
                if (not parms["do_negative_density_check"]) and name=="negative_density":
                    continue
                if (not parms["do_fluxfac_check"]) and name=="fluxfac":
                    continue
                batch_loss = batch_loss + torch.sum(model.log_task_weights[name])

        batch_loss.backward()
        optimizer.step()

        #============================#
        # EVALUATION ON FULL DATASET #
        #============================#
        # evaluated separately from the training step above so that the reported
        # losses are taken after the optimizer step and in eval mode
        model.eval()

        # Asymptotic losses
        def accumulate_asymptotic_loss(dataset_list, traintest):
            total_loss = torch.tensor(0.0, requires_grad=False)
            for dataset in dataset_list:
                F4i = dataset.tensors[0].to(parms["device"])
                F4f_true = dataset.tensors[1].to(parms["device"])
                growthrate_true = dataset.tensors[2].to(parms["device"])

                # get predicted values from the model
                F4f_pred, growthrate_pred, _ = model.predict_all(F4i)

                # normalize quantities by ntotal before computing losses to avoid floating point issues
                ntot_t = ntotal(F4f_true)
                ntot_p = ntotal(F4f_pred)
                assert torch.all(ntot_t > 0)
                assert torch.all(ntot_p > 0)
                F4f_true = F4f_true / ntot_t[:,None,None,None]
                F4f_pred = F4f_pred / ntot_p[:,None,None,None]
                growthrate_true = growthrate_true / ntot_t
                growthrate_pred = growthrate_pred / ntot_p

                # accumulate losses
                total_loss = total_loss + torch.exp(-model.log_task_weights["F4"]     ) * contribute_loss(F4f_pred,
                                                                                                          F4f_true,
                                                                                                          traintest, "F4", comparison_loss_fn, max_error)
                total_loss = total_loss + torch.exp(-model.log_task_weights["growthrate"]) * contribute_loss(growthrate_pred, #torch.log
                                                                                                             growthrate_true, #torch.log
                                                                                                             traintest, "growthrate", comparison_loss_fn, max_error)
                negative_density_loss = torch.exp(-model.log_task_weights["negative_density"]) * contribute_loss(F4f_pred,
                                                                                                                 None,
                                                                                                                 traintest, "negative_density", negative_density_loss_fn, negative_density_max)
                fluxfac_loss          = torch.exp(-model.log_task_weights["fluxfac"]         ) * contribute_loss(F4f_pred,
                                                                                                                 None,
                                                                                                                 traintest, "fluxfac", fluxfac_loss_fn, fluxfac_max)
                if parms["do_negative_density_check"]:
                    total_loss = total_loss + negative_density_loss
                if parms["do_fluxfac_check"]:
                    total_loss = total_loss + fluxfac_loss

                # add loss weights to loss
                if parms["do_learn_task_weights"]:
                    for name in model.log_task_weights.keys():
                        if (not parms["do_negative_density_check"]) and name=="negative_density":
                            continue
                        if (not parms["do_fluxfac_check"]) and name=="fluxfac":
                            continue
                        total_loss = total_loss + torch.sum(model.log_task_weights[name])

            # report the mean over datasets rather than the sum, so that train,
            # validation, and test losses are directly comparable to each other
            # and do not depend on how many databases were loaded
            if len(dataset_list) > 0:
                for key in ["F4","growthrate","negative_density","fluxfac"]:
                    loss_dict[key+"_"+traintest+"_loss"] /= len(dataset_list)
                total_loss = total_loss / len(dataset_list)

            return total_loss

        with torch.no_grad():
            train_loss      = accumulate_asymptotic_loss(dataset_asymptotic_train_list     , "train"     )
            validation_loss = accumulate_asymptotic_loss(dataset_asymptotic_validation_list, "validation")
            test_loss       = accumulate_asymptotic_loss(dataset_asymptotic_test_list      , "test"      )

        # track the total loss
        loss_dict["train_loss"]      =      train_loss.item()
        loss_dict["validation_loss"] = validation_loss.item()
        loss_dict["test_loss"]       =       test_loss.item()

        # track the task weights
        for name in model.log_task_weights.keys():
            loss_dict["weight_"+name] = torch.exp(-model.log_task_weights[name]).item()

        #=====================================#
        # ADVANCE THE LEARNING RATE SCHEDULER #
        #=====================================#
        # step on the validation loss so that the learning rate decays, and training
        # stops early, when generalization stalls rather than when optimization does.
        # the test loss is deliberately never used here - it would stop being a
        # held-out estimate the moment it fed back into a training decision.
        if epoch<=parms["warmup_iters"]:
            scheduler = schedulers[0]
            loss_dict["learning_rate"] = scheduler.get_last_lr()[0]
            scheduler.step()
        else:
            scheduler = schedulers[1]
            loss_dict["learning_rate"] = scheduler.get_last_lr()[0]
            scheduler.step(validation_loss.item())

        #==========================================#
        # OUTPUT LOSS METRICS AND MODEL PARAMETERS #
        #==========================================#
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
        loss_file.flush()
        assert(loss_dict["train_loss"]==loss_dict["train_loss"])

        # determine if stopping early
        stop_early = (scheduler.get_last_lr()[0]<=parms["min_lr"]) and (epoch>parms["warmup_iters"])

        # output
        print(f"{epoch:4d}  {loss_dict['learning_rate']:12.5e}  {loss_dict['train_loss']:12.5e}  {loss_dict['validation_loss']:12.5e}  {loss_dict['test_loss']:12.5e}", flush=True)
        if(epoch%parms["output_every"]==0 or stop_early):
            outfilename = os.getcwd()+"/model"+str(epoch)
            save_model(model, outfilename, parms["device"])
            print("Saved",outfilename, flush=True)

        final_metrics = dict(loss_dict)
        if report_fn is not None:
            report_fn(dict(loss_dict))

        # exit the loop if the learning rate is too low
        if stop_early:
            print("Learning rate below minimum threshold - stopping training")
            break
        

    return final_metrics
