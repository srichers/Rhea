'''
Author: Sherwood Richers

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
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler, SequentialSampler

# create an empty dictionary that will eventually contain all of the loss metrics of an iteration
loss_dict = {}

def configure_loader(parms, dataset_train_list):
    # create list of weights
    weights = torch.cat([
        torch.full((len(dataset),), 1.0 / len(dataset), dtype=torch.double)
        for dataset in dataset_train_list
    ])
    
    # combine the datasets
    dataset_train = ConcatDataset(dataset_train_list)

    generator = torch.Generator().manual_seed(parms["random_seed"])
    
    # create sampler and data loader for test data
    if parms["sampler"]==WeightedRandomSampler:
        sampler = WeightedRandomSampler(weights=weights,
                                        num_samples=parms["weightedrandomsampler.epoch_num_samples"],
                                        replacement=True,
                                        generator=generator)
    elif parms["sampler"]==SequentialSampler:
        sampler = SequentialSampler(dataset_train)
    else:
        raise ValueError("Unknown sampler type "+str(parms["sampler"]))

    # set up the data loader
    loader = DataLoader(dataset_train,
                        batch_size=parms["loader.batch_size"],
                        sampler=sampler,
                        num_workers=parms["loader.num_workers"],
                        pin_memory=True,
                        persistent_workers=True,
                        prefetch_factor=parms["loader.prefetch_factor"])

    print("#  Configuring loader with batch_size=",parms["loader.batch_size"],"for a dataset with",len(dataset_train),"samples.")

    return loader


def train_asymptotic_model(parms,
                           dataset_asymptotic_train_list,
                           dataset_asymptotic_test_list,
                           dataset_stable_train_list,
                           dataset_stable_test_list):

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

    #=========================#
    # set up the data loaders #
    #=========================#
    print("#SETTING UP DATA LOADERS")
    loader_asymptotic = configure_loader(parms, dataset_asymptotic_train_list)
    loader_stable     = configure_loader(parms, dataset_stable_train_list    )


    def contribute_loss(pred, true, traintest, key, loss_fn):
        loss = loss_fn(pred, true)
        loss_dict[key+"_"+traintest+"_loss"] += loss.item()
        loss_dict[key+"_"+traintest+"_max"]  = max(max_error(pred, true), loss_dict[key+"_"+traintest+"_max"])
        return loss

    # set up file for writing performance metrics
    loss_file = open(os.getcwd()+"/loss.dat","w")
    
    #===============#
    # training loop #
    #===============#
    print("#STARTING TRAINING LOOP")
    torch.backends.cudnn.benchmark = True # may help with performance
    for epoch in range(1,parms["epochs"]+1):
        # set up the loss dictionary for IO
        loss_dict = {}
        loss_dict["epoch"] = epoch

        #============================#
        # TRAINING LOOP OVER BATCHES #
        #============================#
        assert(len(loader_asymptotic)==len(loader_stable))
        model.train()
        for (F4i_asymptotic_train, F4f_true_train, growthrate_true_train),(F4i_stable_train, stable_true_train) in zip(loader_asymptotic, loader_stable):

            # move the minibatch to the device
            F4i_asymptotic_train = F4i_asymptotic_train.to(parms["device"])
            F4f_true_train = F4f_true_train.to(parms["device"])
            growthrate_true_train = growthrate_true_train.to(parms["device"])
            F4i_stable_train = F4i_stable_train.to(parms["device"])
            stable_true_train = stable_true_train.to(parms["device"])

            # get predicted values from the model
            # note that growthrate is predicted as (e^y)(ntot)(ndens_to_invsec) where y is the output of the ml model
            F4f_pred_train, growthrate_pred_train, _                 = model.predict_all(F4i_asymptotic_train)
            _           , _                      , stable_pred_train = model.predict_all(F4i_stable_train    )

            # convert F4 to densities and fluxes to feed to loss functions
            # note the outputs are all normalized to the total number density
            ntot_i = ntotal(F4i_asymptotic_train)
            ntot_f = ntotal(F4f_true_train)
            ntot_p = ntotal(F4f_pred_train)
            #print("ntot_pred min/max:", ntot_p.min().item(), ntot_p.max().item())
            assert(torch.allclose(ntot_i, torch.ones_like(ntot_i)))
            assert(torch.allclose(ntot_f, torch.ones_like(ntot_f)))

            # reset the loss and gradients
            optimizer.zero_grad()

            # accumulate losses. NOTE - I don't use += because pytorch fails if I do. Just don't do it.
            batch_loss = 0.0
            batch_loss = batch_loss + torch.exp(-model.log_task_weights["stability"] ) * stability_loss_fn(stable_pred_train, stable_true_train)
            batch_loss = batch_loss + torch.exp(-model.log_task_weights["F4"]     ) * comparison_loss_fn(F4f_pred_train, F4f_true_train)
            batch_loss = batch_loss + torch.exp(-model.log_task_weights["growthrate"]) * comparison_loss_fn(growthrate_pred_train, growthrate_true_train)
            if parms["do_unphysical_check"]:
                batch_loss = batch_loss + torch.exp(-model.log_task_weights["unphysical"]) * unphysical_loss_fn(F4f_pred_train, None)

            # add loss weights to loss
            if parms["do_learn_task_weights"]:
                for name in model.log_task_weights.keys():
                    if (not parms["do_unphysical_check"]) and name=="unphysical":
                        continue
                    else:
                        batch_loss = batch_loss + torch.sum(model.log_task_weights[name])

            
            batch_loss.backward()
            optimizer.step()

        #============================#
        # EVALUATION ON FULL DATASET #
        #============================#
        model.eval()

        loss_dict["F4_train_loss"] = 0
        loss_dict["F4_train_max"] = 0
        loss_dict["F4_test_loss"] = 0
        loss_dict["F4_test_max"] = 0
        loss_dict["growthrate_train_loss"] = 0
        loss_dict["growthrate_train_max"] = 0
        loss_dict["growthrate_test_loss"] = 0
        loss_dict["growthrate_test_max"] = 0
        loss_dict["unphysical_train_loss"] = 0
        loss_dict["unphysical_train_max"] = 0
        loss_dict["unphysical_test_loss"] = 0
        loss_dict["unphysical_test_max"] = 0
        loss_dict["stability_train_loss"] = 0
        loss_dict["stability_train_max"] = 0
        loss_dict["stability_test_loss"] = 0
        loss_dict["stability_test_max"] = 0

        # Asymptotic losses
        def accumulate_asymptotic_loss(dataset_list, traintest):
            total_loss = torch.tensor(0.0, requires_grad=False)
            for dataset in dataset_list:
                F4i = dataset.tensors[0].to(parms["device"])
                F4f_true = dataset.tensors[1].to(parms["device"])
                growthrate_true = dataset.tensors[2].to(parms["device"])

                F4f_pred, growthrate_pred, _ = model.predict_all(F4i)

                total_loss = total_loss + torch.exp(-model.log_task_weights["F4"]     ) * contribute_loss(F4f_pred,
                                                                                                          F4f_true,
                                                                                                          traintest, "F4", comparison_loss_fn)
                total_loss = total_loss + torch.exp(-model.log_task_weights["growthrate"]) * contribute_loss(growthrate_pred, #torch.log
                                                                                                             growthrate_true, #torch.log
                                                                                                             traintest, "growthrate", comparison_loss_fn)
                unphysical_loss = torch.exp(-model.log_task_weights["unphysical"]) * contribute_loss(F4f_pred,
                                                                                                     None,
                                                                                                     traintest, "unphysical", unphysical_loss_fn)
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
                F4i = dataset.tensors[0].to(parms["device"])
                stable_true = dataset.tensors[1].to(parms["device"])

                _, _, y_stable_pred = model.predict_all(F4i)

                #print(torch.sum(torch.abs(torch.sigmoid(y_stable_pred)-stable_true)).item()/y_stable_pred.shape[0],"fractional difference in stable points")

                this_loss = torch.exp(-model.log_task_weights["stability"] ) * \
                    contribute_loss(y_stable_pred, stable_true, traintest, "stability", stability_loss_fn)
                #print("  stability loss contribution:", this_loss.item())
                total_loss = total_loss + this_loss
            return total_loss
        
        with torch.no_grad():
            train_loss = train_loss + accumulate_stable_loss(dataset_stable_train_list, "train")
            test_loss  = test_loss  + accumulate_stable_loss(dataset_stable_test_list , "test" )

        # track the total loss
        loss_dict["train_loss"] = train_loss.item()
        loss_dict["test_loss"]  =  test_loss.item()

        # track the task weights
        for name in model.log_task_weights.keys():
            loss_dict["weight_"+name] = torch.exp(-model.log_task_weights[name]).item()

        #=====================================#
        # ADVANCE THE LEARNING RATE SCHEDULER #
        #=====================================#
        if epoch<=parms["warmup_iters"]:
            scheduler = schedulers[0]
            loss_dict["learning_rate"] = scheduler.get_last_lr()[0]
            scheduler.step()
        else:
            scheduler = schedulers[1]
            loss_dict["learning_rate"] = scheduler.get_last_lr()[0]
            scheduler.step(train_loss.item())

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
        print(f"{epoch:4d}  {loss_dict['learning_rate']:12.5e}  {loss_dict['train_loss']:12.5e}  {loss_dict['test_loss']:12.5e}", flush=True)
        if(epoch%parms["output_every"]==0 or stop_early):
            outfilename = os.getcwd()+"/model"+str(epoch)
            F4i = dataset_asymptotic_test_list[0].tensors[0]
            save_model(model, outfilename, parms["device"], F4i)
            print("Saved",outfilename, flush=True)

        # exit the loop if the learning rate is too low
        if stop_early:
            print("Learning rate below minimum threshold - stopping training")
            break
        

    return
