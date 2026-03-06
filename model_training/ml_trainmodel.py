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

def samplewise_mse(pred, true):
    return torch.mean((pred - true).reshape(pred.shape[0], -1) ** 2, dim=1)

def samplewise_constraint_penalty(F4f_pred):
    negative_density_error = torch.minimum(F4f_pred[:, :, :, 3], torch.zeros_like(F4f_pred[:, :, :, 3]))
    negative_density_loss = torch.mean(negative_density_error ** 2, dim=(1, 2))

    flux_mag2 = torch.sum(F4f_pred[:, :, :, 0:3] ** 2, dim=3)
    ndens2 = F4f_pred[:, :, :, 3] ** 2
    fluxfac_error = torch.maximum(flux_mag2 - ndens2, torch.zeros_like(ndens2))
    fluxfac_loss = torch.mean(fluxfac_error, dim=(1, 2))

    return negative_density_loss + fluxfac_loss

def evaluate_asymptotic_datasets(parms, model, dataset_list, traintest):
    eval_batch_size = parms.get("eval.batch_size", parms["loader.batch_size"])
    loss_weights = {
        "F4": torch.exp(-model.log_task_weights["F4"]).item(),
        "growthrate": torch.exp(-model.log_task_weights["growthrate"]).item(),
        "unphysical": torch.exp(-model.log_task_weights["unphysical"]).item(),
    }

    aggregates = {
        "F4_loss_sum": 0.0,
        "growthrate_loss_sum": 0.0,
        "unphysical_loss_sum": 0.0,
        "stability_error_sum": 0.0,
        "control_F4_loss_sum": 0.0,
        "control_growthrate_loss_sum": 0.0,
        "control_unphysical_loss_sum": 0.0,
        "control_stability_error_sum": 0.0,
        "F4_max": 0.0,
        "growthrate_max": 0.0,
        "unphysical_max": 0.0,
        "total_samples": 0,
    }
    combined_errors = []
    control_errors = []
    constraint_errors = []
    F4_errors = []
    growthrate_errors = []
    control_F4_errors = []
    control_growthrate_errors = []
    residual_F4_errors = []
    residual_growthrate_errors = []
    with torch.no_grad():
        for dataset in dataset_list:
            loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False)
            for F4i, F4f_true, growthrate_true in loader:
                F4i = F4i.to(parms["device"])
                F4f_true = F4f_true.to(parms["device"])
                growthrate_true = growthrate_true.to(parms["device"])

                F4f_pred, growthrate_pred, stability_pred = model.predict_all(F4i)
                F4f_control, growthrate_control, stability_control = model.predict_control(F4i)

                ntot_t = ntotal(F4f_true)
                assert torch.all(ntot_t > 0)

                F4f_true_norm = F4f_true / ntot_t[:, None, None, None]
                F4f_pred_norm = F4f_pred / ntot_t[:, None, None, None]
                F4f_control_norm = F4f_control / ntot_t[:, None, None, None]
                growthrate_true_norm = growthrate_true / ntot_t
                growthrate_pred_norm = growthrate_pred / ntot_t
                growthrate_control_norm = growthrate_control / ntot_t
                stability_true = (growthrate_true <= 0).float()

                F4_error = samplewise_mse(F4f_pred_norm, F4f_true_norm)
                growthrate_error = samplewise_mse(growthrate_pred_norm.unsqueeze(1), growthrate_true_norm.unsqueeze(1))
                unphysical_error = samplewise_constraint_penalty(F4f_pred_norm)
                stability_error = torch.abs(stability_pred - stability_true)

                control_F4_error = samplewise_mse(F4f_control_norm, F4f_true_norm)
                control_growthrate_error = samplewise_mse(growthrate_control_norm.unsqueeze(1), growthrate_true_norm.unsqueeze(1))
                control_unphysical_error = samplewise_constraint_penalty(F4f_control_norm)
                control_stability_error = torch.abs(stability_control - stability_true)

                combined_error = F4_error + growthrate_error + unphysical_error + stability_error
                control_error = control_F4_error + control_growthrate_error + control_unphysical_error + control_stability_error

                aggregates["F4_loss_sum"] += F4_error.sum().item()
                aggregates["growthrate_loss_sum"] += growthrate_error.sum().item()
                aggregates["unphysical_loss_sum"] += unphysical_error.sum().item()
                aggregates["stability_error_sum"] += stability_error.sum().item()
                aggregates["control_F4_loss_sum"] += control_F4_error.sum().item()
                aggregates["control_growthrate_loss_sum"] += control_growthrate_error.sum().item()
                aggregates["control_unphysical_loss_sum"] += control_unphysical_error.sum().item()
                aggregates["control_stability_error_sum"] += control_stability_error.sum().item()
                aggregates["F4_max"] = max(aggregates["F4_max"], torch.max(torch.abs(F4f_pred_norm - F4f_true_norm)).item())
                aggregates["growthrate_max"] = max(aggregates["growthrate_max"], torch.max(torch.abs(growthrate_pred_norm - growthrate_true_norm)).item())
                aggregates["unphysical_max"] = max(aggregates["unphysical_max"], unphysical_error.max().item())
                aggregates["total_samples"] += F4i.shape[0]

                combined_errors.append(combined_error.cpu())
                control_errors.append(control_error.cpu())
                constraint_errors.append(unphysical_error.cpu())
                F4_errors.append(F4_error.cpu())
                growthrate_errors.append(growthrate_error.cpu())
                control_F4_errors.append(control_F4_error.cpu())
                control_growthrate_errors.append(control_growthrate_error.cpu())
                residual_F4_errors.append(samplewise_mse(F4f_pred_norm, F4f_control_norm).cpu())
                residual_growthrate_errors.append(samplewise_mse(growthrate_pred_norm.unsqueeze(1), growthrate_control_norm.unsqueeze(1)).cpu())

    total_samples = aggregates["total_samples"]
    if total_samples == 0:
        return {}

    combined_errors = torch.cat(combined_errors)
    control_errors = torch.cat(control_errors)
    constraint_errors = torch.cat(constraint_errors)
    F4_errors = torch.cat(F4_errors)
    growthrate_errors = torch.cat(growthrate_errors)
    control_F4_errors = torch.cat(control_F4_errors)
    control_growthrate_errors = torch.cat(control_growthrate_errors)
    residual_F4_errors = torch.cat(residual_F4_errors)
    residual_growthrate_errors = torch.cat(residual_growthrate_errors)

    results = {
        f"F4_{traintest}_loss": aggregates["F4_loss_sum"] / total_samples,
        f"F4_{traintest}_max": aggregates["F4_max"],
        f"growthrate_{traintest}_loss": aggregates["growthrate_loss_sum"] / total_samples,
        f"growthrate_{traintest}_max": aggregates["growthrate_max"],
        f"unphysical_{traintest}_loss": aggregates["unphysical_loss_sum"] / total_samples,
        f"unphysical_{traintest}_max": aggregates["unphysical_max"],
        f"stability_{traintest}_error": aggregates["stability_error_sum"] / total_samples,
        f"control_F4_{traintest}_loss": aggregates["control_F4_loss_sum"] / total_samples,
        f"control_growthrate_{traintest}_loss": aggregates["control_growthrate_loss_sum"] / total_samples,
        f"control_unphysical_{traintest}_loss": aggregates["control_unphysical_loss_sum"] / total_samples,
        f"control_stability_{traintest}_error": aggregates["control_stability_error_sum"] / total_samples,
        f"control_total_{traintest}_loss": control_errors.mean().item(),
        f"F4_{traintest}_improvement_frac": torch.mean((F4_errors <= control_F4_errors).float()).item(),
        f"growthrate_{traintest}_improvement_frac": torch.mean((growthrate_errors <= control_growthrate_errors).float()).item(),
        f"total_{traintest}_improvement_frac": torch.mean((combined_errors <= control_errors).float()).item(),
        f"F4_{traintest}_residual_rms": torch.sqrt(torch.mean(residual_F4_errors)).item(),
        f"growthrate_{traintest}_residual_rms": torch.sqrt(torch.mean(residual_growthrate_errors)).item(),
        f"{traintest}_p95_error": torch.quantile(combined_errors, 0.95).item(),
        f"{traintest}_p99_error": torch.quantile(combined_errors, 0.99).item(),
        f"{traintest}_control_gap": torch.mean(torch.relu(combined_errors - control_errors)).item(),
        f"{traintest}_exceedance_rate": torch.mean((combined_errors > control_errors).float()).item(),
        f"{traintest}_constraint_exceedance_rate": torch.mean((constraint_errors > 0).float()).item(),
    }

    weighted_loss = (
        loss_weights["F4"] * results[f"F4_{traintest}_loss"]
        + loss_weights["growthrate"] * results[f"growthrate_{traintest}_loss"]
    )
    if parms["do_unphysical_check"]:
        weighted_loss += loss_weights["unphysical"] * results[f"unphysical_{traintest}_loss"]
    if any(parameter.requires_grad for parameter in model.log_task_weights.values()):
        for name, parameter in model.log_task_weights.items():
            if (not parms["do_unphysical_check"]) and name == "unphysical":
                continue
            weighted_loss += parameter.item()

    results[f"{traintest}_loss"] = weighted_loss
    results[f"{traintest}_mean_error"] = combined_errors.mean().item()
    results[f"{traintest}_robust_score"] = (
        results[f"{traintest}_mean_error"]
        + results[f"{traintest}_control_gap"]
        + 0.1 * results[f"{traintest}_p95_error"]
        + 0.2 * results[f"{traintest}_p99_error"]
        + 0.5 * results[f"{traintest}_exceedance_rate"]
        + 0.5 * results[f"{traintest}_constraint_exceedance_rate"]
        + 0.25 * results[f"stability_{traintest}_error"]
    )

    return results

def train_asymptotic_model(parms,
                           dataset_asymptotic_train_list,
                           dataset_asymptotic_test_list,
                           report_fn = None):

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


    # set up file for writing performance metrics
    loss_file = open(os.getcwd()+"/loss.dat","w")
    
    #===============#
    # training loop #
    #===============#
    print("#STARTING TRAINING LOOP")
    torch.backends.cudnn.benchmark = True # may help with performance
    final_metrics = {}
    for epoch in range(1,parms["epochs"]+1):
        # set up the loss dictionary for IO
        loss_dict = {}
        loss_dict["epoch"] = epoch

        #============================#
        # TRAINING LOOP OVER BATCHES #
        #============================#
        model.train()
        for (F4i_asymptotic_train, F4f_true_train, growthrate_true_train) in loader_asymptotic:

            # move the minibatch to the device
            F4i_asymptotic_train = F4i_asymptotic_train.to(parms["device"])
            F4f_true_train = F4f_true_train.to(parms["device"])
            growthrate_true_train = growthrate_true_train.to(parms["device"])

            # get predicted values from the model
            F4f_pred_train, growthrate_pred_train, stable = model.predict_all(F4i_asymptotic_train)

            # convert F4 to densities and fluxes to feed to loss functions
            # note the outputs are all normalized to the total number density
            #ntot_i = ntotal(F4i_asymptotic_train)
            ntot_t = ntotal(F4f_true_train)
            ntot_p = ntotal(F4f_pred_train)
            #print("ntot_pred min/max:", ntot_p.min().item(), ntot_p.max().item())
            assert torch.all(ntot_t > 0)
            assert torch.all(ntot_p > 0)

            # normalize quantities before computing losses
            F4f_true_train = F4f_true_train / ntot_t[:,None,None,None]
            F4f_pred_train = F4f_pred_train / ntot_p[:,None,None,None]
            growthrate_true_train = growthrate_true_train / ntot_t
            growthrate_pred_train = growthrate_pred_train / ntot_p

            # reset the loss and gradients
            optimizer.zero_grad()

            # accumulate losses. NOTE - I don't use += because pytorch fails if I do. Just don't do it.
            batch_loss = 0.0
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

        train_metrics = evaluate_asymptotic_datasets(parms, model, dataset_asymptotic_train_list, "train")
        test_metrics = evaluate_asymptotic_datasets(parms, model, dataset_asymptotic_test_list, "test")
        loss_dict.update(train_metrics)
        loss_dict.update(test_metrics)
        loss_dict["validation_score"] = loss_dict["test_robust_score"]

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
            scheduler.step(loss_dict["validation_score"])

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
        print(
            f"{epoch:4d}  {loss_dict['learning_rate']:12.5e}  {loss_dict['train_loss']:12.5e}  "
            f"{loss_dict['test_loss']:12.5e}  {loss_dict['validation_score']:12.5e}",
            flush=True,
        )
        if(epoch%parms["output_every"]==0 or stop_early):
            outfilename = os.getcwd()+"/model"+str(epoch)
            F4i = dataset_asymptotic_test_list[0].tensors[0]
            save_model(model, outfilename, parms["device"], F4i)
            print("Saved",outfilename, flush=True)

        final_metrics = dict(loss_dict)
        if report_fn is not None:
            report_fn(dict(loss_dict))

        # exit the loop if the learning rate is too low
        if stop_early:
            print("Learning rate below minimum threshold - stopping training")
            break
        

    return final_metrics
