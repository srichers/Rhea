'''
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This is the file contains the main training loop, including accumulation of the loss function from various sources.
'''

import torch
import torch.autograd.profiler as profiler
import pickle
import os
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from ml_loss import *
from ml_neuralnet import *
from ml_tools import *
from ml_read_data import *
from get_current_lr import current_lr


def configure_loader(parms, dataset_train_list, dataset_test_list):
    assert len(dataset_train_list) == len(dataset_test_list)
    assert len(dataset_train_list) > 0

    weights = torch.cat(
        [
            torch.full((len(dataset),), 1.0 / len(dataset), dtype=torch.double)
            for dataset in dataset_train_list
        ]
    )

    dataset_train = ConcatDataset(dataset_train_list)
    dataset_test = ConcatDataset(dataset_test_list)

    num_samples = parms.get("epoch_num_samples", len(dataset_train))
    generator = torch.Generator().manual_seed(parms.get("random_seed", 0))
    sampler = WeightedRandomSampler(
        weights=weights, num_samples=num_samples, replacement=True, generator=generator
    )
    num_workers = parms.get("num_workers", 0)
    loader = DataLoader(
        dataset_train,
        batch_size=parms["batch_size"],
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=parms.get("pin_memory", parms.get("device", "cpu") == "cuda"),
        persistent_workers=num_workers > 0,
    )

    print(
        "#  Configuring loader with num_samples=",
        num_samples,
        "and batch_size=",
        parms["batch_size"],
        "for a dataset with",
        len(dataset_train),
        "samples.",
    )

    return loader, dataset_test


def train_asymptotic_model(parms,
                           dataset_asymptotic_train_list,
                           dataset_asymptotic_test_list,
                           dataset_stable_train_list,
                           dataset_stable_test_list):

    loss_defaults = {
        "loss_multiplier_stable": 1.0,
        "loss_multiplier_ndens": 1.0,
        "loss_multiplier_fluxmag": 1.0,
        "loss_multiplier_direction": 1.0,
        "loss_multiplier_growthrate": 1.0,
        "loss_multiplier_unphysical": 10.0,
    }
    opt_defaults = {
        "op": torch.optim.AdamW,
        "weight_decay": 0.0,
        "amsgrad": False,
        "fused": False,
        "learning_rate": 1e-3,
        "use_pcgrad": False,
    }
    for key, value in loss_defaults.items():
        parms.setdefault(key, value)
    for key, value in opt_defaults.items():
        parms.setdefault(key, value)

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
    optimizer = parms["op"](
        model.parameters(),
        weight_decay=parms["weight_decay"],
        lr=parms["learning_rate"],
        amsgrad=parms["amsgrad"],
        fused=parms["fused"],
    )

    print("#  number of parameters:", sum(p.numel() for p in model.parameters()))

    #=======================#
    # set up the schedulers #
    #=======================#
    print("#SETTING UP SCHEDULERS")
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0 / max(1, parms["warmup_iters"]),
        end_factor=1,
        total_iters=parms["warmup_iters"],
    )
    scheduler_main = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=parms["patience"],
        cooldown=parms["cooldown"],
        factor=parms["factor"],
        min_lr=parms["min_lr"],
    )
    schedulers = [scheduler_warmup, scheduler_main]

    #=========================#
    # set up the data loaders #
    #=========================#
    print("#SETTING UP DATA LOADERS")
    loader_asymptotic, dataset_asymptotic_test = configure_loader(parms, dataset_asymptotic_train_list, dataset_asymptotic_test_list)
    loader_stable, dataset_stable_test = configure_loader(parms, dataset_stable_train_list, dataset_stable_test_list)

    F4i_asymptotic_test = torch.cat([ds.tensors[0] for ds in dataset_asymptotic_test.datasets], dim=0).to(parms["device"])
    F4f_true_test = torch.cat([ds.tensors[1] for ds in dataset_asymptotic_test.datasets], dim=0).to(parms["device"])
    growthrate_true_test = torch.cat([ds.tensors[2] for ds in dataset_asymptotic_test.datasets], dim=0).to(parms["device"])
    F4i_stable_test = torch.cat([ds.tensors[0] for ds in dataset_stable_test.datasets], dim=0).to(parms["device"])
    stable_true_test = torch.cat([ds.tensors[1] for ds in dataset_stable_test.datasets], dim=0).to(parms["device"])
    ntot_test = ntotal(F4i_asymptotic_test)
    ndens_true_test, fluxmag_true_test, Fhat_true_test = get_ndens_fluxmag_fhat(F4f_true_test)

    # set up file for writing performance metrics
    loss_file = open(os.getcwd()+"/loss.dat","w")
    
    #===============#
    # training loop #
    #===============#
    print("#STARTING TRAINING LOOP")
    torch.backends.cudnn.benchmark = True # may help with performance
    for epoch in range(1,parms["epochs"]+1):
        ml_loss.loss_dict = {"epoch": epoch}
        train_loss_total = 0.0
        max_eln_violation = 0.0

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
            ntot_invsec = ntotal(F4i_asymptotic_train) * ndens_to_invsec

            # get predicted values from the model
            # note that growthrate is predicted as (e^y)(ntot)(ndens_to_invsec) where y is the output of the ml model
            F4f_pred_train, growthrate_pred_train, _                 = model.predict_all(F4i_asymptotic_train)
            _           , _                      , stable_pred_train = model.predict_all(F4i_stable_train    )

            # convert F4 to densities and fluxes to feed to loss functions
            # note the outputs are all normalized to the total number density
            ndens_pred_train, fluxmag_pred_train, Fhat_pred_train = get_ndens_fluxmag_fhat(F4f_pred_train)
            ndens_true_train, fluxmag_true_train, Fhat_true_train = get_ndens_fluxmag_fhat(F4f_true_train)

            ntot_train = ntotal(F4i_asymptotic_train)
            ELN_initial = F4i_asymptotic_train[:, 3, 0, :] - F4i_asymptotic_train[:, 3, 1, :]
            ELN_final = F4f_pred_train[:, 3, 0, :] - F4f_pred_train[:, 3, 1, :]
            max_eln_violation = max(
                max_eln_violation,
                torch.max(torch.abs(ELN_final - ELN_initial) / ntot_train[:, None]).item(),
            )

            task_losses = [
                parms["loss_multiplier_stable"]
                * contribute_loss(stable_pred_train, stable_true_train, "train", "stability", stability_loss_fn),
                parms["loss_multiplier_ndens"]
                * contribute_loss(ndens_pred_train, ndens_true_train, "train", "ndens", comparison_loss_fn),
                parms["loss_multiplier_fluxmag"]
                * contribute_loss(fluxmag_pred_train, fluxmag_true_train, "train", "fluxmag", comparison_loss_fn),
                parms["loss_multiplier_direction"]
                * contribute_loss(Fhat_pred_train, Fhat_true_train, "train", "direction", direction_loss_fn),
                parms["loss_multiplier_growthrate"]
                * contribute_loss(
                    growthrate_pred_train / ntot_train,
                    torch.log(growthrate_true_train / ntot_train / ndens_to_invsec),
                    "train",
                    "growthrate",
                    comparison_loss_fn,
                ),
            ]
            if parms["do_unphysical_check"]:
                task_losses.append(
                    parms["loss_multiplier_unphysical"]
                    * contribute_loss(
                        F4f_pred_train / ntot_train[:, None, None, None],
                        None,
                        "train",
                        "unphysical",
                        unphysical_loss_fn,
                    )
                )

            def pcgrad_step(losses, model, optimizer):
                params = [p for p in model.parameters() if p.requires_grad]
                grads = []
                for i, loss in enumerate(losses):
                    retain = i < len(losses) - 1
                    g = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True, create_graph=False)
                    grads.append(g)
                proj_grads = [list(g) for g in grads]
                num = len(grads)
                for i in range(num):
                    for j in torch.randperm(num):
                        if j == i:
                            continue
                        dot = 0.0
                        gj_norm_sq = 0.0
                        for gi, gj in zip(proj_grads[i], grads[j]):
                            if gi is None or gj is None:
                                continue
                            dot += torch.sum(gi * gj)
                            gj_norm_sq += torch.sum(gj * gj)
                        if gj_norm_sq == 0 or dot >= 0:
                            continue
                        proj_factor = dot / (gj_norm_sq + 1e-12)
                        for k, (gi, gj) in enumerate(zip(proj_grads[i], grads[j])):
                            if gi is None or gj is None:
                                continue
                            proj_grads[i][k] = gi - proj_factor * gj
                optimizer.zero_grad()
                for p in params:
                    p.grad = None
                for idx, p in enumerate(params):
                    agg = None
                    for g in proj_grads:
                        gi = g[idx]
                        if gi is None:
                            continue
                        agg = gi if agg is None else agg + gi
                    if agg is not None:
                        agg = agg / len(proj_grads)
                        p.grad = agg
                optimizer.step()

            if parms.get("use_pcgrad", False):
                pcgrad_step(task_losses, model, optimizer)
                train_loss_total += sum([t.detach().item() for t in task_losses])
            else:
                batch_loss = torch.stack(task_losses).sum()
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                train_loss_total += batch_loss.detach().item()

        #============================#
        # EVALUATION ON FULL DATASET #
        #============================#
        model.eval()

        with torch.no_grad():
            F4f_pred_test, growthrate_pred_test, _ = model.predict_all(F4i_asymptotic_test)
            _, _, stable_pred_test = model.predict_all(F4i_stable_test)
            ndens_pred_test, fluxmag_pred_test, Fhat_pred_test = get_ndens_fluxmag_fhat(F4f_pred_test)

            test_loss = torch.tensor(0.0, device=parms["device"])
            test_loss = test_loss + parms["loss_multiplier_stable"] * contribute_loss(
                stable_pred_test, stable_true_test, "test", "stability", stability_loss_fn
            )
            test_loss = test_loss + parms["loss_multiplier_ndens"] * contribute_loss(
                ndens_pred_test, ndens_true_test, "test", "ndens", comparison_loss_fn
            )
            test_loss = test_loss + parms["loss_multiplier_fluxmag"] * contribute_loss(
                fluxmag_pred_test, fluxmag_true_test, "test", "fluxmag", comparison_loss_fn
            )
            test_loss = test_loss + parms["loss_multiplier_direction"] * contribute_loss(
                Fhat_pred_test, Fhat_true_test, "test", "direction", direction_loss_fn
            )
            test_loss = test_loss + parms["loss_multiplier_growthrate"] * contribute_loss(
                growthrate_pred_test / ntot_test,
                torch.log(growthrate_true_test / ntot_test / ndens_to_invsec),
                "test",
                "growthrate",
                comparison_loss_fn,
            )
            if parms["do_unphysical_check"]:
                test_loss = test_loss + parms["loss_multiplier_unphysical"] * contribute_loss(
                    F4f_pred_test / ntot_test[:, None, None, None],
                    None,
                    "test",
                    "unphysical",
                    unphysical_loss_fn,
                )

        ml_loss.loss_dict["train_loss"] = train_loss_total / max(1, len(loader_asymptotic))
        ml_loss.loss_dict["test_loss"] = test_loss.item()
        ml_loss.loss_dict["ELN_violation"] = max_eln_violation

        #=====================================#
        # ADVANCE THE LEARNING RATE SCHEDULER #
        #=====================================#
        if epoch<=parms["warmup_iters"]:
            scheduler = schedulers[0]
            scheduler.step()
        else:
            scheduler = schedulers[1]
            scheduler.step(ml_loss.loss_dict["train_loss"])
        ml_loss.loss_dict["learning_rate"] = current_lr(optimizer, scheduler)

        #==========================================#
        # OUTPUT LOSS METRICS AND MODEL PARAMETERS #
        #==========================================#
        # print headers
        if epoch==1:
            for k,i in zip(ml_loss.loss_dict.keys(), range(len(ml_loss.loss_dict.keys()))):
                loss_file.write(("{:d}:"+k+"\t").format(i+1))
            loss_file.write('\n')

        # print loss values
        for k in ml_loss.loss_dict.keys():
            if k=="epoch":
                loss_file.write("{:<12d}".format(ml_loss.loss_dict[k]))
            else:
                loss_file.write("{:<12.3e}\t".format(ml_loss.loss_dict[k]))
        loss_file.write('\n')
        loss_file.flush()
        assert(ml_loss.loss_dict["train_loss"]==ml_loss.loss_dict["train_loss"])

        # determine if stopping early
        stop_early = (current_lr(optimizer, scheduler)<=parms["min_lr"]) and (epoch>parms["warmup_iters"])

        # output
        print(f"{epoch:4d}  {ml_loss.loss_dict['learning_rate']:12.5e}  {ml_loss.loss_dict['train_loss']:12.5e}  {ml_loss.loss_dict['test_loss']:12.5e}", flush=True)
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
