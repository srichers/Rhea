'''
Authors: Sherwood Richers, John McGuigan

Copyright: GPLv3 (see LICENSE file)

This is the file contains the main training loop, including accumulation of the loss function from various sources.
'''

import torch
from ml_loss import *
from ml_loss import comparison_per_point, negative_density_per_point, fluxfac_per_point
from ml_neuralnet import *
from ml_tools import *
from ml_read_data import *
import torch.autograd.profiler as profiler
import pickle
import os
import copy

# create an empty dictionary that will eventually contain all of the loss metrics of an iteration
loss_dict = {}

# Each step is a single full-batch pass over every point in the split, so concatenate the
# databases once up front and keep them on the device. Every split is built the same way,
# so the training objective and the reported losses use identical weighting.
def configure_split_data(parms, dataset_list, label):
    F4i        = torch.cat([dataset.tensors[0] for dataset in dataset_list]).to(parms["device"])
    F4f_true   = torch.cat([dataset.tensors[1] for dataset in dataset_list]).to(parms["device"])
    growthrate = torch.cat([dataset.tensors[2] for dataset in dataset_list]).to(parms["device"])

    # relative weight of each database. None means weight them all equally.
    database_weight_list = parms["database_weight_list"]
    if database_weight_list == None:
        database_weight_list = [1.0 for dataset in dataset_list]
    assert(len(database_weight_list) == len(dataset_list))
    assert(all([w > 0 for w in database_weight_list]))
    wtot = sum(database_weight_list)

    # Per-point weights, normalized so they sum to one. Each database's share is
    # divided evenly among its points, so the loss is a weighted mean over databases
    # and is independent of both the number of databases and their sizes.
    weight = torch.cat([torch.full((len(dataset),), w/(wtot*len(dataset)))
                        for dataset,w in zip(dataset_list, database_weight_list)]).to(parms["device"])

    # Which database each point came from, so that per-database losses can be segment-reduced
    # out of the same single full-batch pass. This is reporting only - it must not become a
    # loop over databases in either the training step or the evaluation.
    database_index = torch.cat([torch.full((len(dataset),), i, dtype=torch.long)
                                for i,dataset in enumerate(dataset_list)]).to(parms["device"])

    print("#   "+label+":",len(F4i),"points from",len(dataset_list),"databases in a single full batch.")

    return F4i, F4f_true, growthrate, weight, database_index


def train_asymptotic_model(parms,
                           dataset_asymptotic_train_list,
                           dataset_asymptotic_validation_list,
                           dataset_asymptotic_test_list,
                           report_fn=None):

    print("#Using",parms["device"],"device")
    if parms["device"] == "cuda":
        print("# ",torch.cuda.get_device_name(0))

    # all output files go here. Syne Tune's LocalBackend does not set the working
    # directory of a trial subprocess, so without this every trial would truncate
    # every other trial's loss.dat and parameters.txt.
    output_dir = parms["output_dir"] if parms["output_dir"] is not None else os.getcwd()
    print("#Writing output to",output_dir)

    #=======================#
    # instantiate the model #
    #=======================#
    print("#SETTING UP NEURAL NETWORK")
    model = NeuralNetwork(parms).to(parms["device"])
    # a config written before the split would set this and be silently ignored, training with
    # no weight decay at all rather than the intended amount
    assert "adamw.weight_decay" not in parms, \
        "adamw.weight_decay has been replaced by weight_decay_shared / weight_decay_F4 / weight_decay_growthrate"

    if parms["op"] == torch.optim.AdamW:
        # One parameter group per branch, so the trunk and the two heads can be regularized
        # separately. These three stacks partition every parameter in the model, so nothing is
        # left to pick up AdamW's constructor default of 0.01 by accident.
        param_groups = [
            {"params": list(model.TP_activation_stack_shared.parameters()),     "weight_decay": parms["weight_decay_shared"    ]},
            {"params": list(model.TP_activation_stack_F4.parameters()),         "weight_decay": parms["weight_decay_F4"        ]},
            {"params": list(model.TP_activation_stack_growthrate.parameters()), "weight_decay": parms["weight_decay_growthrate"]},
        ]
        ngrouped = sum(sum(p.numel() for p in g["params"]) for g in param_groups)
        assert(all(len(g["params"]) > 0 for g in param_groups)), "every parameter group must be non-empty"
        assert(ngrouped == sum(p.numel() for p in model.parameters())), "parameter groups must cover every parameter"
        optimizer = parms["op"](param_groups,
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
    # One database_weight_list applies to every split, so entry i has to name the same
    # physical database in each list. split_database.py already produces exactly that
    # structure - the three lists are parallel chunks of the same databases.
    assert(len(dataset_asymptotic_train_list) == len(dataset_asymptotic_validation_list))
    assert(len(dataset_asymptotic_train_list) == len(dataset_asymptotic_test_list))
    F4i             = {}
    F4f_true        = {}
    growthrate_true = {}
    weight          = {}
    database_index  = {}
    for traintest, dataset_list in [("train"     , dataset_asymptotic_train_list     ),
                                    ("validation", dataset_asymptotic_validation_list),
                                    ("test"      , dataset_asymptotic_test_list      )]:
        F4i[traintest], F4f_true[traintest], growthrate_true[traintest], weight[traintest], database_index[traintest] = \
            configure_split_data(parms, dataset_list, traintest)

    # Run the model and normalize everything by the total number density. The predicted
    # and true totals are normalized separately to avoid floating point issues. Used by
    # the training step, by the evaluation of every split, and by the Box3D scales below,
    # so all three see identical preprocessing.
    def predict_normalized(traintest, use_network=True):
        F4f_pred, growthrate_pred, _ = model.predict_all(F4i[traintest], use_network)

        ntot_t = ntotal(F4f_true[traintest])
        ntot_p = ntotal(F4f_pred)
        assert torch.all(ntot_t > 0)
        assert torch.all(ntot_p > 0)

        return (F4f_pred                  / ntot_p[:,None,None,None],
                F4f_true[traintest]       / ntot_t[:,None,None,None],
                growthrate_pred           / ntot_p,
                growthrate_true[traintest]/ ntot_t)

    #==========================================================================#
    # set the task scales from the analytic Box3D baseline on the training set #
    #==========================================================================#
    # These depend only on the data, never on the random seed, the initialization, or any
    # optimizer hyperparameter, so the reported losses are comparable between runs and
    # between Syne Tune trials. A scale calibrated from the run's own progress would not
    # be - a trial that trained badly would have a larger scale and so report a smaller
    # loss for the same true performance. Because the network predicts a residual on top
    # of Box3D, a loss of 1 means the network is doing no better than the analytic model.
    print("#SETTING TASK SCALES FROM THE BOX3D BASELINE")
    with torch.no_grad():
        F4f_box3d, F4f_true_norm, growthrate_box3d, growthrate_true_norm = predict_normalized("train", use_network=False)
        scale_F4         = comparison_loss_fn(F4f_box3d       , F4f_true_norm       , weight["train"]).item()
        scale_growthrate = comparison_loss_fn(growthrate_box3d, growthrate_true_norm, weight["train"]).item()
    assert(scale_F4 > 0)
    assert(scale_growthrate > 0)
    print("#  scale_F4        :", scale_F4)
    print("#  scale_growthrate:", scale_growthrate)

    # The weights that turn the four raw losses into the single number that is minimized.
    # F4 and growthrate are measured against the Box3D baseline error and so need no
    # weight of their own. The two unphysical penalties are linear in a violation measured
    # in number-density units, so they are measured against the RMS baseline error rather
    # than against its square, and a weight of 1 means a violation as large as that error
    # costs as much as the entire F4 loss. Setting a weight to zero disables the penalty.
    task_weights = {
        "F4"               : 1.0 / scale_F4,
        "growthrate"       : 1.0 / scale_growthrate,
        "negative_density" : parms["penalty_negative_density"] / scale_F4**0.5,
        "fluxfac"          : parms["penalty_fluxfac"]          / scale_F4**0.5,
    }

    # print out all parameters for the record, including the data-derived task scales
    parmfile = open(output_dir+"/parameters.txt","w")
    for key in parms.keys():
        parmfile.write(key+" = "+str(parms[key])+"\n")
    parmfile.write("scale_F4 = "+str(scale_F4)+"\n")
    parmfile.write("scale_growthrate = "+str(scale_growthrate)+"\n")
    parmfile.close()

    # combine the four raw losses into the single number that is minimized. This is also
    # what is reported for every split, so the objective and the metric that the learning
    # rate scheduler sees are the same fixed function.
    # NOTE - I don't use += because pytorch fails if I do. Just don't do it.
    def total_loss(losses):
        total = 0.0
        for key in ["F4","growthrate","negative_density","fluxfac"]:
            total = total + task_weights[key] * losses[key]
        return total

    def contribute_loss(pred, true, weight, traintest, key, loss_fn, max_fn):
        loss = loss_fn(pred, true, weight)
        loss_dict[key+"_"+traintest+"_loss"] = loss.item()
        loss_dict[key+"_"+traintest+"_max"]  = max_fn(pred, true)
        return loss

    # Short labels for the per-database diagnostic columns. Taken from the training list, since
    # entry i names the same physical database in all three splits.
    database_names = [os.path.basename(d).split("_chunk")[0] for d in parms["train_database_list"]]
    ndatabases     = len(database_names)

    # set up file for writing performance metrics
    loss_file = open(output_dir+"/loss.dat","w")

    #===============#
    # training loop #
    #===============#
    print("#STARTING TRAINING LOOP")
    torch.backends.cudnn.benchmark = True # may help with performance
    final_metrics = {}
    best_validation_loss = float("inf")
    best_epoch          = 0
    best_saved_epoch    = 0
    best_state          = None
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
        loss_dict["learning_rate"] = 0
        # Diagnostics go after learning_rate, so columns 1-29 keep the meaning they have always
        # had and quickplot_loss.gplt's hard-coded column numbers stay correct. The per-database
        # block is last because its width depends on how many databases are configured.
        for traintest in ["train","validation","test"]:
            loss_dict[traintest+"_median"] = 0
        for traintest in ["train","validation","test"]:
            for name in database_names:
                loss_dict[name+"_"+traintest] = 0

        #===================================#
        # TRAINING STEP ON THE FULL DATASET #
        #===================================#
        model.train()

        # get predicted values from the model, normalized by the total number density
        F4f_pred_norm, F4f_true_norm, growthrate_pred_norm, growthrate_true_norm = predict_normalized("train")

        # reset the gradients
        optimizer.zero_grad()

        # accumulate losses
        batch_losses = {}
        batch_losses["F4"]               = comparison_loss_fn(      F4f_pred_norm       , F4f_true_norm       , weight["train"])
        batch_losses["growthrate"]       = comparison_loss_fn(      growthrate_pred_norm, growthrate_true_norm, weight["train"])
        batch_losses["negative_density"] = negative_density_loss_fn(F4f_pred_norm       , None                , weight["train"])
        batch_losses["fluxfac"]          = fluxfac_loss_fn(         F4f_pred_norm       , None                , weight["train"])
        batch_loss = total_loss(batch_losses)

        batch_loss.backward()
        optimizer.step()

        #============================#
        # EVALUATION ON FULL DATASET #
        #============================#
        # evaluated separately from the training step above so that the reported
        # losses are taken after the optimizer step and in eval mode
        model.eval()

        # Asymptotic losses
        def accumulate_asymptotic_loss(traintest):
            # get predicted values from the model, normalized by the total number density
            F4f_pred, F4f_true_n, growthrate_pred, growthrate_true_n = predict_normalized(traintest)

            # accumulate losses
            losses = {}
            losses["F4"]               = contribute_loss(F4f_pred       , F4f_true_n       , weight[traintest], traintest, "F4"              , comparison_loss_fn      , max_error           )
            losses["growthrate"]       = contribute_loss(growthrate_pred, growthrate_true_n, weight[traintest], traintest, "growthrate"      , comparison_loss_fn      , max_error           )
            losses["negative_density"] = contribute_loss(F4f_pred       , None             , weight[traintest], traintest, "negative_density", negative_density_loss_fn, negative_density_max)
            losses["fluxfac"]          = contribute_loss(F4f_pred       , None             , weight[traintest], traintest, "fluxfac"         , fluxfac_loss_fn         , fluxfac_max         )

            # Diagnostics, out of the same pass and never fed back into any training decision.
            # The per-point objective uses the same task weights as the scalar above, so summing
            # weight*per_point reproduces the reported total exactly.
            per_point = (task_weights["F4"]               * comparison_per_point(F4f_pred, F4f_true_n)
                       + task_weights["growthrate"]       * comparison_per_point(growthrate_pred, growthrate_true_n)
                       + task_weights["negative_density"] * negative_density_per_point(F4f_pred)
                       + task_weights["fluxfac"]          * fluxfac_per_point(F4f_pred))

            # The median is far below the mean whenever a thin tail of hard points dominates,
            # so the gap between them is a free read on how tail-driven the current model is.
            loss_dict[traintest+"_median"] = torch.median(per_point).item()

            # Per-database means, segment-reduced rather than looped. Each database's points are
            # averaged among themselves, so this shows a configuration that wins overall by
            # sacrificing one database.
            idx    = database_index[traintest]
            totals = torch.zeros(ndatabases, device=per_point.device).index_add_(0, idx, per_point)
            counts = torch.zeros(ndatabases, device=per_point.device).index_add_(0, idx, torch.ones_like(per_point))
            for i,name in enumerate(database_names):
                loss_dict[name+"_"+traintest] = (totals[i]/counts[i]).item()

            return total_loss(losses)

        with torch.no_grad():
            train_loss      = accumulate_asymptotic_loss("train"     )
            validation_loss = accumulate_asymptotic_loss("validation")
            test_loss       = accumulate_asymptotic_loss("test"      )

        # track the total loss
        loss_dict["train_loss"]      =      train_loss.item()
        loss_dict["validation_loss"] = validation_loss.item()
        loss_dict["test_loss"]       =       test_loss.item()

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
        # Remember the best weights seen so far. Only the state dict is copied here, which
        # costs microseconds - deep-copying or scripting the model costs ~1s, far more than
        # an epoch, so the export itself happens on the output_every cadence below.
        if loss_dict["validation_loss"] < best_validation_loss:
            best_validation_loss = loss_dict["validation_loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if(epoch%parms["output_every"]==0 or stop_early):
            outfilename = output_dir+"/model"+str(epoch)
            save_model(model, outfilename, parms["device"])
            print("Saved",outfilename, flush=True)

            # Write out the best model too. A trial that a hyperparameter tuner stops at an
            # arbitrary epoch is SIGKILLed, so anything not already on disk is lost - without
            # this, a search would return a winning configuration and no weights to go with it.
            if best_epoch > best_saved_epoch:
                best_saved_epoch = best_epoch
                modelbest = copy.deepcopy(model)
                modelbest.load_state_dict(best_state)
                save_model(modelbest, output_dir+"/modelbest", parms["device"])
                print("Saved",output_dir+"/modelbest","from epoch",best_epoch, flush=True)

        final_metrics = dict(loss_dict)
        if report_fn is not None:
            report_fn(dict(loss_dict))

        # exit the loop if the learning rate is too low
        if stop_early:
            print("Learning rate below minimum threshold - stopping training")
            break


    return final_metrics
