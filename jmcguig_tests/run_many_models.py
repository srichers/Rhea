"""
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This is the file that is actually run to train a model. It requires access to various databases that are published elsewhere. All model hyperparameters are listed here.
"""

if __name__ == "__main__":
    import sys
    import os

    # use the local model_training in this repo
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(HERE, "..", "model_training"))
    import os
    import numpy as np
    import torch
    from itertools import product
    from torch import nn
    from ml_trainmodel import *
    import torch.optim

    # create a list of options
    all_widths = 32
    all_nhidden = 4
    parms = {}

    # allow overriding dataset location via DATA_ROOT; otherwise default to ../../datasets
    DATA_ROOT = os.environ.get(
        "DATA_ROOT", os.path.abspath(os.path.join(HERE, "..", "..", "datasets"))
    )

    def normalize(paths):
        result = []
        for p in paths:
            if os.path.isabs(p):
                result.append(p)
            else:
                # strip to basename when DATA_ROOT is provided, otherwise keep relative layout
                candidate = os.path.join(DATA_ROOT, os.path.basename(p))
                result.append(candidate)
        return result

    parms["database_list"] = normalize(
        [
            #"../../datasets/asymptotic_Box3D_M1NuLib7ms_rl2_yslices_adjustLebedev.h5",
            #"../../datasets/asymptotic_M1-NuLib-7ms.h5",
            #"../../datasets/asymptotic_M1-NuLib.h5",
            "../../datasets/asymptotic_M1-NuLib-old.h5",
            #"../../datasets/asymptotic_random.h5",
        ]
    )
    parms["stable_database_list"] = normalize(
        [
            #"../../datasets/stable_Box3D_M1NuLib7ms_rl2_yslices_adjustLebedev.h5",
            #"../../datasets/stable_M1-LeakageRates_rl0.h5",
            #"../../datasets/stable_M1-LeakageRates_rl1.h5",
            #"../../datasets/stable_M1-LeakageRates_rl2.h5",
            #"../../datasets/stable_M1-LeakageRates_rl3.h5",
            #"../../datasets/stable_M1-Nulib-7ms_rl0.h5",
            #"../../datasets/stable_M1-Nulib-7ms_rl1.h5",
            "../../datasets/stable_M1-Nulib-7ms_rl2.h5",
            "../../datasets/stable_M1-Nulib-7ms_rl3.h5",
            #"../../datasets/stable_M1-NuLib-old_rl0.h5",
            #"../../datasets/stable_M1-NuLib_rl0.h5",
            #"../../datasets/stable_M1-NuLib_rl1.h5",
            #"../../datasets/stable_M1-NuLib_rl2.h5",
            #"../../datasets/stable_M1-NuLib_rl3.h5",
            #"../../datasets/stable_oneflavor.h5",
            #"../../datasets/stable_random.h5",
            #"../../datasets/stable_zerofluxfac.h5",
        ]
    )
    missing = [
        p
        for p in parms["database_list"] + parms["stable_database_list"]
        if not os.path.exists(p)
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing dataset files: {missing}. Set DATA_ROOT=/path/to/datasets or update paths."
        )
    parms["samples_per_database"] = 200000
    parms["random_samples_per_database"] = 200000
    parms["test_size"] = 0.2
    parms["epochs"] = 10000
    parms["output_every"] = 1000
    parms["average_heavies_in_final_state"] = False
    parms["conserve_lepton_number"] = "direct"
    parms["random_seed"] = 42
    parms["batch_size"] = 4096
    parms["epoch_num_samples"] = 200000  # matches downsampled size for balanced epochs
    parms["sampler"] = torch.utils.data.WeightedRandomSampler # WeightedRandomSampler, SequentialSampler

    parms["do_learn_task_weights"] = False
    # loss multipliers tuned to keep tasks on similar scales
    parms["loss_multiplier_stable"] = 1.0
    parms["loss_multiplier_ndens"] = 1.0
    parms["loss_multiplier_fluxmag"] = 1.0
    parms["loss_multiplier_direction"] = 1.0
    parms["loss_multiplier_growthrate"] = 1.0
    parms["loss_multiplier_unphysical"] = 10.0
    
    # data augmentation options
    parms["do_augment_permutation"] = False #False  # this is the most expensive option to make true, and seems to make things worse...
    parms["do_augment_final_stable"] = False  # True
    parms["do_unphysical_check"] = False  # True - seems to help prevent crazy results

    # neural network options
    parms["nhidden_shared"] = 0 #all_nhidden
    parms["nhidden_stability"] = all_nhidden
    parms["nhidden_growthrate"] = all_nhidden
    parms["nhidden_asymptotic"] = all_nhidden
    parms["nhidden_density"] = all_nhidden
    parms["nhidden_flux"] = all_nhidden
    parms["width_shared"] = all_widths
    parms["width_stability"] = all_widths
    parms["width_growthrate"] = all_widths
    parms["width_density"] = all_widths
    parms["width_flux"] = all_widths
    parms["dropout_probability"] = 0.0  # 0.1 #0.5 #0.1 # 0.5
    parms["do_batchnorm"] = True
    parms["do_fdotu"] = True
    parms["activation"] = nn.LeakyReLU  # nn.LeakyReLU, nn.ReLU

    # optimizer options
    parms["op"] = torch.optim.AdamW  # AdamW, SGD, RMSprop
    parms["adamw.amsgrad"] = False
    parms["adamw.weight_decay"] = 0 #0.01  # 1e-5
    parms["learning_rate"] = 5e-2  # 1e-3
    parms["adamw.fused"] = True
    parms["factor"] = 0.5
    parms["patience"] = 100
    parms["cooldown"] = 100
    parms["warmup_iters"] = 10
    parms["min_lr"] = 1e-7

    # the number of flavors should be 3
    parms["NF"] = 3

    # ========================#
    # use a GPU if available #
    # ========================#
    parms["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # ===============#
    # read the data #
    # ===============#
    dataset_asymptotic_train_list, dataset_asymptotic_test_list = read_asymptotic_data(
        parms
    )
    dataset_stable_train_list, dataset_stable_test_list = read_stable_data(parms)

    parameter_grid = {"learning_rate": [1e-5, 5e-6]}

    keys = list(parameter_grid.keys())
    values = list(parameter_grid.values())

    for combination in product(*values):
        directory_name = "model"

        print(combination)
        print("Running new model")
        for key, value in zip(keys, combination):
            parms[key] = value
            print("    ", key, value)
            directory_name = directory_name + "_" + key + "_" + str(value)
        print("     " + directory_name)

        # create the new directory and go in
        if os.path.exists(directory_name):
            print("     ALREADY PRESENT - skipping")
            continue
        else:
            os.mkdir(directory_name)
            os.chdir(directory_name)

        # train the model
        train_asymptotic_model(
            parms,
            dataset_asymptotic_train_list,
            dataset_asymptotic_test_list,
            dataset_stable_train_list,
            dataset_stable_test_list,
        )

        # run a command line
        os.system("gnuplot ../../quickplot.gplt")

        # return to the main directory
        os.chdir("..")
