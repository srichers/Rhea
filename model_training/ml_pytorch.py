'''
Authors: Sherwood Richers, John McGuigan

Copyright: GPLv3 (see LICENSE file)

This is the file that is actually run to train a model. It requires access to various databases that are published elsewhere. All model hyperparameters are listed here.
'''

import torch
from torch import nn


def build_default_parms():
    import e3nn.o3

    # create a list of options
    parms = {}

    # lists of asymptotic data, preprocessed by data/split_database.py
    parms["train_database_list"] = [
        "data/dummy_asymptotic_chunk3-0_thin1_maxfluxfac0.9.h5",
    ]
    parms["validation_database_list"] = [
        "data/dummy_asymptotic_chunk3-1_thin1_maxfluxfac0.9.h5",
    ]
    parms["test_database_list"] = [
        "data/dummy_asymptotic_chunk3-2_thin1_maxfluxfac0.9.h5",
    ]

    # relative weight of each training database in the loss. None weights them equally.
    parms["database_weight_list"] = None

    parms["epochs"] = 10
    parms["output_every"] = 10

    # directory for loss.dat, parameters.txt, and model*.pt. None means the current
    # working directory. run_trial.py points this at the trial directory so that
    # concurrent Syne Tune trials do not overwrite each other's output.
    parms["output_dir"] = None

    parms["average_heavies_in_final_state"] = False
    parms["conserve_lepton_number"] = True
    parms["random_seed"] = 42
    parms["scalar_activation"] = nn.functional.silu
    parms["nonscalar_activation"] = torch.sigmoid
    parms["tensor_product_class"] = "norm"

    # Coefficients of the two unphysical-state penalties. The F4 and growthrate losses are
    # normalized by the Box3D baseline error and so need no weight of their own. These two
    # are hinges with no such baseline, so they are measured in units of the RMS Box3D F4
    # error: a weight of 1 means a violation as large as that error costs as much as the
    # entire F4 loss. Both violations share units, so the two are commensurate. Zero
    # disables the penalty.
    parms["penalty_negative_density"] = 1
    parms["penalty_fluxfac"] = 1

    # neural network options
    parms["nhidden_shared"] = 1
    parms["nhidden_growthrate"] = 3
    parms["nhidden_F4"] = 3
    parms["irreps_hidden"] = e3nn.o3.Irreps("4x0e + 4x1o")
    parms["dropout_probability"] = 0.0
    parms["do_batchnorm"] = False
    parms["do_fdotu"] = True
    parms["activation"] = nn.LeakyReLU  # nn.LeakyReLU, nn.ReLU

    # optimizer options
    parms["op"] = torch.optim.AdamW  # Adam, SGD, RMSprop
    parms["adamw.amsgrad"] = False
    parms["adamw.weight_decay"] = 0
    parms["adamw.fused"] = True
    parms["learning_rate"] = 2e-4
    parms["patience"] = 500
    parms["cooldown"] = 500
    parms["factor"] = 0.5
    parms["warmup_iters"] = 0
    parms["min_lr"] = 0

    # the number of flavors should be 3
    parms["NF"] = 3

    #========================#
    # use a GPU if available #
    #========================#
    parms["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # Hyperparameter search, driven by launch_syne_tune.py. The metric is the same fixed
    # objective that is minimized, normalized by the run-independent Box3D baseline errors,
    # so it is comparable across trials. Do not put penalty_negative_density or
    # penalty_fluxfac in the config space - they change what the objective means, which
    # would make validation_loss incomparable between trials.
    parms["syne_tune"] = {
        "report": False,
        "metric": "validation_loss",
        "do_minimize": True,
        "time_attr": "epoch",
        # Report to the tuner only every this many epochs. An epoch is a single full-batch
        # optimizer step, so a real run reports tens of thousands of times otherwise, and
        # each report costs a parsed line of stdout and a searcher update. The final epoch
        # is always reported so that a run stopped early by min_lr still lands its result.
        "report_every": 100,
        "config_space": {
            "learning_rate": {
                "type": "loguniform",
                "lower": 1e-5,
                "upper": 1e-3,
            },
            "adamw.weight_decay": {
                "type": "loguniform",
                "lower": 1e-8,
                "upper": 1e-2,
            },
        },
        "backend": {
            "pass_args_as_json": True,
            "rotate_gpus": True,
            "num_gpus_per_trial": 1,
        },
        # The rung levels are measured in epochs and are grace_period * reduction_factor**k,
        # so grace_period has to be a real fraction of parms["epochs"] - a rung at epoch 1 of
        # a run that takes thousands of optimizer steps carries no signal. It also has to be
        # a multiple of report_every, or the first rung is never observed. The upper limit of
        # the resource, max_t, is parms["epochs"] and is not set here.
        "scheduler": {
            "name": "asha",
            "searcher": "random_search",
            "grace_period": 500,
            "reduction_factor": 3,
        },
        "tuner": {
            "n_workers": 1,
        },
        "stop": {
            "max_wallclock_time": 3600,
        },
    }

    return parms


def run_default_training(parms=None, report_fn=None):
    from ml_read_data import read_asymptotic_data
    from ml_trainmodel import train_asymptotic_model

    if parms is None:
        parms = build_default_parms()

    dataset_asymptotic_train_list, dataset_asymptotic_validation_list, dataset_asymptotic_test_list = read_asymptotic_data(parms)

    return train_asymptotic_model(
        parms,
        dataset_asymptotic_train_list,
        dataset_asymptotic_validation_list,
        dataset_asymptotic_test_list,
        report_fn=report_fn,
    )


def main():
    run_default_training()


if __name__ == "__main__":
    main()
