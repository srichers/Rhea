'''
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This is the file that is actually run to train a model. It requires access to various databases that are published elsewhere. All model hyperparameters are listed here.
'''

import torch
from torch import nn
import e3nn.o3

from ml_read_data import read_asymptotic_data, read_stable_data
from ml_trainmodel import train_asymptotic_model


def build_default_parms():
    # create a list of options
    parms = {}

    # list of asymptotic data
    # First dataset is deemed test data
    parms["database_list"] = [
        "data/dummy_asymptotic.h5",
        "data/dummy_asymptotic.h5",
    ]

    # list of stability data
    # First dataset is deemed test data
    parms["stable_database_list"] = [
        "data/stable_oneflavor_database.h5",
        "data/stable_random_database.h5",
        "data/stable_zerofluxfac_database.h5",
    ]
    parms["samples_per_database"] = 1000000

    parms["test_size"] = 0.1
    parms["epochs"] = 10
    parms["output_every"] = 10
    parms["average_heavies_in_final_state"] = False
    parms["conserve_lepton_number"] = True
    parms["random_seed"] = 42
    parms["loader.batch_size"] = 10
    parms["loader.num_workers"] = 1
    parms["loader.prefetch_factor"] = 1
    parms["sampler"] = torch.utils.data.WeightedRandomSampler  # WeightedRandomSampler, SequentialSampler
    parms["weightedrandomsampler.epoch_num_samples"] = 10  # parms["samples_per_database"]
    parms["scalar_activation"] = nn.functional.silu
    parms["nonscalar_activation"] = torch.sigmoid
    parms["tensor_product_class"] = "norm"

    parms["do_learn_task_weights"] = False
    parms["task_weight_stability"] = 1.0
    parms["task_weight_F4"] = 1.0
    parms["task_weight_unphysical"] = 1
    parms["task_weight_growthrate"] = 1.0

    # data augmentation options
    parms["do_augment_final_stable"] = False  # True
    parms["do_unphysical_check"] = True  # True - seems to help prevent crazy results

    # neural network options
    parms["nhidden_shared"] = 1
    parms["nhidden_stability"] = 3
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

    parms["syne_tune"] = {
        "report": True,
        "metric": "validation_score",
        "mode": "min",
        "resource_attr": "epoch",
        "max_resource_attr": "epochs",
        "config_space": {
            "epochs": 10,
            "learning_rate": {
                "type": "loguniform",
                "lower": 1e-5,
                "upper": 1e-3,
            },
            "loader.batch_size": {
                "type": "randint",
                "lower": 8,
                "upper": 64,
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
        "scheduler": {
            "name": "hyperband",
            "searcher": "random",
            "type": "stopping",
            "grace_period": 1,
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


def run_default_training(parms = None, report_fn = None):
    if parms is None:
        parms = build_default_parms()

    dataset_asymptotic_train_list, dataset_asymptotic_test_list = read_asymptotic_data(parms)

    # Preserve current behavior and dataset validation side effects even though
    # the stable datasets are not yet consumed by the training loop.
    read_stable_data(parms)

    return train_asymptotic_model(
        parms,
        dataset_asymptotic_train_list,
        dataset_asymptotic_test_list,
        report_fn=report_fn,
    )


def main():
    run_default_training()


if __name__ == "__main__":
    main()
