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

    parms["output_dir"] = "output/train_one"

    # list of asymptotic data
    parms["database_train_list"] = [
        "datasets/asymptotic_M1-NuLib-7ms.h5",
        "datasets/asymptotic_random.h5",
    ]
    parms["database_validation_list"] = [
        "datasets/asymptotic_M1-NuLib-old.h5",
    ]
    parms["database_test_list"] = [
        "datasets/asymptotic_M1-NuLib.h5",
    ]

    # list of stability data
    # First dataset is deemed test data
    parms["stable_database_list"] = [
        "datasets/stable_oneflavor.h5",
        "datasets/stable_random.h5",
        "datasets/stable_zerofluxfac.h5",
    ]
    parms["samples_per_database"] = 1000000

    parms["test_size"] = 0.1
    parms["epochs"] = 10
    parms["output_every"] = 10
    parms["average_heavies_in_final_state"] = False
    parms["conserve_lepton_number"] = True
    parms["random_seed"] = 100
    parms["loader.batch_size"] = 10
    parms["loader.num_workers"] = 1
    parms["loader.prefetch_factor"] = 1
    parms["sampler"] = torch.utils.data.WeightedRandomSampler  # WeightedRandomSampler, SequentialSampler
    parms["weightedrandomsampler.epoch_num_samples"] = 10  # parms["samples_per_database"]
    parms["scalar_activation"] = nn.functional.silu
    parms["nonscalar_activation"] = torch.sigmoid
    parms["tensor_product_class"] = "norm"

    parms["do_learn_task_weights"] = False
    parms["task_weight_F4"] = 1.0
    parms["task_weight_unphysical"] = 1
    parms["task_weight_growthrate"] = 1.0

    # data augmentation options
    parms["do_augment_final_stable"] = False  # True
    parms["do_unphysical_check"] = True  # True - seems to help prevent crazy results

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

    hpo_ref_epochs = 20
    hpo_max_budget = 1000

    parms["syne_tune"] = {
        "report": False,
        "metric": "validation_score",
        "mode": "MIN",
        "scheduler": "ASHA",
        "space": {
            "lr": {
                "kind": "loguniform",
                "lower": 1e-5,
                "upper": 1e-2,
            },
            "weight_decay": {
                "kind": "loguniform",
                "lower": 1e-6,
                "upper": 1e-2,
            },
            "batch_size": {
                "kind": "choice",
                "values": [128, 256, 512, 1024],
            },
            "model_tier": {
                "kind": "choice",
                "values": ["tiny", "small"],
            },
            "ref_epochs": {
                "kind": "constant",
                "value": hpo_ref_epochs,
            },
            "max_budget": {
                "kind": "constant",
                "value": hpo_max_budget,
            },
            "device": {
                "kind": "constant",
                "value": parms["device"],
            },
            "seed": {
                "kind": "constant",
                "value": parms["random_seed"],
            },
        },
        "model_tiers": {
            "tiny": {
                "parms": {
                    "irreps_hidden": "2x0e + 2x1o",
                    "nhidden_shared": 2,
                    "nhidden_growthrate": 2,
                    "nhidden_F4": 2,
                    "dropout_probability": 0.025,
                },
            },
            "small": {
                "parms": {
                    "irreps_hidden": "3x0e + 3x1o",
                    "nhidden_shared": 2,
                    "nhidden_growthrate": 2,
                    "nhidden_F4": 2,
                    "dropout_probability": 0.05,
                },
            },
            "medium": {
                "parms": {
                    "irreps_hidden": "5x0e + 5x1o",
                    "nhidden_shared": 2,
                    "nhidden_growthrate": 2,
                    "nhidden_F4": 3,
                    "dropout_probability": 0.075,
                },
            },
            "large": {
                "parms": {
                    "irreps_hidden": "6x0e + 6x1o",
                    "nhidden_shared": 3,
                    "nhidden_growthrate": 2,
                    "nhidden_F4": 3,
                    "dropout_probability": 0.1,
                },
            },
        },
        "base_model_cfg": {
            "input_irreps": "1x1o + 1x0e + 1x1o + 1x0e",
            "growthrate_irreps": "1x0e",
            "F4_irreps": "1x1o + 1x0e",
            "tensor_product_class": parms["tensor_product_class"],
        },
        "resource": {
            "resource_attr": "budget",
            "max_resource_attr": "max_budget",
            "grace_period": 60,
            "reduction_factor": 3,
            "brackets": 1,
        },
        # ASHA schedules on normalized budget. `ref_epochs` defines how many
        # epochs a reference-cost model gets before consuming max_budget.
        "budget": {
            "kind": "normalized_split",
            "ref_epochs": hpo_ref_epochs,
            "ref_params": 300000,
            "ref_batch_size": 256,
            "ref_lr": 1e-3,
            "param_exponent": 0.25,
            "batch_exponent": 0.5,
            "lr_exponent": 0.25,
            "raw_steps_per_budget": hpo_ref_epochs / hpo_max_budget,
            "max_epochs_cap": 100,
        },
        "sampling": {
            "kind": "cost_balanced",
            "cost_exponent": 0.5,
            "candidate_pool_size": 512,
        },
        "plots": {
            "enabled": True,
            "metric": "validation_score",
            "best_over_time": True,
            "trials_over_time": True,
            "output_dir": "plots",
            "best_filename": "best_validation_score.png",
            "trials_filename": "trials_validation_score.png",
            "show": False,
        },
        "bohb_num_min_data_points": 0,
        "bohb_top_n_percent": 15,
        "bohb_min_bandwidth": 1e-3,
        "bohb_num_candidates": 64,
        "bohb_bandwidth_factor": 3,
        "bohb_random_fraction": 0.33,
        "max_wallclock_time": 23 * 60 * 60,
        "max_num_trials_started": 100,
        "max_num_trials_completed": None,
        "n_workers": 2,
        "random_seed": parms["random_seed"],
        "tuner_name": "rhea-asha",
        "results_root": "syne-tune",
        "rotate_gpus": True,
        "delete_checkpoints": False,
        "save_tuner": True,
        "sleep_time": 5.0,
        "results_update_interval": 10.0,
        "print_update_interval": 30.0,
    }

    return parms


def run_default_training(parms=None, report_fn=None):
    from ml_read_data import read_asymptotic_data, read_stable_data
    from ml_trainmodel import train_asymptotic_model

    if parms is None:
        parms = build_default_parms()

    (
        dataset_asymptotic_train_list,
        dataset_asymptotic_validation_list,
        dataset_asymptotic_test_list,
    ) = read_asymptotic_data(parms)

    # Preserve the current stable-dataset loading behavior even though the
    # training loop does not consume those datasets directly.
    read_stable_data(parms)

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
