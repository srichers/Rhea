"""
Author: Sherwood Richers

Copyright: GPLv3 (see LICENSE file)

This is the file that is actually run to train a model. It requires access to various databases that are published elsewhere. All model hyperparameters are listed here.
"""

if __name__ == "__main__":
    import sys
    import os
    import argparse
    import csv
    import numpy as np
    import torch
    from itertools import product
    from torch import nn
    from ml_trainmodel import *
    import torch.optim

    # use the local model_training in this repo
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(HERE, "..", "model_training"))

    parser = argparse.ArgumentParser(description="Hyperparameter sweep runner.")
    parser.add_argument(
        "--learning-rate",
        default="1e-4,5e-5,1e-5",
        help="Comma-separated learning rates to sweep.",
    )
    parser.add_argument(
        "--width-shared",
        default="32,64",
        help="Comma-separated shared widths to sweep.",
    )
    parser.add_argument(
        "--nhidden-shared",
        default="0,1",
        help="Comma-separated number of shared hidden layers to sweep.",
    )
    parser.add_argument(
        "--seeds",
        default="42",
        help="Comma-separated seeds to sweep.",
    )
    parser.add_argument("--epochs", type=int, default=500, help="Epochs per run.")
    parser.add_argument(
        "--output-every", type=int, default=100, help="Checkpoint/print interval."
    )
    parser.add_argument(
        "--summary",
        default="run_summary.csv",
        help="CSV file to append run summaries.",
    )
    parser.add_argument(
        "--use-pcgrad",
        action="store_true",
        help="Enable PCGrad during training.",
    )
    args = parser.parse_args()

    def parse_list(val, cast=float):
        return [cast(x) for x in val.split(",") if x]

    lr_grid = parse_list(args.learning_rate, float)
    width_grid = parse_list(args.width_shared, int)
    nhid_grid = parse_list(args.nhidden_shared, int)
    seed_grid = parse_list(args.seeds, int)

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
                candidate = os.path.join(DATA_ROOT, os.path.basename(p))
                result.append(candidate)
        return result

    parms["database_list"] = normalize(
        [
            "../../datasets/asymptotic_M1-NuLib-old.h5",
        ]
    )
    parms["stable_database_list"] = normalize(
        [
            "../../datasets/stable_M1-Nulib-7ms_rl2.h5",
            "../../datasets/stable_M1-Nulib-7ms_rl3.h5",
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
    parms["epochs"] = args.epochs
    parms["output_every"] = args.output_every
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
    parms["do_augment_permutation"] = False
    parms["do_augment_final_stable"] = False
    parms["do_unphysical_check"] = False

    # neural network options (will be overridden per sweep)
    parms["nhidden_shared"] = 0
    parms["nhidden_stability"] = 0
    parms["nhidden_growthrate"] = 0
    parms["nhidden_asymptotic"] = 0
    parms["nhidden_density"] = 0
    parms["nhidden_flux"] = 0
    parms["width_shared"] = 32
    parms["width_stability"] = 32
    parms["width_growthrate"] = 32
    parms["width_density"] = 32
    parms["width_flux"] = 32
    parms["dropout_probability"] = 0.0
    parms["do_batchnorm"] = True
    parms["do_fdotu"] = True
    parms["activation"] = nn.LeakyReLU

    # optimizer options
    parms["op"] = torch.optim.AdamW
    parms["adamw.amsgrad"] = False
    parms["adamw.weight_decay"] = 0
    parms["learning_rate"] = 5e-2  # overridden per sweep
    parms["adamw.fused"] = True
    parms["factor"] = 0.5
    parms["patience"] = 50
    parms["cooldown"] = 20
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

    summary_rows = []

    for (lr, w, nh, seed) in product(lr_grid, width_grid, nhid_grid, seed_grid):
        directory_name = f"model_lr_{lr}_w_{w}_nh_{nh}_seed_{seed}"
        if args.use_pcgrad:
            directory_name += "_pcgrad"

        print(f"Running {directory_name}")
        parms["learning_rate"] = lr
        parms["width_shared"] = w
        parms["width_stability"] = w
        parms["width_growthrate"] = w
        parms["width_density"] = w
        parms["width_flux"] = w
        parms["nhidden_shared"] = nh
        parms["nhidden_stability"] = nh
        parms["nhidden_growthrate"] = nh
        parms["nhidden_asymptotic"] = nh
        parms["nhidden_density"] = nh
        parms["nhidden_flux"] = nh
        parms["random_seed"] = seed
        parms["use_pcgrad"] = args.use_pcgrad

        # create the new directory and go in
        if not os.path.exists(directory_name):
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

        # parse final losses
        final_train = None
        final_test = None
        loss_path = os.path.join(os.getcwd(), "loss.dat")
        if os.path.exists(loss_path):
            with open(loss_path, "r") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
                if len(lines) > 1:
                    header = [h for h in lines[0].split("\t") if h]
                    cols = [h.split(":", 1)[-1] for h in header]
                    last = [p for p in lines[-1].split("\t") if p]
                    try:
                        train_idx = cols.index("train_loss")
                        test_idx = cols.index("test_loss")
                        final_train = float(last[train_idx])
                        final_test = float(last[test_idx])
                    except Exception:
                        pass
        summary_rows.append(
            {
                "model": directory_name,
                "learning_rate": lr,
                "width_shared": w,
                "nhidden_shared": nh,
                "seed": seed,
                "use_pcgrad": args.use_pcgrad,
                "final_train_loss": final_train,
                "final_test_loss": final_test,
            }
        )

        # return to the main directory
        os.chdir("..")

    # write summary CSV
    fieldnames = [
        "model",
        "learning_rate",
        "width_shared",
        "nhidden_shared",
        "seed",
        "use_pcgrad",
        "final_train_loss",
        "final_test_loss",
    ]
    write_header = not os.path.exists(args.summary)
    with open(args.summary, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
