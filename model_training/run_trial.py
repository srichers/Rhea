'''
Author: Sherwood Richers / contributors

Copyright: GPLv3 (see LICENSE file)

Configurable single-trial entrypoint for training.
'''

import argparse
import ast
import json
import os
from pathlib import Path

import e3nn.o3
import torch
from torch import nn

from ml_read_data import read_asymptotic_data, read_stable_data
from ml_trainmodel import train_asymptotic_model


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")


def parse_cli_value(raw):
    if isinstance(raw, (int, float, bool, list, dict)) or raw is None:
        return raw

    text = str(raw).strip()
    lowered = text.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def parse_unknown_cli(unknown):
    overrides = {}
    i = 0
    while i < len(unknown):
        token = unknown[i]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected positional argument: {token}")
        key = token[2:]
        if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
            value = parse_cli_value(unknown[i + 1])
            i += 2
        else:
            value = True
            i += 1
        overrides[key] = value
    return overrides


def default_parms(repo_root):
    data_dir = repo_root / "model_training" / "data"
    return {
        "database_list": [
            str(data_dir / "dummy_asymptotic.h5"),
            str(data_dir / "dummy_asymptotic.h5"),
        ],
        "stable_database_list": [
            str(data_dir / "stable_oneflavor_database.h5"),
            str(data_dir / "stable_random_database.h5"),
            str(data_dir / "stable_zerofluxfac_database.h5"),
        ],
        "samples_per_database": 1000000,
        "test_size": 0.1,
        "epochs": 10,
        "output_every": 10,
        "average_heavies_in_final_state": False,
        "conserve_lepton_number": True,
        "random_seed": 42,
        "loader.batch_size": 10,
        "loader.num_workers": 1,
        "loader.prefetch_factor": 1,
        "sampler": "WeightedRandomSampler",
        "weightedrandomsampler.epoch_num_samples": 10,
        "scalar_activation": "silu",
        "nonscalar_activation": "sigmoid",
        "do_learn_task_weights": False,
        "task_weight_stability": 1.0,
        "task_weight_F4": 1.0,
        "task_weight_unphysical": 1.0,
        "task_weight_growthrate": 1.0,
        "do_augment_final_stable": False,
        "do_unphysical_check": True,
        "nhidden_shared": 1,
        "nhidden_stability": 3,
        "nhidden_growthrate": 3,
        "nhidden_F4": 3,
        "irreps_hidden": "4x0e + 4x1o",
        "dropout_probability": 0.0,
        "do_batchnorm": False,
        "do_fdotu": True,
        "activation": "leaky_relu",
        "op": "AdamW",
        "adamw.amsgrad": False,
        "adamw.weight_decay": 0.0,
        "adamw.fused": True,
        "learning_rate": 2e-4,
        "patience": 500,
        "cooldown": 500,
        "factor": 0.5,
        "warmup_iters": 0,
        "min_lr": 0.0,
        "NF": 3,
        "device": "auto",
        "use_box3d_residual": True,
        "use_box3d_growth_baseline": True,
        "report_box3d_control_metrics": True,
        "box3d_resolution_theta": 21,
        "box3d_resolution_phi": 41,
    }


def apply_env_overrides(parms):
    env_map = {
        "RHEA_BATCH_SIZE": ("loader.batch_size", int),
        "RHEA_NUM_WORKERS": ("loader.num_workers", int),
        "RHEA_PREFETCH_FACTOR": ("loader.prefetch_factor", int),
        "RHEA_EPOCHS": ("epochs", int),
        "RHEA_OUTPUT_EVERY": ("output_every", int),
        "RHEA_EPOCH_SAMPLES": ("weightedrandomsampler.epoch_num_samples", int),
        "RHEA_LR": ("learning_rate", float),
        "RHEA_WEIGHT_DECAY": ("adamw.weight_decay", float),
        "RHEA_BATCHNORM": ("do_batchnorm", parse_bool),
        "RHEA_LEARN_TASK_WEIGHTS": ("do_learn_task_weights", parse_bool),
        "RHEA_SAMPLES_PER_DB": ("samples_per_database", int),
        "RHEA_RANDOM_SEED": ("random_seed", int),
        "RHEA_DEVICE": ("device", str),
        "RHEA_USE_BOX3D_RESIDUAL": ("use_box3d_residual", parse_bool),
        "RHEA_USE_BOX3D_GROWTH_BASELINE": ("use_box3d_growth_baseline", parse_bool),
        "RHEA_REPORT_BOX3D_CONTROL_METRICS": ("report_box3d_control_metrics", parse_bool),
        "RHEA_BOX3D_RESOL_THETA": ("box3d_resolution_theta", int),
        "RHEA_BOX3D_RESOL_PHI": ("box3d_resolution_phi", int),
    }
    for env_key, (parm_key, caster) in env_map.items():
        if env_key in os.environ and os.environ[env_key] != "":
            parms[parm_key] = caster(os.environ[env_key])

    if "RHEA_DATABASE_LIST" in os.environ and os.environ["RHEA_DATABASE_LIST"].strip():
        parms["database_list"] = [x.strip() for x in os.environ["RHEA_DATABASE_LIST"].split(",") if x.strip()]
    if "RHEA_STABLE_DATABASE_LIST" in os.environ and os.environ["RHEA_STABLE_DATABASE_LIST"].strip():
        parms["stable_database_list"] = [x.strip() for x in os.environ["RHEA_STABLE_DATABASE_LIST"].split(",") if x.strip()]

    return parms


def _resolve_activation(name):
    if callable(name):
        return name
    table = {
        "silu": nn.functional.silu,
        "relu": nn.functional.relu,
        "leaky_relu": nn.functional.leaky_relu,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
    }
    if name not in table:
        raise ValueError(f"Unknown activation: {name}")
    return table[name]


def resolve_runtime_parms(parms):
    if isinstance(parms.get("op"), str):
        optimizers = {
            "AdamW": torch.optim.AdamW,
            "SGD": torch.optim.SGD,
        }
        if parms["op"] not in optimizers:
            raise ValueError(f"Unknown optimizer string: {parms['op']}")
        parms["op"] = optimizers[parms["op"]]

    if isinstance(parms.get("sampler"), str):
        samplers = {
            "WeightedRandomSampler": torch.utils.data.WeightedRandomSampler,
            "SequentialSampler": torch.utils.data.SequentialSampler,
        }
        if parms["sampler"] not in samplers:
            raise ValueError(f"Unknown sampler string: {parms['sampler']}")
        parms["sampler"] = samplers[parms["sampler"]]

    if isinstance(parms.get("irreps_hidden"), str):
        parms["irreps_hidden"] = e3nn.o3.Irreps(parms["irreps_hidden"])

    parms["scalar_activation"] = _resolve_activation(parms.get("scalar_activation", "silu"))
    parms["nonscalar_activation"] = _resolve_activation(parms.get("nonscalar_activation", "sigmoid"))

    if isinstance(parms.get("activation"), str):
        activations = {
            "LeakyReLU": nn.LeakyReLU,
            "ReLU": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "relu": nn.ReLU,
        }
        if parms["activation"] not in activations:
            raise ValueError(f"Unknown module activation string: {parms['activation']}")
        parms["activation"] = activations[parms["activation"]]

    parms["do_batchnorm"] = parse_bool(parms.get("do_batchnorm", False))
    parms["do_learn_task_weights"] = parse_bool(parms.get("do_learn_task_weights", False))
    parms["do_unphysical_check"] = parse_bool(parms.get("do_unphysical_check", True))
    parms["average_heavies_in_final_state"] = parse_bool(parms.get("average_heavies_in_final_state", False))
    parms["conserve_lepton_number"] = parse_bool(parms.get("conserve_lepton_number", True))
    parms["use_box3d_residual"] = parse_bool(parms.get("use_box3d_residual", True))
    parms["use_box3d_growth_baseline"] = parse_bool(parms.get("use_box3d_growth_baseline", True))
    parms["report_box3d_control_metrics"] = parse_bool(
        parms.get("report_box3d_control_metrics", parms["use_box3d_residual"])
    )

    if parms.get("device", "auto") == "auto":
        parms["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    return parms


def main():
    parser = argparse.ArgumentParser(description="Run one training trial for Rhea")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--config-json", type=str, default=None, help="Inline JSON config dictionary")
    parser.add_argument("--output-dir", type=str, default=None, help="Working directory for outputs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args, unknown = parser.parse_known_args()

    repo_root = Path(__file__).resolve().parents[1]
    parms = default_parms(repo_root)
    parms = apply_env_overrides(parms)

    if args.config:
        with open(args.config, "r", encoding="utf-8") as fin:
            parms.update(json.load(fin))
    if args.config_json:
        parms.update(json.loads(args.config_json))

    parms.update(parse_unknown_cli(unknown))
    if args.seed is not None:
        parms["random_seed"] = args.seed
    if args.epochs is not None:
        parms["epochs"] = args.epochs

    parms = resolve_runtime_parms(parms)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        os.chdir(args.output_dir)

    dataset_asymptotic_train_list, dataset_asymptotic_test_list = read_asymptotic_data(parms)
    dataset_stable_train_list, dataset_stable_test_list = read_stable_data(parms)

    train_asymptotic_model(
        parms,
        dataset_asymptotic_train_list,
        dataset_asymptotic_test_list,
        dataset_stable_train_list,
        dataset_stable_test_list,
    )


if __name__ == "__main__":
    main()
