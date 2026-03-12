#!/usr/bin/env python3
'''
Author: John McGuigan

Copyright: GPLv3 (see LICENSE file)

Launch Syne Tune using the configuration embedded in the parms dictionary.
'''

from pathlib import Path

from syne_tune import StoppingCriterion, Tuner
from syne_tune.backend import LocalBackend
from syne_tune.config_space import choice, lograndint, loguniform, randint, uniform
from syne_tune.optimizer.schedulers import FIFOScheduler, HyperbandScheduler

from ml_pytorch import build_default_parms


def make_domain(spec):
    if not isinstance(spec, dict) or "type" not in spec:
        return spec

    spec_type = spec["type"]
    if spec_type == "choice":
        return choice(spec["values"])
    if spec_type == "randint":
        return randint(spec["lower"], spec["upper"] + 1)
    if spec_type == "lograndint":
        return lograndint(spec["lower"], spec["upper"] + 1)
    if spec_type == "uniform":
        return uniform(spec["lower"], spec["upper"])
    if spec_type == "loguniform":
        return loguniform(spec["lower"], spec["upper"])

    raise ValueError(f"Unknown Syne Tune config space type '{spec_type}'")


def build_config_space(parms):
    syne_tune_cfg = parms["syne_tune"]
    config_space = {}
    for key, value in syne_tune_cfg["config_space"].items():
        config_space[key] = make_domain(value)

    max_resource_attr = syne_tune_cfg["max_resource_attr"]
    if max_resource_attr not in config_space:
        config_space[max_resource_attr] = parms[max_resource_attr]

    return config_space


def build_scheduler(parms, config_space):
    syne_tune_cfg = parms["syne_tune"]
    scheduler_cfg = dict(syne_tune_cfg["scheduler"])
    scheduler_name = scheduler_cfg.pop("name", "hyperband")

    common_kwargs = {
        "config_space": config_space,
        "metric": syne_tune_cfg["metric"],
        "mode": syne_tune_cfg["mode"],
    }

    if scheduler_name == "fifo":
        return FIFOScheduler(**common_kwargs, **scheduler_cfg)

    if scheduler_name == "hyperband":
        common_kwargs["resource_attr"] = syne_tune_cfg["resource_attr"]
        common_kwargs["max_resource_attr"] = syne_tune_cfg["max_resource_attr"]
        return HyperbandScheduler(**common_kwargs, **scheduler_cfg)

    raise ValueError(f"Unknown Syne Tune scheduler '{scheduler_name}'")


def main():
    parms = build_default_parms()
    syne_tune_cfg = parms["syne_tune"]

    entry_point = Path(__file__).with_name("run_trial.py")
    config_space = build_config_space(parms)
    backend_cfg = dict(syne_tune_cfg["backend"])
    tuner_cfg = dict(syne_tune_cfg["tuner"])
    stop_cfg = dict(syne_tune_cfg["stop"])

    backend = LocalBackend(
        entry_point=str(entry_point),
        **backend_cfg,
    )

    scheduler = build_scheduler(parms, config_space)

    tuner = Tuner(
        trial_backend=backend,
        scheduler=scheduler,
        stop_criterion=StoppingCriterion(**stop_cfg),
        **tuner_cfg,
    )
    tuner.run()


if __name__ == "__main__":
    main()
