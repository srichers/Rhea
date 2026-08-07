'''
Author: John McGuigan

Copyright: GPLv3 (see LICENSE file)

Package exports for Rhea hyperparameter tuning.

This module collects the public tuning API in one import location. It exposes
configuration dataclasses, config builders, budget utilities, Syne Tune
integration, plotting helpers, and cost-balanced sampling helpers used by
`run_trial.py` and external tuning scripts.
'''
from .budget import EpochBudgetTracker, StepBudgetPolicy, build_epoch_budget_tracker
from .callbacks import SyneTuneReporter
from .cfgs import (
    build_budget_policy_cfg,
    build_budget_runtime_cfg,
    build_cost_balanced_sampler_cfg,
    build_resource_cfg,
    build_syne_tune_cfg,
    build_tune_plot_cfg,
)
from .registry import (
    ASHACfg,
    BOHBCfg,
    BudgetCfg,
    BudgetPlan,
    BudgetPolicyCfg,
    BudgetRuntimeCfg,
    CostBalancedSamplerCfg,
    ResourceCfg,
    SamplingCfg,
    SyneTuneCfg,
    TunePlotCfg,
    TuneMode,
    TuneSchedulerKind,
)
from .sampling import CostBalancedSampler
from .space import build_config_space


def __getattr__(name):
    if name == "HyperTunePlotter":
        from .plot import HyperTunePlotter

        return HyperTunePlotter
    if name == "HyperTuner":
        from .syne import HyperTuner

        return HyperTuner
    raise AttributeError(f"module 'hypertuner' has no attribute '{name}'")

__all__ = [
    "ASHACfg",
    "BOHBCfg",
    "BudgetPlan",
    "BudgetCfg",
    "BudgetPolicyCfg",
    "BudgetRuntimeCfg",
    "CostBalancedSampler",
    "CostBalancedSamplerCfg",
    "EpochBudgetTracker",
    "HyperTuner",
    "HyperTunePlotter",
    "ResourceCfg",
    "SamplingCfg",
    "StepBudgetPolicy",
    "SyneTuneCfg",
    "SyneTuneReporter",
    "TunePlotCfg",
    "TuneMode",
    "TuneSchedulerKind",
    "build_budget_policy_cfg",
    "build_budget_runtime_cfg",
    "build_cost_balanced_sampler_cfg",
    "build_config_space",
    "build_epoch_budget_tracker",
    "build_resource_cfg",
    "build_syne_tune_cfg",
    "build_tune_plot_cfg",
]
