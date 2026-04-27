'''
Author: John McGuigan

Copyright: GPLv3 (see LICENSE file)

Typed configuration registry for hyperparameter tuning.

This module defines the dataclasses and enums shared across the tuner:
scheduler kind and metric mode, budget/resource settings, runtime budget
plans, cost-balanced sampling settings, plotting settings, and the top-level
`SyneTuneCfg` consumed by `HyperTuner`.
'''
from dataclasses import dataclass, field
from enum import Enum


class TuneMode(str, Enum):
    MIN = "min"
    MAX = "max"


class TuneSchedulerKind(str, Enum):
    ASHA = "asha"
    BOHB = "bohb"
    RANDOM_SEARCH = "random_search"


@dataclass(frozen=True)
class SamplingCfg:
    kind: str = "cost_balanced"
    cost_exponent: float = 0.5
    candidate_pool_size: int = 256
    num_points: int | None = None


@dataclass(frozen=True)
class BudgetCfg:
    kind: str = "normalized_split"
    ref_params: int = 100000
    ref_batch_size: int = 64
    ref_lr: float = 1e-3
    param_exponent: float = 0.25
    batch_exponent: float = 0.5
    lr_exponent: float = 0.25
    raw_steps_per_budget: float = 1.0


@dataclass(frozen=True)
class ResourceCfg:
    resource_attr: str = "budget"
    max_resource_attr: str = "max_budget"
    max_budget: int = 600
    grace_period: int = 60
    reduction_factor: int = 3
    brackets: int = 1


@dataclass(frozen=True)
class BudgetPolicyCfg:
    budget: BudgetCfg = field(default_factory=BudgetCfg)
    resource: ResourceCfg = field(default_factory=ResourceCfg)


@dataclass(frozen=True)
class BudgetRuntimeCfg:
    train: dict = field(default_factory=dict)
    loader: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BudgetPlan:
    resource_attr: str
    raw_max_steps: int
    val_steps: list[int]
    step_to_resource: dict[int, int]
    compute_cost_scale: float
    lr_need_scale: float
    raw_steps_per_budget: float

    def resource_for_step(self, step: int) -> int:
        if step in self.step_to_resource:
            return self.step_to_resource[step]

        eligible = [
            resource
            for raw_step, resource in self.step_to_resource.items()
            if raw_step <= step
        ]
        if eligible:
            return max(eligible)

        return min(self.step_to_resource.values())


@dataclass(frozen=True)
class CostBalancedSamplerCfg:
    raw_space: dict = field(default_factory=dict)
    model_tiers: dict = field(default_factory=dict)
    budget: BudgetCfg = field(default_factory=BudgetCfg)
    sampling: SamplingCfg = field(default_factory=SamplingCfg)
    base_model_cfg: dict = field(default_factory=dict)
    max_num_trials_started: int | None = None
    random_seed: int = 42


@dataclass(frozen=True)
class TunePlotCfg:
    enabled: bool = False
    metric: str = "val_loss"
    best_over_time: bool = True
    trials_over_time: bool = True
    output_dir: str | None = "plots"
    best_filename: str = "best_val_loss.png"
    trials_filename: str = "trials_val_loss.png"
    show: bool = False
    figsize: list[float] | None = None
    results_root: str | None = "syne-tune"


@dataclass(frozen=True)
class ASHACfg:
    resource_attr: str = "epoch"
    max_resource_attr: str = "max_epochs"
    max_resource: int | None = None
    grace_period: int = 2
    reduction_factor: float = 3
    brackets: int = 1


@dataclass(frozen=True)
class BOHBCfg:
    num_min_data_points: int | None = None
    top_n_percent: int = 15
    min_bandwidth: float = 1e-3
    num_candidates: int = 64
    bandwidth_factor: int = 3
    random_fraction: float = 0.33


@dataclass(frozen=True)
class SyneTuneCfg:
    config_space: dict[str, object] = field(default_factory=dict)
    points_to_evaluate: list[dict] | None = None
    resource: ResourceCfg | None = None
    budget: BudgetCfg | None = None
    sampling: SamplingCfg | None = None
    metric: str = "val_loss"
    mode: TuneMode = TuneMode.MIN
    scheduler: TuneSchedulerKind = TuneSchedulerKind.ASHA
    asha: ASHACfg = field(default_factory=ASHACfg)
    bohb: BOHBCfg = field(default_factory=BOHBCfg)
    max_wallclock_time: float | None = 300
    max_num_trials_started: int | None = 20
    max_num_trials_completed: int | None = None
    n_workers: int = 1
    random_seed: int = 42
    tuner_name: str = "python-backend"
    project_root: str | None = None
    results_root: str | None = "syne-tune"
    rotate_gpus: bool = False
    delete_checkpoints: bool = False
    save_tuner: bool = True
    sleep_time: float = 5.0
    results_update_interval: float = 10.0
    print_update_interval: float = 30.0

    @property
    def do_minimize(self) -> bool:
        return self.mode == TuneMode.MIN
