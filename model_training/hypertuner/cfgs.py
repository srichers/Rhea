'''
Author: John McGuigan

Copyright: GPLv3 (see LICENSE file)

Configuration builders for the hyperparameter tuning stack.

This module translates plain dictionaries from defaults or config files into
typed registry dataclasses. It builds resource, budget, sampling, plotting, and
Syne Tune configuration objects, including optional cost-balanced
`points_to_evaluate` generation.
'''
from .registry import (
    ASHACfg,
    BOHBCfg,
    BudgetCfg,
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


def _enum_value(enum_cls, raw):
    key = raw.upper()
    if key in enum_cls.__members__:
        return enum_cls[key]

    return enum_cls(raw.lower())


def build_resource_cfg(
    *,
    resource: dict,
    space: dict | None = None,
) -> ResourceCfg:
    cfg = dict(resource)
    max_resource_attr = cfg.get("max_resource_attr", "max_budget")
    if "max_budget" not in cfg and space is not None:
        cfg["max_budget"] = int(space[max_resource_attr]["value"])

    return ResourceCfg(**cfg)


def build_budget_policy_cfg(
    *,
    budget: dict,
    resource: dict,
    space: dict | None = None,
) -> BudgetPolicyCfg:
    budget_cfg = BudgetCfg(**budget)
    resource_cfg = build_resource_cfg(resource=resource, space=space)
    return BudgetPolicyCfg(budget=budget_cfg, resource=resource_cfg)


def build_budget_runtime_cfg(cfg: dict) -> BudgetRuntimeCfg:
    return BudgetRuntimeCfg(
        train=cfg["train"],
        loader=cfg["loader"],
    )


def build_cost_balanced_sampler_cfg(
    *,
    raw_space: dict,
    model_tiers: dict,
    budget: BudgetCfg,
    sampling: SamplingCfg,
    base_model_cfg: dict,
    max_num_trials_started: int | None,
    random_seed: int,
) -> CostBalancedSamplerCfg:
    return CostBalancedSamplerCfg(
        raw_space=raw_space,
        model_tiers=model_tiers,
        budget=budget,
        sampling=sampling,
        base_model_cfg=base_model_cfg,
        max_num_trials_started=max_num_trials_started,
        random_seed=random_seed,
    )


def build_tune_plot_cfg(
    *,
    plots: dict,
    results_root: str | None,
) -> TunePlotCfg:
    cfg = dict(plots)
    cfg["results_root"] = results_root
    return TunePlotCfg(**cfg)


def build_syne_tune_cfg(
    *,
    config_space: dict[str, object] | None = None,
    points_to_evaluate: list[dict] | None = None,
    raw_space: dict | None = None,
    model_tiers: dict | None = None,
    budget: dict | None = None,
    sampling: dict | None = None,
    base_model_cfg: dict | None = None,
    resource: dict | None = None,
    metric: str = "val_loss",
    mode: str = "MIN",
    scheduler: str = "ASHA",
    resource_attr: str = "epoch",
    max_resource_attr: str = "max_epochs",
    max_resource: int | None = None,
    grace_period: int = 2,
    reduction_factor: float = 3,
    brackets: int = 1,
    bohb_num_min_data_points: int | None = None,
    bohb_top_n_percent: int = 15,
    bohb_min_bandwidth: float = 1e-3,
    bohb_num_candidates: int = 64,
    bohb_bandwidth_factor: int = 3,
    bohb_random_fraction: float = 0.33,
    max_wallclock_time: float | None = 300,
    max_num_trials_started: int | None = 20,
    max_num_trials_completed: int | None = None,
    n_workers: int = 1,
    random_seed: int = 42,
    tuner_name: str = "python-backend",
    project_root: str | None = None,
    results_root: str | None = "syne-tune",
    rotate_gpus: bool = False,
    delete_checkpoints: bool = False,
    save_tuner: bool = True,
    sleep_time: float = 5.0,
    results_update_interval: float = 10.0,
    print_update_interval: float = 30.0,
) -> SyneTuneCfg:
    mode = _enum_value(TuneMode, mode)
    scheduler = _enum_value(TuneSchedulerKind, scheduler)
    budget_cfg = BudgetCfg(**budget) if budget is not None else None
    sampling_cfg = SamplingCfg(**sampling) if sampling is not None else None
    resource_cfg = (
        build_resource_cfg(resource=resource, space=raw_space)
        if resource is not None
        else None
    )

    if resource_cfg is not None:
        resource_attr = resource_cfg.resource_attr
        max_resource_attr = resource_cfg.max_resource_attr
        max_resource = resource_cfg.max_budget
        grace_period = resource_cfg.grace_period
        reduction_factor = resource_cfg.reduction_factor
        brackets = resource_cfg.brackets

    if points_to_evaluate is None and sampling_cfg is not None:
        sampler_cfg = build_cost_balanced_sampler_cfg(
            raw_space=raw_space or {},
            model_tiers=model_tiers or {},
            budget=budget_cfg,
            sampling=sampling_cfg,
            base_model_cfg=base_model_cfg or {},
            max_num_trials_started=max_num_trials_started,
            random_seed=random_seed,
        )
        points_to_evaluate = CostBalancedSampler(sampler_cfg).points_to_evaluate()

    return SyneTuneCfg(
        config_space=config_space or {},
        points_to_evaluate=points_to_evaluate,
        resource=resource_cfg,
        budget=budget_cfg,
        sampling=sampling_cfg,
        metric=metric,
        mode=mode,
        scheduler=scheduler,
        asha=ASHACfg(
            resource_attr=resource_attr,
            max_resource_attr=max_resource_attr,
            max_resource=max_resource,
            grace_period=grace_period,
            reduction_factor=reduction_factor,
            brackets=brackets,
        ),
        bohb=BOHBCfg(
            num_min_data_points=bohb_num_min_data_points,
            top_n_percent=bohb_top_n_percent,
            min_bandwidth=bohb_min_bandwidth,
            num_candidates=bohb_num_candidates,
            bandwidth_factor=bohb_bandwidth_factor,
            random_fraction=bohb_random_fraction,
        ),
        max_wallclock_time=max_wallclock_time,
        max_num_trials_started=max_num_trials_started,
        max_num_trials_completed=max_num_trials_completed,
        n_workers=n_workers,
        random_seed=random_seed,
        tuner_name=tuner_name,
        project_root=project_root,
        results_root=results_root,
        rotate_gpus=rotate_gpus,
        delete_checkpoints=delete_checkpoints,
        save_tuner=save_tuner,
        sleep_time=sleep_time,
        results_update_interval=results_update_interval,
        print_update_interval=print_update_interval,
    )
