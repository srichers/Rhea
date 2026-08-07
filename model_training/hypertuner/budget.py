'''
Author: John McGuigan

Copyright: GPLv3 (see LICENSE file)

Budget planning utilities for multi-fidelity training.

This module converts a scheduler resource budget into concrete raw training
steps. It scales step counts by model parameter count, batch size, and learning
rate so larger or slower configurations can be compared on a more normalized
compute budget. The main entry point is `StepBudgetPolicy`, which mutates a
runtime train config with `max_steps` and validation checkpoints.
'''
from math import ceil

from .registry import BudgetCfg, BudgetPlan, BudgetPolicyCfg, BudgetRuntimeCfg, ResourceCfg


class StepBudgetPolicy:
    def __init__(self, cfg: BudgetPolicyCfg):
        self.cfg = cfg
        self.budg_cfg = cfg.budget
        self.resource_cfg = cfg.resource
        self.max_budget = self.resource_cfg.max_budget
        self.grace_period = self.resource_cfg.grace_period
        self.reduction_factor = self.resource_cfg.reduction_factor

    def count_trainable_params(self, model):
        self.num_train_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    def compute_cost_scale(self, batch_size: int) -> float:
        param_scale = (float(self.num_train_params) / float(self.budg_cfg.ref_params)) ** float(
            self.budg_cfg.param_exponent
        )
        batch_scale = (float(batch_size) / float(self.budg_cfg.ref_batch_size)) ** float(
            self.budg_cfg.batch_exponent
        )
        return param_scale * batch_scale

    def lr_need_scale(self, lr: float) -> float:
        return (float(self.budg_cfg.ref_lr) / float(lr)) ** float(self.budg_cfg.lr_exponent)

    def budget_rungs(self) -> list[int]:
        self.rungs = []
        self.budget = self.resource_cfg.grace_period

        while self.budget < self.max_budget:
            self.rungs.append(int(round(self.budget)))
            self.budget *= self.reduction_factor

        self.rungs.append(self.max_budget)
        return sorted(set(self.rungs))

    def build_plan(self, cfg: BudgetRuntimeCfg, model) -> BudgetPlan:
        train = cfg.train
        loader = cfg.loader
        self.count_trainable_params(model)
        batch_size = int(loader["batch_size"])
        lr = float(train["lr"])

        compute_cost_scale = self.compute_cost_scale(batch_size=batch_size)
        lr_need_scale = self.lr_need_scale(lr=lr)
        raw_steps_per_budget = (
            float(self.budg_cfg.raw_steps_per_budget)
            * lr_need_scale
            / compute_cost_scale
        )

        step_to_resource = {}
        for budget in self.budget_rungs():
            raw_step = max(1, int(ceil(float(budget) * raw_steps_per_budget)))
            step_to_resource[raw_step] = budget

        raw_max_steps = max(step_to_resource)
        val_steps = sorted(step_to_resource)

        return BudgetPlan(
            resource_attr=self.resource_cfg.resource_attr,
            raw_max_steps=raw_max_steps,
            val_steps=val_steps,
            step_to_resource=step_to_resource,
            compute_cost_scale=compute_cost_scale,
            lr_need_scale=lr_need_scale,
            raw_steps_per_budget=raw_steps_per_budget,
        )

    def apply(self, cfg: BudgetRuntimeCfg, model) -> BudgetPlan:
        plan = self.build_plan(cfg, model)

        cfg.train["max_steps"] = plan.raw_max_steps
        cfg.train["val_steps"] = plan.val_steps
        cfg.train["val_every_steps"] = None

        return plan


class EpochBudgetTracker:
    """Reports normalized scheduler budget from the current epoch trainer."""

    def __init__(
        self,
        *,
        budget_cfg: BudgetCfg,
        resource_attr: str,
        max_resource_attr: str,
        max_budget: int,
        num_train_params: int,
        batch_size: int,
        lr: float,
    ):
        self.budget_cfg = budget_cfg
        self.resource_attr = resource_attr
        self.max_resource_attr = max_resource_attr
        self.max_budget = int(max_budget)
        self.num_train_params = int(num_train_params)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.compute_cost_scale = self._compute_cost_scale()
        self.lr_need_scale = self._lr_need_scale()
        self.raw_steps_per_budget = (
            float(self.budget_cfg.raw_steps_per_budget)
            * self.lr_need_scale
            / self.compute_cost_scale
        )

    def _compute_cost_scale(self) -> float:
        param_scale = (float(self.num_train_params) / float(self.budget_cfg.ref_params)) ** float(
            self.budget_cfg.param_exponent
        )
        batch_scale = (float(self.batch_size) / float(self.budget_cfg.ref_batch_size)) ** float(
            self.budget_cfg.batch_exponent
        )
        return param_scale * batch_scale

    def _lr_need_scale(self) -> float:
        return (float(self.budget_cfg.ref_lr) / float(self.lr)) ** float(
            self.budget_cfg.lr_exponent
        )

    def resource_for_epoch(self, epoch: int) -> int:
        raw_budget = float(epoch) / self.raw_steps_per_budget
        return min(self.max_budget, max(1, int(ceil(raw_budget))))

    def uncapped_max_epoch(self) -> int:
        return max(1, int(ceil(float(self.max_budget) * self.raw_steps_per_budget)))

    def max_epoch(self) -> int:
        max_epoch = self.uncapped_max_epoch()
        if self.budget_cfg.max_epochs_cap is not None:
            max_epoch = min(max_epoch, int(self.budget_cfg.max_epochs_cap))
        return max_epoch

    def is_epoch_capped(self) -> bool:
        return self.max_epoch() < self.uncapped_max_epoch()

    def metrics_for_epoch(self, epoch: int) -> dict[str, int | float]:
        return {
            self.resource_attr: self.resource_for_epoch(epoch),
            self.max_resource_attr: self.max_budget,
            "budget_max_epoch": self.max_epoch(),
            "budget_uncapped_max_epoch": self.uncapped_max_epoch(),
            "budget_epoch_capped": int(self.is_epoch_capped()),
            "budget_ref_epochs": int(self.budget_cfg.ref_epochs),
            "budget_compute_cost_scale": self.compute_cost_scale,
            "budget_lr_need_scale": self.lr_need_scale,
            "budget_raw_steps_per_budget": self.raw_steps_per_budget,
        }


def _resolve_max_budget(parms: dict, syne_tune_cfg: dict, resource_cfg: dict) -> int:
    max_resource_attr = resource_cfg.get("max_resource_attr", "max_budget")
    if max_resource_attr in parms:
        return int(parms[max_resource_attr])

    if "max_budget" in resource_cfg:
        return int(resource_cfg["max_budget"])

    space_cfg = syne_tune_cfg.get("space", {})
    max_resource_space = space_cfg.get(max_resource_attr, {})
    if "value" in max_resource_space:
        return int(max_resource_space["value"])

    return int(ResourceCfg().max_budget)


def build_epoch_budget_tracker(parms: dict, model) -> EpochBudgetTracker | None:
    """Build a normalized budget reporter for the current epoch-based trainer."""

    syne_tune_cfg = parms.get("syne_tune", {})
    resource_cfg = syne_tune_cfg.get("resource", {})
    budget_cfg = syne_tune_cfg.get("budget")
    if not budget_cfg or resource_cfg.get("resource_attr") != "budget":
        return None

    num_train_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return EpochBudgetTracker(
        budget_cfg=BudgetCfg(**budget_cfg),
        resource_attr=resource_cfg.get("resource_attr", "budget"),
        max_resource_attr=resource_cfg.get("max_resource_attr", "max_budget"),
        max_budget=_resolve_max_budget(parms, syne_tune_cfg, resource_cfg),
        num_train_params=num_train_params,
        batch_size=int(parms["loader.batch_size"]),
        lr=float(parms["learning_rate"]),
    )
