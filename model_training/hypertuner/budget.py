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

from .registry import BudgetPlan, BudgetPolicyCfg, BudgetRuntimeCfg


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
