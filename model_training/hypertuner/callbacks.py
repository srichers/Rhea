'''
Author: John McGuigan

Copyright: GPLv3 (see LICENSE file)

Callbacks that adapt Rhea training metrics for Syne Tune.

This module currently provides `SyneTuneReporter`, a small validation-end
callback that collects trainer state, attaches budget-plan metadata when
available, and forwards scalar metrics to a Syne Tune `Reporter`.
'''
class SyneTuneReporter:
    def __init__(self, reporter, budget_plan=None):
        self.reporter = reporter
        self.budget_plan = budget_plan

    def on_validation_end(self, trainer):
        result = {
            "epoch": trainer.epoch + 1,
            "raw_step": trainer.global_step,
            "train_loss": float(trainer.train_loss),
            "val_loss": float(trainer.val_loss),
            "num_params": int(sum(p.numel() for p in trainer.model.parameters())),
            "max_steps": trainer.cfg.max_steps,
        }

        if self.budget_plan is None:
            result["step"] = trainer.global_step
        else:
            result[self.budget_plan.resource_attr] = (
                self.budget_plan.resource_for_step(trainer.global_step)
            )
            result["compute_cost_scale"] = self.budget_plan.compute_cost_scale
            result["lr_need_scale"] = self.budget_plan.lr_need_scale
            result["raw_steps_per_budget"] = self.budget_plan.raw_steps_per_budget

        self.reporter(**result)
