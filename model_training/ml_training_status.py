'''
Author: John McGuigan

Copyright: GPLv3 (see LICENSE file)

Shared training-status helpers.

This module defines structured exceptions and validation helpers used by the
training loop. Policy decisions such as converting a divergence into a Syne Tune
penalty metric should live in the runner layer, not in the core trainer.
'''

import math

import torch


class TrainingDivergedError(RuntimeError):
    def __init__(self, reason, metrics=None):
        super().__init__(reason)
        self.reason = reason
        self.metrics = dict(metrics or {})


def validate_positive_predicted_ntotal(ntot_pred, traintest, epoch):
    invalid = ntot_pred <= 0
    if not torch.any(invalid):
        return

    invalid_count = int(torch.sum(invalid).item())
    raise TrainingDivergedError(
        "predicted non-positive total density",
        {
            "epoch": epoch,
            "diverged": 1,
            "divergence_reason": "nonpositive_predicted_ntotal",
            "invalid_ntot_pred_" + traintest + "_count": invalid_count,
        },
    )


def validate_finite_metrics(loss_dict):
    nonfinite_metrics = [
        key
        for key, value in loss_dict.items()
        if isinstance(value, float) and not math.isfinite(value)
    ]
    if not nonfinite_metrics:
        return

    metrics = dict(loss_dict)
    metrics["diverged"] = 1
    metrics["divergence_reason"] = "nonfinite_metric"
    metrics["nonfinite_metrics"] = ",".join(nonfinite_metrics)
    raise TrainingDivergedError("non-finite training metric", metrics)
