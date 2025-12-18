import torch


def current_lr(optimizer, scheduler=None):
    """Return the current learning rate from the optimizer or scheduler."""
    if scheduler is not None and hasattr(scheduler, "get_last_lr"):
        lrs = scheduler.get_last_lr()
        if lrs:
            return lrs[0]
    if optimizer.param_groups:
        return optimizer.param_groups[0].get("lr", 0.0)
    return 0.0
