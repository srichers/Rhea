#################################################################
# --------->>>| Function that always returns current learning
# --------->>>| rate even for different schedulers
# ###############################################################
def current_lr(optim, sched=None):
    # Valid for modern schedulers: StepLR, CosineAnnealingLR, etc.
    if sched is not None and hasattr(sched, "get_last_lr"):
        try:
            return sched.get_last_lr()[0]
        except Exception:
            pass
    # always valid, including for ReduceLROnPlateau
    return optim.param_groups[0]["lr"]
