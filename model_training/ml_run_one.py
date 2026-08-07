#!/usr/bin/env python3
"""Run one Rhea training trial without launching a Syne Tune experiment."""

from ml_run_hytune import run_single_trial


def main():
    run_single_trial({
        "model_tier": "small",
        "lr": 1e-3,
        "batch_size": 64,
        "seed": 42,
    })


if __name__ == "__main__":
    main()
