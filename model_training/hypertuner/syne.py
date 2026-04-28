'''
Author: John McGuigan

Copyright: GPLv3 (see LICENSE file)

Syne Tune orchestration for Rhea training trials.

This module contains `HyperTuner`, the main adapter between Rhea's callable
training trial and Syne Tune. It configures project/result paths, builds the
Python backend, selects a scheduler, builds stopping criteria, and launches the
Syne Tune `Tuner`.

Supported schedulers are random search, ASHA for aggressive multi-fidelity
early stopping, and BOHB for Bayesian optimization with Hyperband-style
resource allocation.
'''

import os
from pathlib import Path
from collections.abc import Callable

from syne_tune import StoppingCriterion, Tuner
from syne_tune.backend import PythonBackend
from syne_tune.optimizer.baselines import BOHB, RandomSearch
from syne_tune.optimizer.schedulers.asha import AsynchronousSuccessiveHalving
from syne_tune.optimizer.schedulers.searchers.last_value_multi_fidelity_searcher import (
    LastValueMultiFidelitySearcher,
)

from .registry import SyneTuneCfg, TuneSchedulerKind


class HyperTuner:
    def __init__(self, tune_function: Callable, cfg: SyneTuneCfg):
        self.cfg = cfg
        self.tune_function = tune_function
        self.config_space = cfg.config_space
        self.configure_environment()
        self.trial_backend = self.build_backend()
        self.scheduler = self.build_sched()
        self.stop_criterion = self.build_stopping_criterion()

    def configure_environment(self) -> Path:
        project_root = Path(self.cfg.project_root or ".").resolve()
        model_training_root = project_root / "Rhea" / "model_training"
        os.environ["KAGGLE_NUMS_ROOT"] = str(project_root)

        pythonpath = os.environ.get("PYTHONPATH", "")
        paths = [p for p in pythonpath.split(os.pathsep) if p]
        required_paths = [str(model_training_root), str(project_root)]
        paths = required_paths + [p for p in paths if p not in required_paths]
        os.environ["PYTHONPATH"] = os.pathsep.join(paths)

        if self.cfg.results_root is not None:
            results_root = Path(self.cfg.results_root)
            if not results_root.is_absolute():
                results_root = project_root / results_root
            os.environ["SYNETUNE_FOLDER"] = str(results_root.resolve())

        return project_root

    def build_backend(self) -> PythonBackend:
        return PythonBackend(
            tune_function=self.tune_function,
            config_space=self.config_space,
            rotate_gpus=self.cfg.rotate_gpus,
            delete_checkpoints=self.cfg.delete_checkpoints,
        )

    def build_sched(self):
        if self.cfg.scheduler == TuneSchedulerKind.ASHA:
            return self._build_asha()

        if self.cfg.scheduler == TuneSchedulerKind.BOHB:
            return self._build_bohb()

        return self._build_random_search()

    def _max_resource(self) -> int:
        asha = self.cfg.asha
        max_resource = asha.max_resource
        if max_resource is None:
            max_resource = self.config_space.get(asha.max_resource_attr)

        return max_resource

    def _build_asha(self):
        asha = self.cfg.asha
        searcher = "random_search"
        searcher_kwargs = None
        if self.cfg.points_to_evaluate is not None:
            searcher = LastValueMultiFidelitySearcher(
                searcher="random_search",
                config_space=self.config_space,
                random_seed=self.cfg.random_seed,
                points_to_evaluate=self.cfg.points_to_evaluate,
                searcher_kwargs={},
            )

        return AsynchronousSuccessiveHalving(
            config_space=self.config_space,
            metric=self.cfg.metric,
            do_minimize=self.cfg.do_minimize,
            searcher=searcher,
            time_attr=asha.resource_attr,
            max_t=self._max_resource(),
            grace_period=asha.grace_period,
            reduction_factor=asha.reduction_factor,
            brackets=asha.brackets,
            random_seed=self.cfg.random_seed,
            searcher_kwargs=searcher_kwargs,
        )

    def _build_bohb(self):
        asha = self.cfg.asha
        bohb = self.cfg.bohb

        return BOHB(
            config_space=self.config_space,
            metric=self.cfg.metric,
            time_attr=asha.resource_attr,
            max_t=self._max_resource(),
            do_minimize=self.cfg.do_minimize,
            random_seed=self.cfg.random_seed,
            num_min_data_points=bohb.num_min_data_points,
            top_n_percent=bohb.top_n_percent,
            min_bandwidth=bohb.min_bandwidth,
            num_candidates=bohb.num_candidates,
            bandwidth_factor=bohb.bandwidth_factor,
            random_fraction=bohb.random_fraction,
            points_to_evaluate=self.cfg.points_to_evaluate,
        )

    def _build_random_search(self) -> RandomSearch:
        return RandomSearch(
            config_space=self.config_space,
            metrics=[self.cfg.metric],
            do_minimize=self.cfg.do_minimize,
            random_seed=self.cfg.random_seed,
            points_to_evaluate=self.cfg.points_to_evaluate,
        )

    def build_stopping_criterion(self) -> StoppingCriterion:
        return StoppingCriterion(
            max_wallclock_time=self.cfg.max_wallclock_time,
            max_num_trials_started=self.cfg.max_num_trials_started,
            max_num_trials_completed=self.cfg.max_num_trials_completed,
        )

    def build(self) -> Tuner:
        return Tuner(
            trial_backend=self.trial_backend,
            scheduler=self.scheduler,
            stop_criterion=self.stop_criterion,
            n_workers=self.cfg.n_workers,
            sleep_time=self.cfg.sleep_time,
            results_update_interval=self.cfg.results_update_interval,
            print_update_interval=self.cfg.print_update_interval,
            tuner_name=self.cfg.tuner_name,
            save_tuner=self.cfg.save_tuner,
        )

    def run(self) -> Tuner:
        tuner = self.build()
        tuner.run()
        return tuner
