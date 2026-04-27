'''
Author: John McGuigan

Copyright: GPLv3 (see LICENSE file)

Plotting utilities for completed Syne Tune experiments.

This module loads a named Syne Tune experiment from the configured results
root and writes summary figures such as best-metric-over-time and
trials-over-time plots. It also configures a per-experiment Matplotlib cache
for non-interactive cluster runs.
'''
import os
from pathlib import Path

from syne_tune.experiments import load_experiment

from .registry import TunePlotCfg


class HyperTunePlotter:
    def __init__(self, cfg: TunePlotCfg):
        self.cfg = cfg

    def load_experiment(self, tuner_name: str):
        return load_experiment(
            tuner_name,
            local_path=self.cfg.results_root,
        )

    def figure_path(self, experiment, filename: str) -> str | None:
        if self.cfg.show:
            return None

        output_dir = Path(self.cfg.output_dir)
        if not output_dir.is_absolute():
            output_dir = experiment.path / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        return str(output_dir / filename)

    def plot(self, tuner_name: str) -> list[str]:
        if not self.cfg.enabled:
            return []

        experiment = self.load_experiment(tuner_name)

        if not self.cfg.show:
            mpl_cache = experiment.path / ".matplotlib"
            mpl_cache.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

            import matplotlib

            matplotlib.use("Agg", force=True)

        outputs = []

        if self.cfg.best_over_time:
            best_path = self.figure_path(experiment, self.cfg.best_filename)
            experiment.plot(
                self.cfg.metric,
                figure_path=best_path,
            )
            outputs.append(best_path)

        if self.cfg.trials_over_time:
            trials_path = self.figure_path(experiment, self.cfg.trials_filename)
            experiment.plot_trials_over_time(
                self.cfg.metric,
                figure_path=trials_path,
                figsize=self.cfg.figsize,
            )
            outputs.append(trials_path)

        return outputs
