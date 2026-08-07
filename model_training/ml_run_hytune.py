#!/usr/bin/env python3
'''
Author: John McGuigan

Copyright: GPLv3 (see LICENSE file)

The is the runner called to run the syne tune hpo experiment
'''

import json
import hashlib
from pathlib import Path
from hypertuner.syne import HyperTuner
from hypertuner.cfgs import build_tune_plot_cfg
from hypertuner.plot import HyperTunePlotter
from ml_pytorch import build_default_parms, run_default_training
from hypertuner.cfgs import build_syne_tune_cfg
from hypertuner.space import build_config_space
from syne_tune import Reporter
import e3nn.o3


class TrialOverrideResolver:
    aliases = {
          "lr": "learning_rate",
          "weight_decay": "adamw.weight_decay",
          "batch_size": "loader.batch_size",
          "seed": "random_seed",
    }

    def __init__(self, parms, config):
        self.parms = parms
        self.config = config or {}
        self.syne_tune_cfg = parms.get("syne_tune", {})
        self.model_tiers = self.syne_tune_cfg.get("model_tiers", {})

    @staticmethod
    def is_irreps(value):
        return value.__class__.__name__ == "Irreps"

    def build_overrides(self):
        overrides = {}

        tier_name = self.config.get("model_tier")
        if tier_name is not None:
            if tier_name not in self.model_tiers:
                raise ValueError(f"Unknown model_tier '{tier_name}'")
            overrides.update(self.model_tiers[tier_name].get("parms", {}))

        for key, value in self.config.items():
            if key in self.aliases:
                overrides[self.aliases[key]] = value
            elif key != "model_tier":
                overrides[key] = value

        return overrides

    def resolve(self):
        resolved = dict(self.parms)
        for key, raw_value in self.build_overrides().items():
            resolved[key] = self.coerce_override(key, raw_value, resolved.get(key))
        return resolved

    # turns new config value into the type expected by the existing default parms{}
    def coerce_override(self, key, raw_value, current_value):
        if isinstance(raw_value, (bool, int, float, list, dict)) or raw_value is None:
            return raw_value

        if current_value is None:
            try:
                return json.loads(raw_value)
            except json.JSONDecodeError:
                return raw_value

        if isinstance(current_value, bool):
            lowered = raw_value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
            raise ValueError(f"Could not parse boolean value '{raw_value}'")
        if isinstance(current_value, int) and not isinstance(current_value, bool):
            return int(raw_value)
        if isinstance(current_value, float):
            return float(raw_value)
        if isinstance(current_value, str):
            return raw_value
        if self.is_irreps(current_value):
            return e3nn.o3.Irreps(raw_value)

        raise ValueError(f"Override for '{key}' is only supported for scalar, string, and Irreps values")


class BuildnRun:
    def __init__(self, config=None):
        self.config = config
        self.parms = self.build_parms()
        self.output_dir = self.build_output_dir()
        self.resolved_config_path = self.output_dir / "trial_config_resolved.json"
        self.summary_path = self.output_dir / "trial_summary.json"
        self.history_path = self.output_dir / "trial_history.json"
        self.report_fn, self.history = self.build_reporting()

    def build_parms(self):
        parms = build_default_parms()
        if self.config is not None:
            parms = TrialOverrideResolver(parms, self.config).resolve()
        return parms

    def build_output_dir(self):
        output_dir = Path(self.parms.get("output_dir", Path.cwd()))
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def build_reporting(self):
        return build_report_callback(
            should_report_to_syne_tune(self.parms, self.config)
        )

    def write_resolved_config(self):
        write_json(
            self.resolved_config_path,
            {key: serialize_config_value(value) for key, value in self.parms.items()},
        )

    def write_results(self, final_metrics, status="completed"):
        summary = summarize_history(self.history, final_metrics, self.parms)
        summary["status"] = status
        write_json(self.history_path, build_history_lib(self.history, self.parms, status))
        write_json(self.summary_path, summary)

    def run(self):
        self.write_resolved_config()
        final_metrics = run_default_training(parms=self.parms, report_fn=self.report_fn)
        self.write_results(final_metrics)
        return final_metrics


class HPOExperiment:
    def __init__(self, parms=None):
        self.parms = parms if parms is not None else build_default_parms()
        self.syne_tune_cfg = self.build_syne_tune_cfg()
        self.hyper_tuner = self.build_tuner()

    def build_syne_tune_cfg(self):
        syne_tune_cfg = dict(self.parms.get("syne_tune", {}))
        raw_space = syne_tune_cfg.get("space", {})
        config_space = build_config_space(raw_space)

        cfg_keys = {
            "points_to_evaluate",
            "model_tiers",
            "budget",
            "sampling",
            "base_model_cfg",
            "resource",
            "metric",
            "mode",
            "scheduler",
            "resource_attr",
            "max_resource_attr",
            "max_resource",
            "grace_period",
            "reduction_factor",
            "brackets",
            "bohb_num_min_data_points",
            "bohb_top_n_percent",
            "bohb_min_bandwidth",
            "bohb_num_candidates",
            "bohb_bandwidth_factor",
            "bohb_random_fraction",
            "max_wallclock_time",
            "max_num_trials_started",
            "max_num_trials_completed",
            "n_workers",
            "random_seed",
            "tuner_name",
            "project_root",
            "results_root",
            "rotate_gpus",
            "delete_checkpoints",
            "save_tuner",
            "sleep_time",
            "results_update_interval",
            "print_update_interval",
        }
        cfg_kwargs = {
            key: value
            for key, value in syne_tune_cfg.items()
            if key in cfg_keys
        }
        cfg_kwargs["config_space"] = config_space
        cfg_kwargs["raw_space"] = raw_space

        return build_syne_tune_cfg(**cfg_kwargs)

    def build_tuner(self):
        return HyperTuner(run_training_trial, self.syne_tune_cfg)

    def maybe_plot_results(self, tuner_name=None):
        plots = self.parms.get("syne_tune", {}).get("plots")
        if not plots or not plots.get("enabled", False):
            return []

        plot_cfg = build_tune_plot_cfg(
            plots=plots,
            results_root=self.hyper_tuner.cfg.results_root,
        )
        return HyperTunePlotter(plot_cfg).plot(tuner_name or self.hyper_tuner.cfg.tuner_name)

    def run(self):
        tuner = self.hyper_tuner.run()
        self.maybe_plot_results(tuner.name)
        return tuner


def is_irreps(value):
    return value.__class__.__name__ == "Irreps"


# this one goes the opposite way parms to json
def serialize_config_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [serialize_config_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialize_config_value(item) for key, item in value.items()}
    if is_irreps(value):
        return str(value)
    if hasattr(value, "__module__") and hasattr(value, "__name__"):
        return f"{value.__module__}.{value.__name__}"
    if isinstance(value, Path):
        return str(value)
    return str(value)


def normalize_mode(mode):
    if hasattr(mode, "value"):
        mode = mode.value
    return str(mode).lower()


def build_report_callback(enable_reporting):
    # keeps track of losses, epochs, learning rate
    # will hook box3d up to this eventually
    history = []
    reporter = None
    if enable_reporting:
        reporter = Reporter()

    def report_fn(metrics):
        numeric_metrics = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float, bool))
        }
        history.append(dict(numeric_metrics))
        if reporter is not None:
            reporter(**numeric_metrics)

    return report_fn, history


def summarize_history(history, final_metrics, parms):
    syne_tune_cfg = parms.get("syne_tune", {})
    metric_name = syne_tune_cfg.get("metric", "validation_score")
    mode = normalize_mode(syne_tune_cfg.get("mode", "min"))

    if not history:
        return {
            "status": "completed",
            "metric": metric_name,
            "mode": mode,
            "epochs_reported": 0,
            "history_filename": "trial_history.json",
            "final_metrics": final_metrics,
        }

    if mode == "max":
        best_metrics = max(history, key=lambda metrics: metrics[metric_name])
    else:
        best_metrics = min(history, key=lambda metrics: metrics[metric_name])

    return {
        "status": "completed",
        "metric": metric_name,
        "mode": mode,
        "epochs_reported": len(history),
        "history_filename": "trial_history.json",
        "best_epoch": int(best_metrics["epoch"]),
        "best_metric_value": float(best_metrics[metric_name]),
        "final_epoch": int(final_metrics["epoch"]),
        "final_metric_value": float(final_metrics[metric_name]),
        "best_metrics": best_metrics,
        "final_metrics": final_metrics,
    }


def should_report_to_syne_tune(parms, config):
    # report metrics to synetune
    syne_tune_cfg = parms.get("syne_tune", {})
    return bool(syne_tune_cfg.get("report", False) or config is not None)


def build_trial_output_dir(config):
    serialized = serialize_config_value(config)
    config_blob = json.dumps(serialized, sort_keys=True)
    trial_hash = hashlib.sha1(config_blob.encode("utf-8")).hexdigest()[:12]
    return f"output/train_tune/trial_{trial_hash}"


def build_divergence_metrics(parms, divergence_metrics):
    syne_tune_cfg = parms.get("syne_tune", {})
    resource_cfg = syne_tune_cfg.get("resource", {})
    resource_attr = resource_cfg.get("resource_attr")
    max_resource_attr = resource_cfg.get("max_resource_attr")
    metric_name = syne_tune_cfg.get("metric", "validation_score")
    penalty = float(parms.get("hpo_failure_penalty", 1.0e30))
    metrics = {
        "epoch": int(divergence_metrics.get("epoch", 0)),
        "train_loss": penalty,
        "validation_loss": penalty,
        "test_loss": penalty,
        "validation_score": penalty,
        metric_name: penalty,
        "diverged": 1,
    }
    metrics.update(divergence_metrics)
    metrics[metric_name] = penalty
    if resource_attr and resource_attr not in metrics:
        metrics[resource_attr] = int(resource_cfg.get("grace_period", 1))
    if max_resource_attr and max_resource_attr not in metrics:
        max_resource = parms.get(max_resource_attr)
        if max_resource is None:
            max_resource = resource_cfg.get("max_budget")
        if max_resource is None:
            max_resource = syne_tune_cfg.get("space", {}).get(max_resource_attr, {}).get("value")
        if max_resource is not None:
            metrics[max_resource_attr] = int(max_resource)
    return metrics


def build_history_lib(history, parms, status):
    syne_tune_cfg = parms.get("syne_tune", {})
    return {
        "status": status,
        "metric": syne_tune_cfg.get("metric", "validation_score"),
        "mode": normalize_mode(syne_tune_cfg.get("mode", "min")),
        "epochs_reported": len(history),
        "history": history,
    }


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as outfile:
        json.dump(serialize_config_value(payload), outfile, indent=2, sort_keys=True)
        outfile.write("\n")


def run_single_trial(config=None):
    return BuildnRun(config).run()


def run_training_trial(**config):
    # Syne Tune's PythonBackend calls this function with one keyword per
    # hyperparameter and serializes it without module globals.
    from ml_run_hytune import BuildnRun, build_divergence_metrics, build_trial_output_dir
    from ml_training_status import TrainingDivergedError

    trial_config = dict(config) or None
    if trial_config is not None and "output_dir" not in trial_config:
        trial_config["output_dir"] = build_trial_output_dir(trial_config)

    runner = BuildnRun(trial_config)
    try:
        return runner.run()
    except TrainingDivergedError as exc:
        final_metrics = build_divergence_metrics(runner.parms, exc.metrics)
        runner.report_fn(final_metrics)
        runner.write_results(final_metrics, status="diverged")
        return final_metrics


def build_syne_tune_cfg_from_parms(parms):
    # builds syne tune configs
    return HPOExperiment(parms).syne_tune_cfg


def build_tuner_from_parms(parms=None):
    # this function actually builds the tuner itself
    return HPOExperiment(parms).hyper_tuner


def maybe_plot_tuning_results(parms, tuner_name, results_root):
    # responsible for plot if enabled
    plots = parms.get("syne_tune", {}).get("plots")
    if not plots or not plots.get("enabled", False):
        return []

    plot_cfg = build_tune_plot_cfg(plots=plots, results_root=results_root)
    return HyperTunePlotter(plot_cfg).plot(tuner_name)


def run_syne_tune(parms=None):
    return HPOExperiment(parms).run()


def main():
    run_syne_tune()


if __name__ == "__main__":
    main()
