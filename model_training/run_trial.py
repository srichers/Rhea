#!/usr/bin/env python3
'''
Author: John McGuigan

Copyright: GPLv3 (see LICENSE file)

Minimal trial runner for parms-driven training and Syne Tune integration.
'''

import argparse
import json
from pathlib import Path

from ml_pytorch import build_default_parms, run_default_training


def load_json_dict(filename):
    with open(filename, "r", encoding="utf-8") as infile:
        data = json.load(infile)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {filename}")
    return data


def coerce_override(key, raw_value, current_value):
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
    if current_value.__class__.__name__ == "Irreps":
        import e3nn.o3

        return e3nn.o3.Irreps(raw_value)

    raise ValueError(f"Override for '{key}' is only supported for scalar, string, and Irreps values")


def apply_overrides(parms, overrides):
    resolved = dict(parms)
    for key, raw_value in overrides.items():
        current_value = resolved.get(key)
        resolved[key] = coerce_override(key, raw_value, current_value)
    return resolved


def is_irreps(value):
    return value.__class__.__name__ == "Irreps"


def serialize_config_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
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


def build_report_callback(enable_reporting):
    history = []
    reporter = None
    if enable_reporting:
        try:
            from syne_tune import Reporter
        except ImportError as exc:
            raise RuntimeError(
                "Syne Tune reporting was requested, but syne_tune is not installed"
            ) from exc
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
    mode = syne_tune_cfg.get("mode", "min")

    if not history:
        return {
            "status": "completed",
            "metric": metric_name,
            "mode": mode,
            "epochs_reported": 0,
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
        "best_epoch": int(best_metrics["epoch"]),
        "best_metric_value": float(best_metrics[metric_name]),
        "final_epoch": int(final_metrics["epoch"]),
        "final_metric_value": float(final_metrics[metric_name]),
        "best_metrics": best_metrics,
        "final_metrics": final_metrics,
    }


def load_trial_parms():
    parser = argparse.ArgumentParser(description="Run one training trial using the parms dictionary")
    parser.add_argument(
        "--st_config_json_filename",
        type=Path,
        help="Syne Tune JSON config handoff. No manual CLI overrides are supported.",
    )
    args = parser.parse_args()

    parms = build_default_parms()
    if args.st_config_json_filename is not None:
        parms = apply_overrides(parms, load_json_dict(args.st_config_json_filename))
    return args, parms


def should_report_to_syne_tune(parms, st_config_json_filename):
    syne_tune_cfg = parms.get("syne_tune", {})
    return bool(syne_tune_cfg.get("report", False) or st_config_json_filename is not None)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, indent=2, sort_keys=True)
        outfile.write("\n")


def main():
    args, parms = load_trial_parms()
    output_dir = Path.cwd()
    resolved_config_path = output_dir / "trial_config_resolved.json"
    summary_path = output_dir / "trial_summary.json"

    write_json(
        resolved_config_path,
        {key: serialize_config_value(value) for key, value in parms.items()},
    )

    report_fn, history = build_report_callback(
        should_report_to_syne_tune(parms, args.st_config_json_filename)
    )

    try:
        final_metrics = run_default_training(parms=parms, report_fn=report_fn)
        summary = summarize_history(history, final_metrics, parms)
    except Exception as exc:
        summary = {
            "status": "failed",
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "epochs_reported": len(history),
            "partial_history": history,
        }
        write_json(summary_path, summary)
        raise

    write_json(summary_path, summary)


if __name__ == "__main__":
    main()
