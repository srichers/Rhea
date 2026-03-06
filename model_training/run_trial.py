#!/usr/bin/env python3
'''
Author: OpenAI Codex

Copyright: GPLv3 (see LICENSE file)

Minimal trial runner for config-driven training and Syne Tune integration.
'''

import argparse
import json
from pathlib import Path

import e3nn.o3

from ml_pytorch import build_default_parms, run_default_training


def parse_key_value(raw):
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Overrides must be formatted as key=value")
    key, value = raw.split("=", 1)
    if not key:
        raise argparse.ArgumentTypeError("Override key cannot be empty")
    return key, value


def parse_bool(raw_value):
    lowered = raw_value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Could not parse boolean value '{raw_value}'")


def parse_unknown_args(unknown_args):
    parsed = {}
    i = 0
    while i < len(unknown_args):
        token = unknown_args[i]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected argument '{token}'")
        if i + 1 >= len(unknown_args):
            raise ValueError(f"Missing value for argument '{token}'")
        key = token[2:]
        parsed[key] = unknown_args[i + 1]
        i += 2
    return parsed


def pop_json_config_arg(override_map):
    return override_map.pop("st_config_json_filename", None)


def pop_bool_override(override_map, key):
    if key not in override_map:
        return False
    raw_value = override_map.pop(key)
    if isinstance(raw_value, bool):
        return raw_value
    return parse_bool(raw_value)


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
        return parse_bool(raw_value)
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(raw_value)
    if isinstance(current_value, float):
        return float(raw_value)
    if isinstance(current_value, str):
        return raw_value
    if current_value.__class__.__name__ == "Irreps":
        return e3nn.o3.Irreps(raw_value)

    raise ValueError(f"Override for '{key}' is only supported for scalar, string, and Irreps values")


def apply_overrides(parms, overrides):
    resolved = dict(parms)
    for key, raw_value in overrides.items():
        current_value = resolved.get(key)
        resolved[key] = coerce_override(key, raw_value, current_value)
    return resolved


def serialize_config_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [serialize_config_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialize_config_value(item) for key, item in value.items()}
    if current_is_irreps(value):
        return str(value)
    if hasattr(value, "__module__") and hasattr(value, "__name__"):
        return f"{value.__module__}.{value.__name__}"
    if isinstance(value, Path):
        return str(value)
    return str(value)


def current_is_irreps(value):
    return value.__class__.__name__ == "Irreps"


def build_report_callback(enable_syne_tune):
    history = []
    reporter = None
    if enable_syne_tune:
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


def summarize_history(history, final_metrics):
    if not history:
        return {
            "status": "completed",
            "epochs_reported": 0,
            "final_metrics": final_metrics,
        }

    best_metrics = min(history, key=lambda metrics: metrics["validation_score"])
    return {
        "status": "completed",
        "epochs_reported": len(history),
        "best_epoch": int(best_metrics["epoch"]),
        "best_validation_score": float(best_metrics["validation_score"]),
        "final_epoch": int(final_metrics["epoch"]),
        "final_validation_score": float(final_metrics["validation_score"]),
        "best_metrics": best_metrics,
        "final_metrics": final_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Run one training trial with optional Syne Tune reporting")
    parser.add_argument("--config", type=Path, help="Path to a JSON file containing flat parameter overrides")
    parser.add_argument("--set", dest="set_overrides", action="append", type=parse_key_value, default=[],
                        help="Flat key=value override. May be passed multiple times.")
    parser.add_argument("--output-dir", type=Path, default=Path("."),
                        help="Directory for resolved config and trial summary artifacts")
    parser.add_argument("--report-to-syne-tune", action="store_true",
                        help="Emit per-epoch metrics through syne_tune.Reporter")
    args, unknown_args = parser.parse_known_args()

    cli_overrides = parse_unknown_args(unknown_args)
    syne_tune_json_path = pop_json_config_arg(cli_overrides)
    report_from_cli_config = pop_bool_override(cli_overrides, "report_to_syne_tune")

    overrides = {}
    if args.config is not None:
        overrides.update(load_json_dict(args.config))
    if syne_tune_json_path is not None:
        overrides.update(load_json_dict(syne_tune_json_path))
    overrides.update(cli_overrides)
    overrides.update(dict(args.set_overrides))
    report_from_overrides = pop_bool_override(overrides, "report_to_syne_tune")

    parms = apply_overrides(build_default_parms(), overrides)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = output_dir / "trial_config_resolved.json"
    summary_path = output_dir / "trial_summary.json"

    with open(resolved_config_path, "w", encoding="utf-8") as outfile:
        json.dump({key: serialize_config_value(value) for key, value in parms.items()}, outfile, indent=2, sort_keys=True)
        outfile.write("\n")

    enable_syne_tune_reporting = (
        args.report_to_syne_tune
        or report_from_cli_config
        or report_from_overrides
        or syne_tune_json_path is not None
    )
    report_fn, history = build_report_callback(enable_syne_tune_reporting)

    try:
        final_metrics = run_default_training(parms=parms, report_fn=report_fn)
        summary = summarize_history(history, final_metrics)
    except Exception as exc:
        summary = {
            "status": "failed",
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "epochs_reported": len(history),
            "partial_history": history,
        }
        with open(summary_path, "w", encoding="utf-8") as outfile:
            json.dump(summary, outfile, indent=2, sort_keys=True)
            outfile.write("\n")
        raise

    with open(summary_path, "w", encoding="utf-8") as outfile:
        json.dump(summary, outfile, indent=2, sort_keys=True)
        outfile.write("\n")


if __name__ == "__main__":
    main()
