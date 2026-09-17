from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

from ..artifacts import atomic_write_json, read_json


DEFAULT_METRICS = (
    "brier",
    "ece",
    "adaptive_ece",
    "auroc",
    "aurc",
    "accuracy",
    "rouge_1_f1",
    "rouge_l_f1",
)


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default="runs")
    parser.add_argument("--metrics-file", default="evaluation/test_metrics.json")
    parser.add_argument("--metric", action="append", default=[])
    parser.add_argument("--output-prefix", default="runs/summary")
    parser.add_argument(
        "--seed-pattern",
        default=r"_s\d+$",
        help="Regex removed from a run-directory name to form the method group",
    )


def _summary(values):
    count = len(values)
    mean = math.fsum(values) / count
    if count < 2:
        return {"n": count, "mean": mean, "std": None, "ci95": None}
    variance = math.fsum((value - mean) ** 2 for value in values) / (count - 1)
    std = math.sqrt(variance)
    return {"n": count, "mean": mean, "std": std, "ci95": 1.96 * std / math.sqrt(count)}


def run(args: argparse.Namespace):
    root = Path(args.root)
    metrics = tuple(args.metric or DEFAULT_METRICS)
    grouped = {}
    for path in sorted(root.glob(f"*/{args.metrics_file}")):
        run_name = path.relative_to(root).parts[0]
        group = re.sub(args.seed_pattern, "", run_name)
        payload = read_json(path)
        grouped.setdefault(group, []).append((run_name, payload))
    if not grouped:
        raise FileNotFoundError(f"No {args.metrics_file} files found below {root}")

    report = {"root": str(root), "groups": {}}
    flat_rows = []
    for group, runs in grouped.items():
        group_report = {"runs": [name for name, _ in runs], "metrics": {}}
        row = {"group": group, "runs": len(runs)}
        for metric in metrics:
            values = [
                float(payload[metric])
                for _, payload in runs
                if isinstance(payload.get(metric), (int, float))
            ]
            if not values:
                continue
            item = _summary(values)
            group_report["metrics"][metric] = item
            row[f"{metric}_mean"] = item["mean"]
            row[f"{metric}_std"] = item["std"]
            row[f"{metric}_ci95"] = item["ci95"]
        report["groups"][group] = group_report
        flat_rows.append(row)

    prefix = Path(args.output_prefix)
    atomic_write_json(prefix.with_suffix(".json"), report)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in flat_rows for key in row})
    with prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)

    lines = ["\\begin{tabular}{l" + "c" * len(metrics) + "}", "\\toprule"]
    lines.append(
        "Method & "
        + " & ".join(metric.replace("_", "\\_") for metric in metrics)
        + " \\\\"
    )
    lines.append("\\midrule")
    for group, values in report["groups"].items():
        cells = []
        for metric in metrics:
            item = values["metrics"].get(metric)
            if item is None:
                cells.append("--")
            elif item["ci95"] is None:
                cells.append(f"{item['mean']:.4f}")
            else:
                cells.append(f"{item['mean']:.4f} $\\pm$ {item['ci95']:.4f}")
        lines.append(group.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    prefix.with_suffix(".tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report
