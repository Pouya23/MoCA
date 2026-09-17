from __future__ import annotations

import argparse
import math
from pathlib import Path

from ..artifacts import atomic_write_json, read_jsonl
from ..calibration import load_calibrator
from ..evaluation import auroc_score, false_positive_rate_at_recall
from ..utils import LOGGER
from .common import add_config_arguments, config_from_args


def configure_parser(parser: argparse.ArgumentParser) -> None:
    add_config_arguments(parser)
    parser.add_argument("--id-predictions", required=True)
    parser.add_argument(
        "--ood-predictions",
        action="append",
        required=True,
        help="OOD prediction JSONL; repeat for multiple corpora",
    )
    parser.add_argument("--output")


def _calibrator_path(config) -> Path:
    if config.evaluation.calibrator_path:
        return Path(config.evaluation.calibrator_path)
    return config.run_dir / "evaluation" / "moca_1p_calibrator.json"


def _confidence(config, rows):
    source = config.evaluation.confidence_source
    if source == "moca_1p":
        return load_calibrator(_calibrator_path(config)).predict(rows)
    if source == "first_token_entropy":
        return [
            float(
                row["non_special_first_token_confidence"]
                if "non_special_first_token_confidence" in row
                else row["first_token_confidence"]
            )
            for row in rows
        ]
    if source == "sequence_probability":
        return [
            math.exp(
                float(row["sequence_log_probability"])
                / max(1, int(row.get("generated_token_count", 1)))
            )
            for row in rows
        ]
    raise ValueError(
        "OOD evaluation requires moca_1p, first_token_entropy, or sequence_probability"
    )


def _minmax(values):
    low = min(values)
    high = max(values)
    if high <= low:
        return [0.5 for _ in values]
    return [(value - low) / (high - low) for value in values]


def run(args: argparse.Namespace):
    config = config_from_args(args)
    id_rows = [dict(row) for row in read_jsonl(args.id_predictions)]
    if not id_rows:
        raise ValueError("ID predictions are empty")
    id_confidence = _confidence(config, id_rows)
    reports = []
    for ood_path in args.ood_predictions:
        ood_rows = [dict(row) for row in read_jsonl(ood_path)]
        if not ood_rows:
            raise ValueError(f"OOD predictions are empty: {ood_path}")
        ood_confidence = _confidence(config, ood_rows)
        labels = [0] * len(id_rows) + [1] * len(ood_rows)
        uncertainty = [1.0 - value for value in id_confidence + ood_confidence]
        distance_values = [
            float(row["routing_distance"]) for row in id_rows + ood_rows
        ]
        distance_scores = _minmax(distance_values)
        margin_values = [
            -float(row.get("routing_margin", 0.0)) for row in id_rows + ood_rows
        ]
        margin_scores = _minmax(margin_values)
        reports.append(
            {
                "ood_source": str(ood_path),
                "num_id": len(id_rows),
                "num_ood": len(ood_rows),
                "confidence_source": config.evaluation.confidence_source,
                "ood_auroc": auroc_score(uncertainty, labels),
                "fpr95": false_positive_rate_at_recall(uncertainty, labels),
                "distance_ood_auroc": auroc_score(distance_scores, labels),
                "margin_ood_auroc": auroc_score(margin_scores, labels),
                "mean_id_confidence": sum(id_confidence) / len(id_confidence),
                "mean_ood_confidence": sum(ood_confidence) / len(ood_confidence),
            }
        )
    output = {
        "id_source": str(args.id_predictions),
        "comparisons": reports,
    }
    destination = (
        Path(args.output)
        if args.output
        else config.run_dir / "evaluation" / "ood_metrics.json"
    )
    atomic_write_json(destination, output)
    LOGGER.info("Wrote OOD evaluation to %s", destination)
    return output
