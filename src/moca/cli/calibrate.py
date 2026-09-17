from __future__ import annotations

import argparse
from pathlib import Path

from ..artifacts import atomic_write_json, read_jsonl, write_jsonl
from ..calibration import fit_logistic_calibrator, save_calibrator
from ..evaluation import (
    auroc_score,
    brier_score,
    expected_calibration_error,
    reference_correctness_score,
)
from ..utils import LOGGER, files_fingerprint, seed_everything
from .common import add_config_arguments, config_from_args
from .evaluate import _nli_scorer, _prediction, _references


def configure_parser(parser: argparse.ArgumentParser) -> None:
    add_config_arguments(parser)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--predictions")
    parser.add_argument("--output")
    parser.add_argument(
        "--allow-nonvalidation-fit",
        action="store_true",
        help=(
            "Explicitly permit fitting outside the validation split "
            "(never use for test reporting)"
        ),
    )


def calibrator_path(config, explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    if config.evaluation.calibrator_path:
        return Path(config.evaluation.calibrator_path)
    return config.run_dir / "evaluation" / "moca_1p_calibrator.json"


def run(args: argparse.Namespace):
    config = config_from_args(args)
    if args.split not in {"validation", "val", "dev"} and not args.allow_nonvalidation_fit:
        raise ValueError(
            "Refusing to fit a calibrator outside validation; pass "
            "--allow-nonvalidation-fit only for diagnostics"
        )
    seed_everything(config.runtime.seed, deterministic=config.runtime.deterministic)
    source = (
        Path(args.predictions)
        if args.predictions
        else config.run_dir / "predictions" / f"{args.split}.jsonl"
    )
    if not source.is_file():
        raise FileNotFoundError(f"Missing calibration predictions: {source}")
    rows = [dict(row) for row in read_jsonl(source)]
    if not rows:
        raise ValueError("No validation predictions were found")
    scorer = _nli_scorer(config)
    outcomes = [
        reference_correctness_score(
            _prediction(row),
            _references(row),
            scorer,
            prompt=row.get("prompt"),
            mode="semantic",
        )
        for row in rows
    ]
    calibrator = fit_logistic_calibrator(
        rows,
        outcomes,
        feature_names=config.evaluation.calibration_features,
        l2=config.evaluation.calibration_l2,
        max_iter=config.evaluation.calibration_max_iter,
        tolerance=config.evaluation.calibration_tolerance,
        source_fingerprint=files_fingerprint([source]),
        config_fingerprint=config.fingerprint(),
    )
    destination = calibrator_path(config, args.output)
    save_calibrator(calibrator, destination)
    confidences = calibrator.predict(rows)
    try:
        auroc = auroc_score(confidences, outcomes)
    except ValueError:
        auroc = None
    report = {
        "split": args.split,
        "source": str(source),
        "calibrator": str(destination),
        "num_examples": len(rows),
        "positive_rate": sum(outcomes) / len(outcomes),
        "brier": brier_score(confidences, outcomes),
        "ece": expected_calibration_error(
            confidences,
            outcomes,
            num_bins=config.evaluation.ece_bins,
        ),
        "auroc": auroc,
        "converged": calibrator.converged,
        "features": list(calibrator.feature_names),
    }
    atomic_write_json(destination.with_name("moca_1p_calibration_fit.json"), report)
    write_jsonl(
        destination.with_name("moca_1p_calibration_examples.jsonl"),
        (
            {
                "example_id": str(row.get("example_id", index)),
                "confidence": confidence,
                "correctness": outcome,
            }
            for index, (row, confidence, outcome) in enumerate(
                zip(rows, confidences, outcomes, strict=True)
            )
        ),
    )
    LOGGER.info("Saved MoCA-1P calibrator to %s", destination)
    return destination
