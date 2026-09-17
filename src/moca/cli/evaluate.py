from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Any

from ..artifacts import (
    atomic_write_json,
    read_json,
    read_jsonl,
    write_jsonl,
)
from ..calibration import load_calibrator
from ..evaluation import (
    BidirectionalNLIScorer,
    auroc_score,
    binary_log_loss,
    brier_score,
    expected_calibration_error,
    evaluate_generation_records,
    max_rouge_1_f1,
    max_rouge_l_f1,
    reference_correctness_score,
    reliability_bins,
    risk_coverage_curve,
    spearman_correlation,
)
from ..inference import RoutedMoCA
from ..utils import LOGGER, seed_everything, stable_fingerprint
from .common import add_config_arguments, config_from_args

_SEMANTIC_CACHE_SCHEMA_VERSION = 1


def configure_parser(parser: argparse.ArgumentParser) -> None:
    add_config_arguments(parser)
    parser.add_argument("--split", default="test")
    parser.add_argument("--predictions")
    parser.add_argument("--output")
    parser.add_argument(
        "--skip-semantic-sampling",
        action="store_true",
        help="Require cached semantic_samples instead of generating them",
    )


def _references(row: dict[str, Any]) -> list[str]:
    values = row.get("references")
    if values is None:
        target = row.get("target", row.get("gold_response"))
        if target is None:
            raise ValueError(f"Prediction {row.get('example_id')} has no references")
        values = [target]
    if isinstance(values, str):
        values = [values]
    result = [str(value) for value in values]
    if not result:
        raise ValueError(f"Prediction {row.get('example_id')} has empty references")
    return result


def _prediction(row: dict[str, Any]) -> str:
    for key in ("generation", "prediction", "generated_response", "output"):
        if row.get(key) is not None:
            return str(row[key])
    raise ValueError(f"Prediction {row.get('example_id')} has no generated text")


def _nli_scorer(config):
    device = None if config.runtime.device == "auto" else config.runtime.device
    return BidirectionalNLIScorer(
        config.evaluation.nli_model,
        revision=config.evaluation.nli_revision,
        threshold=config.evaluation.entailment_threshold,
        decision_rule=config.evaluation.nli_decision_rule,
        include_prompt=config.evaluation.include_prompt_in_nli,
        device=device,
        batch_size=config.evaluation.nli_batch_size,
        max_length=config.evaluation.nli_max_length,
        truncation_side=config.evaluation.nli_truncation_side,
    )


def _add_semantic_samples(
    config,
    rows: list[dict[str, Any]],
    *,
    force: bool = False,
):
    if force:
        for row in rows:
            row.pop("semantic_samples", None)
            row.pop("semantic_expert_id", None)
            row.pop("semantic_routing_distance", None)
    missing = [row for row in rows if not row.get("semantic_samples")]
    if not missing:
        return rows
    system = RoutedMoCA(config)
    for index, row in enumerate(rows, start=1):
        if row.get("semantic_samples"):
            continue
        expert_id, distance, samples = system.sample_for_semantic_entropy(
            str(row["prompt"]),
            config.evaluation.semantic_samples,
        )
        row["semantic_samples"] = [sample.to_dict() for sample in samples]
        row["semantic_expert_id"] = expert_id
        row["semantic_routing_distance"] = distance
        LOGGER.info("Semantic samples %d/%d", index, len(rows))
    return rows


def _semantic_cache_manifest_path(semantic_path: Path) -> Path:
    return semantic_path.with_suffix(".manifest.json")


def _semantic_cache_identity(
    config,
    source_path: Path,
    source_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": _SEMANTIC_CACHE_SCHEMA_VERSION,
        "source_path": str(source_path.resolve()),
        "source_fingerprint": stable_fingerprint(source_rows),
        "config_fingerprint": config.fingerprint(),
    }


def _semantic_samples_are_compatible(
    rows: list[dict[str, Any]],
    expected_samples: int,
) -> bool:
    if not rows or expected_samples < 1:
        return False
    for row in rows:
        samples = row.get("semantic_samples")
        if not isinstance(samples, (list, tuple)) or len(samples) != expected_samples:
            return False
    return True


def _load_compatible_semantic_cache(
    semantic_path: Path,
    identity: dict[str, Any],
    *,
    expected_samples: int,
) -> list[dict[str, Any]] | None:
    manifest_path = _semantic_cache_manifest_path(semantic_path)
    if not semantic_path.exists() or not manifest_path.exists():
        return None
    try:
        manifest = read_json(manifest_path)
        if not isinstance(manifest, dict):
            return None
        if any(manifest.get(key) != value for key, value in identity.items()):
            return None
        cached_rows = [dict(row) for row in read_jsonl(semantic_path)]
        if manifest.get("row_count") != len(cached_rows):
            return None
        if manifest.get("cache_fingerprint") != stable_fingerprint(cached_rows):
            return None
        if not _semantic_samples_are_compatible(
            cached_rows,
            expected_samples,
        ):
            return None
    except (OSError, TypeError, ValueError):
        return None
    LOGGER.info("Reusing compatible semantic-generation cache: %s", semantic_path)
    return cached_rows


def _write_semantic_cache(
    semantic_path: Path,
    rows: list[dict[str, Any]],
    identity: dict[str, Any],
) -> None:
    write_jsonl(semantic_path, rows)
    manifest = {
        **identity,
        "row_count": len(rows),
        "cache_fingerprint": stable_fingerprint(rows),
    }
    atomic_write_json(
        _semantic_cache_manifest_path(semantic_path),
        manifest,
    )


def _semantic_rows_with_cache(
    config,
    source_rows: list[dict[str, Any]],
    *,
    source_path: Path,
    semantic_path: Path,
    skip_semantic_sampling: bool,
) -> list[dict[str, Any]]:
    rows = copy.deepcopy(source_rows)
    expected_samples = int(config.evaluation.semantic_samples)
    cache_enabled = bool(config.evaluation.cache_generations)
    identity = _semantic_cache_identity(config, source_path, source_rows)
    cache_artifact_exists = (
        semantic_path.exists() or _semantic_cache_manifest_path(semantic_path).exists()
    )

    if cache_enabled:
        cached_rows = _load_compatible_semantic_cache(
            semantic_path,
            identity,
            expected_samples=expected_samples,
        )
        if cached_rows is not None:
            return cached_rows

    source_has_compatible_samples = _semantic_samples_are_compatible(
        rows,
        expected_samples,
    )
    if skip_semantic_sampling and not source_has_compatible_samples:
        raise ValueError(
            "No compatible semantic-generation cache is available and "
            "the source predictions do not contain compatible semantic_samples"
        )
    if skip_semantic_sampling:
        if cache_enabled:
            _write_semantic_cache(semantic_path, rows, identity)
        return rows

    if not source_has_compatible_samples:
        rows = _add_semantic_samples(config, rows, force=True)
    elif cache_enabled and cache_artifact_exists:
        # A stale/corrupt cache must never silently fall back to potentially
        # stale embedded samples from the same source.
        rows = _add_semantic_samples(config, rows, force=True)

    if not _semantic_samples_are_compatible(rows, expected_samples):
        raise RuntimeError("Semantic generation did not produce the configured sample count")
    if cache_enabled:
        _write_semantic_cache(semantic_path, rows, identity)
    return rows


def _calibrator_path(config) -> Path:
    if config.evaluation.calibrator_path:
        return Path(config.evaluation.calibrator_path)
    return config.run_dir / "evaluation" / "moca_1p_calibrator.json"


def _single_pass_evaluation(config, rows, scorer):
    examples = []
    confidences: list[float] = []
    outcomes: list[float] = []
    rouge_l: list[float] = []
    rouge_1: list[float] = []
    calibrated_confidences = None
    if config.evaluation.confidence_source == "moca_1p":
        path = _calibrator_path(config)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing MoCA-1P calibrator: {path}. Generate validation predictions "
                "and run `moca calibrate` first."
            )
        calibrator = load_calibrator(path)
        if tuple(config.evaluation.calibration_features) != calibrator.feature_names:
            raise ValueError(
                "Calibrator features differ from evaluation.calibration_features; "
                "refit the validation calibrator"
            )
        calibrated_confidences = calibrator.predict(rows)

    for index, row in enumerate(rows):
        prediction = _prediction(row)
        references = _references(row)
        if calibrated_confidences is not None:
            confidence = calibrated_confidences[index]
        elif config.evaluation.confidence_source == "first_token_entropy":
            confidence = float(
                row["non_special_first_token_confidence"]
                if "non_special_first_token_confidence" in row
                else row["first_token_confidence"]
            )
        elif config.evaluation.confidence_source == "sequence_probability":
            log_probability = float(row["sequence_log_probability"])
            token_count = max(1, int(row.get("generated_token_count", 1)))
            confidence = math.exp(log_probability / token_count)
        elif config.evaluation.confidence_source == "provided":
            confidence = float(row["confidence"])
        else:
            raise ValueError(
                f"Unsupported confidence source: {config.evaluation.confidence_source}"
            )
        confidence = min(1.0, max(0.0, confidence))
        correctness = reference_correctness_score(
            prediction,
            references,
            scorer,
            prompt=row.get("prompt"),
            mode="semantic",
        )
        example_rouge_l = max_rouge_l_f1(prediction, references)
        example_rouge_1 = max_rouge_1_f1(prediction, references)
        confidences.append(confidence)
        outcomes.append(correctness)
        rouge_l.append(example_rouge_l)
        rouge_1.append(example_rouge_1)
        examples.append(
            {
                "example_id": str(row.get("example_id", index)),
                "prediction": prediction,
                "references": references,
                "confidence": confidence,
                "correctness": correctness,
                "rouge_l_f1": example_rouge_l,
                "rouge_1_f1": example_rouge_1,
                "routing_distance": row.get("routing_distance"),
                "routing_margin": row.get("routing_margin"),
                "first_token_entropy": row.get("first_token_entropy"),
            }
        )
    try:
        auroc = auroc_score(confidences, outcomes)
    except ValueError:
        auroc = None
    count = len(rows)
    return {
        "metrics": {
            "num_examples": count,
            "brier": brier_score(confidences, outcomes),
            "auroc": auroc,
            # Table 1's "Acc" is ROUGE-L F1. Semantic equality remains the
            # binary correctness target used for Brier and AUROC.
            "semantic_correctness_rate": sum(outcomes) / count,
            "accuracy": sum(rouge_l) / count,
            "rouge_l_f1": sum(rouge_l) / count,
            "rouge_1_f1": sum(rouge_1) / count,
            "mean_confidence": sum(confidences) / count,
        },
        "examples": examples,
    }


def _add_calibration_diagnostics(config, result):
    examples = result["examples"]
    confidences = [float(row["confidence"]) for row in examples]
    outcomes = [float(row["correctness"]) for row in examples]
    metrics = result["metrics"]
    bins = config.evaluation.ece_bins
    metrics["ece"] = expected_calibration_error(
        confidences,
        outcomes,
        num_bins=bins,
    )
    metrics["adaptive_ece"] = expected_calibration_error(
        confidences,
        outcomes,
        num_bins=bins,
        adaptive=True,
    )
    metrics["binary_log_loss"] = binary_log_loss(confidences, outcomes)
    metrics["reliability_bins"] = reliability_bins(
        confidences,
        outcomes,
        num_bins=bins,
    )
    curve, aurc = risk_coverage_curve(confidences, outcomes)
    stride = max(1, len(curve) // 100)
    summarized = curve[::stride]
    if summarized[-1] != curve[-1]:
        summarized.append(curve[-1])
    metrics["aurc"] = aurc
    metrics["risk_coverage_curve"] = summarized
    threshold = config.evaluation.abstain_threshold
    if threshold is not None:
        accepted = [
            row for row in examples if float(row["confidence"]) >= threshold
        ]
        metrics["abstain_threshold"] = threshold
        metrics["selective_coverage"] = len(accepted) / len(examples)
        metrics["selective_accuracy"] = (
            sum(float(row["correctness"]) for row in accepted) / len(accepted)
            if accepted
            else None
        )
    semantic_entropies = [row.get("semantic_entropy") for row in examples]
    first_entropies = [row.get("normalized_first_token_entropy") for row in examples]
    if all(value is not None for value in semantic_entropies + first_entropies):
        try:
            metrics["first_token_semantic_entropy_spearman"] = spearman_correlation(
                [float(value) for value in first_entropies],
                [float(value) for value in semantic_entropies],
            )
        except ValueError:
            metrics["first_token_semantic_entropy_spearman"] = None


def run(args: argparse.Namespace):
    config = config_from_args(args)
    seed_everything(
        config.runtime.seed,
        deterministic=config.runtime.deterministic,
    )
    prediction_path = (
        Path(args.predictions)
        if args.predictions
        else config.run_dir / "predictions" / f"{args.split}.jsonl"
    )
    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Missing predictions: {prediction_path}. Run `moca generate` first."
        )
    rows = [dict(row) for row in read_jsonl(prediction_path)]
    scorer = _nli_scorer(config)
    if config.evaluation.confidence_source == "semantic_entropy":
        semantic_path = config.run_dir / "predictions" / f"{args.split}_semantic.jsonl"
        rows = _semantic_rows_with_cache(
            config,
            rows,
            source_path=prediction_path,
            semantic_path=semantic_path,
            skip_semantic_sampling=args.skip_semantic_sampling,
        )
        result = evaluate_generation_records(
            rows,
            scorer,
            config=config,
            correctness_scorer=scorer,
            correctness_mode="semantic",
        )
    else:
        result = _single_pass_evaluation(config, rows, scorer)

    _add_calibration_diagnostics(config, result)

    metrics = result["metrics"]
    metrics["paper_scale_percent"] = {
        key: (value * 100.0 if isinstance(value, (int, float)) else value)
        for key, value in metrics.items()
        if key not in {"num_examples"}
    }
    metrics["dataset"] = config.data.name
    metrics["model"] = config.model.name_or_path
    metrics["method"] = config.objective.method
    metrics["confidence_source"] = config.evaluation.confidence_source
    destination = (
        Path(args.output)
        if args.output
        else config.run_dir / "evaluation" / f"{args.split}_metrics.json"
    )
    atomic_write_json(destination, metrics)
    write_jsonl(
        destination.with_name(f"{destination.stem}_examples.jsonl"),
        result["examples"],
    )
    LOGGER.info("Evaluation metrics: %s", metrics)
    return result
