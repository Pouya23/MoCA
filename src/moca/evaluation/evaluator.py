"""Dataset-level evaluation for cached JSONL generation records.

This module intentionally evaluates records that have already been generated;
it does not own model loading or generation orchestration.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .correctness import reference_correctness_score
from .metrics import auroc_score, brier_score, max_rouge_1_f1, max_rouge_l_f1
from .semantic import (
    SemanticScorer,
    entropy_to_confidence,
    kuhn_eq4_semantic_entropy,
    semantic_cluster_ids,
    semantic_confidence,
    semantic_entropy,
)

_PREDICTION_FIELDS = (
    "prediction",
    "generated_text",
    "generated_response",
    "generation",
    "greedy_generation",
    "greedy_response",
    "output",
    "response",
)
_REFERENCE_FIELDS = (
    "references",
    "reference",
    "answers",
    "gold_answers",
    "targets",
    "target",
    "gold_response",
)
_SAMPLE_FIELDS = (
    "semantic_samples",
    "sampled_responses",
    "sampled_generations",
    "sample_generations",
    "samples",
    "generations",
)
_TEXT_FIELDS = (
    "text",
    "prediction",
    "generated_text",
    "generated_response",
    "generation",
    "response",
    "output",
)
_LOG_PROBABILITY_FIELDS = (
    "length_normalized_log_probability",
    "mean_log_probability",
    "sequence_log_probability",
    "log_probability",
    "log_prob",
)
_LOG_PROBABILITY_LIST_FIELDS = (
    "length_normalized_log_probabilities",
    "mean_log_probabilities",
    "sample_log_probabilities",
    "sample_log_probs",
    "sequence_log_probabilities",
    "generation_log_probabilities",
    "log_probabilities",
)


def _record_mapping(record: Any) -> dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    if is_dataclass(record):
        return asdict(record)
    to_dict = getattr(record, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            return dict(value)
    raise TypeError("Generation records must be mappings, dataclasses, or expose to_dict()")


def _first_present(
    mapping: Mapping[str, Any],
    fields: Sequence[str],
) -> tuple[str | None, Any]:
    for field in fields:
        if field in mapping and mapping[field] is not None:
            return field, mapping[field]
    return None, None


def _text_from_value(value: Any, *, field_name: str) -> str:
    if isinstance(value, str):
        return value
    if is_dataclass(value):
        value = asdict(value)
    elif not isinstance(value, Mapping):
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            value = to_dict()
    if isinstance(value, Mapping):
        _, nested = _first_present(value, _TEXT_FIELDS)
        if nested is not None:
            return str(nested)
    raise ValueError(f"{field_name} must contain text")


def _prediction(record: Mapping[str, Any]) -> tuple[str, str]:
    field, value = _first_present(record, _PREDICTION_FIELDS)
    if field is None:
        # A sample-only cache can still be evaluated using its first sample.
        sample_field, samples = _first_present(record, _SAMPLE_FIELDS)
        if sample_field is not None and isinstance(samples, Sequence) and samples:
            return _text_from_value(samples[0], field_name=sample_field), sample_field
        raise ValueError("Generation record has no prediction field")
    return _text_from_value(value, field_name=field), field


def _reference_values(
    record: Mapping[str, Any],
    *,
    prediction_field: str,
) -> list[str]:
    field, value = _first_present(record, _REFERENCE_FIELDS)
    # ``PromptResponse``-style joined records may retain the gold response
    # while storing the generated output under a dedicated prediction field.
    if field is None and prediction_field != "response" and "response" in record:
        value = record["response"]
        field = "response"
    if field is None:
        raise ValueError("Generation record has no references")

    if isinstance(value, str):
        references = [value]
    elif isinstance(value, Mapping):
        references = [
            _text_from_value(value, field_name=field),
        ]
    else:
        references = [
            _text_from_value(item, field_name=field) if isinstance(item, Mapping) else str(item)
            for item in value
        ]
    if not references:
        raise ValueError("Generation record references must not be empty")
    return references


def _log_probability_from_sample(
    sample: Mapping[str, Any],
    *,
    length_normalize: bool,
) -> float | None:
    for field in _LOG_PROBABILITY_FIELDS:
        if field not in sample or sample[field] is None:
            continue
        value = float(sample[field])
        already_normalized = field in {
            "length_normalized_log_probability",
            "mean_log_probability",
        }
        if length_normalize and not already_normalized:
            token_count = (
                sample.get("generated_token_count")
                or sample.get("token_count")
                or sample.get("num_tokens")
            )
            if token_count is not None:
                count = int(token_count)
                if count <= 0:
                    raise ValueError("sample token_count must be positive")
                value /= count
        return value

    token_values = sample.get("token_log_probabilities")
    if token_values is None:
        token_values = sample.get("token_log_probs")
    if token_values is not None:
        values = [float(value) for value in token_values]
        if not values:
            raise ValueError("token log probabilities must not be empty")
        return math.fsum(values) / len(values) if length_normalize else math.fsum(values)
    return None


def _samples_and_log_probabilities(
    record: Mapping[str, Any],
    prediction: str,
    *,
    length_normalize: bool,
) -> tuple[list[str], list[float] | None]:
    sample_field, raw_samples = _first_present(record, _SAMPLE_FIELDS)
    if sample_field is None:
        return [prediction], None
    if isinstance(raw_samples, (str, bytes)) or not isinstance(raw_samples, Sequence):
        raise TypeError(f"{sample_field} must be a sequence")
    if not raw_samples:
        raise ValueError(f"{sample_field} must not be empty")

    samples: list[str] = []
    embedded_log_probabilities: list[float | None] = []
    for sample in raw_samples:
        samples.append(_text_from_value(sample, field_name=sample_field))
        if is_dataclass(sample):
            sample_mapping: Mapping[str, Any] | None = asdict(sample)
        elif isinstance(sample, Mapping):
            sample_mapping = sample
        else:
            to_dict = getattr(sample, "to_dict", None)
            sample_mapping = to_dict() if callable(to_dict) else None
        if isinstance(sample_mapping, Mapping):
            embedded_log_probabilities.append(
                _log_probability_from_sample(
                    sample_mapping,
                    length_normalize=length_normalize,
                )
            )
        else:
            embedded_log_probabilities.append(None)

    log_probability_field, record_log_probabilities = _first_present(
        record,
        _LOG_PROBABILITY_LIST_FIELDS,
    )
    if record_log_probabilities is not None:
        log_probabilities = [float(value) for value in record_log_probabilities]
        if len(log_probabilities) != len(samples):
            raise ValueError("Sample log-probability list must align with semantic samples")
        token_counts = record.get("sample_token_counts")
        already_normalized = log_probability_field in {
            "length_normalized_log_probabilities",
            "mean_log_probabilities",
        }
        if length_normalize and token_counts is not None and not already_normalized:
            counts = [int(value) for value in token_counts]
            if len(counts) != len(samples) or any(count <= 0 for count in counts):
                raise ValueError("sample_token_counts must be aligned and positive")
            log_probabilities = [
                value / count
                for value, count in zip(
                    log_probabilities,
                    counts,
                    strict=True,
                )
            ]
        return samples, log_probabilities

    if all(value is not None for value in embedded_log_probabilities):
        return samples, [float(value) for value in embedded_log_probabilities]
    return samples, None


def _config_value(config: Any | None, name: str, default: Any) -> Any:
    if config is None:
        return default
    evaluation = getattr(config, "evaluation", config)
    return getattr(evaluation, name, default)


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def evaluate_generation_records(
    records: Iterable[Any],
    semantic_scorer: SemanticScorer | None = None,
    *,
    config: Any | None = None,
    semantic_estimator: str | None = None,
    confidence_mapping: str | None = None,
    correctness_scorer: SemanticScorer | None = None,
    correctness_mode: str = "semantic",
    correctness_threshold: float = 0.5,
    length_normalize_log_probabilities: bool | None = None,
    max_examples: int | None = None,
) -> dict[str, Any]:
    """Evaluate already-generated records and return metrics plus examples.

    Without an experiment config, the likelihood-based Kuhn et al. Eq. 4
    estimator is the default and requires one log probability per sample.
    ``frequency`` remains available as the distinct cluster-assignment entropy
    ablation.  The returned ``accuracy`` follows Table 1 and is ROUGE-L F1;
    binary semantic correctness is reported as ``semantic_correctness_rate``.
    """

    estimator = (
        semantic_estimator
        if semantic_estimator is not None
        else _config_value(config, "semantic_estimator", "kuhn_eq4")
    )
    mapping = (
        confidence_mapping
        if confidence_mapping is not None
        else _config_value(
            config,
            "semantic_confidence_mapping",
            "exp_negative_entropy",
        )
    )
    if length_normalize_log_probabilities is None:
        length_normalize_log_probabilities = bool(
            _config_value(
                config,
                "length_normalize_sequence_probability",
                True,
            )
        )
    if max_examples is None:
        max_examples = _config_value(config, "max_eval_examples", None)
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive")

    normalized_estimator = str(estimator).casefold().replace("-", "_")
    clustering_algorithm = _config_value(
        config,
        "semantic_clustering_algorithm",
        "representative_greedy",
    )
    if normalized_estimator not in {
        "frequency",
        "sample_frequency",
        "kuhn",
        "kuhn_eq4",
        "semantic_likelihood",
        "model_probability",
        "model_log_probability",
        "normalized_class_entropy",
        "rao",
        "sequence_probability",
    }:
        raise ValueError(f"Unknown semantic estimator: {estimator!r}")

    example_rows: list[dict[str, Any]] = []
    for index, raw_record in enumerate(records):
        if max_examples is not None and index >= max_examples:
            break
        record = _record_mapping(raw_record)
        prediction, prediction_field = _prediction(record)
        references = _reference_values(
            record,
            prediction_field=prediction_field,
        )
        prompt = str(record.get("prompt", "")) or None
        samples, available_log_probabilities = _samples_and_log_probabilities(
            record,
            prediction,
            length_normalize=length_normalize_log_probabilities,
        )
        cluster_ids = semantic_cluster_ids(
            samples,
            semantic_scorer,
            prompt=prompt,
            algorithm=clustering_algorithm,
        )

        if normalized_estimator in {"frequency", "sample_frequency"}:
            selected_log_probabilities = None
        else:
            if available_log_probabilities is None:
                raise ValueError(
                    "Model-probability semantic entropy requires one log "
                    "probability per semantic sample"
                )
            selected_log_probabilities = available_log_probabilities
        if normalized_estimator in {
            "kuhn",
            "kuhn_eq4",
            "semantic_likelihood",
        }:
            assert selected_log_probabilities is not None
            entropy = kuhn_eq4_semantic_entropy(
                cluster_ids,
                selected_log_probabilities,
            )
            normalized_mapping = str(mapping).casefold().replace("-", "_")
            if normalized_mapping in {
                "maximum_class_probability",
                "max_class_probability",
                "sample_consistency",
            }:
                confidence = semantic_confidence(
                    cluster_ids,
                    selected_log_probabilities,
                    mapping=mapping,
                )
            else:
                confidence = entropy_to_confidence(
                    max(0.0, entropy),
                    mapping=mapping,
                    num_classes=len(set(cluster_ids)),
                )
        else:
            entropy = semantic_entropy(
                cluster_ids,
                selected_log_probabilities,
            )
            confidence = semantic_confidence(
                cluster_ids,
                selected_log_probabilities,
                mapping=mapping,
            )
        correctness = reference_correctness_score(
            prediction,
            references,
            correctness_scorer if correctness_scorer is not None else semantic_scorer,
            prompt=prompt,
            mode=correctness_mode,
            threshold=correctness_threshold,
        )

        example_id = record.get("example_id", record.get("id", index))
        example_row = {
                "example_id": str(example_id),
                "prediction": prediction,
                "references": references,
                "correctness": correctness,
                "semantic_entropy": entropy,
                "semantic_estimator": normalized_estimator,
                "confidence": confidence,
                "semantic_class_ids": cluster_ids,
                "num_semantic_classes": len(set(cluster_ids)),
                "rouge_l_f1": max_rouge_l_f1(prediction, references),
                "rouge_1_f1": max_rouge_1_f1(prediction, references),
            }
        for field in (
            "routing_distance",
            "second_routing_distance",
            "routing_margin",
            "first_token_entropy",
            "normalized_first_token_entropy",
            "first_token_confidence",
            "non_special_first_token_entropy",
            "normalized_non_special_first_token_entropy",
            "non_special_first_token_confidence",
        ):
            if field in record:
                example_row[field] = record[field]
        example_rows.append(example_row)

    if not example_rows:
        raise ValueError("No generation records were evaluated")

    confidences = [float(row["confidence"]) for row in example_rows]
    outcomes = [float(row["correctness"]) for row in example_rows]
    try:
        auroc: float | None = auroc_score(confidences, outcomes)
    except ValueError:
        # A one-class slice has no ROC curve, but all other metrics remain valid.
        auroc = None

    metrics = {
        "num_examples": len(example_rows),
        "brier": brier_score(confidences, outcomes),
        "auroc": auroc,
        # Table 1 names ROUGE-L F1 "Acc"; binary semantic equality is used
        # only as the correctness target for Brier/AUROC.
        "semantic_correctness_rate": _mean(outcomes),
        "rouge_l_f1": _mean([float(row["rouge_l_f1"]) for row in example_rows]),
        "rouge_1_f1": _mean([float(row["rouge_1_f1"]) for row in example_rows]),
        "mean_semantic_entropy": _mean([float(row["semantic_entropy"]) for row in example_rows]),
        "mean_confidence": _mean(confidences),
    }
    metrics["accuracy"] = metrics["rouge_l_f1"]
    return {
        "metrics": metrics,
        "examples": example_rows,
    }


def iter_generation_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    """Yield generation records with line-numbered JSONL errors."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {source}:{line_number}") from error
            if not isinstance(value, Mapping):
                raise TypeError(f"Generation JSONL row {source}:{line_number} is not an object")
            yield dict(value)


def evaluate_generation_jsonl(
    path: str | Path,
    semantic_scorer: SemanticScorer | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Evaluate cached generation records from a JSONL file."""

    return evaluate_generation_records(
        iter_generation_jsonl(path),
        semantic_scorer,
        **kwargs,
    )


evaluate_jsonl = evaluate_generation_jsonl
evaluate_dataset = evaluate_generation_records


__all__ = [
    "evaluate_dataset",
    "evaluate_generation_jsonl",
    "evaluate_generation_records",
    "evaluate_jsonl",
    "iter_generation_jsonl",
]
