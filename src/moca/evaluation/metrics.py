"""Dependency-light calibration and text-overlap metrics.

The functions in this module deliberately operate on ordinary Python
sequences.  ROUGE is the only optional dependency: when ``rouge_score`` is
installed we use its reference implementation, otherwise a deterministic
token-level implementation is used.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from functools import lru_cache


def _validated_binary_inputs(
    confidences: Iterable[float],
    outcomes: Iterable[bool | int | float],
) -> tuple[list[float], list[int]]:
    probabilities = [float(value) for value in confidences]
    labels: list[int] = []
    for value in outcomes:
        numeric = float(value)
        if numeric not in (0.0, 1.0):
            raise ValueError(f"Binary outcomes must be 0 or 1, got {value!r}")
        labels.append(int(numeric))

    if not probabilities:
        raise ValueError("At least one prediction is required")
    if len(probabilities) != len(labels):
        raise ValueError(
            "Confidences and outcomes must have equal length, got "
            f"{len(probabilities)} and {len(labels)}"
        )
    for probability in probabilities:
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"Confidence values must be finite and in [0, 1], got {probability!r}")
    return probabilities, labels


def brier_score(
    confidences: Iterable[float],
    outcomes: Iterable[bool | int | float],
) -> float:
    """Return the mean binary Brier score.

    A lower value is better.  ``confidences`` are probabilities assigned to
    the event represented by an outcome of one.
    """

    probabilities, labels = _validated_binary_inputs(confidences, outcomes)
    return math.fsum(
        (probability - label) ** 2 for probability, label in zip(probabilities, labels, strict=True)
    ) / len(labels)


def auroc_score(
    confidences: Iterable[float],
    outcomes: Iterable[bool | int | float],
) -> float:
    """Return binary AUROC using average ranks for tied confidence values.

    This is equivalent to the Mann-Whitney U statistic and does not require
    scikit-learn.  AUROC is undefined when only one outcome class is present,
    in which case a :class:`ValueError` is raised.
    """

    probabilities, labels = _validated_binary_inputs(confidences, outcomes)
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires at least one positive and one negative outcome")

    ordered = sorted(
        enumerate(probabilities),
        key=lambda item: (item[1], item[0]),
    )
    positive_rank_sum = 0.0
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        # Ranks are one-indexed.  All members of a tie receive the average.
        average_rank = ((cursor + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(labels[index] for index, _ in ordered[cursor:end])
        cursor = end

    statistic = positive_rank_sum - positives * (positives + 1) / 2.0
    return statistic / (positives * negatives)


def reliability_bins(
    confidences: Iterable[float],
    outcomes: Iterable[bool | int | float],
    *,
    num_bins: int = 15,
    adaptive: bool = False,
) -> list[dict[str, float | int]]:
    """Return fixed-width or equal-count reliability bins."""

    probabilities, labels = _validated_binary_inputs(confidences, outcomes)
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    bins: list[list[int]] = [[] for _ in range(min(num_bins, len(labels)))]
    if adaptive:
        ordered = sorted(range(len(labels)), key=lambda index: (probabilities[index], index))
        for rank, index in enumerate(ordered):
            bins[min(len(bins) - 1, rank * len(bins) // len(ordered))].append(index)
    else:
        for index, probability in enumerate(probabilities):
            bin_index = min(len(bins) - 1, int(probability * len(bins)))
            bins[bin_index].append(index)

    result: list[dict[str, float | int]] = []
    for bin_index, indices in enumerate(bins):
        if not indices:
            continue
        mean_confidence = math.fsum(probabilities[index] for index in indices) / len(indices)
        accuracy = math.fsum(labels[index] for index in indices) / len(indices)
        result.append(
            {
                "bin": bin_index,
                "count": len(indices),
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "absolute_gap": abs(mean_confidence - accuracy),
            }
        )
    return result


def expected_calibration_error(
    confidences: Iterable[float],
    outcomes: Iterable[bool | int | float],
    *,
    num_bins: int = 15,
    adaptive: bool = False,
) -> float:
    probabilities, labels = _validated_binary_inputs(confidences, outcomes)
    bins = reliability_bins(probabilities, labels, num_bins=num_bins, adaptive=adaptive)
    return math.fsum(float(item["count"]) * float(item["absolute_gap"]) for item in bins) / len(
        labels
    )


def binary_log_loss(
    confidences: Iterable[float],
    outcomes: Iterable[bool | int | float],
    *,
    epsilon: float = 1e-12,
) -> float:
    probabilities, labels = _validated_binary_inputs(confidences, outcomes)
    return -math.fsum(
        label * math.log(min(1.0 - epsilon, max(epsilon, probability)))
        + (1 - label) * math.log(min(1.0 - epsilon, max(epsilon, 1.0 - probability)))
        for probability, label in zip(probabilities, labels, strict=True)
    ) / len(labels)


def risk_coverage_curve(
    confidences: Iterable[float],
    outcomes: Iterable[bool | int | float],
) -> tuple[list[dict[str, float | int]], float]:
    """Return selective risk as successively lower-confidence examples are included."""

    probabilities, labels = _validated_binary_inputs(confidences, outcomes)
    ordered = sorted(range(len(labels)), key=lambda index: (-probabilities[index], index))
    errors = 0
    curve: list[dict[str, float | int]] = []
    risks: list[float] = []
    for rank, index in enumerate(ordered, start=1):
        errors += 1 - labels[index]
        risk = errors / rank
        risks.append(risk)
        curve.append(
            {
                "accepted": rank,
                "coverage": rank / len(labels),
                "risk": risk,
                "threshold": probabilities[index],
            }
        )
    aurc = math.fsum(risks) / len(risks)
    return curve, aurc


def spearman_correlation(left: Iterable[float], right: Iterable[float]) -> float:
    """Spearman correlation with average ranks for ties."""

    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    if len(left_values) != len(right_values) or len(left_values) < 2:
        raise ValueError("Spearman correlation requires equal sequences of length at least two")
    if any(not math.isfinite(value) for value in left_values + right_values):
        raise ValueError("Spearman inputs must be finite")

    def ranks(values: list[float]) -> list[float]:
        ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
        result = [0.0] * len(values)
        cursor = 0
        while cursor < len(ordered):
            end = cursor + 1
            while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
                end += 1
            average = ((cursor + 1) + end) / 2.0
            for index in ordered[cursor:end]:
                result[index] = average
            cursor = end
        return result

    left_ranks = ranks(left_values)
    right_ranks = ranks(right_values)
    left_mean = math.fsum(left_ranks) / len(left_ranks)
    right_mean = math.fsum(right_ranks) / len(right_ranks)
    numerator = math.fsum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_ranks, right_ranks, strict=True)
    )
    left_norm = math.sqrt(math.fsum((value - left_mean) ** 2 for value in left_ranks))
    right_norm = math.sqrt(math.fsum((value - right_mean) ** 2 for value in right_ranks))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("Spearman correlation is undefined for a constant sequence")
    return numerator / (left_norm * right_norm)


def false_positive_rate_at_recall(
    scores: Iterable[float],
    outcomes: Iterable[bool | int | float],
    *,
    recall: float = 0.95,
) -> float:
    """Return negative-class FPR at the first threshold reaching target recall."""

    probabilities, labels = _validated_binary_inputs(scores, outcomes)
    if not 0.0 < recall <= 1.0:
        raise ValueError("recall must be in (0, 1]")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("FPR@recall requires both outcome classes")
    ordered = sorted(range(len(labels)), key=lambda index: (-probabilities[index], index))
    true_positives = 0
    false_positives = 0
    for index in ordered:
        if labels[index]:
            true_positives += 1
        else:
            false_positives += 1
        if true_positives / positives >= recall:
            return false_positives / negatives
    return 1.0


_TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def _fallback_tokens(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(str(text).casefold())


def _f1(overlap: int, prediction_count: int, reference_count: int) -> float:
    if overlap <= 0 or prediction_count == 0 or reference_count == 0:
        return 0.0
    precision = overlap / prediction_count
    recall = overlap / reference_count
    return 2.0 * precision * recall / (precision + recall)


def _fallback_rouge_1_f1(prediction: str, reference: str) -> float:
    prediction_tokens = _fallback_tokens(prediction)
    reference_tokens = _fallback_tokens(reference)
    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    return _f1(overlap, len(prediction_tokens), len(reference_tokens))


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    # Keep only one dynamic-programming row.  Swapping does not alter LCS.
    if len(right) > len(left):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _fallback_rouge_l_f1(prediction: str, reference: str) -> float:
    prediction_tokens = _fallback_tokens(prediction)
    reference_tokens = _fallback_tokens(reference)
    overlap = _lcs_length(prediction_tokens, reference_tokens)
    return _f1(overlap, len(prediction_tokens), len(reference_tokens))


@lru_cache(maxsize=4)
def _rouge_scorer(metric: str, use_stemmer: bool):
    """Load the optional reference implementation only when first needed."""

    try:
        from rouge_score import rouge_scorer
    except (ImportError, ModuleNotFoundError):
        return None
    return rouge_scorer.RougeScorer([metric], use_stemmer=use_stemmer)


def rouge_1_f1(
    prediction: str,
    reference: str,
    *,
    use_stemmer: bool = False,
) -> float:
    """Return token-unigram ROUGE-1 F1 for one prediction/reference pair."""

    scorer = _rouge_scorer("rouge1", use_stemmer)
    if scorer is None:
        return _fallback_rouge_1_f1(prediction, reference)
    return float(scorer.score(str(reference), str(prediction))["rouge1"].fmeasure)


def rouge_l_f1(
    prediction: str,
    reference: str,
    *,
    use_stemmer: bool = False,
) -> float:
    """Return sentence-level ROUGE-L F1 for one pair."""

    scorer = _rouge_scorer("rougeL", use_stemmer)
    if scorer is None:
        return _fallback_rouge_l_f1(prediction, reference)
    return float(scorer.score(str(reference), str(prediction))["rougeL"].fmeasure)


def _coerce_references(references: Iterable[str] | str) -> list[str]:
    if isinstance(references, str):
        values = [references]
    else:
        values = [str(reference) for reference in references]
    if not values:
        raise ValueError("At least one reference is required")
    return values


def max_rouge_1_f1(
    prediction: str,
    references: Iterable[str] | str,
    *,
    use_stemmer: bool = False,
) -> float:
    """Return the best ROUGE-1 F1 over all valid references."""

    return max(
        rouge_1_f1(prediction, reference, use_stemmer=use_stemmer)
        for reference in _coerce_references(references)
    )


def max_rouge_l_f1(
    prediction: str,
    references: Iterable[str] | str,
    *,
    use_stemmer: bool = False,
) -> float:
    """Return the best ROUGE-L F1 over all valid references."""

    return max(
        rouge_l_f1(prediction, reference, use_stemmer=use_stemmer)
        for reference in _coerce_references(references)
    )


# Familiar aliases for callers that use scikit-learn or paper terminology.
binary_brier_score = brier_score
roc_auc_score = auroc_score
rouge1_f1 = rouge_1_f1
rougeL_f1 = rouge_l_f1


__all__ = [
    "auroc_score",
    "binary_log_loss",
    "binary_brier_score",
    "brier_score",
    "expected_calibration_error",
    "false_positive_rate_at_recall",
    "max_rouge_1_f1",
    "max_rouge_l_f1",
    "roc_auc_score",
    "reliability_bins",
    "risk_coverage_curve",
    "rouge1_f1",
    "rougeL_f1",
    "rouge_1_f1",
    "rouge_l_f1",
    "spearman_correlation",
]
