"""Correctness decisions for free-form generations."""

from __future__ import annotations

from collections.abc import Iterable

from .metrics import max_rouge_1_f1, max_rouge_l_f1
from .semantic import SemanticScorer, _equivalent_pairs


def _references(references: Iterable[str] | str) -> list[str]:
    if isinstance(references, str):
        values = [references]
    else:
        values = [str(reference) for reference in references]
    if not values:
        raise ValueError("At least one reference is required")
    return values


def correctness_against_references(
    prediction: str,
    references: Iterable[str] | str,
    scorer: SemanticScorer | None = None,
    *,
    prompt: str | None = None,
) -> bool:
    """Return true when a prediction is equivalent to any valid reference."""

    reference_values = _references(references)
    decisions = _equivalent_pairs(
        scorer,
        [(str(prediction), reference) for reference in reference_values],
        prompt,
    )
    return any(decisions)


def reference_correctness_score(
    prediction: str,
    references: Iterable[str] | str,
    scorer: SemanticScorer | None = None,
    *,
    prompt: str | None = None,
    mode: str = "semantic",
    threshold: float = 0.5,
) -> float:
    """Return binary correctness with a configurable reference comparison.

    ROUGE threshold modes are useful for summarization ablations.  The paper's
    QA evaluation uses the default semantic-equivalence mode.
    """

    normalized_mode = mode.casefold().replace("-", "_")
    reference_values = _references(references)
    if normalized_mode in {"semantic", "semantic_equivalence", "nli"}:
        return float(
            correctness_against_references(
                prediction,
                reference_values,
                scorer,
                prompt=prompt,
            )
        )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if normalized_mode in {"rouge_l", "rougel", "rouge_l_f1"}:
        return float(max_rouge_l_f1(prediction, reference_values) >= threshold)
    if normalized_mode in {"rouge_1", "rouge1", "rouge_1_f1"}:
        return float(max_rouge_1_f1(prediction, reference_values) >= threshold)
    raise ValueError(f"Unknown correctness mode: {mode!r}")


semantic_correctness = correctness_against_references


__all__ = [
    "correctness_against_references",
    "reference_correctness_score",
    "semantic_correctness",
]
