"""Helpers for token-distribution entropy by generation position."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


def _as_numpy(value: Any) -> np.ndarray:
    # This handles CPU/GPU torch tensors without importing torch eagerly.
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def categorical_entropy(
    values: Any,
    *,
    input_type: str = "logits",
    axis: int = -1,
    base: float = math.e,
) -> np.ndarray:
    """Compute categorical entropy along ``axis``.

    ``input_type`` may be ``"logits"``, ``"log_probabilities"``, or
    ``"probabilities"``.  Probability-like inputs are normalized defensively.
    """

    array = _as_numpy(values)
    if array.size == 0:
        raise ValueError("values must not be empty")
    if not math.isfinite(base) or base <= 0.0 or base == 1.0:
        raise ValueError("Entropy base must be positive and not equal to one")
    normalized_type = input_type.casefold().replace("-", "_")

    if normalized_type == "logits":
        if np.any(np.isnan(array) | np.isposinf(array)):
            raise ValueError("logits may be finite or negative infinity")
        maximum = np.max(array, axis=axis, keepdims=True)
        if np.any(~np.isfinite(maximum)):
            raise ValueError("Every distribution must contain a finite logit")
        shifted = array - maximum
        weights = np.exp(shifted)
        probabilities = weights / np.sum(weights, axis=axis, keepdims=True)
    elif normalized_type in {
        "log_probabilities",
        "log_probs",
        "log_probability",
    }:
        if np.any(np.isnan(array) | np.isposinf(array)):
            raise ValueError("log probabilities may be finite or negative infinity")
        maximum = np.max(array, axis=axis, keepdims=True)
        if np.any(~np.isfinite(maximum)):
            raise ValueError("Every distribution must contain finite log probability")
        weights = np.exp(array - maximum)
        probabilities = weights / np.sum(weights, axis=axis, keepdims=True)
    elif normalized_type in {"probabilities", "probs", "probability"}:
        if np.any(~np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError("probabilities must be finite and non-negative")
        totals = np.sum(array, axis=axis, keepdims=True)
        if np.any(totals <= 0.0):
            raise ValueError("Every distribution must have positive probability mass")
        probabilities = array / totals
    else:
        raise ValueError(f"Unknown input_type: {input_type!r}")

    terms = np.zeros_like(probabilities)
    positive = probabilities > 0.0
    terms[positive] = probabilities[positive] * np.log(probabilities[positive])
    return -np.sum(terms, axis=axis) / math.log(base)


def token_position_entropy(
    distributions: Any,
    attention_mask: Any | None = None,
    *,
    input_type: str = "logits",
    reduce_batch: bool = True,
    base: float = math.e,
) -> list[float] | list[list[float]]:
    """Return predictive entropy at each token position.

    Inputs may have shape ``[positions, vocabulary]`` or
    ``[batch, positions, vocabulary]``.  Batched inputs are averaged at each
    position by default, respecting an optional ``[batch, positions]`` mask.
    Set ``reduce_batch=False`` to retain one entropy trajectory per sample.
    """

    array = _as_numpy(distributions)
    if array.ndim not in (2, 3):
        raise ValueError(
            "distributions must have shape [positions, vocabulary] or "
            "[batch, positions, vocabulary]"
        )
    entropies = categorical_entropy(
        array,
        input_type=input_type,
        axis=-1,
        base=base,
    )

    if array.ndim == 2:
        if attention_mask is not None:
            mask = _as_numpy(attention_mask).astype(bool)
            if mask.shape != entropies.shape:
                raise ValueError("attention_mask has incompatible shape")
            entropies = np.where(mask, entropies, np.nan)
        return [float(value) for value in entropies.tolist()]

    if attention_mask is None:
        mask = np.ones(entropies.shape, dtype=bool)
    else:
        mask = _as_numpy(attention_mask).astype(bool)
        if mask.shape != entropies.shape:
            raise ValueError("attention_mask has incompatible shape")

    if not reduce_batch:
        masked = np.where(mask, entropies, np.nan)
        return [[float(value) for value in row] for row in masked.tolist()]

    result: list[float] = []
    for position in range(entropies.shape[1]):
        valid = entropies[:, position][mask[:, position]]
        result.append(float(np.mean(valid)) if valid.size else math.nan)
    return result


def mean_entropy_by_position(
    entropy_sequences: Sequence[Sequence[float]],
) -> list[float]:
    """Average precomputed, possibly variable-length entropy trajectories."""

    sequences = [[float(value) for value in sequence] for sequence in entropy_sequences]
    if not sequences:
        return []
    maximum_length = max((len(sequence) for sequence in sequences), default=0)
    result: list[float] = []
    for position in range(maximum_length):
        values = [
            sequence[position]
            for sequence in sequences
            if position < len(sequence) and math.isfinite(sequence[position])
        ]
        result.append(math.fsum(values) / len(values) if values else math.nan)
    return result


token_position_entropies = token_position_entropy
aggregate_token_position_entropy = mean_entropy_by_position


__all__ = [
    "aggregate_token_position_entropy",
    "categorical_entropy",
    "mean_entropy_by_position",
    "token_position_entropies",
    "token_position_entropy",
]
