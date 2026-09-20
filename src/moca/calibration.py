"""Validation-fitted, single-generation confidence for MoCA.

The paper's multi-sample semantic entropy remains available as an offline
diagnostic.  This module turns signals already produced by one routed
generation into a probability of semantic correctness without fitting on the
test set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import atomic_write_json, read_json


DEFAULT_FEATURES = (
    "non_special_first_token_confidence",
    "negative_log1p_routing_distance",
    "routing_margin",
)


def _feature_value(row: Mapping[str, Any], name: str) -> float:
    if name == "negative_log1p_routing_distance":
        value = -math.log1p(max(0.0, float(row["routing_distance"])))
    elif name == "length_normalized_sequence_log_probability":
        token_count = max(1, int(row.get("generated_token_count", 1)))
        value = float(row["sequence_log_probability"]) / token_count
    elif name == "relative_routing_margin":
        d1 = max(0.0, float(row["routing_distance"]))
        d2 = max(0.0, float(row["second_routing_distance"]))
        if d2 <= 1e-12:
            value = 0.0
        else:
            value = (d2 - d1) / d2
    elif name == "router_confidence":
        # Normalized router entropy lies in [0, 1].
        # Turning it into confidence makes larger = more decisive.
        value = 1.0 - float(row["router_entropy"])
    elif name == "negative_routing_distance_z":
        value = -float(row["routing_distance_z"])
    else:
        value = float(row[name])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite calibration feature {name!r}: {value!r}")
    return value


def feature_matrix(
    rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str] = DEFAULT_FEATURES,
) -> np.ndarray:
    names = tuple(feature_names)
    if not rows:
        raise ValueError("At least one row is required")
    if not names:
        raise ValueError("At least one calibration feature is required")
    return np.asarray(
        [[_feature_value(row, name) for name in names] for row in rows],
        dtype=np.float64,
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _objective(
    design: np.ndarray,
    labels: np.ndarray,
    coefficients: np.ndarray,
    l2: float,
) -> float:
    logits = design @ coefficients
    losses = np.logaddexp(0.0, logits) - labels * logits
    return float(losses.mean() + 0.5 * l2 * np.square(coefficients[1:]).sum())


@dataclass(frozen=True)
class SinglePassCalibrator:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    intercept: float
    coefficients: tuple[float, ...]
    method: str = "logistic"
    l2: float = 1e-3
    num_validation_examples: int = 0
    validation_positive_rate: float = 0.0
    converged: bool = True
    source_fingerprint: str | None = None
    config_fingerprint: str | None = None

    def predict(self, rows: Sequence[Mapping[str, Any]]) -> list[float]:
        values = feature_matrix(rows, self.feature_names)
        means = np.asarray(self.means, dtype=np.float64)
        scales = np.asarray(self.scales, dtype=np.float64)
        weights = np.asarray(self.coefficients, dtype=np.float64)
        standardized = (values - means) / scales
        probabilities = _sigmoid(self.intercept + standardized @ weights)
        return [float(value) for value in probabilities]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "method": self.method,
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "scales": list(self.scales),
            "intercept": self.intercept,
            "coefficients": list(self.coefficients),
            "l2": self.l2,
            "num_validation_examples": self.num_validation_examples,
            "validation_positive_rate": self.validation_positive_rate,
            "converged": self.converged,
            "source_fingerprint": self.source_fingerprint,
            "config_fingerprint": self.config_fingerprint,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "SinglePassCalibrator":
        if int(values.get("format_version", 0)) != 1:
            raise ValueError("Unsupported calibrator artifact format")
        feature_names = tuple(str(value) for value in values["feature_names"])
        means = tuple(float(value) for value in values["means"])
        scales = tuple(float(value) for value in values["scales"])
        coefficients = tuple(float(value) for value in values["coefficients"])
        if not (len(feature_names) == len(means) == len(scales) == len(coefficients)):
            raise ValueError("Calibrator feature dimensions do not agree")
        if any(scale <= 0.0 or not math.isfinite(scale) for scale in scales):
            raise ValueError("Calibrator scales must be finite and positive")
        return cls(
            feature_names=feature_names,
            means=means,
            scales=scales,
            intercept=float(values["intercept"]),
            coefficients=coefficients,
            method=str(values.get("method", "logistic")),
            l2=float(values.get("l2", 1e-3)),
            num_validation_examples=int(values.get("num_validation_examples", 0)),
            validation_positive_rate=float(values.get("validation_positive_rate", 0.0)),
            converged=bool(values.get("converged", True)),
            source_fingerprint=values.get("source_fingerprint"),
            config_fingerprint=values.get("config_fingerprint"),
        )


def fit_logistic_calibrator(
    rows: Sequence[Mapping[str, Any]],
    outcomes: Sequence[bool | int | float],
    *,
    feature_names: Sequence[str] = DEFAULT_FEATURES,
    l2: float = 1e-3,
    max_iter: int = 200,
    tolerance: float = 1e-8,
    source_fingerprint: str | None = None,
    config_fingerprint: str | None = None,
) -> SinglePassCalibrator:
    """Fit a regularized logistic calibrator using damped Newton steps."""

    if l2 < 0.0 or not math.isfinite(l2):
        raise ValueError("l2 must be finite and non-negative")
    if max_iter <= 0 or tolerance <= 0.0:
        raise ValueError("max_iter and tolerance must be positive")
    labels = np.asarray([float(value) for value in outcomes], dtype=np.float64)
    if labels.ndim != 1 or len(labels) != len(rows):
        raise ValueError("Rows and outcomes must have equal non-zero length")
    if not len(labels) or np.any((labels != 0.0) & (labels != 1.0)):
        raise ValueError("Outcomes must be non-empty binary labels")

    names = tuple(feature_names)
    values = feature_matrix(rows, names)
    means = values.mean(axis=0)
    scales = values.std(axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    standardized = (values - means) / scales
    design = np.column_stack((np.ones(len(values), dtype=np.float64), standardized))

    # Jeffreys-smoothed prevalence gives a finite intercept even for a
    # one-class validation slice.  Such a slice cannot identify feature
    # weights, so use the honest constant-probability model.
    prevalence = float((labels.sum() + 0.5) / (len(labels) + 1.0))
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    coefficients[0] = math.log(prevalence / (1.0 - prevalence))
    converged = len(np.unique(labels)) < 2

    if not converged:
        identity = np.eye(design.shape[1], dtype=np.float64)
        identity[0, 0] = 0.0
        for _ in range(max_iter):
            logits = design @ coefficients
            probabilities = _sigmoid(logits)
            gradient = design.T @ (probabilities - labels) / len(labels)
            gradient += l2 * (identity @ coefficients)
            curvature = probabilities * (1.0 - probabilities)
            hessian = (design.T * curvature) @ design / len(labels)
            hessian += l2 * identity + 1e-9 * np.eye(design.shape[1])
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]

            previous = _objective(design, labels, coefficients, l2)
            step_scale = 1.0
            candidate = coefficients - step
            while _objective(design, labels, candidate, l2) > previous and step_scale > 1e-6:
                step_scale *= 0.5
                candidate = coefficients - step_scale * step
            change = float(np.max(np.abs(candidate - coefficients)))
            coefficients = candidate
            if change <= tolerance:
                converged = True
                break

    return SinglePassCalibrator(
        feature_names=names,
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        intercept=float(coefficients[0]),
        coefficients=tuple(float(value) for value in coefficients[1:]),
        l2=l2,
        num_validation_examples=len(labels),
        validation_positive_rate=float(labels.mean()),
        converged=converged,
        source_fingerprint=source_fingerprint,
        config_fingerprint=config_fingerprint,
    )


def save_calibrator(calibrator: SinglePassCalibrator, path: str | Path) -> None:
    atomic_write_json(path, calibrator.to_dict())


def load_calibrator(path: str | Path) -> SinglePassCalibrator:
    return SinglePassCalibrator.from_dict(read_json(path))


__all__ = [
    "DEFAULT_FEATURES",
    "SinglePassCalibrator",
    "feature_matrix",
    "fit_logistic_calibrator",
    "load_calibrator",
    "save_calibrator",
]
