"""Semantic equivalence clustering, entropy, and confidence utilities."""

from __future__ import annotations

import inspect
import math
import unicodedata
from collections.abc import Callable, Hashable, Iterable, Sequence
from typing import Any

SemanticScorer = Any


class _UnionFind:
    """Union-find whose representative is always the smallest member index."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        lower, higher = sorted((left_root, right_root))
        self.parent[higher] = lower


def normalize_semantic_text(text: str) -> str:
    """Return a conservative canonical form used when no NLI scorer is given."""

    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return " ".join(normalized.split())


def _accepts_keyword(function: Callable[..., Any], keyword: str) -> bool:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return True
    return keyword in signature.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _single_equivalent(
    scorer: SemanticScorer | None,
    left: str,
    right: str,
    prompt: str | None,
) -> bool:
    if scorer is None:
        return normalize_semantic_text(left) == normalize_semantic_text(right)

    function = getattr(scorer, "equivalent", None)
    if function is None:
        function = getattr(scorer, "is_equivalent", None)
    if function is None and callable(scorer):
        function = scorer
    if function is None:
        raise TypeError("Semantic scorer must be callable or expose equivalent/is_equivalent")
    if _accepts_keyword(function, "prompt"):
        return bool(function(left, right, prompt=prompt))
    return bool(function(left, right))


def _equivalent_pairs(
    scorer: SemanticScorer | None,
    pairs: Sequence[tuple[str, str]],
    prompt: str | None,
) -> list[bool]:
    if not pairs:
        return []
    if scorer is not None:
        batch_function = getattr(scorer, "equivalent_many", None)
        if batch_function is not None:
            if _accepts_keyword(batch_function, "prompts"):
                values = batch_function(
                    pairs,
                    prompts=[prompt] * len(pairs),
                )
            else:
                values = batch_function(pairs)
            materialized = [bool(value) for value in values]
            if len(materialized) != len(pairs):
                raise ValueError("Semantic scorer returned a different number of pair labels")
            return materialized
    return [_single_equivalent(scorer, left, right, prompt) for left, right in pairs]


def semantic_cluster_ids(
    responses: Sequence[str],
    scorer: SemanticScorer | None = None,
    *,
    prompt: str | None = None,
    algorithm: str = "representative_greedy",
) -> list[int]:
    """Assign semantic classes with Kuhn's greedy rule or transitive closure."""

    values = [str(response) for response in responses]
    if not values:
        return []
    normalized_algorithm = algorithm.casefold().replace("-", "_")

    if normalized_algorithm in {
        "representative_greedy",
        "kuhn_greedy",
        "greedy",
    }:
        representatives: list[int] = []
        labels: list[int] = []
        for index, value in enumerate(values):
            pairs = [(values[representative], value) for representative in representatives]
            decisions = _equivalent_pairs(scorer, pairs, prompt)
            matching = next(
                (class_id for class_id, equivalent in enumerate(decisions) if equivalent),
                None,
            )
            if matching is None:
                matching = len(representatives)
                representatives.append(index)
            labels.append(matching)
        return labels

    if normalized_algorithm not in {
        "transitive_closure",
        "pairwise_transitive",
        "union_find",
    }:
        raise ValueError(f"Unknown semantic clustering algorithm: {algorithm!r}")

    index_pairs = [
        (left, right) for left in range(len(values)) for right in range(left + 1, len(values))
    ]
    text_pairs = [(values[left], values[right]) for left, right in index_pairs]
    decisions = _equivalent_pairs(scorer, text_pairs, prompt)

    union_find = _UnionFind(len(values))
    for (left, right), equivalent in zip(
        index_pairs,
        decisions,
        strict=True,
    ):
        if equivalent:
            union_find.union(left, right)

    root_to_label: dict[int, int] = {}
    labels: list[int] = []
    for index in range(len(values)):
        root = union_find.find(index)
        if root not in root_to_label:
            root_to_label[root] = len(root_to_label)
        labels.append(root_to_label[root])
    return labels


def semantic_clusters(
    responses: Sequence[str],
    scorer: SemanticScorer | None = None,
    *,
    prompt: str | None = None,
    algorithm: str = "representative_greedy",
) -> list[list[int]]:
    """Return response-index components rather than a label per response."""

    labels = semantic_cluster_ids(
        responses,
        scorer,
        prompt=prompt,
        algorithm=algorithm,
    )
    clusters: list[list[int]] = []
    for index, label in enumerate(labels):
        while len(clusters) <= label:
            clusters.append([])
        clusters[label].append(index)
    return clusters


def _ordered_classes(cluster_ids: Sequence[Hashable]) -> list[Hashable]:
    classes: list[Hashable] = []
    seen: set[Hashable] = set()
    for cluster_id in cluster_ids:
        if cluster_id not in seen:
            seen.add(cluster_id)
            classes.append(cluster_id)
    return classes


def semantic_class_probabilities(
    cluster_ids: Sequence[Hashable],
    log_probabilities: Sequence[float] | None = None,
) -> dict[Hashable, float]:
    """Estimate semantic-class probabilities from counts or model log weights.

    When ``log_probabilities`` is supplied, sequence weights are normalized
    with a stable softmax and then summed within each semantic class.  This
    supports either raw sequence log likelihoods or length-normalized values,
    depending on what the generation record stores.
    """

    labels = list(cluster_ids)
    if not labels:
        raise ValueError("At least one semantic sample is required")
    classes = _ordered_classes(labels)

    if log_probabilities is None:
        counts = {cluster_id: 0 for cluster_id in classes}
        for cluster_id in labels:
            counts[cluster_id] += 1
        total = len(labels)
        return {cluster_id: counts[cluster_id] / total for cluster_id in classes}

    log_weights = [float(value) for value in log_probabilities]
    if len(log_weights) != len(labels):
        raise ValueError("cluster_ids and log_probabilities must have equal length")
    if any(math.isnan(value) or value == math.inf for value in log_weights):
        raise ValueError("log_probabilities may be finite or negative infinity")
    finite = [value for value in log_weights if math.isfinite(value)]
    if not finite:
        raise ValueError("At least one log probability must be finite")
    maximum = max(finite)
    weights = [0.0 if value == -math.inf else math.exp(value - maximum) for value in log_weights]
    total_weight = math.fsum(weights)
    class_weights = {cluster_id: 0.0 for cluster_id in classes}
    for cluster_id, weight in zip(labels, weights, strict=True):
        class_weights[cluster_id] += weight
    return {cluster_id: class_weights[cluster_id] / total_weight for cluster_id in classes}


def entropy_from_probabilities(
    probabilities: Iterable[float],
    *,
    base: float = math.e,
) -> float:
    """Return Shannon entropy for a normalized categorical distribution."""

    values = [float(probability) for probability in probabilities]
    if not values:
        raise ValueError("At least one probability is required")
    if not math.isfinite(base) or base <= 0.0 or base == 1.0:
        raise ValueError("Entropy base must be positive and not equal to one")
    if any(not math.isfinite(probability) or probability < 0.0 for probability in values):
        raise ValueError("Probabilities must be finite and non-negative")
    total = math.fsum(values)
    if total <= 0.0:
        raise ValueError("Probability mass must be positive")
    normalized = [probability / total for probability in values]
    entropy_nats = -math.fsum(
        probability * math.log(probability) for probability in normalized if probability > 0.0
    )
    return entropy_nats / math.log(base)


def semantic_entropy(
    cluster_ids: Sequence[Hashable],
    log_probabilities: Sequence[float] | None = None,
    *,
    normalize: bool = False,
    base: float = math.e,
) -> float:
    """Return entropy over sampled semantic-equivalence classes."""

    probabilities = semantic_class_probabilities(
        cluster_ids,
        log_probabilities=log_probabilities,
    )
    entropy = entropy_from_probabilities(probabilities.values(), base=base)
    if not normalize or len(probabilities) <= 1:
        return entropy
    return entropy / (math.log(len(probabilities)) / math.log(base))


def kuhn_eq4_semantic_entropy(
    cluster_ids: Sequence[Hashable],
    log_probabilities: Sequence[float],
    *,
    base: float = math.e,
) -> float:
    """Return the likelihood-based estimator from Kuhn et al. (2023), Eq. 4.

    Sequence likelihoods are summed in probability space within each semantic
    class using log-sum-exp.  The estimator is the negative arithmetic mean of
    those semantic-class log likelihoods.  Inputs should therefore be model
    log probabilities, optionally length-normalized as the cited paper's
    documented heuristic, rather than cluster frequencies.
    """

    labels = list(cluster_ids)
    log_weights = [float(value) for value in log_probabilities]
    if not labels:
        raise ValueError("At least one semantic sample is required")
    if len(labels) != len(log_weights):
        raise ValueError("cluster_ids and log_probabilities must have equal length")
    if any(math.isnan(value) or value == math.inf for value in log_weights):
        raise ValueError("log_probabilities may be finite or negative infinity")
    if not math.isfinite(base) or base <= 0.0 or base == 1.0:
        raise ValueError("Entropy base must be positive and not equal to one")

    class_log_probabilities: list[float] = []
    for cluster_id in _ordered_classes(labels):
        values = [
            log_probability
            for label, log_probability in zip(
                labels,
                log_weights,
                strict=True,
            )
            if label == cluster_id
        ]
        finite = [value for value in values if math.isfinite(value)]
        if not finite:
            class_log_probabilities.append(-math.inf)
            continue
        maximum = max(finite)
        class_log_probabilities.append(
            maximum
            + math.log(
                math.fsum(math.exp(value - maximum) for value in values if value != -math.inf)
            )
        )
    if any(value == -math.inf for value in class_log_probabilities):
        return math.inf
    return -math.fsum(class_log_probabilities) / len(class_log_probabilities) / math.log(base)


def semantic_entropy_from_clusters(
    clusters: Sequence[Sequence[Any]],
    *,
    normalize: bool = False,
    base: float = math.e,
) -> float:
    """Convenience wrapper for explicit groups of equivalent samples."""

    labels = [cluster_index for cluster_index, cluster in enumerate(clusters) for _ in cluster]
    if not labels:
        raise ValueError("At least one semantic sample is required")
    return semantic_entropy(labels, normalize=normalize, base=base)


def entropy_to_confidence(
    entropy: float,
    *,
    mapping: str = "exp_negative_entropy",
    num_classes: int | None = None,
    max_entropy: float | None = None,
) -> float:
    """Map semantic entropy to a correctness confidence.

    ``exp_negative_entropy`` is this reproduction's explicit default and
    equals inverse semantic perplexity.  The paper names semantic entropy but
    does not specify this conversion.  ``one_minus_normalized_entropy`` is
    provided for ablations and requires a normalization bound.
    """

    value = float(entropy)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("entropy must be finite and non-negative")
    normalized_mapping = mapping.casefold().replace("-", "_")
    if normalized_mapping in {
        "exp_negative_entropy",
        "exp_neg_entropy",
        "inverse_perplexity",
    }:
        return math.exp(-value)
    if normalized_mapping in {
        "one_minus_normalized_entropy",
        "linear_normalized",
    }:
        bound = max_entropy
        if bound is None:
            if num_classes is None or num_classes < 1:
                raise ValueError("num_classes or max_entropy is required for normalized mapping")
            if num_classes == 1:
                return 1.0
            bound = math.log(num_classes)
        if not math.isfinite(bound) or bound <= 0.0:
            raise ValueError("max_entropy must be finite and positive")
        return min(1.0, max(0.0, 1.0 - value / bound))
    raise ValueError(f"Unknown semantic confidence mapping: {mapping!r}")


def semantic_confidence(
    cluster_ids: Sequence[Hashable],
    log_probabilities: Sequence[float] | None = None,
    *,
    mapping: str = "exp_negative_entropy",
) -> float:
    """Estimate confidence directly from semantic class assignments."""

    probabilities = semantic_class_probabilities(cluster_ids, log_probabilities)
    normalized_mapping = mapping.casefold().replace("-", "_")
    if normalized_mapping in {
        "maximum_class_probability",
        "max_class_probability",
        "sample_consistency",
    }:
        return max(probabilities.values())
    entropy = entropy_from_probabilities(probabilities.values())
    return entropy_to_confidence(
        entropy,
        mapping=mapping,
        num_classes=len(probabilities),
    )


cluster_generations = semantic_cluster_ids
cluster_semantic_equivalence = semantic_clusters
rao_semantic_entropy = semantic_entropy


__all__ = [
    "SemanticScorer",
    "cluster_generations",
    "cluster_semantic_equivalence",
    "entropy_from_probabilities",
    "entropy_to_confidence",
    "kuhn_eq4_semantic_entropy",
    "normalize_semantic_text",
    "rao_semantic_entropy",
    "semantic_class_probabilities",
    "semantic_cluster_ids",
    "semantic_clusters",
    "semantic_confidence",
    "semantic_entropy",
    "semantic_entropy_from_clusters",
]
