from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

Transform = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class Ablation:
    name: str
    description: str
    touched_paths: frozenset[str]
    transform: Transform


def _set(path: str, value: Any) -> Transform:
    def transform(config: dict[str, Any]) -> None:
        cursor = config
        parts = path.split(".")
        for part in parts[:-1]:
            cursor = cursor[part]
        cursor[parts[-1]] = value

    return transform


def _compose(*transforms: Transform) -> Transform:
    def transform(config: dict[str, Any]) -> None:
        for item in transforms:
            item(config)

    return transform


ABLATIONS: dict[str, Ablation] = {
    "paper": Ablation(
        "paper",
        "Equation (1): clustered experts plus first-token D_KL(p||U).",
        frozenset(),
        lambda config: None,
    ),
    "k_ft": Ablation(
        "k_ft",
        "Clustered/routed experts trained only with in-cluster sequence NLL.",
        frozenset({"objective.method", "objective.lambda_kl"}),
        _compose(
            _set("objective.method", "k_ft"),
            _set("objective.lambda_kl", 0.0),
        ),
    ),
    "vanilla_ft": Ablation(
        "vanilla_ft",
        "One LoRA adapter trained on all examples with sequence NLL.",
        frozenset(
            {
                "objective.method",
                "objective.lambda_kl",
                "clustering.num_clusters",
            }
        ),
        _compose(
            _set("objective.method", "vanilla_ft"),
            _set("objective.lambda_kl", 0.0),
            _set("clustering.num_clusters", 1),
        ),
    ),
    "first_5_tokens": Ablation(
        "first_5_tokens",
        "Regularize the first five response-token distributions (non-paper ablation).",
        frozenset({"objective.num_ood_tokens"}),
        _set("objective.num_ood_tokens", 5),
    ),
    "uniform_to_model_kl": Ablation(
        "uniform_to_model_kl",
        "Use D_KL(U||p), reversing the paper's KL direction.",
        frozenset({"objective.kl_direction"}),
        _set("objective.kl_direction", "uniform_to_model"),
    ),
    "uniform_cluster_ood": Ablation(
        "uniform_cluster_ood",
        "Sample an OOD cluster uniformly before sampling an example.",
        frozenset({"sampling.pseudo_ood_distribution"}),
        _set("sampling.pseudo_ood_distribution", "uniform_cluster"),
    ),
    "normalized_embeddings": Ablation(
        "normalized_embeddings",
        "L2-normalize embeddings before k-means and routing.",
        frozenset({"embedding.normalize"}),
        _set("embedding.normalize", True),
    ),
    "last_token_pooling": Ablation(
        "last_token_pooling",
        "Use the last prompt token as the base-model embedding.",
        frozenset({"embedding.pooling"}),
        _set("embedding.pooling", "last"),
    ),
    "mean_token_pooling": Ablation(
        "mean_token_pooling",
        "Use mean prompt-token pooling as the reviewer-requested comparison.",
        frozenset({"embedding.pooling"}),
        _set("embedding.pooling", "mean"),
    ),
    "non_special_uniform": Ablation(
        "non_special_uniform",
        "Uniformize only over non-special output tokens.",
        frozenset({"objective.uniform_scope"}),
        _set("objective.uniform_scope", "non_special_tokens"),
    ),
    "frequency_semantic_entropy": Ablation(
        "frequency_semantic_entropy",
        "Use sample-frequency class entropy instead of Kuhn et al. Eq. 4.",
        frozenset({"evaluation.semantic_estimator"}),
        _set("evaluation.semantic_estimator", "frequency"),
    ),
    "transitive_semantic_clustering": Ablation(
        "transitive_semantic_clustering",
        "Use pairwise transitive closure instead of Kuhn's greedy classes.",
        frozenset({"evaluation.semantic_clustering_algorithm"}),
        _set("evaluation.semantic_clustering_algorithm", "transitive_closure"),
    ),
    "threshold_nli": Ablation(
        "threshold_nli",
        "Threshold entailment probability instead of using the NLI argmax.",
        frozenset({"evaluation.nli_decision_rule"}),
        _set("evaluation.nli_decision_rule", "probability_threshold"),
    ),
    "semantic_entropy_evaluation": Ablation(
        "semantic_entropy_evaluation",
        "Use ten-sample semantic entropy as an offline diagnostic.",
        frozenset({"evaluation.confidence_source"}),
        _set("evaluation.confidence_source", "semantic_entropy"),
    ),
    "first_token_only_confidence": Ablation(
        "first_token_only_confidence",
        "Use the raw single-pass first-token entropy score without calibration.",
        frozenset({"evaluation.confidence_source"}),
        _set("evaluation.confidence_source", "first_token_entropy"),
    ),
    "provided_confidence": Ablation(
        "provided_confidence",
        "Evaluate a confidence field supplied by an external baseline.",
        frozenset({"evaluation.confidence_source"}),
        _set("evaluation.confidence_source", "provided"),
    ),
    "sequence_probability_confidence": Ablation(
        "sequence_probability_confidence",
        "Use length-normalized sequence probability as a one-pass baseline.",
        frozenset({"evaluation.confidence_source"}),
        _set("evaluation.confidence_source", "sequence_probability"),
    ),
    "random_partition": Ablation(
        "random_partition",
        "Use a deterministic balanced random partition; set clustering.num_clusters explicitly.",
        frozenset({"clustering.strategy"}),
        _set("clustering.strategy", "random_balanced"),
    ),
    "matched_rank_80": Ablation(
        "matched_rank_80",
        "Parameter-matched single-adapter control with LoRA rank 80.",
        frozenset({"lora.rank", "lora.alpha"}),
        _compose(_set("lora.rank", 80), _set("lora.alpha", 160)),
    ),
}


def apply_ablations(config: dict[str, Any], names: list[str]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    touched: dict[str, str] = {}
    for name in names:
        if name not in ABLATIONS:
            raise ValueError(f"Unknown ablation {name!r}; available: {sorted(ABLATIONS)}")
        ablation = ABLATIONS[name]
        for path in ablation.touched_paths:
            if path in touched:
                raise ValueError(f"Ablations {touched[path]!r} and {name!r} both change {path}")
            touched[path] = name
        ablation.transform(result)
    return result


def describe_ablations() -> list[dict[str, str]]:
    return [{"name": item.name, "description": item.description} for item in ABLATIONS.values()]
