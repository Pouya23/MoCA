from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name_or_path: str = "Qwen/Qwen2.5-7B"
    revision: str | None = None
    tokenizer_revision: str | None = None
    trust_remote_code: bool = False
    torch_dtype: str = "bfloat16"
    attn_implementation: str | None = None
    use_cache: bool = True


@dataclass(frozen=True)
class DataConfig:
    name: str = "coqa"
    dataset_id: str = "stanfordnlp/coqa"
    dataset_config: str | None = None
    dataset_revision: str | None = None
    local_jsonl: str | None = None
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 2026
    split_unit: str = "group"
    use_official_splits: bool = False
    include_dialogue_history: bool = False
    max_examples: int | None = None


@dataclass(frozen=True)
class TokenizationConfig:
    max_prompt_tokens: int = 1536
    max_response_tokens: int = 256
    prompt_truncation_side: str = "left"
    add_special_tokens_to_prompt: bool = True
    response_prefix: str = " "
    append_eos: bool = True
    include_eos_in_nll: bool = True
    pad_to_multiple_of: int = 8


@dataclass(frozen=True)
class EmbeddingConfig:
    layer: int = -1
    pooling: str = "last"
    normalize: bool = False
    max_length: int = 1536
    batch_size: int = 8
    dtype: str = "float32"


@dataclass(frozen=True)
class ClusteringConfig:
    strategy: str = "kmeans"
    num_clusters: int | None = None
    min_clusters: int = 2
    max_clusters: int = 10
    init: str = "k-means++"
    n_init: int = 10
    max_iter: int = 300
    tolerance: float = 1e-4
    random_seed: int = 2026
    silhouette_sample_size: int | None = 10000
    fit_scope: str = "train"


@dataclass(frozen=True)
class LoraConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    bias: str = "none"
    init_lora_weights: bool | str = True


@dataclass(frozen=True)
class ObjectiveConfig:
    method: str = "moca"
    lambda_kl: float = 0.1
    num_ood_tokens: int = 1
    kl_direction: str = "model_to_uniform"
    uniform_scope: str = "full_vocabulary"
    sequence_nll_reduction: str = "sequence_sum_then_batch_mean"
    later_token_conditioning: str = "gold"


@dataclass(frozen=True)
class SamplingConfig:
    in_batch_size: int = 2
    out_batch_size: int = 2
    pseudo_ood_distribution: str = "empirical_complement"
    replacement: bool = True
    dataloader_workers: int = 0


@dataclass(frozen=True)
class OptimizationConfig:
    optimizer: str = "adamw"
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    scheduler: str = "cosine"
    warmup_ratio: float = 0.0
    max_epochs: int = 10
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 2
    validation_batches: int | None = None
    save_every_epoch: bool = False


@dataclass(frozen=True)
class GenerationConfig:
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    max_new_tokens: int = 128
    repetition_penalty: float = 1.0
    batch_size: int = 4


@dataclass(frozen=True)
class EvaluationConfig:
    confidence_source: str = "moca_1p"
    semantic_samples: int = 10
    semantic_estimator: str = "kuhn_eq4"
    semantic_confidence_mapping: str = "exp_negative_entropy"
    nli_model: str = "microsoft/deberta-large-mnli"
    nli_revision: str | None = None
    entailment_threshold: float = 0.5
    equivalence_rule: str = "bidirectional_entailment"
    semantic_clustering_algorithm: str = "representative_greedy"
    nli_decision_rule: str = "argmax_entailment"
    include_prompt_in_nli: bool = True
    nli_batch_size: int = 16
    nli_max_length: int = 512
    nli_truncation_side: str = "left"
    length_normalize_sequence_probability: bool = True
    max_eval_examples: int | None = None
    cache_generations: bool = True
    calibration_features: tuple[str, ...] = (
        "non_special_first_token_confidence",
        "negative_log1p_routing_distance",
        "routing_margin",
    )
    calibration_l2: float = 1e-3
    calibration_max_iter: int = 200
    calibration_tolerance: float = 1e-8
    calibrator_path: str | None = None
    ece_bins: int = 15
    abstain_threshold: float | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 2026
    device: str = "cuda"
    mixed_precision: str = "bf16"
    deterministic: bool = False
    log_every_steps: int = 10
    overwrite_existing: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str = "coqa_qwen_moca"
    output_root: str = "runs"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tokenization: TokenizationConfig = field(default_factory=TokenizationConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    ablations: tuple[str, ...] = ()

    @property
    def run_dir(self) -> Path:
        return Path(self.output_root) / self.experiment_name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def training_fingerprint(self) -> str:
        """Hash every setting that can change trained adapter parameters.

        Generation and evaluation settings are intentionally excluded so an
        existing adapter can be evaluated with different, explicitly selected
        decoding or scoring settings. Operational runtime fields that do not
        affect optimization are excluded for the same reason.
        """

        model_values = asdict(self.model)
        # Training always disables KV caching, irrespective of deployment use.
        model_values.pop("use_cache")
        payload = {
            "model": model_values,
            "data": asdict(self.data),
            "tokenization": asdict(self.tokenization),
            "embedding": asdict(self.embedding),
            "clustering": asdict(self.clustering),
            "lora": asdict(self.lora),
            "objective": asdict(self.objective),
            "sampling": asdict(self.sampling),
            "optimization": asdict(self.optimization),
            "runtime": {
                "seed": self.runtime.seed,
                "mixed_precision": self.runtime.mixed_precision,
                "deterministic": self.runtime.deterministic,
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def validate(self) -> None:
        ratios = self.data.train_ratio + self.data.validation_ratio + self.data.test_ratio
        if any(
            ratio < 0
            for ratio in (
                self.data.train_ratio,
                self.data.validation_ratio,
                self.data.test_ratio,
            )
        ):
            raise ValueError("Data split ratios must be non-negative")
        if abs(ratios - 1.0) > 1e-8:
            raise ValueError(f"Data split ratios must sum to 1, got {ratios}")
        if self.data.split_unit not in {"example", "group"}:
            raise ValueError("split_unit must be example or group")
        if self.data.max_examples is not None and self.data.max_examples <= 0:
            raise ValueError("max_examples must be positive when set")
        if self.model.torch_dtype.lower() not in {
            "float32",
            "fp32",
            "float16",
            "fp16",
            "bfloat16",
            "bf16",
        }:
            raise ValueError("Unsupported model torch_dtype")
        if self.objective.method not in {"moca", "k_ft", "vanilla_ft"}:
            raise ValueError(f"Unknown method: {self.objective.method}")
        if self.objective.lambda_kl < 0:
            raise ValueError("lambda_kl must be non-negative")
        if self.objective.method != "moca" and self.objective.lambda_kl != 0:
            raise ValueError(f"{self.objective.method} requires lambda_kl=0")
        if self.objective.num_ood_tokens < 1:
            raise ValueError("num_ood_tokens must be at least one")
        if self.objective.num_ood_tokens > 1 and self.objective.later_token_conditioning != "gold":
            raise ValueError("Only gold later-token conditioning is implemented")
        if self.objective.kl_direction not in {
            "model_to_uniform",
            "uniform_to_model",
        }:
            raise ValueError("Unsupported KL direction")
        if self.objective.uniform_scope not in {
            "full_vocabulary",
            "non_special_tokens",
        }:
            raise ValueError("Unsupported uniform_scope")
        if (
            self.clustering.num_clusters is not None
            and self.clustering.num_clusters < 2
            and self.objective.method != "vanilla_ft"
        ):
            raise ValueError("Clustered methods require at least two clusters")
        if self.clustering.min_clusters < 2:
            raise ValueError("min_clusters must be at least two")
        if self.clustering.max_clusters < self.clustering.min_clusters:
            raise ValueError("max_clusters must not be less than min_clusters")
        if self.clustering.n_init <= 0 or self.clustering.max_iter <= 0:
            raise ValueError("k-means n_init and max_iter must be positive")
        if self.clustering.tolerance < 0:
            raise ValueError("k-means tolerance must be non-negative")
        if self.clustering.strategy not in {"kmeans", "random_balanced"}:
            raise ValueError("Unsupported clustering strategy")
        if self.clustering.strategy == "random_balanced" and self.clustering.num_clusters is None:
            raise ValueError("random_balanced clustering requires a fixed num_clusters")
        if self.clustering.strategy == "random_balanced" and self.objective.method == "vanilla_ft":
            raise ValueError("random_partition is not meaningful for vanilla_ft")
        if (
            self.clustering.silhouette_sample_size is not None
            and self.clustering.silhouette_sample_size < 2
        ):
            raise ValueError("silhouette_sample_size must be at least two")
        if self.lora.rank <= 0 or self.lora.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0 <= self.lora.dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if self.sampling.in_batch_size <= 0 or self.sampling.out_batch_size <= 0:
            raise ValueError("Batch sizes must be positive")
        if self.sampling.dataloader_workers < 0:
            raise ValueError("dataloader_workers must be non-negative")
        if self.sampling.pseudo_ood_distribution not in {
            "empirical_complement",
            "uniform_cluster",
        }:
            raise ValueError("Unsupported pseudo-OOD sampling distribution")
        if self.optimization.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.optimization.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.optimization.optimizer.lower() != "adamw":
            raise ValueError("Only AdamW is implemented")
        if self.optimization.scheduler.lower() != "cosine":
            raise ValueError("Only the cosine scheduler is implemented")
        if self.optimization.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.optimization.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.optimization.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")
        if not 0 <= self.optimization.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.objective.sequence_nll_reduction not in {
            "sequence_sum_then_batch_mean",
            "token_mean",
        }:
            raise ValueError("Unsupported sequence NLL reduction")
        if self.embedding.pooling not in {"mean", "last", "first"}:
            raise ValueError(f"Unsupported embedding pooling: {self.embedding.pooling}")
        if self.embedding.dtype not in {"float32", "float16"}:
            raise ValueError("Embedding dtype must be float32 or float16")
        if self.embedding.max_length <= 0 or self.embedding.batch_size <= 0:
            raise ValueError("Embedding max_length and batch_size must be positive")
        if self.tokenization.max_prompt_tokens <= 0:
            raise ValueError("max_prompt_tokens must be positive")
        if self.tokenization.max_response_tokens <= 0:
            raise ValueError("max_response_tokens must be positive")
        if self.tokenization.prompt_truncation_side not in {"left", "right"}:
            raise ValueError("prompt_truncation_side must be left or right")
        if self.tokenization.pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of must be positive")
        if self.clustering.fit_scope not in {"train", "all"}:
            raise ValueError("fit_scope must be train or all")
        if self.generation.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.generation.batch_size <= 0:
            raise ValueError("generation batch_size must be positive")
        if self.generation.do_sample and self.generation.temperature <= 0:
            raise ValueError("Sampling temperature must be positive")
        if not 0 < self.generation.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.generation.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if self.evaluation.semantic_samples < 2:
            raise ValueError("semantic_samples must be at least two")
        if self.evaluation.max_eval_examples is not None and self.evaluation.max_eval_examples <= 0:
            raise ValueError("max_eval_examples must be positive when set")
        if self.evaluation.nli_batch_size <= 0 or self.evaluation.nli_max_length <= 0:
            raise ValueError("NLI batch size and max length must be positive")
        if self.evaluation.nli_truncation_side not in {"left", "right"}:
            raise ValueError("NLI truncation side must be left or right")
        if not 0 <= self.evaluation.entailment_threshold <= 1:
            raise ValueError("entailment_threshold must be in [0, 1]")
        if self.evaluation.confidence_source not in {
            "semantic_entropy",
            "first_token_entropy",
            "sequence_probability",
            "moca_1p",
            "provided",
        }:
            raise ValueError("Unsupported confidence_source")
        if self.evaluation.semantic_estimator not in {
            "kuhn_eq4",
            "frequency",
            "model_probability",
        }:
            raise ValueError("Unsupported semantic_estimator")
        if self.evaluation.semantic_confidence_mapping not in {
            "exp_negative_entropy",
            "maximum_class_probability",
            "one_minus_normalized_entropy",
        }:
            raise ValueError("Unsupported semantic_confidence_mapping")
        if self.evaluation.equivalence_rule != "bidirectional_entailment":
            raise ValueError("Only bidirectional_entailment equivalence is implemented")
        if self.evaluation.semantic_clustering_algorithm not in {
            "representative_greedy",
            "transitive_closure",
        }:
            raise ValueError("Unsupported semantic_clustering_algorithm")
        if self.evaluation.nli_decision_rule not in {
            "argmax_entailment",
            "probability_threshold",
        }:
            raise ValueError("Unsupported nli_decision_rule")
        allowed_calibration_features = {
            "first_token_confidence",
            "non_special_first_token_confidence",
            "normalized_first_token_entropy",
            "normalized_non_special_first_token_entropy",
            "negative_log1p_routing_distance",
            "routing_margin",
            "length_normalized_sequence_log_probability",
            "relative_routing_margin",
        }
        if not self.evaluation.calibration_features:
            raise ValueError("calibration_features must not be empty")
        if not set(self.evaluation.calibration_features) <= allowed_calibration_features:
            raise ValueError("Unsupported calibration feature")
        if self.evaluation.calibration_l2 < 0:
            raise ValueError("calibration_l2 must be non-negative")
        if self.evaluation.calibration_max_iter <= 0:
            raise ValueError("calibration_max_iter must be positive")
        if self.evaluation.calibration_tolerance <= 0:
            raise ValueError("calibration_tolerance must be positive")
        if self.evaluation.ece_bins <= 0:
            raise ValueError("ece_bins must be positive")
        if (
            self.evaluation.abstain_threshold is not None
            and not 0.0 <= self.evaluation.abstain_threshold <= 1.0
        ):
            raise ValueError("abstain_threshold must be in [0, 1]")
        if self.runtime.log_every_steps <= 0:
            raise ValueError("log_every_steps must be positive")
        if self.runtime.mixed_precision not in {
            "bf16",
            "bfloat16",
            "fp16",
            "float16",
            "fp32",
            "float32",
            "none",
            "no",
        }:
            raise ValueError("Unsupported mixed_precision")
        if not self.runtime.device:
            raise ValueError("runtime.device must not be empty")


SECTION_TYPES = {
    "model": ModelConfig,
    "data": DataConfig,
    "tokenization": TokenizationConfig,
    "embedding": EmbeddingConfig,
    "clustering": ClusteringConfig,
    "lora": LoraConfig,
    "objective": ObjectiveConfig,
    "sampling": SamplingConfig,
    "optimization": OptimizationConfig,
    "generation": GenerationConfig,
    "evaluation": EvaluationConfig,
    "runtime": RuntimeConfig,
}


def _strict_construct(cls: type[Any], values: Mapping[str, Any]) -> Any:
    allowed = {item.name for item in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    converted = dict(values)
    if cls is LoraConfig and "target_modules" in converted:
        converted["target_modules"] = tuple(converted["target_modules"])
    if cls is EvaluationConfig and "calibration_features" in converted:
        converted["calibration_features"] = tuple(converted["calibration_features"])
    return cls(**converted)


def _deep_merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _set_dotted(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = mapping
    for part in parts[:-1]:
        next_value = cursor.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise TypeError(f"Cannot set {dotted_key}: {part} is not a mapping")
        cursor = next_value
    cursor[parts[-1]] = value


def parse_overrides(overrides: list[str] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must be KEY=VALUE: {override}")
        key, raw_value = override.split("=", 1)
        _set_dotted(result, key, yaml.safe_load(raw_value))
    return result


def experiment_config_from_dict(values: Mapping[str, Any]) -> ExperimentConfig:
    root_allowed = {
        "experiment_name",
        "output_root",
        "ablations",
        *SECTION_TYPES.keys(),
    }
    unknown = set(values) - root_allowed
    if unknown:
        raise ValueError(f"Unknown ExperimentConfig fields: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name in ("experiment_name", "output_root"):
        if name in values:
            kwargs[name] = values[name]
    if "ablations" in values:
        kwargs["ablations"] = tuple(values["ablations"])
    for name, cls in SECTION_TYPES.items():
        if name in values:
            kwargs[name] = _strict_construct(cls, values[name])
    config = ExperimentConfig(**kwargs)
    config.validate()
    return config


def load_experiment_config(
    path: str | Path,
    overrides: list[str] | None = None,
    ablations: list[str] | None = None,
) -> ExperimentConfig:
    from .ablations import apply_ablations

    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    saved_fingerprint = loaded.pop("_config_fingerprint", None)
    if saved_fingerprint is not None:
        saved_config = experiment_config_from_dict(
            _deep_merge(ExperimentConfig().to_dict(), loaded)
        )
        if saved_config.fingerprint() != saved_fingerprint:
            raise ValueError(
                f"Resolved config fingerprint mismatch in {path}; "
                "the file may have been edited or corrupted"
            )
    merged = _deep_merge(ExperimentConfig().to_dict(), loaded)
    merged = _deep_merge(merged, parse_overrides(overrides))
    selected = list(merged.get("ablations", []))
    selected.extend(ablations or [])
    selected = list(dict.fromkeys(selected))
    merged = apply_ablations(merged, selected)
    merged["ablations"] = selected
    return experiment_config_from_dict(merged)


def save_resolved_config(config: ExperimentConfig, destination: str | Path) -> None:
    from .artifacts import atomic_write_yaml

    payload = config.to_dict()
    payload["_config_fingerprint"] = config.fingerprint()
    atomic_write_yaml(destination, payload)
