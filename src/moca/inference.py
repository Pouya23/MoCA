from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .calibration import load_calibrator
from .clustering import (
    cluster_fingerprint,
    load_cluster_artifacts,
    validate_router_compatibility,
)
from .config import ExperimentConfig
from .embeddings import BaseModelPromptEmbedder
from .modeling import (
    load_all_experts,
    load_tokenizer,
    model_device,
    require_torch,
)
from .utils import files_fingerprint
from .artifacts import read_jsonl


@dataclass
class GeneratedResponse:
    prompt: str
    response: str
    expert_id: int
    natural_expert_id: int
    route_was_forced: bool
    routing_distance: float
    second_routing_distance: float
    routing_margin: float
    first_token_entropy: float
    normalized_first_token_entropy: float
    first_token_confidence: float
    non_special_first_token_entropy: float
    normalized_non_special_first_token_entropy: float
    non_special_first_token_confidence: float
    sequence_log_probability: float | None
    generated_token_count: int
    calibrated_confidence: float | None
    abstained: bool
    routing_distance_z: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SampledResponse:
    text: str
    sequence_log_probability: float
    generated_token_count: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["token_count"] = self.generated_token_count
        return payload


def _generation_kwargs(
    config: ExperimentConfig,
    *,
    force_sampling: bool = False,
):
    do_sample = config.generation.do_sample or force_sampling
    kwargs: dict[str, Any] = {
        "do_sample": do_sample,
        "max_new_tokens": config.generation.max_new_tokens,
        "repetition_penalty": config.generation.repetition_penalty,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    if do_sample:
        kwargs.update(
            {
                "temperature": config.generation.temperature,
                "top_p": config.generation.top_p,
                "top_k": config.generation.top_k,
            }
        )
    return kwargs


def _lengths_from_generated_tokens(token_ids, tokenizer) -> list[int]:
    lengths: list[int] = []
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    for row in token_ids.tolist():
        length = 0
        for token in row:
            if eos_id is not None and token == eos_id:
                length += 1
                break
            if pad_id is not None and token == pad_id:
                break
            length += 1
        lengths.append(length)
    return lengths


def _transition_log_probabilities(model, outputs, lengths: Sequence[int]) -> list[float]:
    if not outputs.scores:
        return [0.0 for _ in lengths]
    transitions = model.compute_transition_scores(
        outputs.sequences,
        outputs.scores,
        normalize_logits=True,
    )
    values = []
    for row, length in enumerate(lengths):
        values.append(float(transitions[row, :length].float().sum().item()))
    return values


def _cluster_distance_statistics(run_dir, num_clusters):
    grouped: list[list[float]] = [
        [] for _ in range(num_clusters)
    ]

    for row in read_jsonl(run_dir / "clusters" / "train.jsonl"):
        cluster_id = int(row["cluster_id"])
        metadata = row.get("metadata", {})

        distance = float(
            metadata["routing_distance_to_assigned_centroid"]
        )

        grouped[cluster_id].append(distance)

    statistics = []

    for values in grouped:
        array = np.asarray(values, dtype=np.float64)

        if len(array) == 0:
            statistics.append((0.0, 1.0))
            continue

        mean = float(array.mean())
        std = float(array.std())

        if std < 1e-8:
            std = 1.0

        statistics.append((mean, std))

    return statistics


class RoutedMoCA:
    """Single-adapter hard-routed MoCA inference.

    Semantic-entropy sampling is intentionally exposed through a separate method;
    regular deployment generation calls `generate` once per prompt.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.clusters = load_cluster_artifacts(config.run_dir / "clusters")
        validate_router_compatibility(
            self.clusters,
            config.embedding,
            config.model.name_or_path,
            config.model.revision,
            config.tokenization,
        )
        expected_experts = (
            1 if config.objective.method == "vanilla_ft" else self.clusters.num_clusters
        )
        self.tokenizer = load_tokenizer(config, padding_side="left")
        self.model = load_all_experts(
            config,
            expected_experts,
            expected_cluster_fingerprint=cluster_fingerprint(self.clusters),
            expected_data_fingerprint=files_fingerprint(
                [
                    config.run_dir / "clusters" / "train.jsonl",
                    config.run_dir / "clusters" / "validation.jsonl",
                ]
            ),
        )
        self.embedder = BaseModelPromptEmbedder(
            self.model,
            self.tokenizer,
            config.embedding,
            config.tokenization,
        )
        self.calibrator = None
        if config.evaluation.confidence_source == "moca_1p":
            calibrator_path = (
                config.run_dir / "evaluation" / "moca_1p_calibrator.json"
                if config.evaluation.calibrator_path is None
                else config.evaluation.calibrator_path
            )
            from pathlib import Path

            resolved_calibrator_path = Path(calibrator_path)
            if resolved_calibrator_path.is_file():
                self.calibrator = load_calibrator(resolved_calibrator_path)
                if self.calibrator.feature_names != tuple(
                    config.evaluation.calibration_features
                ):
                    raise ValueError(
                        "Saved calibrator features do not match evaluation.calibration_features"
                    )
        self.cluster_distance_statistics = (
            _cluster_distance_statistics(
                config.run_dir,
                self.clusters.num_clusters,
            )
        )

    def route(self, prompts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        assignments, nearest, _, _ = self.route_with_geometry(prompts)
        return assignments, nearest

    def route_with_geometry(
        self,
        prompts: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.config.objective.method == "vanilla_ft":
            return (
                np.zeros(len(prompts), dtype=np.int64),
                np.zeros(len(prompts), dtype=np.float32),
                np.zeros(len(prompts), dtype=np.float32),
                np.zeros(len(prompts), dtype=np.float32),
            )
        embeddings = self.embedder.encode(prompts)
        return self.clusters.route_with_geometry(embeddings)

    def _encode_prompts(self, prompts: Sequence[str]) -> dict[str, Any]:
        original_side = self.tokenizer.truncation_side
        self.tokenizer.truncation_side = self.config.tokenization.prompt_truncation_side
        try:
            encoded = self.tokenizer(
                list(prompts),
                padding=True,
                truncation=True,
                max_length=self.config.tokenization.max_prompt_tokens,
                add_special_tokens=(self.config.tokenization.add_special_tokens_to_prompt),
                pad_to_multiple_of=self.config.tokenization.pad_to_multiple_of,
                return_tensors="pt",
            )
        finally:
            self.tokenizer.truncation_side = original_side
        device = model_device(self.model)
        return {key: value.to(device) for key, value in encoded.items()}

    def _generate_group(
        self,
        prompts: Sequence[str],
        expert_id: int,
    ) -> list[GeneratedResponse]:
        torch = require_torch()
        self.model.set_adapter(f"expert_{expert_id}")
        encoded = self._encode_prompts(prompts)
        with torch.inference_mode():
            outputs = self.model.generate(
                **encoded,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **_generation_kwargs(self.config),
            )
            if not outputs.scores:
                raise RuntimeError("Transformers generation did not return first-step scores")
            # With the paper-default neutral decoding settings, the first
            # generation score is exactly p_theta(y_1 | x). Reusing it avoids
            # an additional adapter forward pass at deployment.
            next_logits = outputs.scores[0].float()
            log_p = next_logits.log_softmax(dim=-1)
            p = log_p.exp()
            entropy = -torch.where(
                p > 0,
                p * log_p,
                torch.zeros_like(p),
            ).sum(dim=-1)
            normalized_entropy = entropy / math.log(next_logits.shape[-1])
            allowed = torch.ones(next_logits.shape[-1], dtype=torch.bool, device=next_logits.device)
            for token_id in getattr(self.tokenizer, "all_special_ids", ()):
                if 0 <= int(token_id) < next_logits.shape[-1]:
                    allowed[int(token_id)] = False
            if not allowed.any():
                raise ValueError("No non-special tokens remain for first-token entropy")
            non_special_log_p = next_logits[:, allowed].log_softmax(dim=-1)
            non_special_p = non_special_log_p.exp()
            non_special_entropy = -torch.where(
                non_special_p > 0,
                non_special_p * non_special_log_p,
                torch.zeros_like(non_special_p),
            ).sum(dim=-1)
            normalized_non_special_entropy = non_special_entropy / math.log(
                int(allowed.sum().item())
            )
        input_length = encoded["input_ids"].shape[1]
        generated_ids = outputs.sequences[:, input_length:]
        lengths = _lengths_from_generated_tokens(generated_ids, self.tokenizer)
        log_probabilities = _transition_log_probabilities(self.model, outputs, lengths)
        texts = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return [
            GeneratedResponse(
                prompt=prompt,
                response=text.strip(),
                expert_id=expert_id,
                natural_expert_id=expert_id,
                route_was_forced=False,
                routing_distance=0.0,
                second_routing_distance=0.0,
                routing_margin=0.0,
                first_token_entropy=float(entropy[index].item()),
                normalized_first_token_entropy=float(normalized_entropy[index].item()),
                first_token_confidence=float(
                    1.0 - normalized_entropy[index].clamp(0.0, 1.0).item()
                ),
                non_special_first_token_entropy=float(non_special_entropy[index].item()),
                normalized_non_special_first_token_entropy=float(
                    normalized_non_special_entropy[index].item()
                ),
                non_special_first_token_confidence=float(
                    1.0 - normalized_non_special_entropy[index].clamp(0.0, 1.0).item()
                ),
                sequence_log_probability=log_probabilities[index],
                generated_token_count=lengths[index],
                calibrated_confidence=None,
                abstained=False,
            )
            for index, (prompt, text) in enumerate(zip(prompts, texts))
        ]

    def generate(
        self,
        prompts: Sequence[str],
        *,
        force_wrong_route: bool = False,
    ) -> list[GeneratedResponse]:
        if not prompts:
            return []
        natural_assignments, distances, second_distances, margins = self.route_with_geometry(
            prompts
        )
        if force_wrong_route:
            if self.clusters.num_clusters < 2 or self.config.objective.method == "vanilla_ft":
                raise ValueError("Forced wrong routing requires at least two routed experts")
            assignments = (natural_assignments + 1) % self.clusters.num_clusters
        else:
            assignments = natural_assignments
        results: list[GeneratedResponse | None] = [None] * len(prompts)
        for expert_id in sorted(set(assignments.tolist())):
            indices = [
                index
                for index, assignment in enumerate(assignments)
                if int(assignment) == expert_id
            ]
            for start in range(0, len(indices), self.config.generation.batch_size):
                batch_indices = indices[start : start + self.config.generation.batch_size]
                generated = self._generate_group(
                    [prompts[index] for index in batch_indices],
                    expert_id,
                )
                for output, original_index in zip(generated, batch_indices):
                    output.routing_distance = float(distances[original_index])
                    natural_expert = int(natural_assignments[original_index])
                    distance_mean, distance_std = (
                        self.cluster_distance_statistics[natural_expert]
                    )
                    output.routing_distance_z = (
                        float(distances[original_index]) - distance_mean
                    ) / distance_std
                    output.second_routing_distance = float(second_distances[original_index])
                    output.routing_margin = float(margins[original_index])
                    output.natural_expert_id = int(natural_assignments[original_index])
                    output.route_was_forced = force_wrong_route
                    results[original_index] = output
        if any(result is None for result in results):
            raise RuntimeError("Generation did not produce every requested output")
        completed = [result for result in results if result is not None]
        if self.calibrator is not None:
            confidences = self.calibrator.predict([result.to_dict() for result in completed])
            threshold = self.config.evaluation.abstain_threshold
            for result, confidence in zip(completed, confidences, strict=True):
                result.calibrated_confidence = confidence
                result.abstained = threshold is not None and confidence < threshold
        return completed

    def sample_for_semantic_entropy(
        self,
        prompt: str,
        num_samples: int | None = None,
    ) -> tuple[int, float, list[SampledResponse]]:
        torch = require_torch()
        sample_count = num_samples or self.config.evaluation.semantic_samples
        if sample_count < 2:
            raise ValueError("Semantic entropy requires at least two samples")
        assignments, distances = self.route([prompt])
        expert_id = int(assignments[0])
        self.model.set_adapter(f"expert_{expert_id}")
        encoded = self._encode_prompts([prompt])
        with torch.inference_mode():
            outputs = self.model.generate(
                **encoded,
                num_return_sequences=sample_count,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **_generation_kwargs(self.config, force_sampling=True),
            )
        input_length = encoded["input_ids"].shape[1]
        generated_ids = outputs.sequences[:, input_length:]
        lengths = _lengths_from_generated_tokens(generated_ids, self.tokenizer)
        log_probabilities = _transition_log_probabilities(self.model, outputs, lengths)
        texts = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        samples = [
            SampledResponse(
                text=text.strip(),
                sequence_log_probability=log_probability,
                generated_token_count=length,
            )
            for text, log_probability, length in zip(texts, log_probabilities, lengths)
        ]
        return expert_id, float(distances[0]), samples
