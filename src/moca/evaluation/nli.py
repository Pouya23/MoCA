"""Lazy NLI-based semantic equivalence scoring."""

from __future__ import annotations

import os
from ..utils import LOGGER
import math
import threading
from collections.abc import Callable, Sequence
from typing import Any

EntailmentPredictor = Callable[[Sequence[tuple[str, str]]], Sequence[float]]


class BidirectionalNLIScorer:
    """Score semantic equivalence with entailment in both directions.

    ``transformers`` and ``torch`` are imported, and model weights are loaded,
    only on the first call that needs inference.  A lightweight ``predictor``
    can be injected for tests or for integration with an existing inference
    service.  The predictor receives ``(premise, hypothesis)`` pairs and must
    return entailment probabilities in the same order.
    """

    def __init__(
        self,
        model_name_or_path: str = "microsoft/deberta-large-mnli",
        *,
        revision: str | None = None,
        threshold: float = 0.5,
        decision_rule: str = "argmax_entailment",
        include_prompt: bool = True,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 512,
        truncation_side: str = "left",
        trust_remote_code: bool = False,
        predictor: EntailmentPredictor | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if truncation_side not in {"left", "right"}:
            raise ValueError("truncation_side must be left or right")
        if decision_rule not in {
            "argmax_entailment",
            "probability_threshold",
        }:
            raise ValueError("Unsupported NLI decision_rule")

        self.model_name_or_path = model_name_or_path
        self.revision = revision
        self.threshold = float(threshold)
        self.decision_rule = decision_rule
        self.include_prompt = include_prompt
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.truncation_side = truncation_side
        self.trust_remote_code = trust_remote_code
        self._predictor = predictor
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._entailment_index: int | None = None
        self._load_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """Whether a local transformers model has been loaded."""

        return self._model is not None

    def _load(self) -> None:
        if self._predictor is not None or self._model is not None:
            return

        with self._load_lock:
            if self._predictor is not None or self._model is not None:
                return

            try:
                import torch
                from transformers import (
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )
            except (ImportError, ModuleNotFoundError) as error:
                raise RuntimeError(
                    "NLI scoring requires torch and transformers; "
                    "install the project dependencies or inject an "
                    "entailment predictor"
                ) from error
    
            load_kwargs: dict[str, Any] = {
                "revision": self.revision,
                "trust_remote_code": self.trust_remote_code,
            }

            load_kwargs = {
                key: value
                for key, value in load_kwargs.items()
                if value is not None
            }

            LOGGER.info(
                "Loading NLI tokenizer: model=%s revision=%s",
                self.model_name_or_path,
                self.revision or "main",
            )

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                **load_kwargs,
            )
            tokenizer.truncation_side = self.truncation_side

            target_device = self.device
            if target_device is None:
                target_device = (
                    "cuda"
                    if torch.cuda.is_available()
                    else "cpu"
                )

            LOGGER.info(
                "Loading NLI model: model=%s device=%s",
                self.model_name_or_path,
                target_device,
            )

            # microsoft/deberta-large-mnli currently ships a
            # pytorch_model.bin but no model.safetensors.
            #
            # Recent Transformers versions otherwise launch a background
            # safetensors-conversion request. A failure of that optional
            # conversion service produces a noisy Thread-auto_conversion
            # traceback even though the PyTorch checkpoint loads correctly.
            previous_conversion_setting = os.environ.get(
                "DISABLE_SAFETENSORS_CONVERSION"
            )
            os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"

            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name_or_path,
                    low_cpu_mem_usage=True,
                    **load_kwargs,
                )
            finally:
                if previous_conversion_setting is None:
                    os.environ.pop(
                        "DISABLE_SAFETENSORS_CONVERSION",
                        None,
                    )
                else:
                    os.environ[
                        "DISABLE_SAFETENSORS_CONVERSION"
                    ] = previous_conversion_setting

            model.to(target_device)
            model.eval()

            entailment_index = _find_entailment_index(
                model.config
            )

            LOGGER.info(
                "Loaded NLI model successfully: "
                "class=%s device=%s entailment_index=%d labels=%s",
                model.__class__.__name__,
                target_device,
                entailment_index,
                getattr(model.config, "id2label", None),
            )

            self.device = target_device
            self._torch = torch
            self._tokenizer = tokenizer
            self._model = model
            self._entailment_index = entailment_index

    def _statement(self, answer: str, prompt: str | None) -> str:
        answer = str(answer).strip()
        if not self.include_prompt or not prompt:
            return answer
        prompt_text = str(prompt).rstrip()
        # Dataset prompts already end in "Answer:" or "Summary:". Appending
        # the candidate completes that statement without inventing a second
        # answer marker. Generic prompts get an explicit answer marker.
        if prompt_text.endswith(("Answer:", "Summary:")):
            return f"{prompt_text} {answer}"
        return f"{prompt_text}\n\nAnswer: {answer}"

    def entailment_probabilities(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> list[float]:
        """Return directional entailment probabilities for premise/hypothesis pairs."""

        materialized = [(str(left), str(right)) for left, right in pairs]
        if not materialized:
            return []

        if self._predictor is not None:
            probabilities = [float(value) for value in self._predictor(materialized)]
            _validate_probabilities(probabilities, len(materialized))
            return probabilities

        self._load()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None
        assert self._entailment_index is not None

        probabilities: list[float] = []
        for start in range(0, len(materialized), self.batch_size):
            batch = materialized[start : start + self.batch_size]
            encoded = self._tokenizer(
                [pair[0] for pair in batch],
                [pair[1] for pair in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self._torch.inference_mode():
                logits = self._model(**encoded).logits
                batch_probabilities = self._torch.softmax(logits, dim=-1)[:, self._entailment_index]
            probabilities.extend(
                float(value) for value in batch_probabilities.detach().cpu().tolist()
            )
        _validate_probabilities(probabilities, len(materialized))
        return probabilities

    def equivalence_probabilities(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        prompts: Sequence[str | None] | None = None,
    ) -> list[float]:
        """Return ``min(p(a entails b), p(b entails a))`` for each pair."""

        materialized = [(str(left), str(right)) for left, right in pairs]
        if prompts is None:
            prompt_values: list[str | None] = [None] * len(materialized)
        else:
            prompt_values = list(prompts)
            if len(prompt_values) != len(materialized):
                raise ValueError("prompts and pairs must have equal length")

        directional_pairs: list[tuple[str, str]] = []
        for (left, right), prompt in zip(materialized, prompt_values, strict=True):
            left_statement = self._statement(left, prompt)
            right_statement = self._statement(right, prompt)
            directional_pairs.append((left_statement, right_statement))
            directional_pairs.append((right_statement, left_statement))

        directional = self.entailment_probabilities(directional_pairs)
        return [
            min(directional[index], directional[index + 1])
            for index in range(0, len(directional), 2)
        ]

    def entailment_decisions(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> list[bool]:
        """Return whether entailment is the model's argmax NLI label.

        Injected predictors expose only entailment probabilities, so they use
        the configured threshold as a documented testing/service fallback.
        """

        materialized = [(str(left), str(right)) for left, right in pairs]
        if not materialized:
            return []
        if self._predictor is not None:
            return [
                probability >= self.threshold
                for probability in self.entailment_probabilities(materialized)
            ]

        self._load()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None
        assert self._entailment_index is not None

        decisions: list[bool] = []
        for start in range(0, len(materialized), self.batch_size):
            batch = materialized[start : start + self.batch_size]
            encoded = self._tokenizer(
                [pair[0] for pair in batch],
                [pair[1] for pair in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self._torch.inference_mode():
                predicted = self._model(**encoded).logits.argmax(dim=-1)
            decisions.extend(
                int(value) == self._entailment_index for value in predicted.detach().cpu().tolist()
            )
        return decisions

    def equivalent_many(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        prompts: Sequence[str | None] | None = None,
    ) -> list[bool]:
        """Classify several response pairs as semantically equivalent."""

        if self.decision_rule == "probability_threshold":
            return [
                probability >= self.threshold
                for probability in self.equivalence_probabilities(
                    pairs,
                    prompts=prompts,
                )
            ]

        materialized = [(str(left), str(right)) for left, right in pairs]
        prompt_values = [None] * len(materialized) if prompts is None else list(prompts)
        if len(prompt_values) != len(materialized):
            raise ValueError("prompts and pairs must have equal length")
        directional_pairs: list[tuple[str, str]] = []
        for (left, right), prompt in zip(
            materialized,
            prompt_values,
            strict=True,
        ):
            left_statement = self._statement(left, prompt)
            right_statement = self._statement(right, prompt)
            directional_pairs.extend(
                [
                    (left_statement, right_statement),
                    (right_statement, left_statement),
                ]
            )
        directional = self.entailment_decisions(directional_pairs)
        return [
            directional[index] and directional[index + 1] for index in range(0, len(directional), 2)
        ]

    def score(
        self,
        left: str,
        right: str,
        *,
        prompt: str | None = None,
    ) -> float:
        """Return the bidirectional entailment score for one pair."""

        return self.equivalence_probabilities(
            [(left, right)],
            prompts=[prompt],
        )[0]

    def equivalent(
        self,
        left: str,
        right: str,
        *,
        prompt: str | None = None,
    ) -> bool:
        """Return whether one response pair passes the entailment threshold."""

        return self.equivalent_many(
            [(left, right)],
            prompts=[prompt],
        )[0]

    is_equivalent = equivalent
    __call__ = equivalent


def _validate_probabilities(probabilities: Sequence[float], expected: int) -> None:
    if len(probabilities) != expected:
        raise ValueError(
            f"Entailment predictor returned {len(probabilities)} values for {expected} pairs"
        )
    for probability in probabilities:
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"Entailment probabilities must be finite and in [0, 1], got {probability!r}"
            )


def _find_entailment_index(config: Any) -> int:
    label2id = getattr(config, "label2id", None) or {}
    for label, index in label2id.items():
        if "entail" in str(label).casefold():
            return int(index)

    id2label = getattr(config, "id2label", None) or {}
    for index, label in id2label.items():
        if "entail" in str(label).casefold():
            return int(index)

    # Standard MNLI ordering is contradiction, neutral, entailment.
    if int(getattr(config, "num_labels", 0)) == 3:
        return 2
    raise ValueError("Could not identify the entailment label from the NLI model config")


NLISemanticEquivalenceScorer = BidirectionalNLIScorer


__all__ = [
    "BidirectionalNLIScorer",
    "EntailmentPredictor",
    "NLISemanticEquivalenceScorer",
]
