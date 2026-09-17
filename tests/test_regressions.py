from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from moca.clustering import (
    cluster_fingerprint,
    fit_clusters,
    load_cluster_artifacts,
    save_cluster_artifacts,
)
from moca.config import (
    ClusteringConfig,
    DataConfig,
    EmbeddingConfig,
    EvaluationConfig,
    ExperimentConfig,
    GenerationConfig,
    LoraConfig,
    OptimizationConfig,
    RuntimeConfig,
    SamplingConfig,
    TokenizationConfig,
    load_experiment_config,
    save_resolved_config,
)
from moca.inference import _lengths_from_generated_tokens
from moca.utils import stable_fingerprint


class FingerprintTests(unittest.TestCase):
    def test_training_fingerprint_ignores_inference_only_changes(self) -> None:
        base = ExperimentConfig()
        changed_generation = replace(
            base,
            generation=replace(
                base.generation,
                temperature=0.7,
                max_new_tokens=37,
            ),
        )
        changed_evaluation = replace(
            base,
            evaluation=replace(
                base.evaluation,
                semantic_samples=17,
                max_eval_examples=23,
            ),
        )
        changed_training = replace(
            base,
            optimization=replace(base.optimization, learning_rate=1.0e-4),
        )

        self.assertNotEqual(changed_generation.fingerprint(), base.fingerprint())
        self.assertNotEqual(changed_evaluation.fingerprint(), base.fingerprint())
        self.assertEqual(
            changed_generation.training_fingerprint(),
            base.training_fingerprint(),
        )
        self.assertEqual(
            changed_evaluation.training_fingerprint(),
            base.training_fingerprint(),
        )
        self.assertNotEqual(
            changed_training.training_fingerprint(),
            base.training_fingerprint(),
        )

    def test_cluster_fingerprint_is_stable_and_round_trips(self) -> None:
        embeddings = np.asarray(
            [
                [-2.1, -2.0],
                [-2.0, -2.1],
                [-1.9, -2.0],
                [2.0, 2.1],
                [2.1, 2.0],
                [1.9, 2.0],
            ],
            dtype=np.float32,
        )
        example_ids = [f"example-{index}" for index in range(len(embeddings))]
        clustering = ClusteringConfig(
            num_clusters=2,
            random_seed=17,
            n_init=5,
            silhouette_sample_size=None,
        )
        embedding = EmbeddingConfig()

        first = fit_clusters(
            embeddings,
            example_ids,
            clustering,
            embedding,
            "tiny/base",
            "revision-1",
        )
        second = fit_clusters(
            embeddings.copy(),
            list(example_ids),
            clustering,
            embedding,
            "tiny/base",
            "revision-1",
        )
        expected_dataset_fingerprint = stable_fingerprint(example_ids)

        self.assertEqual(
            first.dataset_fingerprint,
            expected_dataset_fingerprint,
        )
        self.assertEqual(
            second.dataset_fingerprint,
            expected_dataset_fingerprint,
        )
        self.assertEqual(
            cluster_fingerprint(first),
            cluster_fingerprint(second),
        )

        with tempfile.TemporaryDirectory() as temporary:
            save_cluster_artifacts(first, temporary)
            restored = load_cluster_artifacts(temporary)
        self.assertEqual(
            restored.dataset_fingerprint,
            expected_dataset_fingerprint,
        )
        self.assertEqual(restored.train_example_ids, tuple(example_ids))
        self.assertEqual(
            cluster_fingerprint(restored),
            cluster_fingerprint(first),
        )


class InferenceRegressionTests(unittest.TestCase):
    def test_generated_length_counts_eos_when_eos_is_padding(self) -> None:
        """Llama-family tokenizers commonly configure EOS as padding."""

        tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=2)
        generated = np.asarray(
            [
                [11, 12, 2, 2],
                [21, 2, 2, 2],
            ],
            dtype=np.int64,
        )

        self.assertEqual(
            _lengths_from_generated_tokens(generated, tokenizer),
            [3, 2],
        )


class ConfigurationValidationTests(unittest.TestCase):
    def test_strictly_invalid_values_are_rejected(self) -> None:
        invalid_sections = [
            (
                "data",
                DataConfig(
                    train_ratio=-0.1,
                    validation_ratio=0.2,
                    test_ratio=0.9,
                ),
            ),
            ("tokenization", TokenizationConfig(max_prompt_tokens=0)),
            ("embedding", EmbeddingConfig(batch_size=0)),
            ("clustering", ClusteringConfig(n_init=0)),
            ("lora", LoraConfig(dropout=1.0)),
            ("sampling", SamplingConfig(dataloader_workers=-1)),
            ("optimization", OptimizationConfig(learning_rate=0.0)),
            ("generation", GenerationConfig(batch_size=0)),
            (
                "evaluation",
                EvaluationConfig(entailment_threshold=1.1),
            ),
            ("runtime", RuntimeConfig(log_every_steps=0)),
        ]

        for field_name, section in invalid_sections:
            with self.subTest(field_name=field_name, section=section):
                config = replace(
                    ExperimentConfig(),
                    **{field_name: section},
                )
                with self.assertRaises(ValueError):
                    config.validate()

    def test_saved_resolved_config_reloads_without_losing_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = replace(
                ExperimentConfig(),
                experiment_name="resolved-roundtrip",
                output_root=str(root / "runs"),
                generation=replace(
                    GenerationConfig(),
                    max_new_tokens=17,
                ),
                evaluation=replace(
                    EvaluationConfig(),
                    max_eval_examples=9,
                ),
            )
            destination = root / "resolved_config.yaml"

            save_resolved_config(config, destination)
            restored = load_experiment_config(destination)

        self.assertEqual(restored, config)
        self.assertEqual(restored.fingerprint(), config.fingerprint())


if __name__ == "__main__":
    unittest.main()
