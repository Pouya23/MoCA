from __future__ import annotations

import tempfile
import unittest

import numpy as np

from moca.ablations import apply_ablations
from moca.clustering import (
    fit_clusters,
    load_cluster_artifacts,
    route_embeddings,
    route_embeddings_with_geometry,
    save_cluster_artifacts,
)
from moca.config import (
    ClusteringConfig,
    EmbeddingConfig,
    ExperimentConfig,
)


class ConfigurationTests(unittest.TestCase):
    def test_paper_defaults(self) -> None:
        config = ExperimentConfig()
        self.assertEqual(config.lora.rank, 16)
        self.assertEqual(config.lora.alpha, 32)
        self.assertEqual(config.lora.target_modules, ("q_proj", "v_proj"))
        self.assertEqual(config.objective.lambda_kl, 0.1)
        self.assertEqual(config.objective.num_ood_tokens, 1)
        self.assertEqual(config.objective.kl_direction, "model_to_uniform")
        self.assertEqual(config.optimization.learning_rate, 2e-4)
        self.assertEqual(config.optimization.max_epochs, 10)

    def test_ablation_conflict_is_rejected(self) -> None:
        base = ExperimentConfig().to_dict()
        with self.assertRaisesRegex(ValueError, "both change"):
            apply_ablations(base, ["k_ft", "vanilla_ft"])

    def test_k_ft_ablation_changes_only_objective(self) -> None:
        result = apply_ablations(ExperimentConfig().to_dict(), ["k_ft"])
        self.assertEqual(result["objective"]["method"], "k_ft")
        self.assertEqual(result["objective"]["lambda_kl"], 0.0)
        self.assertIsNone(result["clustering"]["num_clusters"])


class ClusteringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embeddings = np.asarray(
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
        self.ids = [f"example-{index}" for index in range(len(self.embeddings))]

    def test_router_and_tie_breaking(self) -> None:
        centroids = np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        assignments, distances = route_embeddings(
            np.asarray([[-2.0, 0.0], [2.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            centroids,
        )
        self.assertEqual(assignments.tolist(), [0, 1, 0])
        np.testing.assert_allclose(distances, [1.0, 1.0, 1.0])

    def test_router_exposes_second_distance_and_margin(self) -> None:
        centroids = np.asarray([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        assignments, nearest, second, margins = route_embeddings_with_geometry(
            np.asarray([[0.0, 0.0], [3.0, 0.0]], dtype=np.float32),
            centroids,
        )
        self.assertEqual(assignments.tolist(), [0, 1])
        np.testing.assert_allclose(nearest, [0.0, 1.0])
        np.testing.assert_allclose(second, [2.0, 3.0])
        np.testing.assert_allclose(margins, [2.0, 2.0])

    def test_fit_save_load_and_route(self) -> None:
        clustering = ClusteringConfig(
            num_clusters=2,
            random_seed=7,
            n_init=5,
        )
        embedding = EmbeddingConfig(pooling="mean", normalize=False)
        fitted = fit_clusters(
            self.embeddings,
            self.ids,
            clustering,
            embedding,
            "tiny/base",
            "revision",
        )
        self.assertEqual(fitted.num_clusters, 2)
        self.assertGreater(fitted.selected_silhouette, 0.9)
        self.assertEqual(
            len(set(fitted.train_assignments[:3].tolist())),
            1,
        )
        self.assertNotEqual(
            fitted.train_assignments[0],
            fitted.train_assignments[-1],
        )

        with tempfile.TemporaryDirectory() as temporary:
            save_cluster_artifacts(fitted, temporary)
            restored = load_cluster_artifacts(temporary)
        np.testing.assert_allclose(restored.centroids, fitted.centroids)
        np.testing.assert_array_equal(
            restored.train_assignments,
            fitted.train_assignments,
        )
        self.assertEqual(restored.train_example_ids, tuple(self.ids))

    def test_silhouette_selection_is_deterministic(self) -> None:
        clustering = ClusteringConfig(
            num_clusters=None,
            min_clusters=2,
            max_clusters=3,
            random_seed=11,
            n_init=5,
            silhouette_sample_size=None,
        )
        embedding = EmbeddingConfig()
        first = fit_clusters(
            self.embeddings,
            self.ids,
            clustering,
            embedding,
            "tiny/base",
            None,
        )
        second = fit_clusters(
            self.embeddings,
            self.ids,
            clustering,
            embedding,
            "tiny/base",
            None,
        )
        self.assertEqual(first.num_clusters, second.num_clusters)
        np.testing.assert_allclose(first.centroids, second.centroids)


if __name__ == "__main__":
    unittest.main()
