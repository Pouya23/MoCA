from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .artifacts import atomic_write_json, read_json
from .config import ClusteringConfig, EmbeddingConfig, TokenizationConfig
from .utils import stable_fingerprint


@dataclass(frozen=True)
class ClusterArtifacts:
    num_clusters: int
    centroids: np.ndarray
    selected_silhouette: float | None
    silhouette_by_k: Mapping[int, float]
    embedding_config: Mapping[str, Any]
    tokenization_config: Mapping[str, Any]
    clustering_config: Mapping[str, Any]
    base_model: str
    base_model_revision: str | None
    train_example_ids: tuple[str, ...]
    train_assignments: np.ndarray
    dataset_fingerprint: str

    def route(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return route_embeddings(embeddings, self.centroids)

    def route_with_geometry(
        self,
        embeddings: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return route_embeddings_with_geometry(embeddings, self.centroids)

    def route_with_diagnostics(
        self,
        embeddings: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        return route_embeddings_with_diagnostics(
            embeddings,
             self.centroids,
        )

def cluster_fingerprint(artifacts: ClusterArtifacts) -> str:
    """Return a content hash for a router and its training assignments."""

    metadata = {
        "num_clusters": artifacts.num_clusters,
        "selected_silhouette": artifacts.selected_silhouette,
        "silhouette_by_k": dict(artifacts.silhouette_by_k),
        "embedding_config": dict(artifacts.embedding_config),
        "tokenization_config": dict(artifacts.tokenization_config),
        "clustering_config": dict(artifacts.clustering_config),
        "base_model": artifacts.base_model,
        "base_model_revision": artifacts.base_model_revision,
        "train_example_ids": list(artifacts.train_example_ids),
        "dataset_fingerprint": artifacts.dataset_fingerprint,
    }
    digest = hashlib.sha256(
        json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )
    centroids = np.ascontiguousarray(artifacts.centroids, dtype="<f4")
    assignments = np.ascontiguousarray(artifacts.train_assignments, dtype="<i8")
    digest.update(centroids.tobytes())
    digest.update(assignments.tobytes())
    return digest.hexdigest()


def route_embeddings(
    embeddings: np.ndarray,
    centroids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    assignments, nearest, _, _ = route_embeddings_with_geometry(embeddings, centroids)
    return assignments, nearest


def route_embeddings_with_geometry(
    embeddings: np.ndarray,
    centroids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Route points and return nearest/second-nearest distances and margin.

    The margin is ``d2 - d1``.

    For a one-centroid vanilla baseline, d2 and the margin are defined
    as zero so downstream calibration stays finite.
    """

    distances = centroid_distance_matrix(
        embeddings,
        centroids,
    )

    num_examples, num_centroids = distances.shape

    # np.argmin preserves the previous deterministic lowest-index
    # tie-breaking behavior.
    assignments = np.argmin(
        distances,
        axis=1,
    ).astype(np.int64)

    nearest = distances[
        np.arange(num_examples),
        assignments,
    ].astype(np.float32)

    if num_centroids == 1:
        second_nearest = np.zeros(
            num_examples,
            dtype=np.float32,
        )
        margins = np.zeros(
            num_examples,
            dtype=np.float32,
        )
    else:
        two_smallest = np.partition(
            distances,
            kth=1,
            axis=1,
        )[:, :2]

        two_smallest.sort(axis=1)

        second_nearest = two_smallest[
            :,
            1,
        ].astype(np.float32)

        margins = (
            second_nearest - nearest
        ).astype(np.float32)

    return (
        assignments,
        nearest,
        second_nearest,
        margins,
    )


def router_distribution_from_distances(
    distances: np.ndarray,
) -> np.ndarray:
    """Convert centroid distances into a normalized router distribution.

    This is NOT used for expert selection. Hard routing remains nearest
    centroid exactly as before.

    We normalize by each prompt's mean centroid distance before the
    softmax so router ambiguity is less sensitive to the absolute scale
    of the embedding space.

        r_k(x) ∝ exp(-d_k(x) / mean_j d_j(x))

    Returns:
        probabilities with shape [N, K].
    """

    values = np.asarray(
        distances,
        dtype=np.float64,
    )

    if values.ndim != 2:
        raise ValueError(
            "Router distances must be a rank-two array"
        )

    if values.shape[1] < 1:
        raise ValueError(
            "At least one centroid is required"
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Router distances contain NaN or infinite values"
        )

    if np.any(values < 0):
        raise ValueError(
            "Router distances must be non-negative"
        )

    # Vanilla one-expert baseline.
    if values.shape[1] == 1:
        return np.ones(
            (values.shape[0], 1),
            dtype=np.float32,
        )

    scale = values.mean(
        axis=1,
        keepdims=True,
    )

    scale = np.maximum(
        scale,
        1e-8,
    )

    logits = -values / scale

    # Numerically stable softmax.
    logits -= logits.max(
        axis=1,
        keepdims=True,
    )

    probabilities = np.exp(logits)

    probabilities /= probabilities.sum(
        axis=1,
        keepdims=True,
    )

    return probabilities.astype(
        np.float32,
        copy=False,
    )


def router_uncertainty_from_distances(
    distances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized router entropy and maximum router probability.

    Normalized entropy:

        H_R = -sum_k r_k log(r_k) / log(K)

    Therefore:
        0 -> highly decisive routing
        1 -> maximally ambiguous routing

    max_probability is:
        max_k r_k

    For K=1:
        entropy = 0
        max_probability = 1
    """

    probabilities = (
        router_distribution_from_distances(
            distances
        )
    )

    num_centroids = probabilities.shape[1]

    if num_centroids == 1:
        return (
            np.zeros(
                probabilities.shape[0],
                dtype=np.float32,
            ),
            np.ones(
                probabilities.shape[0],
                dtype=np.float32,
            ),
        )

    safe_probabilities = np.maximum(
        probabilities.astype(np.float64),
        1e-12,
    )

    entropy = -np.sum(
        safe_probabilities
        * np.log(safe_probabilities),
        axis=1,
    )

    entropy /= np.log(num_centroids)

    max_probability = probabilities.max(
        axis=1
    )

    return (
        entropy.astype(np.float32),
        max_probability.astype(np.float32),
    )


def route_embeddings_with_diagnostics(
    embeddings: np.ndarray,
    centroids: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Hard-route embeddings and additionally expose router uncertainty.

    Returns:
        assignments
        nearest_distance
        second_nearest_distance
        routing_margin
        router_entropy
        router_max_probability

    Hard routing itself is unchanged.
    """

    distances = centroid_distance_matrix(
        embeddings,
        centroids,
    )

    num_examples, num_centroids = distances.shape

    assignments = np.argmin(
        distances,
        axis=1,
    ).astype(np.int64)

    nearest = distances[
        np.arange(num_examples),
        assignments,
    ].astype(np.float32)

    if num_centroids == 1:
        second_nearest = np.zeros(
            num_examples,
            dtype=np.float32,
        )
        margins = np.zeros(
            num_examples,
            dtype=np.float32,
        )
    else:
        two_smallest = np.partition(
            distances,
            kth=1,
            axis=1,
        )[:, :2]

        two_smallest.sort(axis=1)

        second_nearest = two_smallest[
            :,
            1,
        ].astype(np.float32)

        margins = (
            second_nearest - nearest
        ).astype(np.float32)

    (
        router_entropy,
        router_max_probability,
    ) = router_uncertainty_from_distances(
        distances
    )

    return (
        assignments,
        nearest,
        second_nearest,
        margins,
        router_entropy,
        router_max_probability,
    )


def _candidate_cluster_counts(config: ClusteringConfig, n_samples: int) -> list[int]:
    upper = min(config.max_clusters, n_samples - 1)
    if config.num_clusters is not None:
        if config.num_clusters == 1:
            return [1]
        if config.num_clusters > upper:
            raise ValueError(
                f"num_clusters={config.num_clusters} is invalid for {n_samples} examples"
            )
        return [config.num_clusters]
    candidates = list(range(config.min_clusters, upper + 1))
    if not candidates:
        raise ValueError(
            f"At least {config.min_clusters + 1} training examples are needed "
            "for silhouette-based cluster selection"
        )
    return candidates


def fit_clusters(
    embeddings: np.ndarray,
    example_ids: Sequence[str],
    config: ClusteringConfig,
    embedding_config: EmbeddingConfig,
    base_model: str,
    base_model_revision: str | None,
    tokenization_config: TokenizationConfig | None = None,
    dataset_fingerprint: str | None = None,
) -> ClusterArtifacts:
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Expected a rank-two embedding array")
    if len(example_ids) != values.shape[0]:
        raise ValueError("example_ids and embeddings have different lengths")
    if not np.isfinite(values).all():
        raise ValueError("Embeddings contain NaN or infinite values")

    if config.strategy == "random_balanced":
        if config.num_clusters is None:
            raise ValueError("random_balanced clustering requires num_clusters")
        if config.num_clusters < 2 or config.num_clusters >= values.shape[0]:
            raise ValueError("random_balanced num_clusters must be in [2, n_samples)")
        generator = np.random.default_rng(config.random_seed)
        order = generator.permutation(values.shape[0])
        assignments = np.empty(values.shape[0], dtype=np.int64)
        assignments[order] = np.arange(values.shape[0], dtype=np.int64) % config.num_clusters
        centroids = np.stack(
            [
                values[assignments == cluster_id].mean(axis=0)
                for cluster_id in range(config.num_clusters)
            ]
        ).astype(np.float32)
        score = float(
            silhouette_score(
                values,
                assignments,
                metric="euclidean",
                sample_size=(
                    min(config.silhouette_sample_size, values.shape[0])
                    if config.silhouette_sample_size is not None
                    else None
                ),
                random_state=config.random_seed,
            )
        )
        return ClusterArtifacts(
            num_clusters=config.num_clusters,
            centroids=centroids,
            selected_silhouette=score,
            silhouette_by_k={config.num_clusters: score},
            embedding_config=asdict(embedding_config),
            tokenization_config=asdict(tokenization_config or TokenizationConfig()),
            clustering_config=asdict(config),
            base_model=base_model,
            base_model_revision=base_model_revision,
            train_example_ids=tuple(example_ids),
            train_assignments=assignments,
            dataset_fingerprint=dataset_fingerprint or stable_fingerprint(list(example_ids)),
        )

    candidates = _candidate_cluster_counts(config, values.shape[0])
    if candidates == [1]:
        assignments = np.zeros(values.shape[0], dtype=np.int64)
        centroids = values.mean(axis=0, keepdims=True)
        return ClusterArtifacts(
            num_clusters=1,
            centroids=centroids,
            selected_silhouette=None,
            silhouette_by_k={},
            embedding_config=asdict(embedding_config),
            tokenization_config=asdict(tokenization_config or TokenizationConfig()),
            clustering_config=asdict(config),
            base_model=base_model,
            base_model_revision=base_model_revision,
            train_example_ids=tuple(example_ids),
            train_assignments=assignments,
            dataset_fingerprint=dataset_fingerprint or stable_fingerprint(list(example_ids)),
        )

    fitted: dict[int, KMeans] = {}
    scores: dict[int, float] = {}
    for k in candidates:
        estimator = KMeans(
            n_clusters=k,
            init=config.init,
            n_init=config.n_init,
            max_iter=config.max_iter,
            tol=config.tolerance,
            random_state=config.random_seed,
            algorithm="lloyd",
        )
        labels = estimator.fit_predict(values)
        distinct_labels = np.unique(labels)
        if len(distinct_labels) < 2 or len(distinct_labels) >= values.shape[0]:
            if config.num_clusters is not None:
                raise ValueError(
                    f"k-means with k={k} produced {len(distinct_labels)} "
                    "non-empty clusters; silhouette is undefined"
                )
            continue
        fitted[k] = estimator
        sample_size = config.silhouette_sample_size
        if sample_size is not None:
            sample_size = min(sample_size, values.shape[0])
        try:
            scores[k] = float(
                silhouette_score(
                    values,
                    labels,
                    metric="euclidean",
                    sample_size=sample_size,
                    random_state=config.random_seed,
                )
            )
        except ValueError:
            if config.num_clusters is not None:
                raise
            fitted.pop(k, None)
    if not scores:
        raise ValueError("No candidate k produced a valid multi-cluster silhouette score")

    # A lower k wins an exact score tie, which is explicit and deterministic.
    best_k = max(sorted(scores), key=lambda k: scores[k])
    best = fitted[best_k]
    return ClusterArtifacts(
        num_clusters=best_k,
        centroids=np.asarray(best.cluster_centers_, dtype=np.float32),
        selected_silhouette=scores[best_k],
        silhouette_by_k=scores,
        embedding_config=asdict(embedding_config),
        tokenization_config=asdict(tokenization_config or TokenizationConfig()),
        clustering_config=asdict(config),
        base_model=base_model,
        base_model_revision=base_model_revision,
        train_example_ids=tuple(example_ids),
        train_assignments=np.asarray(best.labels_, dtype=np.int64),
        dataset_fingerprint=dataset_fingerprint or stable_fingerprint(list(example_ids)),
    )


def save_cluster_artifacts(artifacts: ClusterArtifacts, directory: str | Path) -> None:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "centroids.npy", artifacts.centroids, allow_pickle=False)
    np.save(
        destination / "train_assignments.npy",
        artifacts.train_assignments,
        allow_pickle=False,
    )
    manifest = {
        "format_version": 2,
        "num_clusters": artifacts.num_clusters,
        "selected_silhouette": artifacts.selected_silhouette,
        "silhouette_by_k": {str(key): value for key, value in artifacts.silhouette_by_k.items()},
        "embedding_config": dict(artifacts.embedding_config),
        "tokenization_config": dict(artifacts.tokenization_config),
        "clustering_config": dict(artifacts.clustering_config),
        "base_model": artifacts.base_model,
        "base_model_revision": artifacts.base_model_revision,
        "train_example_ids": list(artifacts.train_example_ids),
        "dataset_fingerprint": artifacts.dataset_fingerprint,
        "centroid_shape": list(artifacts.centroids.shape),
        "cluster_fingerprint": cluster_fingerprint(artifacts),
    }
    atomic_write_json(destination / "manifest.json", manifest)


def load_cluster_artifacts(directory: str | Path) -> ClusterArtifacts:
    source = Path(directory)
    manifest = read_json(source / "manifest.json")
    centroids = np.load(source / "centroids.npy", allow_pickle=False)
    assignments = np.load(source / "train_assignments.npy", allow_pickle=False)
    expected_shape = tuple(manifest["centroid_shape"])
    if centroids.shape != expected_shape:
        raise ValueError(
            f"Centroid shape mismatch: expected {expected_shape}, got {centroids.shape}"
        )
    if len(assignments) != len(manifest["train_example_ids"]):
        raise ValueError("Cluster assignments do not match train example IDs")
    artifacts = ClusterArtifacts(
        num_clusters=int(manifest["num_clusters"]),
        centroids=np.asarray(centroids, dtype=np.float32),
        selected_silhouette=manifest["selected_silhouette"],
        silhouette_by_k={
            int(key): float(value) for key, value in manifest["silhouette_by_k"].items()
        },
        embedding_config=manifest["embedding_config"],
        tokenization_config=manifest.get(
            "tokenization_config",
            asdict(TokenizationConfig()),
        ),
        clustering_config=manifest["clustering_config"],
        base_model=manifest["base_model"],
        base_model_revision=manifest.get("base_model_revision"),
        train_example_ids=tuple(manifest["train_example_ids"]),
        train_assignments=np.asarray(assignments, dtype=np.int64),
        dataset_fingerprint=manifest["dataset_fingerprint"],
    )
    expected_fingerprint = manifest.get("cluster_fingerprint")
    if expected_fingerprint is not None and cluster_fingerprint(artifacts) != expected_fingerprint:
        raise ValueError(
            f"Cluster artifact fingerprint mismatch in {source}; "
            "router files may be stale or corrupted"
        )
    return artifacts


def validate_router_compatibility(
    artifacts: ClusterArtifacts,
    embedding_config: EmbeddingConfig,
    base_model: str,
    base_model_revision: str | None,
    tokenization_config: TokenizationConfig | None = None,
) -> None:
    expected_embedding = asdict(embedding_config)
    if dict(artifacts.embedding_config) != expected_embedding:
        raise ValueError(
            "Router embedding configuration differs from the configuration used "
            "to create its centroids"
        )
    if tokenization_config is not None:
        expected_tokenization = asdict(tokenization_config)
        if dict(artifacts.tokenization_config) != expected_tokenization:
            raise ValueError(
                "Router tokenization configuration differs from the configuration "
                "used to create its prompt embeddings"
            )
    if artifacts.base_model != base_model:
        raise ValueError("Router was created with a different base model")
    if artifacts.base_model_revision != base_model_revision:
        raise ValueError("Router was created with a different base-model revision")


def centroid_distance_matrix(
    embeddings: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """Return Euclidean distance from every embedding to every centroid.

    Shape:
        embeddings: [N, D]
        centroids:  [K, D]
        return:     [N, K]

    This uses O(NK) auxiliary memory rather than explicitly constructing
    an [N, K, D] broadcasted difference tensor.
    """

    points = np.asarray(embeddings, dtype=np.float32)
    centers = np.asarray(centroids, dtype=np.float32)

    if points.ndim != 2 or centers.ndim != 2:
        raise ValueError(
            "Embeddings and centroids must both be rank-two arrays"
        )

    if points.shape[1] != centers.shape[1]:
        raise ValueError(
            f"Embedding dimension {points.shape[1]} does not match "
            f"centroid dimension {centers.shape[1]}"
        )

    squared_distances = (
        np.square(points).sum(axis=1, keepdims=True)
        + np.square(centers).sum(axis=1, keepdims=True).T
        - 2.0 * (points @ centers.T)
    )

    # Floating-point roundoff can turn exact zeros slightly negative.
    np.maximum(
        squared_distances,
        0.0,
        out=squared_distances,
    )

    return np.sqrt(squared_distances).astype(
        np.float32,
        copy=False,
    )
