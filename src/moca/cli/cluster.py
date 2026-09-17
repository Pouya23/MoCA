from __future__ import annotations

import argparse
import dataclasses

import numpy as np

from ..artifacts import atomic_write_json, write_jsonl
from ..clustering import fit_clusters, save_cluster_artifacts
from ..config import save_resolved_config
from ..embeddings import BaseModelPromptEmbedder
from ..modeling import load_base_model, load_tokenizer
from ..records import PromptResponse
from ..utils import LOGGER, iterable_fingerprint, seed_everything
from .common import add_config_arguments, config_from_args


def configure_parser(parser: argparse.ArgumentParser) -> None:
    add_config_arguments(parser)


def _with_cluster(
    record: PromptResponse,
    cluster_id: int,
    distance: float,
) -> PromptResponse:
    metadata = dict(record.metadata)
    metadata["routing_distance_to_assigned_centroid"] = distance
    return dataclasses.replace(
        record,
        cluster_id=cluster_id,
        metadata=metadata,
    )


def run(args: argparse.Namespace):
    from ..data import load_prepared_splits

    config = config_from_args(args)
    seed_everything(
        config.runtime.seed,
        deterministic=config.runtime.deterministic,
    )
    splits = load_prepared_splits(config)
    fit_records = (
        splits["train"]
        if config.clustering.fit_scope == "train"
        else [record for split in ("train", "validation", "test") for record in splits[split]]
    )
    if config.objective.method == "vanilla_ft":
        # The vanilla baseline has a single centroid only for a common artifact API.
        clustering_values = dataclasses.replace(config.clustering, num_clusters=1)
    else:
        clustering_values = config.clustering

    tokenizer = load_tokenizer(config, padding_side="right")
    model = load_base_model(config, for_training=False)
    embedder = BaseModelPromptEmbedder(
        model,
        tokenizer,
        config.embedding,
        config.tokenization,
    )
    LOGGER.info("Embedding %d prompts with the frozen base model", len(fit_records))
    fit_embeddings = embedder.encode([record.prompt for record in fit_records])
    artifacts = fit_clusters(
        fit_embeddings,
        [record.example_id for record in fit_records],
        clustering_values,
        config.embedding,
        config.model.name_or_path,
        config.model.revision,
        config.tokenization,
        iterable_fingerprint(record.to_dict() for record in fit_records),
    )
    cluster_dir = config.run_dir / "clusters"
    save_cluster_artifacts(artifacts, cluster_dir)
    save_resolved_config(config, config.run_dir / "resolved_config.yaml")

    fit_assignment_by_id = dict(
        zip(artifacts.train_example_ids, artifacts.train_assignments.tolist())
    )
    fit_embedding_by_id = {
        example_id: fit_embeddings[index]
        for index, example_id in enumerate(artifacts.train_example_ids)
    }
    split_summary: dict[str, dict[str, int]] = {}
    for split_name, records in splits.items():
        assignments = np.empty(len(records), dtype=np.int64)
        distances = np.empty(len(records), dtype=np.float32)
        missing_indices = [
            index
            for index, record in enumerate(records)
            if record.example_id not in fit_assignment_by_id
        ]
        for index, record in enumerate(records):
            if record.example_id in fit_assignment_by_id:
                assignments[index] = fit_assignment_by_id[record.example_id]
                distances[index] = float(
                    np.linalg.norm(
                        fit_embedding_by_id[record.example_id]
                        - artifacts.centroids[assignments[index]]
                    )
                )
        if missing_indices:
            held_out_embeddings = embedder.encode(
                [records[index].prompt for index in missing_indices]
            )
            held_out_assignments, held_out_distances = artifacts.route(held_out_embeddings)
            for offset, index in enumerate(missing_indices):
                assignments[index] = held_out_assignments[offset]
                distances[index] = held_out_distances[offset]
        write_jsonl(
            cluster_dir / f"{split_name}.jsonl",
            (
                _with_cluster(
                    record,
                    int(assignments[index]),
                    float(distances[index]),
                ).to_dict()
                for index, record in enumerate(records)
            ),
        )
        np.save(
            cluster_dir / f"{split_name}_assignments.npy",
            assignments,
            allow_pickle=False,
        )
        counts = np.bincount(assignments, minlength=artifacts.num_clusters)
        split_summary[split_name] = {str(index): int(count) for index, count in enumerate(counts)}

    atomic_write_json(cluster_dir / "split_cluster_counts.json", split_summary)
    LOGGER.info(
        "Selected k=%d (silhouette=%s)",
        artifacts.num_clusters,
        artifacts.selected_silhouette,
    )
    return artifacts
