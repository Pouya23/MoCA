from __future__ import annotations

import argparse
import gc

from ..clustering import load_cluster_artifacts
from ..training import train_expert
from ..utils import LOGGER
from .common import add_config_arguments, config_from_args


def configure_expert_parser(parser: argparse.ArgumentParser) -> None:
    add_config_arguments(parser)
    parser.add_argument("--expert-id", required=True, type=int)


def run_expert(args: argparse.Namespace):
    config = config_from_args(args)
    manifest = train_expert(config, args.expert_id)
    LOGGER.info(
        "Finished expert %d; best epoch=%d",
        args.expert_id,
        manifest["best_epoch"],
    )
    return manifest


def configure_all_parser(parser: argparse.ArgumentParser) -> None:
    add_config_arguments(parser)


def run_all(args: argparse.Namespace):
    config = config_from_args(args)
    clusters = load_cluster_artifacts(config.run_dir / "clusters")
    num_experts = 1 if config.objective.method == "vanilla_ft" else clusters.num_clusters
    manifests = []
    for expert_id in range(num_experts):
        LOGGER.info("Training expert %d/%d", expert_id + 1, num_experts)
        manifests.append(train_expert(config, expert_id))
        gc.collect()
        try:
            import torch
        except ImportError:
            continue
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return manifests
