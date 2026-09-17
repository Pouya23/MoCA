from __future__ import annotations

import argparse

from ..config import ExperimentConfig, load_experiment_config


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Experiment YAML file")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted configuration value; repeatable",
    )
    parser.add_argument(
        "--ablation",
        action="append",
        default=[],
        help="Apply a registered ablation; repeatable",
    )
    parser.add_argument("--verbose", action="store_true")


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return load_experiment_config(
        args.config,
        overrides=args.set,
        ablations=args.ablation,
    )
