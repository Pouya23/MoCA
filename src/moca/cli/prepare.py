from __future__ import annotations

import argparse

from ..artifacts import atomic_write_json
from ..config import save_resolved_config
from ..utils import LOGGER, package_versions, stable_fingerprint
from .common import add_config_arguments, config_from_args


def configure_parser(parser: argparse.ArgumentParser) -> None:
    add_config_arguments(parser)


def run(args: argparse.Namespace):
    from ..data import prepare_data

    config = config_from_args(args)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, config.run_dir / "resolved_config.yaml")
    splits = prepare_data(config)
    atomic_write_json(
        config.run_dir / "data" / "manifest.json",
        {
            "format_version": 1,
            "dataset_name": config.data.name,
            "dataset_id": config.data.dataset_id,
            "dataset_revision": config.data.dataset_revision,
            "split_seed": config.data.split_seed,
            "split_unit": config.data.split_unit,
            "use_official_splits": config.data.use_official_splits,
            "counts": {name: len(records) for name, records in splits.items()},
            "example_id_fingerprints": {
                name: stable_fingerprint([record.example_id for record in records])
                for name, records in splits.items()
            },
            "package_versions": package_versions(),
        },
    )
    LOGGER.info(
        "Prepared dataset: train=%d validation=%d test=%d",
        len(splits["train"]),
        len(splits["validation"]),
        len(splits["test"]),
    )
    return splits
