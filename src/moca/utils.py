from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger("moca")


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def stable_fingerprint(values: Sequence[Any] | Any) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def iterable_fingerprint(values: Iterable[Any]) -> str:
    """Hash an iterable incrementally without materializing its full JSON."""

    digest = hashlib.sha256()
    for value in values:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def files_fingerprint(paths: Sequence[str | Path]) -> str:
    """Hash an ordered set of files, including names and exact contents."""

    digest = hashlib.sha256()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Cannot fingerprint missing file: {path}")
        encoded_name = path.name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    packages = [
        "accelerate",
        "datasets",
        "numpy",
        "peft",
        "scikit-learn",
        "torch",
        "transformers",
    ]
    versions: dict[str, str] = {}
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        return versions
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def resolve_run_path(run_dir: Path, *parts: str) -> Path:
    path = run_dir.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
