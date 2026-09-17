from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from ..artifacts import read_jsonl
from ..config import DataConfig
from .prompts import canonical_dataset_name

DATASET_IDS = {
    "coqa": "stanfordnlp/coqa",
    "quac": "allenai/quac",
    "xsum": "EdinburghNLP/xsum",
}

RawRecord = Mapping[str, Any]
RawSplits = Mapping[str, Iterable[RawRecord]]

_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "validation": "validation",
    "valid": "validation",
    "val": "validation",
    "dev": "validation",
    "test": "test",
    "testing": "test",
}


def normalize_split_name(name: str, *, allow_all: bool = True) -> str:
    normalized = str(name).strip().lower()
    if allow_all and normalized in {"", "all", "unspecified"}:
        return "all"
    try:
        return _SPLIT_ALIASES[normalized]
    except KeyError as error:
        expected = sorted({*_SPLIT_ALIASES, "all"} if allow_all else _SPLIT_ALIASES)
        raise ValueError(f"Unsupported split {name!r}; expected one of {expected}") from error


def load_local_jsonl(path: str | Path) -> dict[str, list[RawRecord]]:
    """Read raw or already-expanded examples from a local JSONL file."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Local JSONL dataset does not exist: {source}")

    splits: dict[str, list[RawRecord]] = {}
    for row in read_jsonl(source):
        raw_split = row.get("_split") or row.get("split") or "all"
        split = normalize_split_name(raw_split)
        splits.setdefault(split, []).append(row)
    return splits


def _import_huggingface_loader() -> Callable[..., Any]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError(
            "Hugging Face dataset loading requires the optional `datasets` package"
        ) from error
    return load_dataset


def load_raw_dataset(
    config: DataConfig,
    *,
    load_dataset_fn: Callable[..., Any] | None = None,
) -> RawSplits:
    """Load local JSONL or a supported Hugging Face dataset.

    The Hugging Face dependency and network access are both deferred until
    this function is called.  ``load_dataset_fn`` is injectable for offline
    tests and controlled mirrors.
    """

    canonical_dataset_name(config.name)
    if config.local_jsonl:
        return load_local_jsonl(config.local_jsonl)

    loader = load_dataset_fn or _import_huggingface_loader()
    dataset_id = config.dataset_id or DATASET_IDS[canonical_dataset_name(config.name)]
    positional: list[Any] = [dataset_id]
    if config.dataset_config is not None:
        positional.append(config.dataset_config)
    keyword: dict[str, Any] = {}
    if config.dataset_revision is not None:
        keyword["revision"] = config.dataset_revision

    loaded = loader(*positional, **keyword)
    if isinstance(loaded, Mapping):
        return loaded
    return {"all": loaded}
