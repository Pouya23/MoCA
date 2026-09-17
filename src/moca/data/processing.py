from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..artifacts import read_jsonl, write_jsonl
from ..config import DataConfig, ExperimentConfig
from ..records import PromptResponse
from .loading import RawSplits, load_raw_dataset, normalize_split_name
from .prompts import (
    canonical_dataset_name,
    format_coqa_prompt,
    format_quac_prompt,
    format_xsum_prompt,
)

SPLIT_NAMES = ("train", "validation", "test")


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _deduplicate_texts(values: Iterable[Any]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _text(value)
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return tuple(result)


def _extract_texts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        direct_keys = (
            "input_text",
            "text",
            "texts",
            "answer",
            "answers",
            "response",
            "references",
            "summary",
        )
        for key in direct_keys:
            if key in value:
                return _extract_texts(value[key])
        result: list[str] = []
        for nested in value.values():
            result.extend(_extract_texts(nested))
        return result
    if _is_sequence(value):
        result = []
        for nested in value:
            result.extend(_extract_texts(nested))
        return result
    return [str(value)]


def _aligned_value(value: Any, index: int) -> Any:
    """Take one conversational turn from row- or column-oriented features."""

    if isinstance(value, Mapping):
        aligned: dict[str, Any] = {}
        for key, nested in value.items():
            if _is_sequence(nested):
                if index >= len(nested):
                    raise ValueError(f"Feature {key!r} has no item at turn {index}")
                aligned[key] = nested[index]
            else:
                aligned[key] = nested
        return aligned
    if _is_sequence(value):
        if index >= len(value):
            raise ValueError(f"Feature has no item at turn {index}")
        return value[index]
    if index:
        raise ValueError(f"Scalar feature cannot provide turn {index}")
    return value


def _question_text(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("input_text", "question", "text"):
            if key in value:
                return _text(value[key])
    return _text(value)


def _stable_fallback_id(
    dataset_name: str,
    split: str,
    record_index: int,
    source_text: str,
) -> str:
    digest = hashlib.sha1(
        f"{dataset_name}\0{split}\0{record_index}\0{source_text}".encode()
    ).hexdigest()[:16]
    return f"{dataset_name}-{digest}"


def _first_identifier(record: Mapping[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = record.get(key)
        if value is not None and _text(value):
            return _text(value)
    return ""


def _canonical_prompt_response(
    record: Mapping[str, Any],
    *,
    dataset_name: str,
    split: str,
    record_index: int,
) -> PromptResponse | None:
    if "prompt" not in record or "response" not in record:
        return None
    response = _text(record["response"])
    if not response:
        raise ValueError(f"Expanded local record {record_index} has an empty response")
    example_id = _first_identifier(record, ("example_id", "id"))
    if not example_id:
        example_id = _stable_fallback_id(dataset_name, split, record_index, _text(record["prompt"]))
    group_id = _first_identifier(record, ("group_id", "conversation_id", "dialogue_id"))
    if not group_id:
        group_id = example_id
    references = _deduplicate_texts([response, *_extract_texts(record.get("references"))])
    metadata = dict(record.get("metadata") or {})
    metadata.setdefault("dataset", dataset_name)
    metadata.setdefault("raw_split", split)
    cluster = record.get("cluster_id")
    return PromptResponse(
        example_id=example_id,
        prompt=str(record["prompt"]),
        response=response,
        references=references,
        split=normalize_split_name(record.get("split") or split),
        group_id=group_id,
        cluster_id=None if cluster is None else int(cluster),
        metadata=metadata,
    )


def _coqa_additional_references(record: Mapping[str, Any], index: int) -> list[str]:
    additional = record.get("additional_answers")
    if additional is None:
        return []
    result: list[str] = []
    if isinstance(additional, Mapping):
        for annotator_answers in additional.values():
            try:
                result.extend(_extract_texts(_aligned_value(annotator_answers, index)))
            except ValueError:
                continue
        return result
    try:
        return _extract_texts(_aligned_value(additional, index))
    except ValueError:
        return []


def _expand_coqa(
    record: Mapping[str, Any],
    *,
    split: str,
    record_index: int,
    include_dialogue_history: bool,
) -> list[PromptResponse]:
    story = _text(record.get("story", record.get("context")))
    questions = record.get("questions")
    answers = record.get("answers")
    if not story or not _is_sequence(questions) or answers is None:
        raise ValueError("CoQA records require story, questions, and answers")

    group_id = _first_identifier(
        record, ("group_id", "conversation_id", "dialogue_id", "id")
    ) or _stable_fallback_id("coqa", split, record_index, story)
    raw_question_ids = record.get("question_ids", record.get("turn_ids"))
    history: list[tuple[str, str]] = []
    expanded: list[PromptResponse] = []

    for turn_index, raw_question in enumerate(questions):
        question = _question_text(raw_question)
        answer_texts = _extract_texts(_aligned_value(answers, turn_index))
        response_candidates = _deduplicate_texts(answer_texts)
        if not question or not response_candidates:
            raise ValueError(f"CoQA group {group_id!r} has an invalid turn {turn_index}")
        response = response_candidates[0]
        references = _deduplicate_texts(
            [
                response,
                *response_candidates[1:],
                *_coqa_additional_references(record, turn_index),
                *_extract_texts(
                    _aligned_value(record["references"], turn_index)
                    if "references" in record
                    else None
                ),
            ]
        )
        if raw_question_ids is not None:
            example_id = _text(_aligned_value(raw_question_ids, turn_index))
        elif isinstance(raw_question, Mapping):
            example_id = _first_identifier(raw_question, ("id", "question_id", "turn_id"))
        else:
            example_id = ""
        example_id = example_id or f"{group_id}:turn-{turn_index:03d}"
        prompt = format_coqa_prompt(
            story,
            question,
            history if include_dialogue_history else (),
        )
        metadata: dict[str, Any] = {
            "dataset": "coqa",
            "raw_split": split,
            "turn_index": turn_index,
        }
        if record.get("source") is not None:
            metadata["source"] = _text(record["source"])
        expanded.append(
            PromptResponse(
                example_id=example_id,
                prompt=prompt,
                response=response,
                references=references,
                split=split,
                group_id=group_id,
                metadata=metadata,
            )
        )
        history.append((question, response))
    return expanded


def _quac_orig_answer(record: Mapping[str, Any], index: int) -> list[str]:
    original = record.get("orig_answers", record.get("orig_answer"))
    if original is None:
        return []
    try:
        return _extract_texts(_aligned_value(original, index))
    except ValueError:
        return []


def _metadata_turn_value(record: Mapping[str, Any], key: str, index: int) -> Any:
    if key not in record:
        return None
    try:
        return _aligned_value(record[key], index)
    except ValueError:
        return None


def _expand_quac(
    record: Mapping[str, Any],
    *,
    split: str,
    record_index: int,
    include_dialogue_history: bool,
) -> list[PromptResponse]:
    context = _text(record.get("context", record.get("story")))
    questions = record.get("questions")
    answers = record.get("answers")
    if not context or not _is_sequence(questions) or answers is None:
        raise ValueError("QuAC records require context, questions, and answers")

    group_id = _first_identifier(
        record, ("group_id", "dialogue_id", "conversation_id", "id")
    ) or _stable_fallback_id("quac", split, record_index, context)
    turn_ids = record.get("turn_ids", record.get("question_ids"))
    history: list[tuple[str, str]] = []
    expanded: list[PromptResponse] = []

    for turn_index, raw_question in enumerate(questions):
        question = _question_text(raw_question)
        answer_references = _deduplicate_texts(_extract_texts(_aligned_value(answers, turn_index)))
        original = _deduplicate_texts(_quac_orig_answer(record, turn_index))
        candidates = _deduplicate_texts([*original, *answer_references])
        if not question or not candidates:
            raise ValueError(f"QuAC group {group_id!r} has an invalid turn {turn_index}")
        response = candidates[0]
        if turn_ids is not None:
            example_id = _text(_aligned_value(turn_ids, turn_index))
        elif isinstance(raw_question, Mapping):
            example_id = _first_identifier(raw_question, ("id", "question_id", "turn_id"))
        else:
            example_id = ""
        example_id = example_id or f"{group_id}:turn-{turn_index:03d}"
        metadata: dict[str, Any] = {
            "dataset": "quac",
            "raw_split": split,
            "turn_index": turn_index,
        }
        for source_key in ("wikipedia_page_title", "section_title"):
            if record.get(source_key) is not None:
                metadata[source_key] = _text(record[source_key])
        for turn_key in ("followups", "yesnos"):
            turn_value = _metadata_turn_value(record, turn_key, turn_index)
            if turn_value is not None:
                metadata[turn_key.rstrip("s")] = turn_value
        expanded.append(
            PromptResponse(
                example_id=example_id,
                prompt=format_quac_prompt(
                    context,
                    question,
                    history if include_dialogue_history else (),
                ),
                response=response,
                references=candidates,
                split=split,
                group_id=group_id,
                metadata=metadata,
            )
        )
        history.append((question, response))
    return expanded


def _expand_xsum(
    record: Mapping[str, Any],
    *,
    split: str,
    record_index: int,
) -> list[PromptResponse]:
    document = _text(record.get("document", record.get("article")))
    response_candidates = _deduplicate_texts(
        [
            *_extract_texts(record.get("summary", record.get("response"))),
            *_extract_texts(record.get("references")),
        ]
    )
    if not document or not response_candidates:
        raise ValueError("XSum records require document and summary")
    example_id = _first_identifier(record, ("example_id", "id"))
    example_id = example_id or _stable_fallback_id("xsum", split, record_index, document)
    return [
        PromptResponse(
            example_id=example_id,
            prompt=format_xsum_prompt(document),
            response=response_candidates[0],
            references=response_candidates,
            split=split,
            group_id=_first_identifier(record, ("group_id",)) or example_id,
            metadata={"dataset": "xsum", "raw_split": split},
        )
    ]


def expand_record(
    record: Mapping[str, Any],
    dataset_name: str,
    *,
    split: str = "all",
    record_index: int = 0,
    include_dialogue_history: bool = False,
) -> list[PromptResponse]:
    """Expand one dataset row into one or more prompt-response examples."""

    name = canonical_dataset_name(dataset_name)
    normalized_split = normalize_split_name(split)
    canonical = _canonical_prompt_response(
        record,
        dataset_name=name,
        split=normalized_split,
        record_index=record_index,
    )
    if canonical is not None:
        return [canonical]
    if name == "coqa":
        return _expand_coqa(
            record,
            split=normalized_split,
            record_index=record_index,
            include_dialogue_history=include_dialogue_history,
        )
    if name == "quac":
        return _expand_quac(
            record,
            split=normalized_split,
            record_index=record_index,
            include_dialogue_history=include_dialogue_history,
        )
    return _expand_xsum(record, split=normalized_split, record_index=record_index)


def expand_records(
    records: Iterable[Mapping[str, Any]],
    dataset_name: str,
    *,
    split: str = "all",
    include_dialogue_history: bool = False,
) -> list[PromptResponse]:
    """Expand an iterable of raw rows into canonical examples."""

    result: list[PromptResponse] = []
    for record_index, record in enumerate(records):
        result.extend(
            expand_record(
                record,
                dataset_name,
                split=split,
                record_index=record_index,
                include_dialogue_history=include_dialogue_history,
            )
        )
    return result


def _ordered_raw_splits(raw_splits: RawSplits) -> Iterable[tuple[str, Iterable[Any]]]:
    remaining = dict(raw_splits)
    for alias in ("train", "validation", "dev", "val", "test", "all"):
        if alias in remaining:
            yield alias, remaining.pop(alias)
    for name in sorted(remaining):
        yield name, remaining[name]


def expand_raw_splits(raw_splits: RawSplits, config: DataConfig) -> list[PromptResponse]:
    """Expand every raw split with a stable split traversal order."""

    result: list[PromptResponse] = []
    for split, records in _ordered_raw_splits(raw_splits):
        result.extend(
            expand_records(
                records,
                config.name,
                split=split,
                include_dialogue_history=config.include_dialogue_history,
            )
        )
        if config.max_examples is not None and len(result) >= config.max_examples:
            return result[: config.max_examples]
    return result


def _allocation_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if any(ratio < 0 for ratio in ratios) or not math.isclose(
        sum(ratios),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise ValueError(f"Split ratios must be non-negative and sum to one, got {ratios}")
    exact = [total * ratio for ratio in ratios]
    counts = [math.floor(value) for value in exact]
    remainder = total - sum(counts)
    order = sorted(
        range(len(ratios)),
        key=lambda index: (-(exact[index] - counts[index]), index),
    )
    for index in order[:remainder]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def deterministic_split(
    examples: Sequence[PromptResponse],
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 2026,
    split_unit: str = "example",
) -> dict[str, list[PromptResponse]]:
    """Deterministically partition examples by example or conversation group."""

    if split_unit not in {"example", "group"}:
        raise ValueError("split_unit must be 'example' or 'group'")
    seen_examples: set[str] = set()
    unit_to_examples: dict[str, list[PromptResponse]] = {}
    for example in examples:
        if example.example_id in seen_examples:
            raise ValueError(f"Duplicate example_id: {example.example_id!r}")
        seen_examples.add(example.example_id)
        unit_id = example.example_id if split_unit == "example" else example.group_id
        unit_to_examples.setdefault(unit_id, []).append(example)

    units = sorted(unit_to_examples)
    random.Random(seed).shuffle(units)
    counts = _allocation_counts(
        len(units),
        (train_ratio, validation_ratio, test_ratio),
    )
    assignments: dict[str, str] = {}
    offset = 0
    for split, count in zip(SPLIT_NAMES, counts):
        for unit in units[offset : offset + count]:
            assignments[unit] = split
        offset += count

    result = {name: [] for name in SPLIT_NAMES}
    for example in examples:
        unit_id = example.example_id if split_unit == "example" else example.group_id
        assigned = assignments[unit_id]
        result[assigned].append(replace(example, split=assigned))
    return result


def use_official_splits(
    examples: Sequence[PromptResponse],
) -> dict[str, list[PromptResponse]]:
    """Bucket examples by their dataset-provided split labels."""

    result = {name: [] for name in SPLIT_NAMES}
    for example in examples:
        split = normalize_split_name(example.split, allow_all=False)
        result[split].append(replace(example, split=split))
    return result


def split_examples(
    examples: Sequence[PromptResponse],
    config: DataConfig,
) -> dict[str, list[PromptResponse]]:
    """Apply either official splits or the paper's configured random split."""

    if config.use_official_splits:
        return use_official_splits(examples)
    return deterministic_split(
        examples,
        train_ratio=config.train_ratio,
        validation_ratio=config.validation_ratio,
        test_ratio=config.test_ratio,
        seed=config.split_seed,
        split_unit=config.split_unit,
    )


def prepared_data_dir(config: ExperimentConfig) -> Path:
    return config.run_dir / "data"


def prepared_split_path(config: ExperimentConfig, split: str) -> Path:
    normalized = normalize_split_name(split, allow_all=False)
    return prepared_data_dir(config) / f"{normalized}.jsonl"


def write_prepared_splits(
    config: ExperimentConfig,
    splits: Mapping[str, Iterable[PromptResponse]],
) -> dict[str, Path]:
    """Atomically write canonical train/validation/test JSONL artifacts."""

    paths: dict[str, Path] = {}
    for split in SPLIT_NAMES:
        path = prepared_split_path(config, split)

        def rows(split_name: str = split) -> Iterable[dict[str, Any]]:
            for example in splits.get(split_name, ()):
                canonical = (
                    example if example.split == split_name else replace(example, split=split_name)
                )
                yield canonical.to_dict()

        write_jsonl(path, rows())
        paths[split] = path
    return paths


def load_prepared_split(
    config: ExperimentConfig,
    split: str,
) -> list[PromptResponse]:
    path = prepared_split_path(config, split)
    if not path.is_file():
        raise FileNotFoundError(f"Prepared split does not exist: {path}")
    return [PromptResponse.from_dict(row) for row in read_jsonl(path)]


def load_prepared_splits(
    config: ExperimentConfig,
) -> dict[str, list[PromptResponse]]:
    return {split: load_prepared_split(config, split) for split in SPLIT_NAMES}


def prepared_splits_exist(config: ExperimentConfig) -> bool:
    return all(prepared_split_path(config, split).is_file() for split in SPLIT_NAMES)


def prepare_data(
    config: ExperimentConfig,
    *,
    load_dataset_fn: Any | None = None,
) -> dict[str, list[PromptResponse]]:
    """Load, canonicalize, split, persist, and return the experiment data."""

    raw = load_raw_dataset(config.data, load_dataset_fn=load_dataset_fn)
    expanded = expand_raw_splits(raw, config.data)
    splits = split_examples(expanded, config.data)
    write_prepared_splits(config, splits)
    return splits


def load_or_prepare_data(
    config: ExperimentConfig,
    *,
    force: bool = False,
    load_dataset_fn: Any | None = None,
) -> dict[str, list[PromptResponse]]:
    if not force and prepared_splits_exist(config):
        return load_prepared_splits(config)
    return prepare_data(config, load_dataset_fn=load_dataset_fn)


# Concise compatibility names for callers that use "save" or "dataset".
save_prepared_splits = write_prepared_splits
prepare_dataset = prepare_data
