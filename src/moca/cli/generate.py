from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from ..artifacts import atomic_write_json, read_jsonl, write_jsonl
from ..inference import RoutedMoCA
from ..utils import LOGGER, seed_everything
from .common import add_config_arguments, config_from_args


def configure_parser(parser: argparse.ArgumentParser) -> None:
    add_config_arguments(parser)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--input-file",
        help="Optional JSONL with prompt and optional example_id/references",
    )
    parser.add_argument("--output-file")
    parser.add_argument(
        "--force-wrong-route",
        action="store_true",
        help="Route each prompt to the next expert for the reviewer-requested stress test",
    )


def _load_inputs(config, split: str, input_file: str | None):
    if input_file:
        records = []
        for index, row in enumerate(read_jsonl(input_file)):
            prompt = row.get("prompt")
            if not isinstance(prompt, str):
                raise TypeError(f"Input row {index} has no string prompt")
            if not prompt.strip():
                raise ValueError(f"Input row {index} has an empty prompt")
            records.append(
                {
                    "example_id": str(row.get("example_id", index)),
                    "prompt": prompt,
                    "response": row.get("response"),
                    "references": row.get("references", []),
                    "split": row.get("split", "custom"),
                    "metadata": row.get("metadata", {}),
                }
            )
        return records
    path = config.run_dir / "clusters" / f"{split}.jsonl"
    return [row for row in read_jsonl(path)]


def run(args: argparse.Namespace):
    config = config_from_args(args)
    seed_everything(
        config.runtime.seed,
        deterministic=config.runtime.deterministic,
    )
    inputs = _load_inputs(config, args.split, args.input_file)
    if config.evaluation.max_eval_examples is not None:
        inputs = inputs[: config.evaluation.max_eval_examples]
    load_started = time.perf_counter()
    system = RoutedMoCA(config)
    load_seconds = time.perf_counter() - load_started
    outputs: list[dict[str, Any]] = []
    batch_size = config.generation.batch_size
    generation_started = time.perf_counter()
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start : start + batch_size]
        generated = system.generate(
            [row["prompt"] for row in batch],
            force_wrong_route=args.force_wrong_route,
        )
        for source, result in zip(batch, generated):
            row = dict(source)
            row.update(result.to_dict())
            row["generation"] = row.pop("response")
            if source.get("response") is not None:
                row["target"] = source["response"]
            outputs.append(row)
        LOGGER.info("Generated %d/%d responses", len(outputs), len(inputs))
    destination = (
        Path(args.output_file)
        if args.output_file
        else config.run_dir
        / "predictions"
        / (
            f"{args.split}_forced_wrong.jsonl"
            if args.force_wrong_route
            else f"{args.split}.jsonl"
        )
    )
    write_jsonl(destination, outputs)
    generation_seconds = time.perf_counter() - generation_started
    generated_tokens = sum(int(row.get("generated_token_count", 0)) for row in outputs)
    timing_path = destination.with_name(f"{destination.stem}_timing.json")
    atomic_write_json(
        timing_path,
        {
            "num_examples": len(outputs),
            "generated_tokens": generated_tokens,
            "model_and_adapter_load_seconds": load_seconds,
            "generation_seconds": generation_seconds,
            "seconds_per_example": generation_seconds / len(outputs) if outputs else None,
            "tokens_per_second": (
                generated_tokens / generation_seconds if generation_seconds else None
            ),
            "batch_size": batch_size,
            "force_wrong_route": bool(args.force_wrong_route),
            "device": config.runtime.device,
        },
    )
    LOGGER.info("Wrote generations to %s", destination)
    return destination
