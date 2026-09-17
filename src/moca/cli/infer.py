from __future__ import annotations

import argparse
from pathlib import Path

from ..hf_inference import (
    DEFAULT_LOCAL_ROOT,
    run_existing_generate,
)


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Experiment YAML. Its experiment_name is used to find "
            "matching timestamped runs on Hugging Face."
        ),
    )

    parser.add_argument(
        "--run-folder",
        default=None,
        help=(
            "Specific HF run folder to use. "
            "If omitted, the latest run for this experiment is used."
        ),
    )

    parser.add_argument(
        "--local-root",
        default=DEFAULT_LOCAL_ROOT.as_posix(),
        help=(
            "Relative local cache for downloaded runs. "
            f"Default: {DEFAULT_LOCAL_ROOT.as_posix()}"
        ),
    )

    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face repository revision.",
    )

    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force redownload of the selected run.",
    )

    parser.add_argument(
        "--split",
        default="test",
        help="Saved dataset split to generate on when --input-file is omitted.",
    )

    parser.add_argument(
        "--input-file",
        default=None,
        help="Optional JSONL containing formatted prompts.",
    )

    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional output JSONL path.",
    )

    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Inference-time config override; repeatable.",
    )

    parser.add_argument(
        "--force-wrong-route",
        action="store_true",
        help="Route prompts to the next expert for the stress test.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )


def run(args: argparse.Namespace) -> Path:
    return run_existing_generate(
        args.config,
        run_folder=args.run_folder,
        local_root=args.local_root,
        revision=args.revision,
        force_download=args.force_download,
        config_overrides=args.set,
        split=args.split,
        input_file=args.input_file,
        output_file=args.output_file,
        force_wrong_route=args.force_wrong_route,
        verbose=args.verbose,
    )
