from __future__ import annotations

import argparse
import gc

from ..utils import LOGGER
from .common import add_config_arguments


def configure_parser(parser: argparse.ArgumentParser) -> None:
    add_config_arguments(parser)
    parser.add_argument("--split", default="test")
    parser.add_argument("--skip-evaluation", action="store_true")


def _release_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(args: argparse.Namespace):
    from . import calibrate, cluster, evaluate, generate, prepare, train
    from .common import config_from_args

    common = {
        "config": args.config,
        "set": args.set,
        "ablation": args.ablation,
        "verbose": args.verbose,
    }
    config = config_from_args(args)
    fit_calibrator = (
        config.evaluation.confidence_source == "moca_1p" and not args.skip_evaluation
    )
    total_stages = 7 if fit_calibrator else 5
    LOGGER.info("Stage 1/%d: prepare data", total_stages)
    prepare.run(argparse.Namespace(**common))
    LOGGER.info("Stage 2/%d: embed and cluster", total_stages)
    cluster.run(argparse.Namespace(**common))
    _release_accelerator_memory()
    LOGGER.info("Stage 3/%d: train experts", total_stages)
    train.run_all(argparse.Namespace(**common))
    _release_accelerator_memory()
    if fit_calibrator:
        LOGGER.info("Stage 4/7: generate validation predictions for calibration")
        generate.run(
            argparse.Namespace(
                **common,
                split="validation",
                input_file=None,
                output_file=None,
                force_wrong_route=False,
            )
        )
        _release_accelerator_memory()
        LOGGER.info("Stage 5/7: fit validation-only MoCA-1P calibrator")
        calibrate.run(
            argparse.Namespace(
                **common,
                split="validation",
                predictions=None,
                output=None,
                allow_nonvalidation_fit=False,
            )
        )
        _release_accelerator_memory()
        generation_stage = "6/7"
        evaluation_stage = "7/7"
    else:
        generation_stage = "4/5"
        evaluation_stage = "5/5"
    LOGGER.info("Stage %s: generate %s predictions", generation_stage, args.split)
    generate.run(
        argparse.Namespace(
            **common,
            split=args.split,
            input_file=None,
            output_file=None,
            force_wrong_route=False,
        )
    )
    _release_accelerator_memory()
    if args.skip_evaluation:
        return None
    LOGGER.info("Stage %s: evaluate", evaluation_stage)
    return evaluate.run(
        argparse.Namespace(
            **common,
            split=args.split,
            predictions=None,
            output=None,
            skip_semantic_sampling=False,
        )
    )
