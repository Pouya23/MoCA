from __future__ import annotations

import argparse
import json
import sys

from ..ablations import describe_ablations
from ..utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moca",
        description="Mixture of Calibrated Adapters reproduction",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    from . import aggregate, calibrate, cluster, generate, prepare, train

    commands = {
        "prepare": (prepare.configure_parser, prepare.run),
        "cluster": (cluster.configure_parser, cluster.run),
        "train-expert": (train.configure_expert_parser, train.run_expert),
        "train-all": (train.configure_all_parser, train.run_all),
        "generate": (generate.configure_parser, generate.run),
        "calibrate": (calibrate.configure_parser, calibrate.run),
        "aggregate": (aggregate.configure_parser, aggregate.run),
    }
    try:
        from . import evaluate, evaluate_ood

        commands["evaluate"] = (evaluate.configure_parser, evaluate.run)
        commands["evaluate-ood"] = (evaluate_ood.configure_parser, evaluate_ood.run)
    except ImportError:
        pass
    try:
        from . import run_experiment

        commands["run"] = (
            run_experiment.configure_parser,
            run_experiment.run,
        )
    except ImportError:
        pass
    for name, (configure, runner) in commands.items():
        command_parser = subparsers.add_parser(name)
        configure(command_parser)
        command_parser.set_defaults(_runner=runner)

    list_parser = subparsers.add_parser("list-ablations")
    list_parser.set_defaults(_runner=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(getattr(args, "verbose", False))
    if args.command == "list-ablations":
        print(json.dumps(describe_ablations(), indent=2))
        return 0
    try:
        args._runner(args)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        if getattr(args, "verbose", False):
            raise
        parser.exit(1, f"moca: error: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
