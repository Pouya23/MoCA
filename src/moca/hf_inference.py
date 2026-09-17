from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Sequence

from huggingface_hub import HfApi, RepoFolder, snapshot_download

from .config import ExperimentConfig, load_experiment_config
from .hf_hub import HF_REPO_ID
from .inference import GeneratedResponse, RoutedMoCA
from .utils import LOGGER, configure_logging


# This must match the layout used by upload_inference_run_to_hub():
#   runs/<experiment_name>__<UTC timestamp>/...
HF_REPO_TYPE = "model"
REMOTE_RUNS_ROOT = "runs"
DEFAULT_LOCAL_ROOT = Path(".moca_hf_runs")

# upload_inference_run_to_hub() used:
# datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
_TIMESTAMP_FORMATS = (
    "%Y%m%dT%H%M%S_%fZ",
    "%Y%m%dT%H%M%SZ",
)


@dataclass(frozen=True)
class ResolvedHFRun:
    """A downloaded HF run rebased to a local, relative run_dir."""

    selector_config: ExperimentConfig
    config: ExperimentConfig
    remote_run_dir: str
    local_run_dir: Path
    resolved_config_path: Path


@dataclass
class LoadedHFRun:
    """A fully loaded MoCA system ready for repeated inference."""

    resolved: ResolvedHFRun
    system: RoutedMoCA

    @property
    def config(self) -> ExperimentConfig:
        return self.resolved.config

    @property
    def run_dir(self) -> Path:
        return self.resolved.local_run_dir

    def generate(
        self,
        prompts: Sequence[str],
        *,
        force_wrong_route: bool = False,
    ) -> list[GeneratedResponse]:
        return self.system.generate(
            prompts,
            force_wrong_route=force_wrong_route,
        )


def _require_safe_relative_path(path: str | Path, *, name: str) -> Path:
    """Require a path controlled by this helper to remain relative to CWD."""

    value = Path(path)
    if value.is_absolute():
        raise ValueError(f"{name} must be relative, got absolute path: {value}")
    if ".." in value.parts:
        raise ValueError(f"{name} must not contain '..': {value}")
    return value


def _normalize_remote_folder(value: str) -> str:
    """Accept either '<folder>' or 'runs/<folder>', never an absolute path."""

    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(
            "run_folder must be a relative HF repository path or folder name"
        )

    parts = pure.parts
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2 and parts[0] == REMOTE_RUNS_ROOT:
        return parts[1]

    raise ValueError(
        f"run_folder must be '<folder>' or '{REMOTE_RUNS_ROOT}/<folder>', got: {value}"
    )


def _timestamp_for_folder(folder_name: str, experiment_name: str) -> datetime | None:
    prefix = f"{experiment_name}__"
    if not folder_name.startswith(prefix):
        return None

    raw_timestamp = folder_name[len(prefix) :]
    for timestamp_format in _TIMESTAMP_FORMATS:
        try:
            parsed = datetime.strptime(raw_timestamp, timestamp_format)
        except ValueError:
            continue
        return parsed.replace(tzinfo=timezone.utc)
    return None


def _matching_run_folders(
    *,
    repo_id: str,
    experiment_name: str,
    revision: str | None,
    token: str | bool | None,
) -> list[tuple[str, datetime]]:
    """List timestamped remote run folders for exactly one experiment."""

    if "/" in experiment_name or "\\" in experiment_name:
        raise ValueError(
            "experiment_name must be a folder name, not a path: "
            f"{experiment_name!r}"
        )

    api = HfApi(token=token)
    entries = api.list_repo_tree(
        repo_id=repo_id,
        repo_type=HF_REPO_TYPE,
        revision=revision,
        path_in_repo=REMOTE_RUNS_ROOT,
        recursive=False,
        token=token,
    )

    matches: list[tuple[str, datetime]] = []
    malformed_matches: list[str] = []
    prefix = f"{experiment_name}__"

    for entry in entries:
        if not isinstance(entry, RepoFolder):
            continue

        folder_name = PurePosixPath(entry.path).name
        if not folder_name.startswith(prefix):
            continue

        timestamp = _timestamp_for_folder(folder_name, experiment_name)
        if timestamp is None:
            malformed_matches.append(folder_name)
            continue

        matches.append((folder_name, timestamp))

    if not matches:
        message = (
            f"No timestamped run folder for experiment {experiment_name!r} "
            f"was found under {repo_id}/{REMOTE_RUNS_ROOT}."
        )
        if malformed_matches:
            message += (
                " Matching names existed but did not contain a supported UTC timestamp: "
                + ", ".join(sorted(malformed_matches))
            )
        raise FileNotFoundError(message)

    return matches


def select_remote_run_folder(
    *,
    experiment_name: str,
    run_folder: str | None = None,
    repo_id: str = HF_REPO_ID,
    revision: str | None = None,
    token: str | bool | None = None,
) -> str:
    """
    Resolve the remote run folder.

    If run_folder is omitted, the newest timestamped folder whose name starts
    exactly with '<experiment_name>__' is selected.
    """

    matches = _matching_run_folders(
        repo_id=repo_id,
        experiment_name=experiment_name,
        revision=revision,
        token=token,
    )
    available = {name: timestamp for name, timestamp in matches}

    if run_folder is not None:
        requested = _normalize_remote_folder(run_folder)
        if requested not in available:
            choices = "\n".join(
                f"  - {name}"
                for name, _ in sorted(matches, key=lambda item: item[1], reverse=True)
            )
            raise FileNotFoundError(
                f"Requested run folder {requested!r} is not a valid timestamped "
                f"run for experiment {experiment_name!r}. Available runs:\n{choices}"
            )
        return requested

    # Use parsed datetimes, not lexicographic ordering or local filesystem mtimes.
    return max(matches, key=lambda item: item[1])[0]


def _verify_downloaded_run(run_dir: Path, *, confidence_source: str) -> None:
    required_files = (
        run_dir / "resolved_config.yaml",
        run_dir / "clusters" / "manifest.json",
        run_dir / "clusters" / "centroids.npy",
        run_dir / "clusters" / "train_assignments.npy",
        run_dir / "clusters" / "train.jsonl",
        run_dir / "clusters" / "validation.jsonl",
    )
    missing = [path for path in required_files if not path.is_file()]

    adapters_root = run_dir / "adapters"
    best_adapter_configs = sorted(adapters_root.glob("expert_*/best/adapter_config.json"))
    if not best_adapter_configs:
        missing.append(adapters_root / "expert_*/best/adapter_config.json")

    if confidence_source == "moca_1p":
        calibrator = run_dir / "evaluation" / "moca_1p_calibrator.json"
        if not calibrator.is_file():
            missing.append(calibrator)

    if missing:
        formatted = "\n".join(f"  - {path.as_posix()}" for path in missing)
        raise FileNotFoundError(
            "The downloaded run is incomplete for inference. Missing:\n" + formatted
        )


def resolve_hf_run(
    config_path: str | Path,
    *,
    run_folder: str | None = None,
    local_root: str | Path = DEFAULT_LOCAL_ROOT,
    repo_id: str = HF_REPO_ID,
    revision: str | None = None,
    token: str | bool | None = None,
    force_download: bool = False,
    config_overrides: Sequence[str] = (),
) -> ResolvedHFRun:
    """
    Find, download, and rebase one HF run for inference.

    `config_path` may itself be absolute or relative; only its experiment_name
    is used to select the HF folder. Paths created/used for the downloaded run
    are deliberately relative to the current working directory.
    """

    selector_config = load_experiment_config(config_path)
    experiment_name = selector_config.experiment_name

    local_root_path = _require_safe_relative_path(local_root, name="local_root")

    selected_folder = select_remote_run_folder(
        experiment_name=experiment_name,
        run_folder=run_folder,
        repo_id=repo_id,
        revision=revision,
        token=token,
    )
    remote_run_dir = f"{REMOTE_RUNS_ROOT}/{selected_folder}"

    LOGGER.info(
        "Selected HF run %s/%s",
        repo_id,
        remote_run_dir,
    )

    # local_dir preserves the repository-relative structure. Therefore only the
    # selected run appears at:
    #   <local_root>/runs/<experiment_name>__<timestamp>/
    snapshot_download(
        repo_id=repo_id,
        repo_type=HF_REPO_TYPE,
        revision=revision,
        token=token,
        local_dir=local_root_path,
        allow_patterns=f"{remote_run_dir}/**",
        force_download=force_download,
    )

    local_run_dir = local_root_path / REMOTE_RUNS_ROOT / selected_folder
    if not local_run_dir.is_dir():
        raise FileNotFoundError(
            f"HF download completed but run directory was not created: {local_run_dir}"
        )

    resolved_config_path = local_run_dir / "resolved_config.yaml"
    if not resolved_config_path.is_file():
        raise FileNotFoundError(
            f"Downloaded run has no resolved_config.yaml: {resolved_config_path}"
        )

    # Load the config that actually produced the selected artifacts, not the
    # selector YAML. Rebase only runtime location fields to the relative local
    # download. The saved fingerprint is checked before overrides are applied.
    mandatory_overrides = [
        f"output_root={local_run_dir.parent.as_posix()}",
        f"experiment_name={selected_folder}",
    ]

    # A run bundle is self-contained with respect to MoCA-specific artifacts.
    # Avoid an old machine-specific absolute calibrator path from the original
    # config; RoutedMoCA should use <run_dir>/evaluation/moca_1p_calibrator.json.
    bundled_calibrator = local_run_dir / "evaluation" / "moca_1p_calibrator.json"
    if bundled_calibrator.is_file():
        mandatory_overrides.append("evaluation.calibrator_path=null")

    rebased_config = load_experiment_config(
        resolved_config_path,
        overrides=[*config_overrides, *mandatory_overrides],
    )

    if rebased_config.run_dir != local_run_dir:
        raise RuntimeError(
            "Internal run_dir rebasing error: "
            f"config points to {rebased_config.run_dir}, expected {local_run_dir}"
        )

    _verify_downloaded_run(
        local_run_dir,
        confidence_source=rebased_config.evaluation.confidence_source,
    )

    return ResolvedHFRun(
        selector_config=selector_config,
        config=rebased_config,
        remote_run_dir=remote_run_dir,
        local_run_dir=local_run_dir,
        resolved_config_path=resolved_config_path,
    )


def load_hf_run(
    config_path: str | Path,
    *,
    run_folder: str | None = None,
    local_root: str | Path = DEFAULT_LOCAL_ROOT,
    repo_id: str = HF_REPO_ID,
    revision: str | None = None,
    token: str | bool | None = None,
    force_download: bool = False,
    config_overrides: Sequence[str] = (),
) -> LoadedHFRun:
    """Download a run and fully load router + base model + all adapters + calibrator."""

    resolved = resolve_hf_run(
        config_path,
        run_folder=run_folder,
        local_root=local_root,
        repo_id=repo_id,
        revision=revision,
        token=token,
        force_download=force_download,
        config_overrides=config_overrides,
    )
    system = RoutedMoCA(resolved.config)
    return LoadedHFRun(resolved=resolved, system=system)


def run_existing_generate(
    config_path: str | Path,
    *,
    run_folder: str | None = None,
    local_root: str | Path = DEFAULT_LOCAL_ROOT,
    repo_id: str = HF_REPO_ID,
    revision: str | None = None,
    token: str | bool | None = None,
    force_download: bool = False,
    config_overrides: Sequence[str] = (),
    split: str = "test",
    input_file: str | None = None,
    output_file: str | None = None,
    force_wrong_route: bool = False,
    verbose: bool = False,
) -> Path:
    """Download/rebase a run, then call the repository's existing `moca generate` code."""

    from .cli import generate as generate_cli

    resolved = resolve_hf_run(
        config_path,
        run_folder=run_folder,
        local_root=local_root,
        repo_id=repo_id,
        revision=revision,
        token=token,
        force_download=force_download,
        config_overrides=config_overrides,
    )

    if input_file is None:
        split_file = resolved.local_run_dir / "clusters" / f"{split}.jsonl"
        if not split_file.is_file():
            raise FileNotFoundError(
                f"No --input-file was supplied and the downloaded run has no {split!r} "
                f"cluster split: {split_file}"
            )

    # generate.run() loads config itself, so pass the relative-path rebasing
    # through its ordinary --set mechanism. This calls exactly the same code as
    # `moca generate`; no duplicate inference implementation is introduced.
    mandatory_overrides = [
        f"output_root={resolved.local_run_dir.parent.as_posix()}",
        f"experiment_name={resolved.local_run_dir.name}",
    ]
    if (
        resolved.local_run_dir / "evaluation" / "moca_1p_calibrator.json"
    ).is_file():
        mandatory_overrides.append("evaluation.calibrator_path=null")

    args = argparse.Namespace(
        config=str(resolved.resolved_config_path),
        set=[*config_overrides, *mandatory_overrides],
        ablation=[],
        verbose=verbose,
        split=split,
        input_file=input_file,
        output_file=output_file,
        force_wrong_route=force_wrong_route,
    )
    return generate_cli.run(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m moca.hf_inference",
        description=(
            "Select a timestamped MoCA run from Hugging Face, download it to a "
            "relative local cache, then execute the existing MoCA generation path."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Any MoCA YAML for the desired experiment. Its experiment_name selects "
            "the matching timestamped HF run."
        ),
    )
    parser.add_argument(
        "--run-folder",
        help=(
            "Specific timestamped folder to use. Accepts either '<folder>' or "
            "'runs/<folder>'. Omit to use the latest matching run."
        ),
    )
    parser.add_argument(
        "--local-root",
        default=DEFAULT_LOCAL_ROOT.as_posix(),
        help=(
            "Relative directory used for HF downloads. Absolute paths and '..' are "
            f"rejected. Default: {DEFAULT_LOCAL_ROOT.as_posix()}"
        ),
    )
    parser.add_argument("--revision", default=None, help="HF repo revision; default: main")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload files even if the local HF metadata says they are current.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--input-file",
        help="Same JSONL input accepted by `moca generate --input-file`.",
    )
    parser.add_argument(
        "--output-file",
        help="Same output path accepted by `moca generate --output-file`.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Inference-time config override, e.g. --set generation.temperature=0.7. "
            "Path rebasing fields are controlled by this loader."
        ),
    )
    parser.add_argument(
        "--force-wrong-route",
        action="store_true",
        help="Pass through the existing forced-wrong-route stress-test option.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    try:
        destination = run_existing_generate(
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
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        if args.verbose:
            raise
        parser.exit(1, f"moca HF inference: error: {error}\n")

    LOGGER.info("Generation complete: %s", destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
