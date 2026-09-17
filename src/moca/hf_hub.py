from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi

from .artifacts import atomic_write_json
from .clustering import load_cluster_artifacts
from .config import ExperimentConfig
from .modeling import adapter_checkpoint_path
from .utils import LOGGER

HF_REPO_ID = "Pouyatr/MoCA"

HF_REPO_TYPE = "model"


def _safe_folder_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    value = value.strip("._-")
    return value or "experiment"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def _check_inference_artifacts(config: ExperimentConfig) -> None:
    """
    Fail before upload if anything required by RoutedMoCA inference is missing.
    """

    run_dir = config.run_dir

    required_files = [
        run_dir / "resolved_config.yaml",
        run_dir / "clusters" / "manifest.json",
        run_dir / "clusters" / "centroids.npy",
        run_dir / "clusters" / "train_assignments.npy",
        run_dir / "clusters" / "train.jsonl",
        run_dir / "clusters" / "validation.jsonl",
    ]

    clusters = load_cluster_artifacts(run_dir / "clusters")

    num_experts = (
        1
        if config.objective.method == "vanilla_ft"
        else clusters.num_clusters
    )

    for expert_id in range(num_experts):
        expert_dir = run_dir / "adapters" / f"expert_{expert_id}"

        required_files.extend(
            [
                expert_dir / "training_manifest.json",
                adapter_checkpoint_path(run_dir, expert_id)
                / "adapter_config.json",
            ]
        )

        checkpoint_dir = adapter_checkpoint_path(run_dir, expert_id)

        has_weights = (
            (checkpoint_dir / "adapter_model.safetensors").is_file()
            or (checkpoint_dir / "adapter_model.bin").is_file()
        )

        if not has_weights:
            raise FileNotFoundError(
                f"Missing adapter weights for expert {expert_id}: "
                f"{checkpoint_dir}"
            )

    if config.evaluation.confidence_source == "moca_1p":
        if config.evaluation.calibrator_path is not None:
            calibrator_path = Path(config.evaluation.calibrator_path)
        else:
            calibrator_path = (
                run_dir
                / "evaluation"
                / "moca_1p_calibrator.json"
            )

        required_files.append(calibrator_path)

    missing = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)

        raise FileNotFoundError(
            "Cannot upload inference bundle because required artifacts "
            f"are missing:\n{formatted}"
        )


def upload_inference_run_to_hub(
    config: ExperimentConfig,
) -> str:
    """
    Upload one completed MoCA run to a unique timestamped folder.

    Remote layout:
        runs/
            <experiment_name>__<UTC timestamp>/
                resolved_config.yaml
                clusters/
                adapters/
                evaluation/
                ...
    """

    _check_inference_artifacts(config)

    run_dir = config.run_dir

    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Run directory does not exist: {run_dir}"
        )

    experiment_name = _safe_folder_name(
        config.experiment_name
    )

    timestamp = _timestamp()

    remote_folder = (
        f"runs/{experiment_name}__{timestamp}"
    )

    # Add useful metadata to the uploaded run itself.
    atomic_write_json(
        run_dir / "hub_upload_manifest.json",
        {
            "format_version": 1,
            "repo_id": HF_REPO_ID,
            "repo_type": HF_REPO_TYPE,
            "path_in_repo": remote_folder,
            "experiment_name": config.experiment_name,
            "timestamp_utc": timestamp,
            "base_model": config.model.name_or_path,
            "base_model_revision": config.model.revision,
            "config_fingerprint": config.fingerprint(),
            "training_fingerprint": config.training_fingerprint(),
        },
    )

    LOGGER.info(
        "Uploading inference run to Hugging Face: %s/%s",
        HF_REPO_ID,
        remote_folder,
    )

    api = HfApi()

    api.upload_folder(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        folder_path=str(run_dir),
        path_in_repo=remote_folder,
        commit_message=(
            f"Upload MoCA run: "
            f"{config.experiment_name} [{timestamp}]"
        ),
    )

    LOGGER.info(
        "Uploaded inference run to %s/%s",
        HF_REPO_ID,
        remote_folder,
    )

    return remote_folder
