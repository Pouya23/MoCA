from __future__ import annotations

import math
import shutil
import time
from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json, read_jsonl
from .clustering import (
    cluster_fingerprint,
    load_cluster_artifacts,
    validate_router_compatibility,
)
from .config import ExperimentConfig, save_resolved_config
from .losses import (
    first_token_uniform_kl,
    response_prefix_uniform_kl,
    sequence_nll,
)
from .modeling import (
    adapter_checkpoint_path,
    create_lora_expert,
    load_base_model,
    load_tokenizer,
    model_device,
    require_torch,
    special_token_ids,
    trainable_parameter_summary,
)
from .records import PromptResponse
from .utils import LOGGER, files_fingerprint, package_versions, seed_everything


@dataclass
class EpochMetrics:
    total: float
    nll: float
    kl: float
    batches: int
    target_tokens: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "total": self.total,
            "nll": self.nll,
            "kl": self.kl,
            "batches": self.batches,
            "target_tokens": self.target_tokens,
        }


class PromptOnlyCollator:
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config

    def __call__(self, examples: Sequence[PromptResponse]) -> dict[str, Any]:
        original_side = self.tokenizer.truncation_side
        self.tokenizer.truncation_side = self.config.prompt_truncation_side
        try:
            batch = self.tokenizer(
                [example.prompt for example in examples],
                padding=True,
                truncation=True,
                max_length=self.config.max_prompt_tokens,
                add_special_tokens=self.config.add_special_tokens_to_prompt,
                pad_to_multiple_of=self.config.pad_to_multiple_of,
                return_tensors="pt",
            )
        finally:
            self.tokenizer.truncation_side = original_side
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }


class _CyclingIterator:
    def __init__(self, loader):
        self.loader = loader
        self.iterator: Iterator[Any] = iter(loader)

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


def _load_clustered_split(run_dir: Path, split: str) -> list[PromptResponse]:
    path = run_dir / "clusters" / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing clustered {split} data: {path}. Run `moca cluster` first."
        )
    return [PromptResponse.from_dict(row) for row in read_jsonl(path)]


def _records_for_expert(
    records: Sequence[PromptResponse],
    expert_id: int,
    method: str,
) -> tuple[list[PromptResponse], list[PromptResponse]]:
    if method == "vanilla_ft":
        return list(records), []
    positive = [record for record in records if record.cluster_id == expert_id]
    negative = [record for record in records if record.cluster_id != expert_id]
    if not positive:
        raise ValueError(f"Expert {expert_id} has no in-cluster examples")
    if method == "moca" and not negative:
        raise ValueError(f"Expert {expert_id} has no pseudo-OOD examples")
    return positive, negative


def _make_response_collator(tokenizer, config):
    try:
        from .data import ResponseOnlyCausalLMCollator
    except ImportError as error:
        raise RuntimeError("The moca.data package is incomplete") from error
    return ResponseOnlyCausalLMCollator(tokenizer, config.tokenization)


def _make_loader(
    records: Sequence[PromptResponse],
    batch_size: int,
    collate_fn,
    *,
    shuffle: bool,
    workers: int,
    seed: int,
    sampling_policy: str = "empirical_complement",
    replacement: bool = False,
):
    torch = require_torch()
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = None
    if sampling_policy == "uniform_cluster":
        counts: dict[int, int] = {}
        for record in records:
            if record.cluster_id is None:
                raise ValueError("Uniform-cluster sampling requires cluster IDs")
            counts[record.cluster_id] = counts.get(record.cluster_id, 0) + 1
        weights = [1.0 / counts[int(record.cluster_id)] for record in records]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights,
            num_samples=(max(len(records), batch_size) if replacement else len(records)),
            replacement=replacement,
            generator=generator,
        )
        shuffle = False
    elif sampling_policy == "empirical_complement" and replacement:
        sampler = torch.utils.data.RandomSampler(
            list(records),
            replacement=True,
            num_samples=max(len(records), batch_size),
            generator=generator,
        )
        shuffle = False
    elif sampling_policy != "empirical_complement":
        raise ValueError(f"Unknown sampling policy: {sampling_policy}")
    return torch.utils.data.DataLoader(
        list(records),
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=workers,
        generator=generator if sampler is None else None,
        pin_memory=True,
        drop_last=False,
    )


def _to_device(batch: dict[str, Any], device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _autocast_context(config: ExperimentConfig, device):
    torch = require_torch()
    precision = config.runtime.mixed_precision
    if precision in {"no", "none", "fp32", "float32"}:
        return nullcontext()
    if device.type not in {"cuda", "cpu"}:
        return nullcontext()
    dtype = torch.bfloat16 if precision in {"bf16", "bfloat16"} else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _excluded_tokens(config: ExperimentConfig, tokenizer) -> tuple[int, ...]:
    if config.objective.uniform_scope == "full_vocabulary":
        return ()
    return special_token_ids(tokenizer)


def _compute_batch_loss(
    model,
    positive_batch: dict[str, Any],
    negative_batch: dict[str, Any] | None,
    config: ExperimentConfig,
    tokenizer,
):
    torch = require_torch()
    positive_outputs = model(
        input_ids=positive_batch["input_ids"],
        attention_mask=positive_batch["attention_mask"],
        use_cache=False,
        return_dict=True,
    )
    nll, target_tokens = sequence_nll(
        positive_outputs.logits,
        positive_batch["labels"],
        reduction=config.objective.sequence_nll_reduction,
    )
    kl = nll.new_zeros(())
    if config.objective.method == "moca":
        if negative_batch is None:
            raise ValueError("MoCA requires a pseudo-OOD batch")
        negative_outputs = model(
            input_ids=negative_batch["input_ids"],
            attention_mask=negative_batch["attention_mask"],
            use_cache=False,
            return_dict=True,
        )
        excluded = _excluded_tokens(config, tokenizer)
        if config.objective.num_ood_tokens == 1:
            kl = first_token_uniform_kl(
                negative_outputs.logits,
                negative_batch["attention_mask"],
                direction=config.objective.kl_direction,
                excluded_token_ids=excluded,
            )
        else:
            if config.objective.later_token_conditioning != "gold":
                raise ValueError(
                    "Only gold conditioning is implemented for the non-paper first-m-token ablation"
                )
            kl = response_prefix_uniform_kl(
                negative_outputs.logits,
                negative_batch["labels"],
                num_tokens=config.objective.num_ood_tokens,
                direction=config.objective.kl_direction,
                excluded_token_ids=excluded,
            )
    total = nll + config.objective.lambda_kl * kl
    if not torch.isfinite(total):
        raise FloatingPointError(
            f"Non-finite loss: total={total.item()}, nll={nll.item()}, kl={kl.item()}"
        )
    return total, nll, kl, target_tokens


def _run_validation(
    model,
    positive_loader,
    negative_loader,
    config: ExperimentConfig,
    tokenizer,
) -> EpochMetrics | None:
    torch = require_torch()
    if positive_loader is None:
        return None
    was_training = model.training
    model.eval()
    device = model_device(model)
    nll_sum = 0.0
    nll_weight = 0
    kl_sum = 0.0
    kl_examples = 0
    positive_batches = 0
    target_tokens = 0
    limit = config.optimization.validation_batches
    with torch.inference_mode():
        for batch_index, positive_batch in enumerate(positive_loader):
            if limit is not None and batch_index >= limit:
                break
            positive_batch = _to_device(positive_batch, device)
            with _autocast_context(config, device):
                outputs = model(
                    input_ids=positive_batch["input_ids"],
                    attention_mask=positive_batch["attention_mask"],
                    use_cache=False,
                    return_dict=True,
                )
                nll, token_count = sequence_nll(
                    outputs.logits,
                    positive_batch["labels"],
                    reduction=config.objective.sequence_nll_reduction,
                )
            batch_examples = int(positive_batch["input_ids"].shape[0])
            batch_weight = (
                token_count
                if config.objective.sequence_nll_reduction == "token_mean"
                else batch_examples
            )
            nll_sum += float(nll.item()) * batch_weight
            nll_weight += batch_weight
            positive_batches += 1
            target_tokens += token_count
        if config.objective.method == "moca":
            if negative_loader is None:
                raise ValueError("MoCA validation requires pseudo-OOD examples")
            excluded = _excluded_tokens(config, tokenizer)
            for batch_index, negative_batch in enumerate(negative_loader):
                if limit is not None and batch_index >= limit:
                    break
                negative_batch = _to_device(negative_batch, device)
                with _autocast_context(config, device):
                    outputs = model(
                        input_ids=negative_batch["input_ids"],
                        attention_mask=negative_batch["attention_mask"],
                        use_cache=False,
                        return_dict=True,
                    )
                    if config.objective.num_ood_tokens == 1:
                        kl = first_token_uniform_kl(
                            outputs.logits,
                            negative_batch["attention_mask"],
                            direction=config.objective.kl_direction,
                            excluded_token_ids=excluded,
                        )
                    else:
                        kl = response_prefix_uniform_kl(
                            outputs.logits,
                            negative_batch["labels"],
                            num_tokens=config.objective.num_ood_tokens,
                            direction=config.objective.kl_direction,
                            excluded_token_ids=excluded,
                        )
                batch_examples = int(negative_batch["input_ids"].shape[0])
                kl_sum += float(kl.item()) * batch_examples
                kl_examples += batch_examples
    if was_training:
        model.train()
    if not nll_weight:
        return None
    nll_mean = nll_sum / nll_weight
    kl_mean = kl_sum / kl_examples if kl_examples else 0.0
    return EpochMetrics(
        total=nll_mean + config.objective.lambda_kl * kl_mean,
        nll=nll_mean,
        kl=kl_mean,
        batches=positive_batches,
        target_tokens=target_tokens,
    )


def _save_adapter(model, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination, safe_serialization=True)
    if not (destination / "adapter_config.json").exists():
        # Defensive compatibility for PEFT versions that nest named adapters.
        nested = destination / "default"
        if nested.exists() and (nested / "adapter_config.json").exists():
            for source in nested.iterdir():
                shutil.copy2(source, destination / source.name)
    if not (destination / "adapter_config.json").exists():
        raise RuntimeError(f"PEFT did not save a loadable adapter to {destination}")


def train_expert(config: ExperimentConfig, expert_id: int) -> dict[str, Any]:
    torch = require_torch()
    config.validate()
    seed_everything(
        config.runtime.seed + expert_id,
        deterministic=config.runtime.deterministic,
    )
    cluster_artifacts = load_cluster_artifacts(config.run_dir / "clusters")
    validate_router_compatibility(
        cluster_artifacts,
        config.embedding,
        config.model.name_or_path,
        config.model.revision,
        config.tokenization,
    )
    expected_experts = (
        1 if config.objective.method == "vanilla_ft" else cluster_artifacts.num_clusters
    )
    if expert_id < 0 or expert_id >= expected_experts:
        raise ValueError(f"expert_id must be in [0, {expected_experts - 1}], got {expert_id}")

    train_records = _load_clustered_split(config.run_dir, "train")
    validation_records = _load_clustered_split(config.run_dir, "validation")
    clustered_data_fingerprint = files_fingerprint(
        [
            config.run_dir / "clusters" / "train.jsonl",
            config.run_dir / "clusters" / "validation.jsonl",
        ]
    )
    if config.clustering.fit_scope == "train":
        clustered_ids = [record.example_id for record in train_records]
        artifact_ids = list(cluster_artifacts.train_example_ids)
        if (
            len(clustered_ids) != len(set(clustered_ids))
            or len(artifact_ids) != len(set(artifact_ids))
            or set(clustered_ids) != set(artifact_ids)
        ):
            raise ValueError(
                "Clustered training records do not have the exact example-ID set "
                "used to fit the persisted router"
            )
    assignment_by_id = dict(
        zip(
            cluster_artifacts.train_example_ids,
            cluster_artifacts.train_assignments.tolist(),
        )
    )
    inconsistent = [
        record.example_id
        for record in train_records
        if record.example_id in assignment_by_id
        and record.cluster_id != assignment_by_id[record.example_id]
    ]
    if inconsistent:
        raise ValueError(
            "Clustered training JSONL disagrees with the persisted k-means "
            f"assignments (first IDs: {inconsistent[:5]})"
        )
    train_positive, train_negative = _records_for_expert(
        train_records, expert_id, config.objective.method
    )
    if config.objective.method == "vanilla_ft":
        validation_positive, validation_negative = list(validation_records), []
    else:
        validation_positive = [
            record for record in validation_records if record.cluster_id == expert_id
        ]
        validation_negative = [
            record for record in validation_records if record.cluster_id != expert_id
        ]

    run_directory = config.run_dir / "adapters" / f"expert_{expert_id}"
    manifest_path = run_directory / "training_manifest.json"
    best_path = adapter_checkpoint_path(config.run_dir, expert_id)
    if (manifest_path.exists() or best_path.exists()) and not (config.runtime.overwrite_existing):
        raise FileExistsError(
            f"Expert {expert_id} already has training artifacts under "
            f"{run_directory}. Use a new experiment_name or explicitly set "
            "runtime.overwrite_existing=true."
        )
    if config.runtime.overwrite_existing and best_path.exists():
        shutil.rmtree(best_path)

    tokenizer = load_tokenizer(config, padding_side="right")
    response_collator = _make_response_collator(tokenizer, config)
    negative_collator = (
        PromptOnlyCollator(tokenizer, config.tokenization)
        if config.objective.num_ood_tokens == 1
        else response_collator
    )
    train_positive_loader = _make_loader(
        train_positive,
        config.sampling.in_batch_size,
        response_collator,
        shuffle=True,
        workers=config.sampling.dataloader_workers,
        seed=config.runtime.seed + expert_id,
    )
    train_negative_loader = (
        _make_loader(
            train_negative,
            config.sampling.out_batch_size,
            negative_collator,
            shuffle=True,
            workers=config.sampling.dataloader_workers,
            seed=config.runtime.seed + 1000 + expert_id,
            sampling_policy=config.sampling.pseudo_ood_distribution,
            replacement=config.sampling.replacement,
        )
        if config.objective.method == "moca"
        else None
    )
    has_complete_validation_objective = bool(validation_positive) and (
        config.objective.method != "moca" or bool(validation_negative)
    )
    validation_positive_loader = (
        _make_loader(
            validation_positive,
            config.sampling.in_batch_size,
            response_collator,
            shuffle=False,
            workers=config.sampling.dataloader_workers,
            seed=config.runtime.seed,
        )
        if has_complete_validation_objective
        else None
    )
    validation_negative_loader = (
        _make_loader(
            validation_negative,
            config.sampling.out_batch_size,
            negative_collator,
            shuffle=False,
            workers=config.sampling.dataloader_workers,
            seed=config.runtime.seed,
            sampling_policy=config.sampling.pseudo_ood_distribution,
            replacement=(config.sampling.pseudo_ood_distribution == "uniform_cluster"),
        )
        if config.objective.method == "moca" and validation_negative
        else None
    )

    base_model = load_base_model(config, for_training=True)
    model = create_lora_expert(base_model, config, expert_id)
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.optimization.learning_rate,
        betas=(
            config.optimization.adam_beta1,
            config.optimization.adam_beta2,
        ),
        eps=config.optimization.adam_epsilon,
        weight_decay=config.optimization.weight_decay,
    )
    updates_per_epoch = math.ceil(
        len(train_positive_loader) / config.optimization.gradient_accumulation_steps
    )
    total_updates = max(1, updates_per_epoch * config.optimization.max_epochs)
    warmup_steps = int(total_updates * config.optimization.warmup_ratio)
    try:
        from transformers import get_cosine_schedule_with_warmup
    except ImportError as error:
        raise RuntimeError("transformers is required") from error
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    use_gradient_scaler = (
        config.runtime.mixed_precision in {"fp16", "float16"} and model_device(model).type == "cuda"
    )
    try:
        gradient_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=use_gradient_scaler,
        )
    except (AttributeError, TypeError):
        gradient_scaler = torch.cuda.amp.GradScaler(
            enabled=use_gradient_scaler,
        )

    run_directory.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, run_directory / "resolved_config.yaml")
    manifest: dict[str, Any] = {
        "format_version": 2,
        "expert_id": expert_id,
        "method": config.objective.method,
        "config_fingerprint": config.fingerprint(),
        "training_fingerprint": config.training_fingerprint(),
        "cluster_fingerprint": cluster_fingerprint(cluster_artifacts),
        "clustered_data_fingerprint": clustered_data_fingerprint,
        "completed": False,
        "train_positive_examples": len(train_positive),
        "train_negative_examples": len(train_negative),
        "validation_positive_examples": len(validation_positive),
        "validation_negative_examples": len(validation_negative),
        "parameter_summary": trainable_parameter_summary(model),
        "package_versions": package_versions(),
        "epochs": [],
        "started_at_unix": time.time(),
    }
    atomic_write_json(manifest_path, manifest)

    device = model_device(model)
    negative_iterator = _CyclingIterator(train_negative_loader) if train_negative_loader else None
    best_metric = float("inf")
    best_epoch = 0
    stale_epochs = 0
    optimizer.zero_grad(set_to_none=True)
    global_update = 0

    for epoch in range(1, config.optimization.max_epochs + 1):
        totals: list[float] = []
        nlls: list[float] = []
        kls: list[float] = []
        target_tokens = 0
        for batch_index, positive_batch in enumerate(train_positive_loader, start=1):
            positive_batch = _to_device(positive_batch, device)
            negative_batch = (
                _to_device(negative_iterator.next(), device)
                if negative_iterator is not None
                else None
            )
            with _autocast_context(config, device):
                total, nll, kl, token_count = _compute_batch_loss(
                    model,
                    positive_batch,
                    negative_batch,
                    config,
                    tokenizer,
                )
            accumulation_start = (
                (batch_index - 1)
                // config.optimization.gradient_accumulation_steps
                * config.optimization.gradient_accumulation_steps
            )
            accumulation_group_size = min(
                config.optimization.gradient_accumulation_steps,
                len(train_positive_loader) - accumulation_start,
            )
            scaled_loss = total / accumulation_group_size
            gradient_scaler.scale(scaled_loss).backward()
            should_update = (
                batch_index % config.optimization.gradient_accumulation_steps == 0
                or batch_index == len(train_positive_loader)
            )
            if should_update:
                gradient_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, config.optimization.max_grad_norm)
                scale_before = gradient_scaler.get_scale()
                gradient_scaler.step(optimizer)
                gradient_scaler.update()
                step_was_skipped = (
                    use_gradient_scaler and gradient_scaler.get_scale() < scale_before
                )
                if not step_was_skipped:
                    scheduler.step()
                    global_update += 1
                optimizer.zero_grad(set_to_none=True)
            totals.append(float(total.detach().item()))
            nlls.append(float(nll.detach().item()))
            kls.append(float(kl.detach().item()))
            target_tokens += token_count
            if batch_index % config.runtime.log_every_steps == 0:
                LOGGER.info(
                    "expert=%d epoch=%d batch=%d/%d total=%.4f nll=%.4f kl=%.4f",
                    expert_id,
                    epoch,
                    batch_index,
                    len(train_positive_loader),
                    totals[-1],
                    nlls[-1],
                    kls[-1],
                )

        train_metrics = EpochMetrics(
            total=sum(totals) / len(totals),
            nll=sum(nlls) / len(nlls),
            kl=sum(kls) / len(kls),
            batches=len(totals),
            target_tokens=target_tokens,
        )
        validation_metrics = _run_validation(
            model,
            validation_positive_loader,
            validation_negative_loader,
            config,
            tokenizer,
        )
        monitor = (
            validation_metrics.total if validation_metrics is not None else train_metrics.total
        )
        epoch_record = {
            "epoch": epoch,
            "global_updates": global_update,
            "learning_rate": scheduler.get_last_lr()[0],
            "train": train_metrics.to_dict(),
            "validation": (validation_metrics.to_dict() if validation_metrics else None),
            "monitor": monitor,
        }
        manifest["epochs"].append(epoch_record)
        LOGGER.info(
            "expert=%d epoch=%d train=%.4f validation=%s",
            expert_id,
            epoch,
            train_metrics.total,
            f"{validation_metrics.total:.4f}" if validation_metrics else "n/a",
        )

        if monitor < best_metric:
            best_metric = monitor
            best_epoch = epoch
            stale_epochs = 0
            _save_adapter(model, adapter_checkpoint_path(config.run_dir, expert_id))
        else:
            stale_epochs += 1
        if config.optimization.save_every_epoch:
            _save_adapter(model, run_directory / f"epoch_{epoch:02d}")

        manifest["best_epoch"] = best_epoch
        manifest["best_monitor"] = best_metric
        atomic_write_json(manifest_path, manifest)
        if (
            config.optimization.early_stopping_patience > 0
            and stale_epochs >= config.optimization.early_stopping_patience
        ):
            LOGGER.info("Early stopping expert %d at epoch %d", expert_id, epoch)
            break

    manifest["finished_at_unix"] = time.time()
    manifest["best_epoch"] = best_epoch
    manifest["best_monitor"] = best_metric
    manifest["completed"] = True
    atomic_write_json(manifest_path, manifest)
    return manifest
