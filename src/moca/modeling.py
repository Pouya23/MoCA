from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import read_json
from .config import ExperimentConfig


def require_torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required for model training/inference. Run scripts/setup.sh."
        ) from error
    return torch


def resolve_torch_dtype(name: str):
    torch = require_torch()
    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return aliases[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported torch dtype: {name}") from error


def load_tokenizer(config: ExperimentConfig, padding_side: str = "right"):
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("transformers is required; run scripts/setup.sh") from error

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path,
        revision=config.model.tokenizer_revision or config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def load_base_model(config: ExperimentConfig, *, for_training: bool):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise RuntimeError("transformers is required; run scripts/setup.sh") from error

    kwargs: dict[str, Any] = {
        "revision": config.model.revision,
        "trust_remote_code": config.model.trust_remote_code,
        "torch_dtype": resolve_torch_dtype(config.model.torch_dtype),
        "low_cpu_mem_usage": True,
    }
    if config.model.attn_implementation:
        kwargs["attn_implementation"] = config.model.attn_implementation
    if config.runtime.device == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(config.model.name_or_path, **kwargs)
    model.config.use_cache = False if for_training else config.model.use_cache
    model.requires_grad_(False)
    device = config.runtime.device
    if device != "auto":
        model.to(device)
    return model


def create_lora_expert(base_model, config: ExperimentConfig, expert_id: int):
    try:
        from peft import LoraConfig as PeftLoraConfig
        from peft import TaskType, get_peft_model
    except ImportError as error:
        raise RuntimeError("peft is required; run scripts/setup.sh") from error

    # Every expert is trained in an independent process. Saving the training
    # adapter under PEFT's canonical "default" name keeps adapter_config.json at
    # the checkpoint root. At routed inference it is loaded under expert_<id>.
    adapter_name = "default"
    peft_config = PeftLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=list(config.lora.target_modules),
        bias=config.lora.bias,
        init_lora_weights=config.lora.init_lora_weights,
    )
    model = get_peft_model(base_model, peft_config, adapter_name=adapter_name)
    model.set_adapter(adapter_name)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters were created")
    invalid = [name for name in trainable if "lora_" not in name and "modules_to_save" not in name]
    if invalid:
        raise RuntimeError(f"Unexpected trainable base parameters: {invalid[:5]}")
    return model


def adapter_checkpoint_path(run_dir: Path, expert_id: int) -> Path:
    return run_dir / "adapters" / f"expert_{expert_id}" / "best"


def load_all_experts(
    config: ExperimentConfig,
    num_experts: int,
    *,
    expected_cluster_fingerprint: str,
    expected_data_fingerprint: str,
):
    try:
        from peft import PeftModel
    except ImportError as error:
        raise RuntimeError("peft is required; run scripts/setup.sh") from error

    reference_versions: dict[str, str] | None = None
    for expert_id in range(num_experts):
        manifest_path = (
            config.run_dir / "adapters" / f"expert_{expert_id}" / "training_manifest.json"
        )
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing expert manifest: {manifest_path}")
        manifest = read_json(manifest_path)
        if int(manifest.get("expert_id", -1)) != expert_id:
            raise ValueError(f"Expert manifest ID mismatch: {manifest_path}")
        if manifest.get("method") != config.objective.method:
            raise ValueError(
                f"Expert {expert_id} was trained with method "
                f"{manifest.get('method')!r}, not {config.objective.method!r}"
            )
        if manifest.get("completed") is not True:
            raise ValueError(f"Expert {expert_id} training is incomplete: {manifest_path}")
        if manifest.get("training_fingerprint") != config.training_fingerprint():
            raise ValueError(
                f"Expert {expert_id} was trained with a different training-relevant configuration"
            )
        if manifest.get("cluster_fingerprint") != expected_cluster_fingerprint:
            raise ValueError(f"Expert {expert_id} was trained against different router artifacts")
        if manifest.get("clustered_data_fingerprint") != expected_data_fingerprint:
            raise ValueError(
                f"Expert {expert_id} was trained against different clustered train/validation data"
            )
        versions = dict(manifest.get("package_versions") or {})
        if reference_versions is None:
            reference_versions = versions
        elif versions != reference_versions:
            raise ValueError("Expert manifests report different training package versions")

    base_model = load_base_model(config, for_training=False)
    first_path = adapter_checkpoint_path(config.run_dir, 0)
    if not first_path.exists():
        raise FileNotFoundError(f"Missing expert checkpoint: {first_path}")
    model = PeftModel.from_pretrained(
        base_model,
        first_path,
        adapter_name="expert_0",
        is_trainable=False,
    )
    for expert_id in range(1, num_experts):
        path = adapter_checkpoint_path(config.run_dir, expert_id)
        if not path.exists():
            raise FileNotFoundError(f"Missing expert checkpoint: {path}")
        model.load_adapter(path, adapter_name=f"expert_{expert_id}", is_trainable=False)
    model.eval()
    return model


def model_device(model):
    torch = require_torch()
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def trainable_parameter_summary(model) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / total if total else 0.0,
    }


def special_token_ids(tokenizer) -> tuple[int, ...]:
    values: list[int | None] = list(getattr(tokenizer, "all_special_ids", None) or [])
    values.extend(
        [
            getattr(tokenizer, "bos_token_id", None),
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "pad_token_id", None),
            getattr(tokenizer, "unk_token_id", None),
        ]
    )
    return tuple(sorted({int(value) for value in values if value is not None}))
