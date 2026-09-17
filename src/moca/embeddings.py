from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext

import numpy as np

from .config import EmbeddingConfig, TokenizationConfig
from .modeling import model_device, require_torch


def pool_hidden_states(hidden, attention_mask, pooling: str):
    torch = require_torch()
    mask = attention_mask.to(dtype=hidden.dtype)
    if pooling == "mean":
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1)
        return (hidden * mask.unsqueeze(-1)).sum(dim=1) / denominator
    positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
    if pooling == "last":
        indices = positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
    elif pooling == "first":
        indices = positions.masked_fill(attention_mask == 0, hidden.shape[1]).min(dim=1).values
    else:
        raise ValueError(f"Unsupported pooling: {pooling}")
    if (indices < 0).any() or (indices >= hidden.shape[1]).any():
        raise ValueError("Cannot pool an empty token sequence")
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), indices]


class BaseModelPromptEmbedder:
    """Prompt embeddings e(x) from the frozen base LLM M_phi."""

    def __init__(
        self,
        model,
        tokenizer,
        config: EmbeddingConfig,
        tokenization_config: TokenizationConfig | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.tokenization_config = tokenization_config or TokenizationConfig()

    def _hidden_states(self, encoded):
        """Run the decoder directly to avoid materializing vocabulary logits."""

        base_model = (
            self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        )
        decoder = getattr(base_model, "model", None)
        using_decoder = callable(decoder)
        forward_model = decoder if using_decoder else self.model
        needs_all_layers = self.config.layer != -1 or not using_decoder
        outputs = forward_model(
            **encoded,
            output_hidden_states=needs_all_layers,
            use_cache=False,
            return_dict=True,
        )
        if not needs_all_layers:
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is not None:
                return hidden
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states")
        return hidden_states[self.config.layer]

    def encode(self, prompts: Sequence[str]) -> np.ndarray:
        torch = require_torch()
        if not prompts:
            return np.empty((0, 0), dtype=np.float32)
        self.model.eval()
        device = model_device(self.model)
        all_embeddings: np.ndarray | None = None
        original_truncation_side = self.tokenizer.truncation_side
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.truncation_side = self.tokenization_config.prompt_truncation_side
        # Keep routing representations invariant between clustering (training
        # tokenizer) and generation (left-padded tokenizer).
        self.tokenizer.padding_side = "right"
        try:
            for start in range(0, len(prompts), self.config.batch_size):
                batch = list(prompts[start : start + self.config.batch_size])
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    add_special_tokens=(self.tokenization_config.add_special_tokens_to_prompt),
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                disable_context = (
                    self.model.disable_adapter()
                    if hasattr(self.model, "disable_adapter")
                    else nullcontext()
                )
                with torch.inference_mode(), disable_context:
                    hidden = self._hidden_states(encoded).float()
                    pooled = pool_hidden_states(
                        hidden,
                        encoded["attention_mask"],
                        self.config.pooling,
                    )
                    if self.config.normalize:
                        pooled = torch.nn.functional.normalize(pooled, dim=-1)
                numpy_dtype = np.float16 if self.config.dtype == "float16" else np.float32
                batch_embeddings = (
                    pooled.cpu()
                    .numpy()
                    .astype(
                        numpy_dtype,
                        copy=False,
                    )
                )
                if all_embeddings is None:
                    all_embeddings = np.empty(
                        (len(prompts), batch_embeddings.shape[1]),
                        dtype=numpy_dtype,
                    )
                all_embeddings[start : start + len(batch_embeddings)] = batch_embeddings
        finally:
            self.tokenizer.truncation_side = original_truncation_side
            self.tokenizer.padding_side = original_padding_side
        if all_embeddings is None:
            raise RuntimeError("Prompt embedding produced no batches")
        return all_embeddings
