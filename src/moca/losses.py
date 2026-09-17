from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .modeling import require_torch


@dataclass
class LossBreakdown:
    total: object
    in_cluster_nll: object
    cross_cluster_kl: object
    num_examples: int
    num_target_tokens: int


def sequence_nll(logits, labels, reduction: str = "sequence_sum_then_batch_mean"):
    """Response-only causal NLL.

    Labels must have -100 at prompt and padding positions. The literal paper
    expectation is a mean over per-example *summed* sequence NLLs.
    """

    torch = require_torch()
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("Expected logits [B,L,V] and labels [B,L]")
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    flat = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(shift_labels.shape)
    valid = shift_labels.ne(-100)
    token_counts = valid.sum(dim=1)
    if (token_counts == 0).any():
        raise ValueError("Every in-cluster example must contain a target token")
    sequence_sums = (flat * valid).sum(dim=1)
    if reduction == "sequence_sum_then_batch_mean":
        loss = sequence_sums.mean()
    elif reduction == "token_mean":
        loss = sequence_sums.sum() / token_counts.sum()
    else:
        raise ValueError(f"Unsupported sequence NLL reduction: {reduction}")
    return loss, int(valid.sum().item())


def _allowed_mask(logits, excluded_token_ids: Sequence[int] | None):
    torch = require_torch()
    if not excluded_token_ids:
        return torch.ones(logits.shape[-1], device=logits.device, dtype=torch.bool)
    allowed = torch.ones(logits.shape[-1], device=logits.device, dtype=torch.bool)
    valid_ids = [token_id for token_id in excluded_token_ids if 0 <= token_id < logits.shape[-1]]
    if valid_ids:
        allowed[torch.tensor(valid_ids, device=logits.device)] = False
    if not allowed.any():
        raise ValueError("No vocabulary entries remain in the uniform support")
    return allowed


def kl_to_uniform(
    logits,
    direction: str = "model_to_uniform",
    excluded_token_ids: Sequence[int] | None = None,
):
    """Compute KL on the final dimension in float32.

    Paper direction: D_KL(p || U) = log(V) - H(p).
    """

    allowed = _allowed_mask(logits, excluded_token_ids)
    selected = logits.float()[..., allowed]
    log_p = selected.log_softmax(dim=-1)
    support_size = selected.shape[-1]
    if direction == "model_to_uniform":
        p = log_p.exp()
        return (p * log_p).sum(dim=-1) + math.log(support_size)
    if direction == "uniform_to_model":
        return -math.log(support_size) - log_p.mean(dim=-1)
    raise ValueError(f"Unsupported KL direction: {direction}")


def logits_after_last_prompt_token(logits, attention_mask):
    torch = require_torch()
    if logits.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("Expected logits [B,L,V] and attention mask [B,L]")
    positions = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
    last_indices = positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
    if (last_indices < 0).any():
        raise ValueError("OOD prompts must not be empty")
    return logits[
        torch.arange(logits.shape[0], device=logits.device),
        last_indices,
    ]


def first_token_uniform_kl(
    logits,
    attention_mask,
    direction: str = "model_to_uniform",
    excluded_token_ids: Sequence[int] | None = None,
):
    first_logits = logits_after_last_prompt_token(logits, attention_mask)
    return kl_to_uniform(
        first_logits,
        direction=direction,
        excluded_token_ids=excluded_token_ids,
    ).mean()


def response_prefix_uniform_kl(
    logits,
    labels,
    num_tokens: int,
    direction: str = "model_to_uniform",
    excluded_token_ids: Sequence[int] | None = None,
):
    """Non-paper first-m-token ablation using gold-prefix conditioning."""

    torch = require_torch()
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    per_example = []
    for row in range(shift_labels.shape[0]):
        target_positions = torch.nonzero(shift_labels[row].ne(-100), as_tuple=False).flatten()[
            :num_tokens
        ]
        if target_positions.numel() == 0:
            raise ValueError("OOD response prefix has no target tokens")
        values = kl_to_uniform(
            shift_logits[row, target_positions],
            direction=direction,
            excluded_token_ids=excluded_token_ids,
        )
        per_example.append(values.mean())
    return torch.stack(per_example).mean()
