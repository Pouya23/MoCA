from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..config import TokenizationConfig
from ..records import PromptResponse

IGNORE_INDEX = -100


class ResponseOnlyCausalLMCollator:
    """Tokenize prompt/response pairs while supervising only response tokens.

    PyTorch is intentionally imported inside ``__call__`` so data preparation,
    prompt inspection, and unit tests do not require the training stack.
    """

    def __init__(
        self,
        tokenizer: Any,
        config: TokenizationConfig | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config or TokenizationConfig()
        if self.config.max_prompt_tokens <= 0:
            raise ValueError("max_prompt_tokens must be positive")
        if self.config.max_response_tokens <= 0:
            raise ValueError("max_response_tokens must be positive")
        if self.config.prompt_truncation_side not in {"left", "right"}:
            raise ValueError("prompt_truncation_side must be left or right")
        if self.config.pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of must be positive")

    def _encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        if hasattr(self.tokenizer, "encode"):
            encoded = self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
            )
        else:
            encoded = self.tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                truncation=False,
            )
        if isinstance(encoded, Mapping):
            encoded = encoded["input_ids"]
        if encoded and isinstance(encoded[0], Sequence):
            if len(encoded) != 1:
                raise ValueError("Tokenizer returned a batched encoding for one string")
            encoded = encoded[0]
        return [int(token) for token in encoded]

    def _truncate_prompt(self, token_ids: list[int]) -> list[int]:
        limit = self.config.max_prompt_tokens
        if len(token_ids) <= limit:
            return token_ids
        if self.config.prompt_truncation_side == "left":
            return token_ids[-limit:]
        return token_ids[:limit]

    def _prompt_tokens(self, prompt: str) -> list[int]:
        """Match the tokenizer's normal special-token/truncation behavior."""

        if callable(self.tokenizer):
            original_side = getattr(self.tokenizer, "truncation_side", "right")
            self.tokenizer.truncation_side = self.config.prompt_truncation_side
            try:
                encoded = self.tokenizer(
                    prompt,
                    add_special_tokens=self.config.add_special_tokens_to_prompt,
                    truncation=True,
                    max_length=self.config.max_prompt_tokens,
                )
            finally:
                self.tokenizer.truncation_side = original_side
            if isinstance(encoded, Mapping):
                encoded = encoded["input_ids"]
            if encoded and isinstance(encoded[0], Sequence):
                if len(encoded) != 1:
                    raise ValueError("Tokenizer returned a batched encoding for one prompt")
                encoded = encoded[0]
            return [int(token) for token in encoded]
        return self._truncate_prompt(
            self._encode(
                prompt,
                add_special_tokens=self.config.add_special_tokens_to_prompt,
            )
        )

    def _response_tokens(self, response: str) -> tuple[list[int], int | None]:
        response_ids = self._encode(f"{self.config.response_prefix}{response}")
        eos_position: int | None = None
        if self.config.append_eos:
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_token_id is None:
                raise ValueError("append_eos=True requires tokenizer.eos_token_id")
            content_limit = self.config.max_response_tokens - 1
            response_ids = response_ids[:content_limit]
            response_ids.append(int(eos_token_id))
            eos_position = len(response_ids) - 1
        else:
            response_ids = response_ids[: self.config.max_response_tokens]
        return response_ids, eos_position

    @staticmethod
    def _fields(item: PromptResponse | Mapping[str, Any]) -> tuple[str, str]:
        if isinstance(item, PromptResponse):
            return item.prompt, item.response
        try:
            return str(item["prompt"]), str(item["response"])
        except KeyError as error:
            raise ValueError("Each batch item requires prompt and response") from error

    def _padding_token_id(self) -> int:
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise ValueError("Tokenizer requires pad_token_id or eos_token_id")
        return int(pad_token_id)

    def __call__(
        self,
        items: Sequence[PromptResponse | Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not items:
            raise ValueError("Cannot collate an empty batch")
        try:
            import torch
        except ImportError as error:
            raise ImportError("Collation requires PyTorch (`torch`)") from error

        encoded_rows: list[tuple[list[int], list[int], list[int]]] = []
        for item in items:
            prompt, response = self._fields(item)
            prompt_ids = self._prompt_tokens(prompt)
            response_ids, eos_position = self._response_tokens(response)
            input_ids = [*prompt_ids, *response_ids]
            labels = [IGNORE_INDEX] * len(prompt_ids) + list(response_ids)
            if eos_position is not None and not self.config.include_eos_in_nll:
                labels[len(prompt_ids) + eos_position] = IGNORE_INDEX
            attention_mask = [1] * len(input_ids)
            encoded_rows.append((input_ids, attention_mask, labels))

        longest = max(len(row[0]) for row in encoded_rows)
        multiple = self.config.pad_to_multiple_of
        padded_length = ((longest + multiple - 1) // multiple) * multiple
        pad_token_id = self._padding_token_id()
        padding_side = getattr(self.tokenizer, "padding_side", "right")
        if padding_side not in {"left", "right"}:
            raise ValueError("tokenizer.padding_side must be left or right")

        input_batch: list[list[int]] = []
        mask_batch: list[list[int]] = []
        label_batch: list[list[int]] = []
        for input_ids, attention_mask, labels in encoded_rows:
            padding = padded_length - len(input_ids)
            if padding_side == "left":
                input_batch.append([pad_token_id] * padding + input_ids)
                mask_batch.append([0] * padding + attention_mask)
                label_batch.append([IGNORE_INDEX] * padding + labels)
            else:
                input_batch.append(input_ids + [pad_token_id] * padding)
                mask_batch.append(attention_mask + [0] * padding)
                label_batch.append(labels + [IGNORE_INDEX] * padding)

        return {
            "input_ids": torch.tensor(input_batch, dtype=torch.long),
            "attention_mask": torch.tensor(mask_batch, dtype=torch.long),
            "labels": torch.tensor(label_batch, dtype=torch.long),
        }


ResponseOnlyCollator = ResponseOnlyCausalLMCollator
