from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PromptResponse:
    example_id: str
    prompt: str
    response: str
    references: tuple[str, ...]
    split: str
    group_id: str
    cluster_id: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> PromptResponse:
        payload = dict(values)
        payload["references"] = tuple(payload.get("references") or [payload["response"]])
        return cls(**payload)
