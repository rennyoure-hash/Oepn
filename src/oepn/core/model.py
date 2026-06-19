from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator

from oepn.core.message import Message


@dataclass
class ModelResponse:
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    usage: dict[str, int] | None = None
    finish_reason: str | None = None


class BaseModel(ABC):
    model_id: str
    provider: str

    @abstractmethod
    def invoke(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        ...

    @abstractmethod
    def invoke_stream(self, messages: list[Message], **kwargs: Any) -> Iterator[ModelResponse]:
        ...

    @abstractmethod
    async def ainvoke(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        ...

    @abstractmethod
    async def ainvoke_stream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[ModelResponse]:
        ...

    @abstractmethod
    def count_tokens(self, messages: list[Message]) -> int:
        ...

    def supports_tool_calling(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True
