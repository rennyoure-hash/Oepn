from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator

from oepn.core.message import Message
from oepn.core.model import BaseModel, ModelResponse
from oepn.core.tool import BaseTool, ToolDefinition


@dataclass
class AgentRunResult:
    content: str | None = None
    messages: list[Message] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    tool_calls: int = 0

    @property
    def text(self) -> str:
        return self.content or ""


class BaseAgent(ABC):
    name: str
    instructions: str | None = None

    @abstractmethod
    def run(self, prompt: str, **kwargs: Any) -> AgentRunResult:
        ...

    @abstractmethod
    def run_stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        ...

    @abstractmethod
    async def arun(self, prompt: str, **kwargs: Any) -> AgentRunResult:
        ...

    @abstractmethod
    async def arun_stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        ...

    @abstractmethod
    def get_tools(self) -> list[ToolDefinition]:
        ...
