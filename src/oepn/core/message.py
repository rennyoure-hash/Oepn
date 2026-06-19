from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    role: str
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMessage(Message):
    role: str = "system"


@dataclass
class UserMessage(Message):
    role: str = "user"


@dataclass
class AssistantMessage(Message):
    role: str = "assistant"


@dataclass
class ToolMessage(Message):
    role: str = "tool"
    tool_call_id: str = ""
