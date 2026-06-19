from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Knowledge(ABC):
    @abstractmethod
    def query(self, query: str) -> str:
        ...

    @abstractmethod
    def add(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        ...


@dataclass
class SimpleKnowledge(Knowledge):
    _entries: list[dict[str, Any]] = field(default_factory=list)

    def query(self, query: str) -> str:
        results = [e["content"] for e in self._entries if query.lower() in e["content"].lower()]
        return "\n".join(results[:5]) if results else ""

    def add(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        self._entries.append({"content": content, "metadata": metadata or {}})
