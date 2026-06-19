from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Memory(ABC):
    max_entries: int = 100

    @abstractmethod
    def add(self, key: str, value: Any) -> None:
        ...

    @abstractmethod
    def get(self, key: str) -> Any | None:
        ...

    @abstractmethod
    def search(self, query: str) -> list[Any]:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...


@dataclass
class InMemory(Memory):
    _store: dict[str, list[Any]] = field(default_factory=dict)

    def add(self, key: str, value: Any) -> None:
        if key not in self._store:
            self._store[key] = []
        self._store[key].append(value)
        if len(self._store[key]) > self.max_entries:
            self._store[key] = self._store[key][-self.max_entries:]

    def get(self, key: str) -> Any | None:
        entries = self._store.get(key)
        if not entries:
            return None
        return entries[-1]

    def search(self, query: str) -> list[Any]:
        results = []
        for key, entries in self._store.items():
            if query.lower() in key.lower():
                results.extend(entries)
        return results

    def clear(self) -> None:
        self._store.clear()
