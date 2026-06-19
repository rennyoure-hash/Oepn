from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Node:
    name: str
    handler: Callable[..., Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    source: str
    target: str
    condition: Callable[..., bool] | None = None


@dataclass
class StateGraph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    start: str | None = None

    def add_node(self, name: str, handler: Callable[..., Any], **metadata: Any) -> None:
        self.nodes[name] = Node(name=name, handler=handler, metadata=metadata)
        if self.start is None:
            self.start = name

    def add_edge(self, source: str, target: str, condition: Callable[..., bool] | None = None) -> None:
        self.edges.append(Edge(source=source, target=target, condition=condition))

    def compile(self) -> CompiledGraph:
        return CompiledGraph(self)


@dataclass
class CompiledGraph:
    graph: StateGraph

    def invoke(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        current = self.graph.start
        state = state or {}
        visited: set[str] = set()

        while current and current not in visited:
            visited.add(current)
            node = self.graph.nodes.get(current)
            if node is None:
                break
            state = node.handler(state)
            next_node = self._route(current, state)
            current = next_node

        return state

    def _route(self, source: str, state: dict[str, Any]) -> str | None:
        candidates = [e for e in self.graph.edges if e.source == source]
        for edge in candidates:
            if edge.condition is None or edge.condition(state):
                return edge.target
        return None
