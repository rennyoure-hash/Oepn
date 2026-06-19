from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class WorkflowStep:
    name: str
    fn: Callable[[dict[str, Any]], dict[str, Any]]
    depends_on: list[str] = field(default_factory=list)


@dataclass
class Workflow:
    steps: dict[str, WorkflowStep] = field(default_factory=dict)

    def add_step(self, name: str, fn: Callable[[dict[str, Any]], dict[str, Any]], depends_on: list[str] | None = None) -> None:
        self.steps[name] = WorkflowStep(name=name, fn=fn, depends_on=depends_on or [])

    def run(self, initial: dict[str, Any] | None = None) -> dict[str, Any]:
        state = initial or {}
        executed: set[str] = set()

        while len(executed) < len(self.steps):
            for name, step in self.steps.items():
                if name in executed:
                    continue
                if all(dep in executed for dep in step.depends_on):
                    result = step.fn(state)
                    if isinstance(result, dict):
                        state.update(result)
                    executed.add(name)

        return state
