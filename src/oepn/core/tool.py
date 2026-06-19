from __future__ import annotations
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints

from pydantic import BaseModel, create_model


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any] | None = None
    async_fn: Callable[..., Any] | None = None


class BaseTool(ABC):
    name: str
    description: str

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        ...

    @abstractmethod
    async def arun(self, **kwargs: Any) -> Any:
        ...

    def to_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=_get_parameters_schema(self.run),
            fn=self.run,
            async_fn=self.arun,
        )


class Tool(BaseTool):
    def __init__(
        self,
        name: str,
        description: str,
        fn: Callable[..., Any],
        async_fn: Callable[..., Any] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self._fn = fn
        self._async_fn = async_fn

    def run(self, **kwargs: Any) -> Any:
        return self._fn(**kwargs)

    async def arun(self, **kwargs: Any) -> Any:
        if self._async_fn:
            return await self._async_fn(**kwargs)
        return self._fn(**kwargs)


def tool(
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[..., Any]], Tool]:
    def decorator(fn: Callable[..., Any]) -> Tool:
        return Tool(
            name=name or fn.__name__,
            description=description or (fn.__doc__ or "").strip(),
            fn=fn,
        )

    return decorator


def _get_parameters_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)
    fields: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name in ("self", "cls", "kwargs", "args"):
            continue
        annotation = hints.get(name, str)
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[name] = (annotation, default)

    if fields:
        model = create_model("ToolArgs", **fields)
        schema = model.model_json_schema()
        return {"type": "object", "properties": schema.get("properties", {}), "required": schema.get("required", [])}
    return {"type": "object", "properties": {}}
