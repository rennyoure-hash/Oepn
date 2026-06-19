from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from oepn.core.message import Message
from oepn.core.tool import ToolDefinition
from oepn.core.model import ModelResponse


class Capability(ABC):
    name: str

    @abstractmethod
    def before_run(self, prompt: str) -> str:
        return prompt

    @abstractmethod
    def after_run(self, result: Any) -> Any:
        return result

    @abstractmethod
    def before_tool_call(self, tool: ToolDefinition, args: dict[str, Any]) -> dict[str, Any]:
        return args

    @abstractmethod
    def after_tool_call(self, tool: ToolDefinition, result: Any) -> Any:
        return result

    @abstractmethod
    def before_model_request(self, messages: list[Message]) -> list[Message]:
        return messages

    @abstractmethod
    def after_model_request(self, response: ModelResponse) -> ModelResponse:
        return response


@dataclass
class HookCapability(Capability):
    name: str = "hooks"
    pre_run: list[Any] = field(default_factory=list)
    post_run: list[Any] = field(default_factory=list)

    def before_run(self, prompt: str) -> str:
        for hook in self.pre_run:
            result = hook(prompt) if callable(hook) else prompt
            if isinstance(result, str):
                prompt = result
        return prompt

    def after_run(self, result: Any) -> Any:
        for hook in self.post_run:
            result = hook(result) if callable(hook) else result
        return result

    def before_tool_call(self, tool: ToolDefinition, args: dict[str, Any]) -> dict[str, Any]:
        return args

    def after_tool_call(self, tool: ToolDefinition, result: Any) -> Any:
        return result

    def before_model_request(self, messages: list[Message]) -> list[Message]:
        return messages

    def after_model_request(self, response: ModelResponse) -> ModelResponse:
        return response
