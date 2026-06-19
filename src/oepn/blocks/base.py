from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator

from oepn.models import infer_model
from oepn.core.message import Message


class BaseBlock(ABC):
    name: str
    description: str = ""

    @abstractmethod
    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        ...

    @abstractmethod
    async def aexecute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass
class InputBlock(BaseBlock):
    name: str = "input"
    description: str = "Graph input"

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return inputs

    async def aexecute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return inputs


@dataclass
class OutputBlock(BaseBlock):
    name: str = "output"
    description: str = "Graph output"

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"output": inputs}

    async def aexecute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"output": inputs}


@dataclass
class LLMBlock(BaseBlock):
    name: str = "llm"
    description: str = "LLM call"
    model: str = "openai:gpt-4o"
    system_prompt: str = ""

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        model = infer_model(self.model)
        prompt = inputs.get("prompt", "")
        messages = []
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))
        messages.append(Message(role="user", content=prompt))
        response = model.invoke(messages)
        return {"response": response.content, **inputs}

    async def aexecute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        model = infer_model(self.model)
        prompt = inputs.get("prompt", "")
        messages = []
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))
        messages.append(Message(role="user", content=prompt))
        response = await model.ainvoke(messages)
        return {"response": response.content, **inputs}
