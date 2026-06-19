from __future__ import annotations
from typing import Any, AsyncIterator, Iterator

from anthropic import Anthropic, AsyncAnthropic

from oepn.core.message import Message
from oepn.core.model import BaseModel, ModelResponse


class AnthropicModel(BaseModel):
    provider = "anthropic"

    def __init__(self, model_id: str = "claude-sonnet-4-20250514") -> None:
        self.model_id = model_id
        self._client = Anthropic()
        self._async_client = AsyncAnthropic()

    def invoke(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        system, msgs = self._split_system(messages)
        raw = self._client.messages.create(
            model=self.model_id,
            system=system or [],
            messages=msgs,
            max_tokens=kwargs.get("max_tokens", 4096),
        )
        return self._parse(raw)

    def invoke_stream(self, messages: list[Message], **kwargs: Any) -> Iterator[ModelResponse]:
        system, msgs = self._split_system(messages)
        with self._client.messages.stream(
            model=self.model_id,
            system=system or [],
            messages=msgs,
            max_tokens=kwargs.get("max_tokens", 4096),
        ) as stream:
            for text in stream.text_stream:
                yield ModelResponse(content=text)

    async def ainvoke(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        system, msgs = self._split_system(messages)
        raw = await self._async_client.messages.create(
            model=self.model_id,
            system=system or [],
            messages=msgs,
            max_tokens=kwargs.get("max_tokens", 4096),
        )
        return self._parse(raw)

    async def ainvoke_stream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[ModelResponse]:
        system, msgs = self._split_system(messages)
        async with self._async_client.messages.stream(
            model=self.model_id,
            system=system or [],
            messages=msgs,
            max_tokens=kwargs.get("max_tokens", 4096),
        ) as stream:
            async for text in stream.text_stream:
                yield ModelResponse(content=text)

    def count_tokens(self, messages: list[Message]) -> int:
        return sum(len(m.content or "") for m in messages)

    def _split_system(self, messages: list[Message]) -> tuple[list[dict], list[dict]]:
        system = []
        msgs = []
        for m in messages:
            d = {"role": "user" if m.role == "user" else "assistant", "content": m.content or ""}
            if m.role == "system":
                system.append({"type": "text", "text": m.content or ""})
            else:
                msgs.append(d)
        return system, msgs

    def _parse(self, raw: Any) -> ModelResponse:
        text = ""
        for block in raw.content:
            if hasattr(block, "text"):
                text += block.text
        return ModelResponse(
            content=text or None,
            usage={"input_tokens": raw.usage.input_tokens, "output_tokens": raw.usage.output_tokens} if raw.usage else None,
        )
