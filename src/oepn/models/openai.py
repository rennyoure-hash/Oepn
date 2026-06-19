from __future__ import annotations
import json
from typing import Any, AsyncIterator, Iterator

from openai import OpenAI, AsyncOpenAI

from oepn.core.message import Message
from oepn.core.model import BaseModel, ModelResponse


class OpenAIModel(BaseModel):
    provider = "openai"

    def __init__(self, model_id: str = "gpt-4o", base_url: str | None = None) -> None:
        self.model_id = model_id
        self._client = OpenAI(base_url=base_url) if base_url else OpenAI()
        self._async_client = AsyncOpenAI(base_url=base_url) if base_url else AsyncOpenAI()

    def invoke(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        raw = self._client.chat.completions.create(
            model=self.model_id,
            messages=[self._to_dict(m) for m in messages],
            **kwargs,
        )
        return self._parse(raw)

    def invoke_stream(self, messages: list[Message], **kwargs: Any) -> Iterator[ModelResponse]:
        stream = self._client.chat.completions.create(
            model=self.model_id,
            messages=[self._to_dict(m) for m in messages],
            stream=True,
            **kwargs,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta:
                yield ModelResponse(
                    content=delta.content,
                    tool_calls=[t.model_dump() for t in delta.tool_calls] if delta.tool_calls else None,
                )

    async def ainvoke(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        raw = await self._async_client.chat.completions.create(
            model=self.model_id,
            messages=[self._to_dict(m) for m in messages],
            **kwargs,
        )
        return self._parse(raw)

    async def ainvoke_stream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[ModelResponse]:
        stream = await self._async_client.chat.completions.create(
            model=self.model_id,
            messages=[self._to_dict(m) for m in messages],
            stream=True,
            **kwargs,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta:
                yield ModelResponse(
                    content=delta.content,
                    tool_calls=[t.model_dump() for t in delta.tool_calls] if delta.tool_calls else None,
                )

    def count_tokens(self, messages: list[Message]) -> int:
        return sum(len(m.content or "") for m in messages)

    def _parse(self, raw: Any) -> ModelResponse:
        choice = raw.choices[0] if raw.choices else None
        if not choice:
            return ModelResponse()
        msg = choice.message
        return ModelResponse(
            content=msg.content,
            tool_calls=[t.model_dump() for t in msg.tool_calls] if msg.tool_calls else None,
            usage={"prompt_tokens": raw.usage.prompt_tokens, "completion_tokens": raw.usage.completion_tokens} if raw.usage else None,
            finish_reason=choice.finish_reason,
        )

    def _to_dict(self, m: Message) -> dict[str, Any]:
        d: dict[str, Any] = {"role": m.role}
        if m.content is not None:
            d["content"] = m.content
        if m.tool_calls is not None:
            d["tool_calls"] = m.tool_calls
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        if m.name:
            d["name"] = m.name
        return d
