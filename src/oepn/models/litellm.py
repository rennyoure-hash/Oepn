from __future__ import annotations
from typing import Any, AsyncIterator, Iterator

from oepn.core.message import Message
from oepn.core.model import BaseModel, ModelResponse


class LiteLLMModel(BaseModel):
    provider = "litellm"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def invoke(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        from litellm import completion

        raw = completion(
            model=self.model_id,
            messages=[{"role": m.role, "content": m.content or ""} for m in messages],
            **kwargs,
        )
        choice = raw.choices[0] if raw.choices else None
        if not choice:
            return ModelResponse()
        return ModelResponse(
            content=choice.message.content,
            tool_calls=[t.model_dump() for t in choice.message.tool_calls] if choice.message.tool_calls else None,
            usage={"prompt_tokens": raw.usage.prompt_tokens, "completion_tokens": raw.usage.completion_tokens} if raw.usage else None,
        )

    def invoke_stream(self, messages: list[Message], **kwargs: Any) -> Iterator[ModelResponse]:
        from litellm import completion

        stream = completion(
            model=self.model_id,
            messages=[{"role": m.role, "content": m.content or ""} for m in messages],
            stream=True,
            **kwargs,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield ModelResponse(content=chunk.choices[0].delta.content)

    async def ainvoke(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        return self.invoke(messages, **kwargs)

    async def ainvoke_stream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[ModelResponse]:
        for r in self.invoke_stream(messages, **kwargs):
            yield r

    def count_tokens(self, messages: list[Message]) -> int:
        return sum(len(m.content or "") for m in messages)
