from __future__ import annotations
from typing import Any, AsyncIterator, Iterator

import google.generativeai as genai

from oepn.core.message import Message
from oepn.core.model import BaseModel, ModelResponse


class GeminiModel(BaseModel):
    provider = "google"

    def __init__(self, model_id: str = "gemini-2.0-flash") -> None:
        self.model_id = model_id
        self._client = genai.GenerativeModel(model_id)

    def invoke(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        history, prompt = self._build_history(messages)
        chat = self._client.start_chat(history=history)
        response = chat.send_message(prompt, **kwargs)
        return ModelResponse(
            content=response.text,
            usage={"prompt_tokens": 0, "completion_tokens": 0},
        )

    def invoke_stream(self, messages: list[Message], **kwargs: Any) -> Iterator[ModelResponse]:
        history, prompt = self._build_history(messages)
        chat = self._client.start_chat(history=history)
        response = chat.send_message(prompt, stream=True, **kwargs)
        for chunk in response:
            if chunk.text:
                yield ModelResponse(content=chunk.text)

    async def ainvoke(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        return self.invoke(messages, **kwargs)

    async def ainvoke_stream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[ModelResponse]:
        for r in self.invoke_stream(messages, **kwargs):
            yield r

    def count_tokens(self, messages: list[Message]) -> int:
        return sum(len(m.content or "") for m in messages)

    def _build_history(self, messages: list[Message]) -> tuple[list[dict], str]:
        history = []
        prompt = ""
        for m in messages:
            role = "user" if m.role in ("user", "system") else "model"
            content = m.content or ""
            if m.role == "system":
                prompt = f"[System: {content}]\n"
            elif m is messages[-1]:
                prompt += content
            else:
                history.append({"role": role, "parts": [content]})
        return history, prompt
