from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator

from oepn.core.agent import BaseAgent, AgentRunResult
from oepn.core.message import Message, SystemMessage, UserMessage, AssistantMessage, ToolMessage
from oepn.core.model import BaseModel, ModelResponse
from oepn.core.tool import BaseTool, ToolDefinition
from oepn.models import infer_model


@dataclass
class Agent(BaseAgent):
    name: str = "Agent"
    model: BaseModel | str | None = None
    instructions: str | None = None
    tools: list[BaseTool | ToolDefinition] = field(default_factory=list)
    max_tool_rounds: int = 10

    def __post_init__(self) -> None:
        if isinstance(self.model, str):
            self.model = infer_model(self.model)
        if self.model is None:
            self.model = infer_model("openai:gpt-4o")

    def run(self, prompt: str, **kwargs: Any) -> AgentRunResult:
        messages = self._build_messages(prompt)
        return self._run_loop(messages, **kwargs)

    def run_stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        messages = self._build_messages(prompt)
        yield from self._stream_loop(messages, **kwargs)

    async def arun(self, prompt: str, **kwargs: Any) -> AgentRunResult:
        messages = self._build_messages(prompt)
        return await self._arun_loop(messages, **kwargs)

    async def arun_stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        messages = self._build_messages(prompt)
        async for chunk in self._astream_loop(messages, **kwargs):
            yield chunk

    def get_tools(self) -> list[ToolDefinition]:
        result: list[ToolDefinition] = []
        for t in self.tools:
            if isinstance(t, ToolDefinition):
                result.append(t)
            elif isinstance(t, BaseTool):
                result.append(t.to_definition())
        return result

    def _build_messages(self, prompt: str) -> list[Message]:
        messages: list[Message] = []
        if self.instructions:
            messages.append(SystemMessage(content=self.instructions))
        messages.append(UserMessage(content=prompt))
        return messages

    def _run_loop(self, messages: list[Message], **kwargs: Any) -> AgentRunResult:
        model = self.model
        assert model is not None
        tool_defs = self.get_tools()
        tools_kwargs = self._build_tools_kwargs(tool_defs) if tool_defs else {}
        tool_count = 0

        for _ in range(self.max_tool_rounds):
            response = model.invoke(messages, **tools_kwargs, **kwargs)
            messages.append(AssistantMessage(content=response.content, tool_calls=response.tool_calls))

            if not response.tool_calls:
                return AgentRunResult(
                    content=response.content,
                    messages=messages,
                    usage=response.usage or {},
                    tool_calls=tool_count,
                )

            tool_results = self._execute_tool_calls(response.tool_calls, tool_defs)
            for tc, result in zip(response.tool_calls, tool_results):
                messages.append(ToolMessage(content=json.dumps(result) if not isinstance(result, str) else result, tool_call_id=tc.get("id", "")))
            tool_count += len(response.tool_calls)

        final = model.invoke(messages)
        return AgentRunResult(content=final.content, messages=messages, usage=final.usage or {}, tool_calls=tool_count)

    def _stream_loop(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        model = self.model
        assert model is not None
        tool_defs = self.get_tools()
        tools_kwargs = self._build_tools_kwargs(tool_defs) if tool_defs else {}

        full_content = ""
        for _ in range(self.max_tool_rounds):
            for chunk in model.invoke_stream(messages, **tools_kwargs, **kwargs):
                if chunk.content:
                    full_content += chunk.content
                    yield chunk.content
            messages.append(AssistantMessage(content=full_content))
            break

    async def _arun_loop(self, messages: list[Message], **kwargs: Any) -> AgentRunResult:
        model = self.model
        assert model is not None
        tool_defs = self.get_tools()
        tools_kwargs = self._build_tools_kwargs(tool_defs) if tool_defs else {}
        tool_count = 0

        for _ in range(self.max_tool_rounds):
            response = await model.ainvoke(messages, **tools_kwargs, **kwargs)
            messages.append(AssistantMessage(content=response.content, tool_calls=response.tool_calls))
            if not response.tool_calls:
                return AgentRunResult(content=response.content, messages=messages, usage=response.usage or {}, tool_calls=tool_count)
            tool_results = await self._aexecute_tool_calls(response.tool_calls, tool_defs)
            for tc, result in zip(response.tool_calls, tool_results):
                messages.append(ToolMessage(content=json.dumps(result) if not isinstance(result, str) else result, tool_call_id=tc.get("id", "")))
            tool_count += len(response.tool_calls)

        final = await model.ainvoke(messages)
        return AgentRunResult(content=final.content, messages=messages, usage=final.usage or {}, tool_calls=tool_count)

    async def _astream_loop(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[str]:
        model = self.model
        assert model is not None
        tool_defs = self.get_tools()
        tools_kwargs = self._build_tools_kwargs(tool_defs) if tool_defs else {}
        async for chunk in model.ainvoke_stream(messages, **tools_kwargs, **kwargs):
            if chunk.content:
                yield chunk.content

    def _build_tools_kwargs(self, tool_defs: list[ToolDefinition]) -> dict[str, Any]:
        tools = []
        for td in tool_defs:
            tools.append({"type": "function", "function": {"name": td.name, "description": td.description, "parameters": td.parameters}})
        return {"tools": tools} if tools else {}

    def _execute_tool_calls(self, tool_calls: list[dict[str, Any]], tool_defs: list[ToolDefinition]) -> list[Any]:
        def_map = {td.name: td for td in tool_defs}
        results = []
        for tc in tool_calls:
            fn_info = tc.get("function", {})
            name = fn_info.get("name", "")
            args = json.loads(fn_info.get("arguments", "{}"))
            td = def_map.get(name)
            if td and td.fn:
                results.append(td.fn(**args))
            else:
                results.append(f"Tool '{name}' not found")
        return results

    async def _aexecute_tool_calls(self, tool_calls: list[dict[str, Any]], tool_defs: list[ToolDefinition]) -> list[Any]:
        import asyncio

        def_map = {td.name: td for td in tool_defs}
        async def execute_one(tc: dict[str, Any]) -> Any:
            fn_info = tc.get("function", {})
            name = fn_info.get("name", "")
            args = json.loads(fn_info.get("arguments", "{}"))
            td = def_map.get(name)
            if td:
                if td.async_fn:
                    return await td.async_fn(**args)
                return td.fn(**args) if td.fn else f"Tool '{name}' not found"
            return f"Tool '{name}' not found"
        return await asyncio.gather(*[execute_one(tc) for tc in tool_calls])
