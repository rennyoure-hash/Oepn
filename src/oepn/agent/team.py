from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Iterator

from oepn.core.agent import BaseAgent, AgentRunResult
from oepn.core.message import Message, SystemMessage, UserMessage, AssistantMessage
from oepn.core.tool import ToolDefinition


class TeamMode(Enum):
    SUPERVISOR = "supervisor"
    BROADCAST = "broadcast"
    SEQUENTIAL = "sequential"


@dataclass
class Team(BaseAgent):
    name: str = "Team"
    members: list[BaseAgent] = field(default_factory=list)
    mode: TeamMode = TeamMode.SUPERVISOR
    instructions: str | None = None
    supervisor: BaseAgent | None = None

    def run(self, prompt: str, **kwargs: Any) -> AgentRunResult:
        if self.mode == TeamMode.SEQUENTIAL:
            return self._run_sequential(prompt, **kwargs)
        if self.mode == TeamMode.BROADCAST:
            return self._run_broadcast(prompt, **kwargs)
        return self._run_supervisor(prompt, **kwargs)

    def run_stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        result = self.run(prompt, **kwargs)
        if result.content:
            yield result.content

    async def arun(self, prompt: str, **kwargs: Any) -> AgentRunResult:
        if self.mode == TeamMode.SEQUENTIAL:
            return await self._arun_sequential(prompt, **kwargs)
        if self.mode == TeamMode.BROADCAST:
            return await self._arun_broadcast(prompt, **kwargs)
        return await self._arun_supervisor(prompt, **kwargs)

    async def arun_stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        result = await self.arun(prompt, **kwargs)
        if result.content:
            yield result.content

    def get_tools(self) -> list[ToolDefinition]:
        return []

    def _run_supervisor(self, prompt: str, **kwargs: Any) -> AgentRunResult:
        sup = self.supervisor or self.members[0] if self.members else None
        if not sup:
            return AgentRunResult(content="No supervisor or members configured")
        instructions = f"You are a supervisor. Delegate to team members: {[m.name for m in self.members]}\n\n"
        if self.instructions:
            instructions += self.instructions + "\n\n"
        sup.instructions = (sup.instructions or "") + "\n" + instructions
        return sup.run(prompt, **kwargs)

    def _run_sequential(self, prompt: str, **kwargs: Any) -> AgentRunResult:
        current = prompt
        for member in self.members:
            result = member.run(current, **kwargs)
            current = f"Previous step ({member.name}): {result.content}\n\nContinue: {current}"
        return AgentRunResult(content=current)

    def _run_broadcast(self, prompt: str, **kwargs: Any) -> AgentRunResult:
        results = [member.run(prompt, **kwargs) for member in self.members]
        combined = "\n\n".join(f"[{member.name}]: {r.content}" for member, r in zip(self.members, results))
        return AgentRunResult(content=combined, usage={"tool_calls": sum(r.tool_calls for r in results)})

    async def _arun_supervisor(self, prompt: str, **kwargs: Any) -> AgentRunResult:
        sup = self.supervisor or self.members[0] if self.members else None
        if not sup:
            return AgentRunResult(content="No supervisor or members configured")
        instructions = f"You are a supervisor. Delegate to team members: {[m.name for m in self.members]}\n\n"
        if self.instructions:
            instructions += self.instructions + "\n\n"
        sup.instructions = (sup.instructions or "") + "\n" + instructions
        return await sup.arun(prompt, **kwargs)

    async def _arun_sequential(self, prompt: str, **kwargs: Any) -> AgentRunResult:
        current = prompt
        for member in self.members:
            result = await member.arun(current, **kwargs)
            current = f"Previous step ({member.name}): {result.content}\n\nContinue: {current}"
        return AgentRunResult(content=current)

    async def _arun_broadcast(self, prompt: str, **kwargs: Any) -> AgentRunResult:
        import asyncio
        results = await asyncio.gather(*[member.arun(prompt, **kwargs) for member in self.members])
        combined = "\n\n".join(f"[{member.name}]: {r.content}" for member, r in zip(self.members, results))
        return AgentRunResult(content=combined, usage={"tool_calls": sum(r.tool_calls for r in results)})
