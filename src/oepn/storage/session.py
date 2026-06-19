from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from oepn.core.message import Message


@dataclass
class Session:
    session_id: str
    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class SessionStore:
    _sessions: dict[str, Session] = {}

    def save(self, session: Session) -> None:
        self._sessions[session.session_id] = session

    def load(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def list(self) -> list[Session]:
        return list(self._sessions.values())
