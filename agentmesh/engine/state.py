"""Deterministic state models and immutable event logs for AgentMesh."""

from __future__ import annotations

import hashlib
import json
import time
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EventType(str, Enum):
    WORKFLOW_START = "WORKFLOW_START"
    NODE_START = "NODE_START"
    NODE_COMPLETE = "NODE_COMPLETE"
    NODE_FAILED = "NODE_FAILED"
    TOOL_CALL = "TOOL_CALL"
    CHECKPOINT = "CHECKPOINT"
    WORKFLOW_COMPLETE = "WORKFLOW_COMPLETE"


class Event(BaseModel):
    """Immutable event record in the event stream."""

    event_id: str
    event_type: EventType
    timestamp: float = Field(default_factory=time.time)
    node_id: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    state_hash: Optional[str] = None

    def calculate_hash(self, prev_hash: str = "") -> str:
        content = f"{prev_hash}|{self.event_id}|{self.event_type}|{self.timestamp}|{json.dumps(self.payload, sort_keys=True)}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


class State(BaseModel):
    """Workflow state container with deterministic hashing and event tracking."""

    workflow_id: str
    version: int = 0
    data: Dict[str, Any] = Field(default_factory=dict)
    events: List[Event] = Field(default_factory=list)
    last_hash: str = ""

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.version += 1

    def append_event(self, event_type: EventType, node_id: Optional[str] = None, payload: Optional[Dict[str, Any]] = None) -> Event:
        evt_id = f"evt-{len(self.events) + 1}-{int(time.time() * 1000)}"
        event = Event(
            event_id=evt_id,
            event_type=event_type,
            node_id=node_id,
            payload=payload or {},
        )
        event.state_hash = event.calculate_hash(self.last_hash)
        self.last_hash = event.state_hash
        self.events.append(event)
        return event

    def snapshot(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "version": self.version,
            "data": self.data.copy(),
            "last_hash": self.last_hash,
            "events_count": len(self.events),
        }
