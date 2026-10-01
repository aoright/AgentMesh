"""Deterministic state models and immutable event logs for AgentMesh."""

from __future__ import annotations

import hashlib
import json
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

GENESIS_HASH: str = "0" * 64


def canonical_json_dumps(obj: Any) -> str:
    """Serializes a Python object into an RFC 8785 canonical JSON string.

    Guarantees bit-for-bit identical serialization regardless of dict insertion order:
    - Sorted keys.
    - Zero whitespace separators.
    - UTF-8 characters preserved.
    - Recursive serialization of Enums, Pydantic models, paths, and sets.
    """

    def _default_serializer(val: Any) -> Any:
        if isinstance(val, Enum):
            return val.value
        if hasattr(val, "model_dump"):
            return val.model_dump()
        if hasattr(val, "to_dict"):
            return val.to_dict()
        if hasattr(val, "__fspath__"):
            return str(val)
        if isinstance(val, (set, frozenset)):
            return sorted(val)
        if isinstance(val, bytes):
            return val.hex()
        if hasattr(val, "__dict__"):
            return val.__dict__
        return repr(val)

    return json.dumps(
        obj,
        default=_default_serializer,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


class EventType(str, Enum):
    # Workflow Execution Lifecycle
    WORKFLOW_START = "WORKFLOW_START"
    WORKFLOW_COMPLETE = "WORKFLOW_COMPLETE"
    WORKFLOW_FAILED = "WORKFLOW_FAILED"
    WORKFLOW_PAUSED = "WORKFLOW_PAUSED"

    # DAG Node Lifecycle
    NODE_START = "NODE_START"
    NODE_COMPLETE = "NODE_COMPLETE"
    NODE_FAILED = "NODE_FAILED"
    NODE_SKIPPED = "NODE_SKIPPED"

    # Activity Isolation & Memoizer Lifecycle
    ACTIVITY_SCHEDULED = "ACTIVITY_SCHEDULED"
    ACTIVITY_STARTED = "ACTIVITY_STARTED"
    ACTIVITY_COMPLETED = "ACTIVITY_COMPLETED"
    ACTIVITY_FAILED = "ACTIVITY_FAILED"
    ACTIVITY_MEMOIZED_HIT = "ACTIVITY_MEMOIZED_HIT"

    # State & Snapshot Management
    STATE_MUTATED = "STATE_MUTATED"
    CHECKPOINT = "CHECKPOINT"
    CHECKPOINT_SAVED = "CHECKPOINT_SAVED"
    TOOL_CALL = "TOOL_CALL"

    # Anti-Tamper & Security Audit
    INTEGRITY_VIOLATION_DETECTED = "INTEGRITY_VIOLATION_DETECTED"



class CryptographicIntegrityError(RuntimeError):
    """Raised immediately when cryptographic verification fails or event tampering is detected.

    Enforces instant execution blocking to protect workflow state against unauthorized
    modifications, replay attacks, or silent data corruption.
    """

    def __init__(
        self,
        message: str,
        violation_type: str = "GENERIC_TAMPER",
        sequence_num: int | None = None,
        event_id: str | None = None,
        expected_hash: str | None = None,
        actual_hash: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.violation_type = violation_type
        self.sequence_num = sequence_num
        self.event_id = event_id
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash
        self.details = details or {}

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "violation_type": self.violation_type,
            "sequence_num": self.sequence_num,
            "event_id": self.event_id,
            "expected_hash": self.expected_hash,
            "actual_hash": self.actual_hash,
            "message": str(self),
            "details": self.details,
        }


class Event(BaseModel):
    """Immutable event record in the event stream with cryptographic hash linkage."""

    event_id: str
    sequence_num: int = 1
    workflow_id: str = ""
    run_id: str = ""
    event_type: EventType
    timestamp_ns: int = Field(default_factory=time.time_ns)
    node_id: str | None = None
    activity_id: str | None = None
    idempotency_key: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    prev_hash: str = Field(default="")
    state_hash: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _handle_legacy_timestamp(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "timestamp_ns" not in data and "timestamp" in data:
                ts = data.pop("timestamp")
                if isinstance(ts, (int, float)):
                    data["timestamp_ns"] = int(ts * 1_000_000_000)
            elif "timestamp" in data:
                data.pop("timestamp", None)
        return data

    @property
    def timestamp(self) -> float:
        """Backward-compatible float timestamp in seconds."""
        return self.timestamp_ns / 1_000_000_000.0

    def calculate_hash(self, prev_hash: str | None = None) -> str:
        """Computes deterministic SHA-256 digest linking prev_hash, identity context, and event data.

        Uses canonical JSON array encoding (RFC 8785) to eliminate second-preimage delimiter collisions
        and binds workflow_id and run_id to prevent cross-workflow event replay and transplantation.
        """
        parent_hash = prev_hash if prev_hash is not None else (self.prev_hash or "")
        event_type_str = self.event_type.value if isinstance(self.event_type, Enum) else str(self.event_type)
        preimage = [
            parent_hash,
            self.workflow_id,
            self.run_id,
            self.event_id,
            event_type_str,
            self.sequence_num,
            self.timestamp_ns,
            self.node_id or "",
            self.activity_id or "",
            self.idempotency_key or "",
            self.payload,
        ]
        content = canonical_json_dumps(preimage)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()



class State(BaseModel):
    """Workflow state container with deterministic hashing and cryptographic event chaining."""

    workflow_id: str
    run_id: str = ""
    version: int = 0
    data: dict[str, Any] = Field(default_factory=dict)
    events: list[Event] = Field(default_factory=list)
    last_hash: str = ""
    memoizer: Any = Field(default=None, exclude=True)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.version += 1

    def append_event(
        self,
        event_type: EventType,
        node_id: str | None = None,
        payload: dict[str, Any] | None = None,
        activity_id: str | None = None,
        idempotency_key: str | None = None,
        timestamp_ns: int | None = None,
    ) -> Event:
        seq = len(self.events) + 1
        evt_id = f"evt-{seq:08d}"
        prev_hash = (self.events[-1].state_hash or "") if self.events else (self.last_hash or "")

        ts_ns = timestamp_ns if timestamp_ns is not None else time.time_ns()
        if self.events and ts_ns <= self.events[-1].timestamp_ns:
            ts_ns = self.events[-1].timestamp_ns + 1

        event = Event(
            event_id=evt_id,
            sequence_num=seq,
            workflow_id=self.workflow_id,
            run_id=self.run_id,
            event_type=event_type,
            timestamp_ns=ts_ns,
            node_id=node_id,
            activity_id=activity_id,
            idempotency_key=idempotency_key,
            payload=payload or {},
            prev_hash=prev_hash,
        )
        event.state_hash = event.calculate_hash(prev_hash)
        self.last_hash = event.state_hash
        self.events.append(event)
        return event

    async def execute_activity(
        self,
        activity_name: str,
        fn: Any,
        args: tuple[Any, ...] | list[Any] = (),
        kwargs: dict[str, Any] | None = None,
        *,
        idempotency_key: str | None = None,
        timeout_seconds: float | None = None,
        retry_policy: Any = None,
    ) -> Any:
        from agentmesh.engine.activity import (
            ActivityExecutionContext,
            current_activity_context,
            execute_activity,
        )

        ctx = current_activity_context.get()
        if ctx is None and getattr(self, "memoizer", None) is not None:
            ctx = ActivityExecutionContext(
                workflow_id=self.workflow_id,
                node_id="",
                memoizer=self.memoizer,
                state=self,
            )
        return await execute_activity(
            activity_name=activity_name,
            fn=fn,
            args=args,
            kwargs=kwargs,
            idempotency_key=idempotency_key,
            timeout_seconds=timeout_seconds,
            retry_policy=retry_policy,
            context=ctx,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "run_id": self.run_id,
            "version": self.version,
            "data": self.data.copy(),
            "last_hash": self.last_hash,
            "events_count": len(self.events),
            "last_sequence_num": self.events[-1].sequence_num if self.events else 0,
        }
