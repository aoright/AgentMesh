"""Agent-to-Agent (A2A) cross-agent communication protocol and negotiation engine."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, Field

logger = logging.getLogger("agentmesh.mesh.a2a")


class Performative(str, Enum):
    """FIPA ACL-inspired communicative acts defining message intent."""

    REQUEST = "REQUEST"
    PROPOSE = "PROPOSE"
    AGREE = "AGREE"
    REFUSE = "REFUSE"
    INFORM = "INFORM"
    FAILURE = "FAILURE"


class AgentCapability(BaseModel):
    """Discrete, versioned functional capability declared by an agent."""

    name: str = Field(..., description="Unique capability name")
    version: str = Field(default="1.0.0", description="Semantic version string")
    description: str = Field(default="", description="Capability description")
    parameters_schema: dict[str, Any] = Field(default_factory=dict, description="JSON Schema for parameters")
    return_schema: dict[str, Any] = Field(default_factory=dict, description="JSON Schema for output")
    tags: list[str] = Field(default_factory=list, description="Discoverability tags")


class AgentCard(BaseModel):
    """Self-describing identity and capability manifest of an autonomous agent."""

    agent_id: str = Field(..., description="Globally unique agent identifier")
    name: str = Field(..., description="Human-readable display name")
    description: str = Field(default="", description="Agent role and scope")
    version: str = Field(default="1.0.0", description="Agent release version")
    capabilities: list[AgentCapability] = Field(default_factory=list, description="Capabilities exposed")
    input_schema: dict[str, Any] = Field(default_factory=dict, description="Top-level input schema")
    output_schema: dict[str, Any] = Field(default_factory=dict, description="Top-level output schema")
    endpoints: list[str] = Field(default_factory=lambda: ["in-process"], description="Supported transport URIs")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary operational metadata")

    def has_capability(self, capability_name: str, min_version: str | None = None) -> bool:
        """Checks if the agent provides the specified capability with optional semver requirement."""
        for cap in self.capabilities:
            if cap.name == capability_name:
                if min_version is None:
                    return True
                return self._parse_version(cap.version) >= self._parse_version(min_version)
        return False

    def get_capability(self, capability_name: str) -> AgentCapability | None:
        """Retrieves definition for named capability."""
        for cap in self.capabilities:
            if cap.name == capability_name:
                return cap
        return None

    @staticmethod
    def _parse_version(v: str) -> tuple[int, ...]:
        clean = v.split("-")[0].split("+")[0]
        parts = []
        for part in clean.split("."):
            try:
                parts.append(int(part))
            except ValueError:
                parts.append(0)
        return tuple(parts)


class A2AMessage(BaseModel):
    """Standardized wire envelope for inter-agent communication."""

    message_id: str = Field(default_factory=lambda: uuid.uuid4().hex, description="Unique message UUID")
    sender_id: str = Field(..., description="Agent ID of the sender")
    recipient_id: str = Field(..., description="Agent ID of the recipient")
    performative: Performative = Field(..., description="Communicative intent")
    payload: dict[str, Any] = Field(default_factory=dict, description="Message data body")
    conversation_id: str = Field(default_factory=lambda: uuid.uuid4().hex, description="Conversation session UUID")
    reply_to_id: str | None = Field(default=None, description="UUID of parent message if replying")
    traceparent: str | None = Field(default=None, description="W3C TraceContext traceparent string")
    timestamp_ns: int = Field(default_factory=time.time_ns, description="Integer nanosecond timestamp")

    def create_reply(
        self,
        performative: Performative,
        payload: dict[str, Any],
        traceparent: str | None = None,
    ) -> A2AMessage:
        """Constructs a responsive reply in the same conversation."""
        return A2AMessage(
            sender_id=self.recipient_id,
            recipient_id=self.sender_id,
            performative=performative,
            payload=payload,
            conversation_id=self.conversation_id,
            reply_to_id=self.message_id,
            traceparent=traceparent or self.traceparent,
        )


class NegotiationState(str, Enum):
    """States of the agent negotiation lifecycle."""

    INIT = "INIT"
    REQUESTED = "REQUESTED"
    PROPOSED = "PROPOSED"
    AGREED = "AGREED"
    REFUSED = "REFUSED"
    EXECUTING = "EXECUTING"
    INFORMED = "INFORMED"
    FAILED = "FAILED"


class A2AProtocolError(Exception):
    """Base exception for A2A protocol violations."""


class InvalidStateTransitionError(A2AProtocolError):
    """Raised when an illegal performative is received for current negotiation state."""


class NegotiationSession:
    """Tracks state and message history for an active negotiation conversation."""

    VALID_TRANSITIONS: ClassVar[dict[NegotiationState, set[Performative]]] = {
        NegotiationState.INIT: {Performative.REQUEST},
        NegotiationState.REQUESTED: {
            Performative.PROPOSE,
            Performative.AGREE,
            Performative.REFUSE,
            Performative.FAILURE,
        },
        NegotiationState.PROPOSED: {
            Performative.AGREE,
            Performative.REFUSE,
            Performative.PROPOSE,
            Performative.FAILURE,
        },
        NegotiationState.AGREED: {Performative.INFORM, Performative.FAILURE},
        NegotiationState.EXECUTING: {Performative.INFORM, Performative.FAILURE},
        NegotiationState.REFUSED: set(),
        NegotiationState.INFORMED: set(),
        NegotiationState.FAILED: set(),
    }

    def __init__(
        self,
        conversation_id: str,
        requester_id: str,
        provider_id: str,
        timeout_sec: float = 60.0,
        metadata: dict[str, Any] | None = None,
    ):
        self.conversation_id = conversation_id
        self.requester_id = requester_id
        self.provider_id = provider_id
        self.current_state = NegotiationState.INIT
        self.history: list[A2AMessage] = []
        self.created_at_ns = time.time_ns()
        self.updated_at_ns = self.created_at_ns
        self.timeout_sec = timeout_sec
        self.metadata: dict[str, Any] = metadata or {}

    def is_terminal(self) -> bool:
        """Returns True if negotiation has reached an end state."""
        return self.current_state in {
            NegotiationState.REFUSED,
            NegotiationState.INFORMED,
            NegotiationState.FAILED,
        }

    def is_expired(self, now_ns: int | None = None) -> bool:
        """Evaluates whether session has exceeded its timeout."""
        now = now_ns if now_ns is not None else time.time_ns()
        elapsed_sec = (now - self.updated_at_ns) / 1_000_000_000.0
        return elapsed_sec > self.timeout_sec

    def mark_executing(self) -> None:
        """Transitions state from AGREED to EXECUTING."""
        if self.current_state != NegotiationState.AGREED:
            raise InvalidStateTransitionError(
                f"Cannot transition to EXECUTING from state '{self.current_state}'."
            )
        self.current_state = NegotiationState.EXECUTING
        self.updated_at_ns = time.time_ns()

    def process_message(self, message: A2AMessage) -> NegotiationState:
        """Validates incoming message performative, applies state transition, and logs history."""
        if self.is_terminal():
            raise InvalidStateTransitionError(
                f"Session '{self.conversation_id}' is in terminal state '{self.current_state}'."
            )
        if self.is_expired():
            self.current_state = NegotiationState.FAILED
            raise InvalidStateTransitionError(
                f"Session '{self.conversation_id}' has timed out."
            )

        allowed = self.VALID_TRANSITIONS.get(self.current_state, set())
        if message.performative not in allowed:
            allowed_names = sorted([p.value for p in allowed])
            raise InvalidStateTransitionError(
                f"Invalid performative '{message.performative}' in state '{self.current_state}'. Allowed: {allowed_names}"
            )

        if message.performative == Performative.REQUEST:
            self.current_state = NegotiationState.REQUESTED
        elif message.performative == Performative.PROPOSE:
            self.current_state = NegotiationState.PROPOSED
        elif message.performative == Performative.AGREE:
            self.current_state = NegotiationState.AGREED
        elif message.performative == Performative.REFUSE:
            self.current_state = NegotiationState.REFUSED
        elif message.performative == Performative.INFORM:
            self.current_state = NegotiationState.INFORMED
        elif message.performative == Performative.FAILURE:
            self.current_state = NegotiationState.FAILED

        self.history.append(message)
        self.updated_at_ns = time.time_ns()
        return self.current_state


class NegotiationManager:
    """Thread-safe manager for concurrent multi-agent negotiation sessions."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, NegotiationSession] = {}

    def create_session(
        self,
        requester_id: str,
        provider_id: str,
        conversation_id: str | None = None,
        timeout_sec: float = 60.0,
        metadata: dict[str, Any] | None = None,
    ) -> NegotiationSession:
        cid = conversation_id or uuid.uuid4().hex
        session = NegotiationSession(
            conversation_id=cid,
            requester_id=requester_id,
            provider_id=provider_id,
            timeout_sec=timeout_sec,
            metadata=metadata,
        )
        with self._lock:
            self._sessions[cid] = session
        return session

    def get_session(self, conversation_id: str) -> NegotiationSession | None:
        with self._lock:
            return self._sessions.get(conversation_id)

    def process_message(self, message: A2AMessage) -> NegotiationSession:
        with self._lock:
            session = self._sessions.get(message.conversation_id)
            if not session:
                if message.performative == Performative.REQUEST:
                    session = self.create_session(
                        requester_id=message.sender_id,
                        provider_id=message.recipient_id,
                        conversation_id=message.conversation_id,
                    )
                else:
                    raise A2AProtocolError(
                        f"Unknown conversation_id '{message.conversation_id}' for non-REQUEST performative."
                    )
            session.process_message(message)
            return session

    def evict_expired_sessions(self, now_ns: int | None = None) -> list[str]:
        with self._lock:
            now = now_ns if now_ns is not None else time.time_ns()
            evicted: list[str] = []
            for cid, session in list(self._sessions.items()):
                if session.is_expired(now):
                    session.current_state = NegotiationState.FAILED
                    del self._sessions[cid]
                    evicted.append(cid)
            return evicted
