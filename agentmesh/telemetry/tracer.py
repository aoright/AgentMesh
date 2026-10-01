"""Telemetry, token accounting, and OpenTelemetry-aligned tracing."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("agentmesh.telemetry")


class Span(BaseModel):
    span_id: str
    trace_id: str
    parent_span_id: Optional[str] = None
    name: str
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None
    status: str = "OK"
    attributes: Dict[str, Any] = Field(default_factory=dict)
    events: List[Dict[str, Any]] = Field(default_factory=list)

    def finish(self, status: str = "OK", error: Optional[str] = None):
        self.end_time = time.time()
        self.status = status
        if error:
            self.attributes["error.message"] = error


class TraceContext:
    def __init__(self, trace_id: Optional[str] = None):
        self.trace_id = trace_id or uuid.uuid4().hex
        self.spans: List[Span] = []
        self.total_tokens_consumed: int = 0
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0

    def start_span(self, name: str, parent_span_id: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None) -> Span:
        span = Span(
            span_id=uuid.uuid4().hex[:16],
            trace_id=self.trace_id,
            parent_span_id=parent_span_id,
            name=name,
            attributes=attributes or {},
        )
        self.spans.append(span)
        return span

    def record_token_usage(self, prompt: int, completion: int):
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens_consumed += (prompt + completion)


class AgentTracer:
    """Global tracer registry for AgentMesh observability."""

    def __init__(self):
        self.active_contexts: Dict[str, TraceContext] = {}

    def get_or_create_context(self, trace_id: Optional[str] = None) -> TraceContext:
        ctx = TraceContext(trace_id=trace_id)
        self.active_contexts[ctx.trace_id] = ctx
        return ctx

    def export_summary(self, trace_id: str) -> Dict[str, Any]:
        ctx = self.active_contexts.get(trace_id)
        if not ctx:
            return {}
        return {
            "trace_id": ctx.trace_id,
            "span_count": len(ctx.spans),
            "total_tokens": ctx.total_tokens_consumed,
            "prompt_tokens": ctx.prompt_tokens,
            "completion_tokens": ctx.completion_tokens,
            "spans": [s.model_dump() for s in ctx.spans],
        }
