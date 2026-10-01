"""Telemetry, token accounting, structured audit logging, and W3C-aligned tracing."""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Self

from pydantic import BaseModel, Field

from agentmesh.telemetry.w3c import (
    TraceContextValidationError,
    TraceState,
    extract_w3c_trace_context,
    validate_traceparent,
)

logger = logging.getLogger("agentmesh.telemetry")


class QuotaExceededError(Exception):
    """Raised when an operation would exceed the configured token budget limit."""

    def __init__(
        self,
        budget_limit: int,
        current_tokens: int,
        requested_tokens: int,
        message: str | None = None,
    ) -> None:
        self.budget_limit = budget_limit
        self.current_tokens = current_tokens
        self.requested_tokens = requested_tokens
        msg = message or (
            f"QuotaExceededError: Token quota exceeded. "
            f"Current={current_tokens}, Requested={requested_tokens}, "
            f"Projected={current_tokens + requested_tokens}, BudgetLimit={budget_limit}"
        )
        super().__init__(msg)


class TokenTracker:
    """Thread-safe token usage accounting with quota enforcement and cost estimation."""

    DEFAULT_RATES: ClassVar[dict[str, dict[str, float]]] = {
        "default": {"prompt": 0.002 / 1000, "completion": 0.002 / 1000},
        "gpt-4": {"prompt": 0.03 / 1000, "completion": 0.06 / 1000},
        "gpt-4o": {"prompt": 0.005 / 1000, "completion": 0.015 / 1000},
        "gpt-3.5-turbo": {"prompt": 0.0015 / 1000, "completion": 0.002 / 1000},
        "claude-3-5-sonnet": {"prompt": 0.003 / 1000, "completion": 0.015 / 1000},
    }

    def __init__(
        self,
        budget_limit: int | None = None,
        custom_rates: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self._budget_limit = budget_limit
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        self._total_tokens: int = 0
        self._cost_estimate: float = 0.0
        self._model_breakdown: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()
        self._rates = dict(self.DEFAULT_RATES)
        if custom_rates:
            self._rates.update(custom_rates)

    @property
    def prompt_tokens(self) -> int:
        with self._lock:
            return self._prompt_tokens

    @property
    def completion_tokens(self) -> int:
        with self._lock:
            return self._completion_tokens

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return self._total_tokens

    @property
    def cost_estimate(self) -> float:
        with self._lock:
            return round(self._cost_estimate, 6)

    @property
    def budget_limit(self) -> int | None:
        with self._lock:
            return self._budget_limit

    def set_budget(self, limit: int | None) -> None:
        with self._lock:
            self._budget_limit = limit

    def record(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        model: str = "default",
    ) -> None:
        """Records token usage, validating non-negative counts and enforcing budget."""
        if prompt_tokens < 0 or completion_tokens < 0:
            raise ValueError(
                f"Token counts cannot be negative: prompt={prompt_tokens}, completion={completion_tokens}"
            )

        requested = prompt_tokens + completion_tokens
        with self._lock:
            if self._budget_limit is not None and (self._total_tokens + requested) > self._budget_limit:
                raise QuotaExceededError(
                    budget_limit=self._budget_limit,
                    current_tokens=self._total_tokens,
                    requested_tokens=requested,
                )

            self._prompt_tokens += prompt_tokens
            self._completion_tokens += completion_tokens
            self._total_tokens += requested

            model_rates = self._rates.get(model, self._rates["default"])
            incremental_cost = (
                prompt_tokens * model_rates.get("prompt", 0.0)
                + completion_tokens * model_rates.get("completion", 0.0)
            )
            self._cost_estimate += incremental_cost

            if model not in self._model_breakdown:
                self._model_breakdown[model] = {"prompt": 0, "completion": 0, "total": 0}
            self._model_breakdown[model]["prompt"] += prompt_tokens
            self._model_breakdown[model]["completion"] += completion_tokens
            self._model_breakdown[model]["total"] += requested

    def get_metrics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "total_tokens": self._total_tokens,
                "cost_estimate": round(self._cost_estimate, 6),
                "budget_limit": self._budget_limit,
                "model_breakdown": {k: dict(v) for k, v in self._model_breakdown.items()},
            }

    def reset(self) -> None:
        with self._lock:
            self._prompt_tokens = 0
            self._completion_tokens = 0
            self._total_tokens = 0
            self._cost_estimate = 0.0
            self._model_breakdown.clear()


class AuditRecord(BaseModel):
    """Immutable structured audit log record."""

    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    timestamp_ns: int = Field(default_factory=time.time_ns)
    event_type: str
    trace_id: str
    span_id: str = ""
    actor_id: str
    target: str
    action: str
    status: str
    details: dict[str, Any] = Field(default_factory=dict)


class AuditLogger:
    """Thread-safe structured audit logger adhering to physical state separation."""

    def __init__(self, log_file: Path | str | None = None) -> None:
        self._lock = threading.Lock()
        self._records: list[AuditRecord] = []
        self._log_file: Path | None = Path(log_file) if log_file else None
        if self._log_file:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: AuditRecord) -> AuditRecord:
        with self._lock:
            self._records.append(record)
            if self._log_file:
                with open(self._log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record.model_dump()) + "\n")
        logger.info(
            "AUDIT: [%s] actor=%s target=%s action=%s status=%s trace=%s",
            record.event_type,
            record.actor_id,
            record.target,
            record.action,
            record.status,
            record.trace_id,
        )
        return record

    def log_tool_call(
        self,
        trace_id: str,
        span_id: str,
        actor_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        status: str = "SUCCESS",
        result: Any = None,
        error: str | None = None,
    ) -> AuditRecord:
        details: dict[str, Any] = {"arguments": arguments}
        if result is not None:
            details["result"] = result
        if error:
            details["error"] = error
        rec = AuditRecord(
            event_type="TOOL_CALL",
            trace_id=trace_id,
            span_id=span_id,
            actor_id=actor_id,
            target=tool_name,
            action="tools/call",
            status=status,
            details=details,
        )
        return self.log(rec)

    def log_agent_interaction(
        self,
        trace_id: str,
        span_id: str,
        sender_id: str,
        recipient_id: str,
        performative: str,
        action: str = "send_message",
        status: str = "SUCCESS",
        payload: dict[str, Any] | None = None,
    ) -> AuditRecord:
        rec = AuditRecord(
            event_type="AGENT_INTERACTION",
            trace_id=trace_id,
            span_id=span_id,
            actor_id=sender_id,
            target=recipient_id,
            action=action,
            status=status,
            details={"performative": performative, "payload": payload or {}},
        )
        return self.log(rec)

    def log_security_check(
        self,
        trace_id: str,
        span_id: str,
        actor_id: str,
        check_type: str,
        target: str,
        status: str = "SUCCESS",
        details: dict[str, Any] | None = None,
    ) -> AuditRecord:
        rec = AuditRecord(
            event_type="SECURITY_CHECK",
            trace_id=trace_id,
            span_id=span_id,
            actor_id=actor_id,
            target=target,
            action=f"security_check:{check_type}",
            status=status,
            details=details or {},
        )
        return self.log(rec)

    def get_records(
        self,
        trace_id: str | None = None,
        event_type: str | None = None,
    ) -> list[AuditRecord]:
        with self._lock:
            res = self._records
            if trace_id:
                res = [r for r in res if r.trace_id == trace_id]
            if event_type:
                res = [r for r in res if r.event_type == event_type]
            return list(res)

    def export_json(self, trace_id: str | None = None) -> list[dict[str, Any]]:
        return [r.model_dump() for r in self.get_records(trace_id=trace_id)]

    def export_jsonl(self, trace_id: str | None = None) -> str:
        return "\n".join(json.dumps(r.model_dump()) for r in self.get_records(trace_id=trace_id))


class Span(BaseModel):
    """Execution span representing an individual unit of work."""

    span_id: str
    trace_id: str
    parent_span_id: str | None = None
    name: str
    start_time: float = Field(default_factory=time.time)
    end_time: float | None = None
    status: str = "OK"
    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)

    def finish(self, status: str = "OK", error: str | None = None) -> None:
        self.end_time = time.time()
        self.status = status
        if error:
            self.attributes["error.message"] = error

    @property
    def duration(self) -> float:
        end = self.end_time if self.end_time is not None else time.time()
        return round(end - self.start_time, 6)

    def set_attribute(self, key: str, value: Any) -> Self:
        self.attributes[key] = value
        return self

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> Self:
        self.events.append(
            {
                "name": name,
                "timestamp": time.time(),
                "attributes": attributes or {},
            }
        )
        return self

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        if exc_val is not None:
            self.finish(status="ERROR", error=str(exc_val))
        else:
            self.finish(status="OK")


class TraceContext:
    """W3C-aligned TraceContext managing span tree, token tracking, and W3C headers."""

    def __init__(
        self,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        trace_flags: str = "01",
        tracestate: TraceState | None = None,
        budget_limit: int | None = None,
    ) -> None:
        if trace_id:
            t_clean = trace_id.lower().replace("-", "")
            if len(t_clean) == 32 and all(c in "0123456789abcdef" for c in t_clean) and t_clean != "0" * 32:
                self.trace_id = t_clean
            else:
                self.trace_id = trace_id
        else:
            self.trace_id = secrets.token_hex(16)

        self.parent_span_id = parent_span_id
        self.trace_flags = trace_flags
        self.tracestate = tracestate or TraceState()
        self.spans: list[Span] = []
        self.token_tracker = TokenTracker(budget_limit=budget_limit)

    @property
    def total_tokens_consumed(self) -> int:
        return self.token_tracker.total_tokens

    @property
    def prompt_tokens(self) -> int:
        return self.token_tracker.prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self.token_tracker.completion_tokens

    @property
    def cost_estimate(self) -> float:
        return self.token_tracker.cost_estimate

    def start_span(
        self,
        name: str,
        parent_span_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Starts a new span linked to this context."""
        span = Span(
            span_id=secrets.token_hex(8),
            trace_id=self.trace_id,
            parent_span_id=parent_span_id or self.parent_span_id,
            name=name,
            attributes=attributes or {},
        )
        self.spans.append(span)
        return span

    def record_token_usage(
        self,
        prompt: int = 0,
        completion: int = 0,
        model: str = "default",
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        """Records token consumption with multi-alias backward compatibility."""
        p = prompt_tokens if prompt_tokens is not None else prompt
        c = completion_tokens if completion_tokens is not None else completion
        self.token_tracker.record(prompt_tokens=p, completion_tokens=c, model=model)

    def to_traceparent(self, current_span_id: str | None = None) -> str:
        """Formats current context into standard W3C traceparent string."""
        sid = (
            current_span_id
            or (self.spans[-1].span_id if self.spans else self.parent_span_id or secrets.token_hex(8))
        )
        return f"00-{self.trace_id}-{sid}-{self.trace_flags}"

    def to_w3c_headers(self, current_span_id: str | None = None) -> dict[str, str]:
        """Exports W3C traceparent and tracestate headers dictionary."""
        headers = {"traceparent": self.to_traceparent(current_span_id=current_span_id)}
        ts_str = self.tracestate.format()
        if ts_str:
            headers["tracestate"] = ts_str
        return headers

    @classmethod
    def from_w3c_headers(
        cls,
        headers: dict[str, str],
        budget_limit: int | None = None,
    ) -> TraceContext:
        """Reconstructs TraceContext from W3C headers with strict validation."""
        tp_val, ts_val = extract_w3c_trace_context(headers)
        if not tp_val:
            raise TraceContextValidationError(
                "Missing required 'traceparent' header in headers dictionary"
            )
        _version, trace_id, span_id, trace_flags = validate_traceparent(tp_val)
        ts = TraceState.parse(ts_val)
        return cls(
            trace_id=trace_id,
            parent_span_id=span_id,
            trace_flags=trace_flags,
            tracestate=ts,
            budget_limit=budget_limit,
        )


class AgentTracer:
    """Global tracer registry for AgentMesh observability."""

    def __init__(self, audit_log_file: Path | str | None = None) -> None:
        self.active_contexts: dict[Any, TraceContext] = {}
        self.audit_logger = AuditLogger(log_file=audit_log_file)
        self._lock = threading.Lock()

    def get_or_create_context(
        self,
        trace_id: str | None = None,
        budget_limit: int | None = None,
    ) -> TraceContext:
        """Gets existing context or safely registers a new context without overwriting."""
        with self._lock:
            if trace_id and trace_id in self.active_contexts:
                ctx = self.active_contexts[trace_id]
                if budget_limit is not None:
                    ctx.token_tracker.set_budget(budget_limit)
                return ctx
            ctx = TraceContext(trace_id=trace_id, budget_limit=budget_limit)
            self.active_contexts[ctx.trace_id] = ctx
            return ctx

    def create_context_from_w3c(
        self,
        headers: dict[str, str],
        budget_limit: int | None = None,
    ) -> TraceContext:
        """Parses W3C headers and registers the context into the active registry."""
        with self._lock:
            ctx = TraceContext.from_w3c_headers(headers, budget_limit=budget_limit)
            self.active_contexts[ctx.trace_id] = ctx
            return ctx

    def record_token_usage(
        self,
        prompt: int = 0,
        completion: int = 0,
        model: str = "default",
        trace_id: str | None = None,
    ) -> None:
        """Delegates token recording to the target context or latest active context."""
        with self._lock:
            if trace_id and trace_id in self.active_contexts:
                self.active_contexts[trace_id].record_token_usage(
                    prompt=prompt, completion=completion, model=model
                )
            elif self.active_contexts:
                latest_ctx = list(self.active_contexts.values())[-1]
                latest_ctx.record_token_usage(
                    prompt=prompt, completion=completion, model=model
                )

    def export_summary(self, trace_id: str) -> dict[str, Any]:
        """Exports audit summary for the designated trace ID."""
        with self._lock:
            ctx = self.active_contexts.get(trace_id)
            if not ctx:
                return {}
            return {
                "trace_id": ctx.trace_id,
                "span_count": len(ctx.spans),
                "total_tokens": ctx.total_tokens_consumed,
                "prompt_tokens": ctx.prompt_tokens,
                "completion_tokens": ctx.completion_tokens,
                "cost_estimate": ctx.cost_estimate,
                "spans": [s.model_dump() for s in ctx.spans],
                "audit_events": [
                    r.model_dump()
                    for r in self.audit_logger.get_records(trace_id=trace_id)
                ],
            }

    def export_audit_logs(self, trace_id: str | None = None) -> list[dict[str, Any]]:
        return self.audit_logger.export_json(trace_id=trace_id)
