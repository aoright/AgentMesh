"""W3C TraceContext Distributed Tracing Specification Compliance.

Implements W3C Trace Context recommendation (traceparent, tracestate, inject/extract)
with strict validation, forward compatibility, and async context propagation.
"""

from __future__ import annotations

import logging
import re
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any

from pydantic import BaseModel, field_validator

logger = logging.getLogger("agentmesh.telemetry.w3c")

# W3C Specification Constants
W3C_VERSION_00 = "00"
INVALID_VERSION_FF = "ff"
TRACEPARENT_HEADER = "traceparent"
TRACESTATE_HEADER = "tracestate"
TRACEPARENT_LEN_V0 = 55
MAX_TRACESTATE_MEMBERS = 32
MAX_TRACESTATE_CHARS = 512

# Validation Regexes
RE_HEX = re.compile(r"^[0-9a-f]+$")
RE_TRACESTATE_KEY = re.compile(r"^[a-z0-9][a-z0-9_\-*/]{0,255}$")
RE_TRACESTATE_TENANT_KEY = re.compile(
    r"^[a-z0-9][a-z0-9_\-*/]{0,240}@[a-z0-9][a-z0-9_\-*/]{0,13}$"
)
RE_TRACESTATE_VALUE = re.compile(r"^[\x21-\x2b\x2d-\x3c\x3e-\x7e]{1,256}$")


class TraceContextError(Exception):
    """Base exception for all W3C TraceContext errors."""


class TraceContextValidationError(TraceContextError):
    """Raised when traceparent or tracestate violates W3C specification syntax."""


def is_valid_hex(val: str) -> bool:
    """Checks whether string consists solely of lowercase hex characters."""
    return bool(RE_HEX.match(val))


def validate_traceparent(traceparent: str) -> tuple[str, str, str, str]:
    """Strictly validates a W3C traceparent header string.

    Returns:
        Tuple of (version, trace_id, span_id, trace_flags).

    Raises:
        TraceContextValidationError on any specification deviation.
    """
    if not isinstance(traceparent, str):
        raise TraceContextValidationError("traceparent header must be a string")

    parts = traceparent.split("-")
    if len(parts) < 4:
        raise TraceContextValidationError(
            f"traceparent must contain at least 4 hyphen-separated fields, got {len(parts)} in '{traceparent}'"
        )

    version = parts[0].lower()
    if version == INVALID_VERSION_FF:
        raise TraceContextValidationError(
            "Version 'ff' is explicitly forbidden by W3C TraceContext specification"
        )

    if len(version) != 2 or not is_valid_hex(version):
        raise TraceContextValidationError(f"Invalid version format: '{version}'")

    if version == W3C_VERSION_00:
        if len(parts) != 4:
            raise TraceContextValidationError(
                f"Version '00' requires exactly 4 hyphen-separated fields, got {len(parts)}"
            )
        if len(traceparent) != TRACEPARENT_LEN_V0:
            raise TraceContextValidationError(
                f"Version '00' traceparent must be exactly {TRACEPARENT_LEN_V0} characters, got {len(traceparent)}"
            )

    trace_id = parts[1].lower()
    if len(trace_id) != 32 or not is_valid_hex(trace_id):
        raise TraceContextValidationError(
            f"trace_id must be exactly 32 lowercase hex characters, got '{trace_id}'"
        )
    if trace_id == "0" * 32:
        raise TraceContextValidationError("trace_id cannot be all zeros")

    span_id = parts[2].lower()
    if len(span_id) != 16 or not is_valid_hex(span_id):
        raise TraceContextValidationError(
            f"span_id must be exactly 16 lowercase hex characters, got '{span_id}'"
        )
    if span_id == "0" * 16:
        raise TraceContextValidationError("span_id cannot be all zeros")

    trace_flags = parts[3].lower()
    if len(trace_flags) != 2 or not is_valid_hex(trace_flags):
        raise TraceContextValidationError(
            f"trace_flags must be 2 lowercase hex characters, got '{trace_flags}'"
        )

    return version, trace_id, span_id, trace_flags


class TraceParent(BaseModel):
    """W3C TraceParent data model."""

    version: str = W3C_VERSION_00
    trace_id: str
    span_id: str
    trace_flags: str = "01"

    @field_validator("version", "trace_id", "span_id", "trace_flags")
    @classmethod
    def normalize_lower(cls, v: str) -> str:
        return v.lower()

    def format(self) -> str:
        """Serializes to canonical W3C traceparent string format."""
        return f"{self.version}-{self.trace_id}-{self.span_id}-{self.trace_flags}"

    def __str__(self) -> str:
        return self.format()

    @property
    def is_sampled(self) -> bool:
        """Checks if recorded/sampled bit (bit 0) is set."""
        try:
            return (int(self.trace_flags, 16) & 1) == 1
        except ValueError:
            return False

    @classmethod
    def create(
        cls,
        trace_id: str | None = None,
        span_id: str | None = None,
        sampled: bool = True,
        version: str = W3C_VERSION_00,
    ) -> TraceParent:
        """Constructs a cryptographically strong valid TraceParent instance."""
        t_id = trace_id.lower() if trace_id else secrets.token_hex(16)
        s_id = span_id.lower() if span_id else secrets.token_hex(8)
        flags = "01" if sampled else "00"
        validate_traceparent(f"{version}-{t_id}-{s_id}-{flags}")
        return cls(version=version, trace_id=t_id, span_id=s_id, trace_flags=flags)

    @classmethod
    def parse(cls, header_value: str) -> TraceParent:
        """Parses and validates a traceparent header string."""
        v, tid, sid, flags = validate_traceparent(header_value)
        return cls(version=v, trace_id=tid, span_id=sid, trace_flags=flags)

    def create_child(self, new_span_id: str | None = None) -> TraceParent:
        """Derives a child span TraceParent with identical trace_id and new span_id."""
        child_sid = new_span_id.lower() if new_span_id else secrets.token_hex(8)
        return TraceParent(
            version=self.version,
            trace_id=self.trace_id,
            span_id=child_sid,
            trace_flags=self.trace_flags,
        )


class TraceState:
    """W3C TraceState vendor list model with strict mutation rules."""

    def __init__(self, entries: list[tuple[str, str]] | None = None) -> None:
        self._entries: list[tuple[str, str]] = list(entries or [])

    @staticmethod
    def _validate_key(key: str) -> None:
        if not (RE_TRACESTATE_KEY.match(key) or RE_TRACESTATE_TENANT_KEY.match(key)):
            raise TraceContextValidationError(f"Invalid tracestate key: '{key}'")

    @staticmethod
    def _validate_value(value: str) -> None:
        if not RE_TRACESTATE_VALUE.match(value):
            raise TraceContextValidationError(f"Invalid tracestate value: '{value}'")

    def get(self, key: str) -> str | None:
        for k, v in self._entries:
            if k == key:
                return v
        return None

    def set(self, key: str, value: str) -> TraceState:
        """Sets a vendor key-value pair, moving it to index 0 (front of list)."""
        self._validate_key(key)
        self._validate_value(value)
        new_entries = [(k, v) for k, v in self._entries if k != key]
        new_entries.insert(0, (key, value))

        # Enforce maximum list members (32)
        while len(new_entries) > MAX_TRACESTATE_MEMBERS:
            new_entries.pop()

        # Enforce maximum character limit (512)
        while (
            new_entries
            and len(",".join(f"{k}={v}" for k, v in new_entries)) > MAX_TRACESTATE_CHARS
        ):
            new_entries.pop()

        self._entries = new_entries
        return self

    def delete(self, key: str) -> TraceState:
        """Removes a vendor key from tracestate."""
        self._entries = [(k, v) for k, v in self._entries if k != key]
        return self

    def format(self) -> str:
        """Serializes tracestate to W3C comma-separated string."""
        return ",".join(f"{k}={v}" for k, v in self._entries)

    def __str__(self) -> str:
        return self.format()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        return any(k == key for k, _ in self._entries)

    @classmethod
    def parse(cls, header_value: str | None) -> TraceState:
        """Parses a W3C tracestate string header."""
        if not header_value or not header_value.strip():
            return cls()

        members = header_value.strip().split(",")
        entries: list[tuple[str, str]] = []
        for m in members:
            m = m.strip()
            if not m or "=" not in m:
                continue
            k, v = m.split("=", 1)
            k, v = k.strip(), v.strip()
            try:
                cls._validate_key(k)
                cls._validate_value(v)
                if not any(ek == k for ek, _ in entries):
                    entries.append((k, v))
            except TraceContextValidationError:
                continue

        new_entries = entries[:MAX_TRACESTATE_MEMBERS]
        while (
            new_entries
            and len(",".join(f"{k}={v}" for k, v in new_entries)) > MAX_TRACESTATE_CHARS
        ):
            new_entries.pop()
        return cls(entries=new_entries)


# Async Context Management
_CURRENT_TRACE_CONTEXT: ContextVar[Any] = ContextVar(
    "agentmesh_current_trace_context", default=None
)


def get_current_trace_context() -> Any:
    """Returns ambient TraceContext for current execution task."""
    return _CURRENT_TRACE_CONTEXT.get()


def set_current_trace_context(ctx: Any) -> Token[Any]:
    """Sets ambient TraceContext for current task."""
    return _CURRENT_TRACE_CONTEXT.set(ctx)


@contextmanager
def trace_scope(ctx: Any) -> Iterator[Any]:
    """Context manager binding ambient TraceContext to current async/thread scope."""
    token = set_current_trace_context(ctx)
    try:
        yield ctx
    finally:
        _CURRENT_TRACE_CONTEXT.reset(token)


# Cross-Agent & MCP Wire Helpers
def inject_w3c_trace_context(
    carrier: dict[str, Any], traceparent: str, tracestate: str | None = None
) -> None:
    """Injects W3C headers into wire carrier dictionary."""
    carrier[TRACEPARENT_HEADER] = traceparent
    if tracestate:
        carrier[TRACESTATE_HEADER] = tracestate


def extract_w3c_trace_context(
    carrier: dict[str, Any],
) -> tuple[str | None, str | None]:
    """Case-insensitively extracts traceparent and tracestate from wire carrier."""
    tp: str | None = None
    ts: str | None = None
    for k, v in carrier.items():
        if k.lower() == TRACEPARENT_HEADER and isinstance(v, str):
            tp = v
        elif k.lower() == TRACESTATE_HEADER and isinstance(v, str):
            ts = v
    return tp, ts
