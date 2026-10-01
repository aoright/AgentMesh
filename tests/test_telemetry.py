"""Comprehensive Unit and Integration Test Suite for Telemetry, W3C Tracing, and Token Accounting."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import threading
from typing import Any

import pytest

from agentmesh.mesh.a2a import A2AMessage, Performative
from agentmesh.telemetry.tracer import (
    AgentTracer,
    AuditLogger,
    QuotaExceededError,
    TokenTracker,
    TraceContext,
)
from agentmesh.telemetry.w3c import (
    TraceContextValidationError,
    TraceParent,
    TraceState,
    extract_w3c_trace_context,
    get_current_trace_context,
    inject_w3c_trace_context,
    trace_scope,
    validate_traceparent,
)

# ==============================================================================
# Category 1: W3C TraceParent Header Parsing, Formatting & Validation Tests
# ==============================================================================


class TestW3CTraceParent:
    def test_valid_traceparent_canonical(self):
        """Validates canonical W3C traceparent example."""
        header = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
        tp = TraceParent.parse(header)
        assert tp.version == "00"
        assert tp.trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"
        assert tp.span_id == "00f067aa0ba902b7"
        assert tp.trace_flags == "01"
        assert tp.is_sampled is True
        assert tp.format() == header

    def test_valid_traceparent_unsampled(self):
        """Validates unsampled traceparent."""
        header = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00"
        tp = TraceParent.parse(header)
        assert tp.trace_flags == "00"
        assert tp.is_sampled is False

    def test_traceparent_create_random(self):
        """Validates automatic generation of cryptographically secure valid TraceParent."""
        tp = TraceParent.create(sampled=True)
        assert tp.version == "00"
        assert len(tp.trace_id) == 32
        assert len(tp.span_id) == 16
        assert tp.trace_flags == "01"
        assert tp.is_sampled is True

    def test_traceparent_create_child(self):
        """Validates deriving child span with same trace_id and new span_id."""
        parent = TraceParent.create(sampled=True)
        child = parent.create_child()
        assert child.trace_id == parent.trace_id
        assert child.span_id != parent.span_id
        assert len(child.span_id) == 16
        assert child.trace_flags == parent.trace_flags

    @pytest.mark.parametrize(
        "invalid_header,reason",
        [
            ("", "Empty string"),
            ("00-1234-5678-01", "Too few fields"),
            (
                "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
                "Forbidden version ff",
            ),
            (
                "0z-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
                "Non-hex version",
            ),
            (
                "00-4bf92f3577b34da6a3ce929d0e0e473-00f067aa0ba902b7-01",
                "trace_id length 31 (too short)",
            ),
            (
                "00-4bf92f3577b34da6a3ce929d0e0e47366-00f067aa0ba902b7-01",
                "trace_id length 33 (too long)",
            ),
            (
                "00-4bf92f3577b34da6a3ce929d0e0e473g-00f067aa0ba902b7-01",
                "Non-hex character in trace_id",
            ),
            (
                "00-00000000000000000000000000000000-00f067aa0ba902b7-01",
                "All-zero trace_id",
            ),
            (
                "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b-01",
                "span_id length 15 (too short)",
            ),
            (
                "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b77-01",
                "span_id length 17 (too long)",
            ),
            (
                "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902bz-01",
                "Non-hex character in span_id",
            ),
            (
                "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",
                "All-zero span_id",
            ),
            (
                "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-1",
                "flags length 1",
            ),
            (
                "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-001",
                "flags length 3",
            ),
            (
                "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-0z",
                "Non-hex flags",
            ),
            (
                "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra",
                "Version 00 with extra fields",
            ),
        ],
    )
    def test_reject_invalid_traceparent(self, invalid_header: str, reason: str):
        """Validates that all W3C violations raise TraceContextValidationError."""
        with pytest.raises(TraceContextValidationError):
            validate_traceparent(invalid_header)

    def test_forward_compatibility_future_version(self):
        """Validates that version > 00 allows forward-compatible extra fields."""
        future_header = (
            "01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-future-data"
        )
        v, tid, sid, flags = validate_traceparent(future_header)
        assert v == "01"
        assert tid == "4bf92f3577b34da6a3ce929d0e0e4736"
        assert sid == "00f067aa0ba902b7"
        assert flags == "01"


# ==============================================================================
# Category 2: W3C TraceState Header Parsing, Mutation & Limits Tests
# ==============================================================================


class TestW3CTraceState:
    def test_tracestate_parse_simple(self):
        """Validates parsing standard comma-separated vendor entries."""
        raw = "rojo=123,congo=456"
        ts = TraceState.parse(raw)
        assert len(ts) == 2
        assert ts.get("rojo") == "123"
        assert ts.get("congo") == "456"
        assert ts.format() == raw

    def test_tracestate_tenant_key(self):
        """Validates parsing multi-tenant key with @ delimiter."""
        raw = "rojo@corp=val1,system=val2"
        ts = TraceState.parse(raw)
        assert len(ts) == 2
        assert ts.get("rojo@corp") == "val1"
        assert ts.get("system") == "val2"

    def test_tracestate_set_prepends_to_front(self):
        """Validates W3C rule: new/updated entry MUST be moved to index 0."""
        ts = TraceState.parse("vendor1=val1,vendor2=val2")
        ts.set("vendor3", "val3")
        assert ts.format().startswith("vendor3=val3")
        assert len(ts) == 3

    def test_tracestate_update_existing_moves_to_front(self):
        """Validates updating existing key moves it from its position to index 0."""
        ts = TraceState.parse("vendor1=val1,vendor2=val2,vendor3=val3")
        ts.set("vendor2", "updated_val2")
        assert ts.format().startswith("vendor2=updated_val2")
        assert ts.get("vendor2") == "updated_val2"
        assert len(ts) == 3

    def test_tracestate_delete(self):
        """Validates deleting a vendor key."""
        ts = TraceState.parse("vendor1=val1,vendor2=val2")
        ts.delete("vendor1")
        assert "vendor1" not in ts
        assert ts.get("vendor1") is None
        assert ts.format() == "vendor2=val2"

    def test_tracestate_member_limit_prunes_oldest(self):
        """Validates 32 member maximum limit prunes right-most tail items."""
        ts = TraceState()
        for i in range(35):
            ts.set(f"v{i}", f"val_{i}")
        assert len(ts) == 32
        assert "v34" in ts
        assert "v0" not in ts

    def test_tracestate_reject_invalid_keys(self):
        """Validates rejection of invalid key characters."""
        ts = TraceState()
        with pytest.raises(TraceContextValidationError):
            ts.set("Invalid_UpperCase", "val")
        with pytest.raises(TraceContextValidationError):
            ts.set("key with spaces", "val")

    def test_tracestate_reject_invalid_values(self):
        """Validates rejection of values with comma or equals."""
        ts = TraceState()
        with pytest.raises(TraceContextValidationError):
            ts.set("key", "val=123")
        with pytest.raises(TraceContextValidationError):
            ts.set("key", "val,123")


# ==============================================================================
# Category 3: TraceContext & Span Hierarchy Tests
# ==============================================================================


class TestTraceContextAndSpans:
    def test_trace_context_initialization(self):
        """Validates TraceContext generation and initial metrics."""
        ctx = TraceContext()
        assert len(ctx.trace_id) == 32
        assert ctx.total_tokens_consumed == 0
        assert ctx.prompt_tokens == 0
        assert ctx.completion_tokens == 0
        assert len(ctx.spans) == 0

    def test_span_start_and_finish(self):
        """Validates span lifecycle."""
        ctx = TraceContext()
        span = ctx.start_span("process_order", attributes={"order_id": "ORD-123"})
        assert span.name == "process_order"
        assert span.trace_id == ctx.trace_id
        assert len(span.span_id) == 16
        assert span.attributes["order_id"] == "ORD-123"
        assert span.end_time is None

        span.finish(status="OK")
        assert span.end_time is not None
        assert span.status == "OK"
        assert span.duration >= 0.0

    def test_span_context_manager(self):
        """Validates span context manager syntax."""
        ctx = TraceContext()
        with ctx.start_span("database_read") as span:
            span.set_attribute("db.table", "users")
            span.add_event("query_executed", {"rows": 10})

        assert span.end_time is not None
        assert span.status == "OK"
        assert span.attributes["db.table"] == "users"
        assert len(span.events) == 1

    def test_span_context_manager_error_handling(self):
        """Validates that exceptions inside context manager auto-finish span with ERROR."""
        ctx = TraceContext()
        try:
            with ctx.start_span("faulty_call") as span:
                raise RuntimeError("Service failure")
        except RuntimeError:
            pass

        assert span.end_time is not None
        assert span.status == "ERROR"
        assert "Service failure" in span.attributes["error.message"]

    def test_parent_child_span_hierarchy(self):
        """Validates parent-child span linking."""
        ctx = TraceContext()
        parent_span = ctx.start_span("parent_job")
        child_span = ctx.start_span("child_job", parent_span_id=parent_span.span_id)

        assert child_span.trace_id == ctx.trace_id
        assert child_span.parent_span_id == parent_span.span_id
        assert child_span.span_id != parent_span.span_id

    def test_w3c_header_round_trip(self):
        """Validates exporting to W3C headers and reconstructing TraceContext."""
        ctx = TraceContext()
        span = ctx.start_span("root_span")
        ctx.tracestate.set("mesh_tenant", "alpha")

        headers = ctx.to_w3c_headers()
        assert "traceparent" in headers
        assert "tracestate" in headers

        imported_ctx = TraceContext.from_w3c_headers(headers)
        assert imported_ctx.trace_id == ctx.trace_id
        assert imported_ctx.parent_span_id == span.span_id
        assert imported_ctx.tracestate.get("mesh_tenant") == "alpha"


# ==============================================================================
# Category 4: Async Context Propagation Tests
# ==============================================================================


class TestAsyncContextPropagation:
    @pytest.mark.asyncio
    async def test_trace_scope_ambient_context(self):
        """Validates ambient context resolution across async task hierarchy."""
        ctx = TraceContext()
        assert get_current_trace_context() is None

        with trace_scope(ctx):
            assert get_current_trace_context() is ctx

            async def inner_task() -> Any:
                return get_current_trace_context()

            retrieved = await inner_task()
            assert retrieved is ctx

        assert get_current_trace_context() is None

    @pytest.mark.asyncio
    async def test_async_task_isolation(self):
        """Validates that concurrent asyncio tasks maintain isolated contexts."""
        ctx_a = TraceContext()
        ctx_b = TraceContext()

        async def worker_a() -> Any:
            with trace_scope(ctx_a):
                await asyncio.sleep(0.01)
                return get_current_trace_context()

        async def worker_b() -> Any:
            with trace_scope(ctx_b):
                await asyncio.sleep(0.01)
                return get_current_trace_context()

        res_a, res_b = await asyncio.gather(worker_a(), worker_b())
        assert res_a is ctx_a
        assert res_b is ctx_b


# ==============================================================================
# Category 5: Token Accounting (TokenTracker) Tests
# ==============================================================================


class TestTokenTracker:
    def test_initial_state(self):
        tracker = TokenTracker()
        assert tracker.prompt_tokens == 0
        assert tracker.completion_tokens == 0
        assert tracker.total_tokens == 0
        assert tracker.cost_estimate == 0.0

    def test_record_accumulates_accurately(self):
        tracker = TokenTracker()
        tracker.record(prompt_tokens=100, completion_tokens=50)
        tracker.record(prompt_tokens=200, completion_tokens=100)

        assert tracker.prompt_tokens == 300
        assert tracker.completion_tokens == 150
        assert tracker.total_tokens == 450
        assert tracker.cost_estimate > 0.0

    def test_record_zero_tokens(self):
        tracker = TokenTracker()
        tracker.record(prompt_tokens=0, completion_tokens=0)
        assert tracker.total_tokens == 0

    def test_reject_negative_tokens(self):
        tracker = TokenTracker()
        with pytest.raises(ValueError):
            tracker.record(prompt_tokens=-10, completion_tokens=50)
        with pytest.raises(ValueError):
            tracker.record(prompt_tokens=50, completion_tokens=-5)

    def test_model_pricing_and_cost_estimation(self):
        tracker = TokenTracker()
        tracker.record(prompt_tokens=1000, completion_tokens=1000, model="gpt-4")
        metrics = tracker.get_metrics()
        assert abs(metrics["cost_estimate"] - 0.09) < 1e-5
        assert "gpt-4" in metrics["model_breakdown"]

    def test_thread_safe_token_accumulation(self):
        """Validates thread-safety under heavy concurrent multithreading."""
        tracker = TokenTracker()
        calls_per_thread = 50
        tokens_per_call = 10

        def worker() -> None:
            for _ in range(calls_per_thread):
                tracker.record(
                    prompt_tokens=tokens_per_call,
                    completion_tokens=tokens_per_call,
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker) for _ in range(10)]
            concurrent.futures.wait(futures)

        expected_total = 10 * calls_per_thread * (tokens_per_call * 2)
        assert tracker.total_tokens == expected_total


# ==============================================================================
# Category 6: Token Budget Limits & Quota Exhaustion Tests
# ==============================================================================


class TestQuotaEnforcement:
    def test_quota_exceeded_error_raised(self):
        """Validates QuotaExceededError is raised when exceeding budget."""
        tracker = TokenTracker(budget_limit=1000)
        tracker.record(prompt_tokens=400, completion_tokens=400)

        with pytest.raises(QuotaExceededError) as exc_info:
            tracker.record(prompt_tokens=150, completion_tokens=100)

        err = exc_info.value
        assert err.budget_limit == 1000
        assert err.current_tokens == 800
        assert err.requested_tokens == 250
        assert tracker.total_tokens == 800

    def test_quota_dynamic_budget_update(self):
        tracker = TokenTracker(budget_limit=500)
        tracker.record(prompt_tokens=200, completion_tokens=200)

        with pytest.raises(QuotaExceededError):
            tracker.record(prompt_tokens=100, completion_tokens=100)

        tracker.set_budget(1000)
        tracker.record(prompt_tokens=100, completion_tokens=100)
        assert tracker.total_tokens == 600

    def test_concurrent_quota_exhaustion(self):
        """Validates concurrent threads respect quota without over-allocation."""
        tracker = TokenTracker(budget_limit=1000)
        successes = 0
        failures = 0
        lock = threading.Lock()

        def worker() -> None:
            nonlocal successes, failures
            try:
                tracker.record(prompt_tokens=50, completion_tokens=50)
                with lock:
                    successes += 1
            except QuotaExceededError:
                with lock:
                    failures += 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker) for _ in range(25)]
            concurrent.futures.wait(futures)

        assert tracker.total_tokens <= 1000
        assert successes == 10
        assert failures == 15


# ==============================================================================
# Category 7: Structured JSON Audit Logging Tests
# ==============================================================================


class TestAuditLogging:
    def test_audit_log_tool_call(self):
        audit = AuditLogger()
        rec = audit.log_tool_call(
            trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
            span_id="00f067aa0ba902b7",
            actor_id="agent_analyst",
            tool_name="query_ledger",
            arguments={"account_id": "ACC-999"},
            status="SUCCESS",
            result={"status": "CLEAN"},
        )
        assert rec.event_type == "TOOL_CALL"
        assert rec.actor_id == "agent_analyst"
        assert rec.target == "query_ledger"
        assert rec.status == "SUCCESS"
        assert rec.details["arguments"]["account_id"] == "ACC-999"

    def test_audit_log_agent_interaction(self):
        audit = AuditLogger()
        rec = audit.log_agent_interaction(
            trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
            span_id="00f067aa0ba902b7",
            sender_id="requester",
            recipient_id="worker",
            performative="REQUEST",
            status="SENT",
            payload={"task": "audit"},
        )
        assert rec.event_type == "AGENT_INTERACTION"
        assert rec.actor_id == "requester"
        assert rec.target == "worker"
        assert rec.details["performative"] == "REQUEST"

    def test_audit_log_security_check(self):
        audit = AuditLogger()
        rec = audit.log_security_check(
            trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
            span_id="00f067aa0ba902b7",
            actor_id="agent_worker",
            check_type="file_access",
            target="/etc/passwd",
            status="DENIED",
            details={"mode": "r", "reason": "Path traversal blocked"},
        )
        assert rec.event_type == "SECURITY_CHECK"
        assert rec.status == "DENIED"
        assert rec.target == "/etc/passwd"

    def test_audit_json_export_and_filtering(self):
        audit = AuditLogger()
        audit.log_tool_call("trace_1", "span_1", "actor_1", "tool_a", {})
        audit.log_tool_call("trace_2", "span_2", "actor_2", "tool_b", {})

        t1_records = audit.export_json(trace_id="trace_1")
        assert len(t1_records) == 1
        assert t1_records[0]["trace_id"] == "trace_1"

        jsonl_str = audit.export_jsonl()
        lines = jsonl_str.strip().split("\n")
        assert len(lines) == 2
        for raw_line in lines:
            parsed = json.loads(raw_line)
            assert "timestamp" in parsed
            assert "event_type" in parsed


# ==============================================================================
# Category 8: Cross-Agent & Protocol Wire Propagation Tests
# ==============================================================================


class TestProtocolWirePropagation:
    def test_a2a_traceparent_wire_propagation(self):
        """Validates traceparent propagation across A2AMessage envelope and replies."""
        trace_ctx = TraceContext()
        root_span = trace_ctx.start_span("agent_a_root")

        msg = A2AMessage(
            sender_id="agent_a",
            recipient_id="agent_b",
            performative=Performative.REQUEST,
            payload={"action": "verify"},
            traceparent=trace_ctx.to_traceparent(),
        )

        assert msg.traceparent is not None
        _v, tid, sid, _flags = validate_traceparent(msg.traceparent)
        assert tid == trace_ctx.trace_id
        assert sid == root_span.span_id

        # Recipient B creates reply
        reply = msg.create_reply(
            performative=Performative.INFORM,
            payload={"status": "VERIFIED"},
        )
        assert reply.traceparent == msg.traceparent

    def test_mcp_meta_trace_propagation(self):
        """Validates injecting and extracting W3C headers into MCP tool request params._meta."""
        trace_ctx = TraceContext()
        trace_ctx.start_span("mcp_caller")

        mcp_params: dict[str, Any] = {
            "name": "query_db",
            "arguments": {"user_id": 42},
            "_meta": {},
        }
        inject_w3c_trace_context(mcp_params["_meta"], trace_ctx.to_traceparent())

        tp, _ts = extract_w3c_trace_context(mcp_params["_meta"])
        assert tp is not None
        assert tp == trace_ctx.to_traceparent()


# ==============================================================================
# Category 9: AgentTracer Registry & Regression Tests
# ==============================================================================


class TestAgentTracerRegistry:
    def test_get_or_create_context_preserves_existing_regression(self):
        """Regression test for TEST_INFRA.md defect escalation:

        Verify get_or_create_context does NOT overwrite existing TraceContext!
        """
        tracer = AgentTracer()
        ctx1 = tracer.get_or_create_context("trace-persistent-100")
        span1 = ctx1.start_span("first_span")
        ctx1.record_token_usage(prompt=100, completion=50)

        # Call again with same trace_id
        ctx2 = tracer.get_or_create_context("trace-persistent-100")
        assert ctx1 is ctx2
        assert len(ctx2.spans) == 1
        assert ctx2.spans[0] is span1
        assert ctx2.total_tokens_consumed == 150

    def test_export_summary_contract(self):
        """Validates export_summary payload matching E2E tier contracts."""
        tracer = AgentTracer()
        ctx = tracer.get_or_create_context("trace-summary-test")
        span = ctx.start_span("work_span")
        ctx.record_token_usage(prompt=250, completion=75)
        span.finish()

        summary = tracer.export_summary("trace-summary-test")
        assert summary["trace_id"] == "trace-summary-test"
        assert summary["span_count"] == 1
        assert summary["total_tokens"] == 325
        assert summary["prompt_tokens"] == 250
        assert summary["completion_tokens"] == 75
        assert summary["cost_estimate"] > 0.0
        assert len(summary["spans"]) == 1

    def test_unknown_trace_id_summary_returns_empty(self):
        tracer = AgentTracer()
        assert tracer.export_summary("nonexistent-trace-id") == {}
