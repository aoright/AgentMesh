"""Empirical stress test suite challenging Milestone 3 (W3C Telemetry & Concurrency).

Adversarially challenges:
1. W3C TraceContext Conformance:
   - Malformed traceparents (bad lengths, non-hex characters, version ff, all-zeros).
   - Future version compatibility (version > 00 with extra fields and core validation).
   - Tracestate limits (32-member maximum limit, 512-character limit, key/value syntax, duplicate keys).
   - Ambient async context propagation across concurrent coroutines and nested scopes.
2. Token Accounting Race Conditions:
   - High-concurrency multithreaded token accumulation across 20+ threads.
   - Exact atomic token counts without loss or double counting.
   - Atomic budget quota enforcement (exact cutoff, zero budget leakage, atomic rollback).
   - Concurrent reader/writer metric consistency.
3. Structured Audit Logging:
   - Concurrent audit log appending across 20+ threads.
   - Physical state separation: zero database binaries or state files inside repository tree.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import secrets
import tempfile
import threading
from pathlib import Path

import pytest

from agentmesh.telemetry.tracer import (
    AuditLogger,
    QuotaExceededError,
    TokenTracker,
    TraceContext,
)
from agentmesh.telemetry.w3c import (
    TraceContextValidationError,
    TraceParent,
    TraceState,
    get_current_trace_context,
    trace_scope,
    validate_traceparent,
)

# ==============================================================================
# Challenge Suite 1: W3C TraceParent Header Conformance & Adversarial Vectors
# ==============================================================================


class TestW3CTraceParentAdversarial:
    @pytest.mark.parametrize(
        "malformed_header,description",
        [
            ("", "Empty string"),
            ("   ", "Whitespace string"),
            ("00", "Only version field"),
            ("00-4bf92f3577b34da6a3ce929d0e0e4736", "Missing span_id and flags"),
            ("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7", "Missing flags"),
            (
                "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-",
                "Trailing hyphen in v00 (5 fields)",
            ),
            (
                "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra",
                "Extra field in v00",
            ),
            (
                "0-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
                "Version length 1",
            ),
            (
                "000-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
                "Version length 3",
            ),
            (
                "0g-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
                "Non-hex version character 'g'",
            ),
            (
                "zz-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
                "Non-hex version 'zz'",
            ),
            (
                "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
                "Explicitly forbidden version 'ff'",
            ),
            (
                "FF-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
                "Uppercase forbidden version 'FF'",
            ),
            (
                "00-4bf92f3577b34da6a3ce929d0e0e473-00f067aa0ba902b7-01",
                "Trace ID length 31",
            ),
            (
                "00-4bf92f3577b34da6a3ce929d0e0e47366-00f067aa0ba902b7-01",
                "Trace ID length 33",
            ),
            (
                "00-4bf92f3577b34da6a3ce929d0e0e473g-00f067aa0ba902b7-01",
                "Non-hex char in trace_id",
            ),
            (
                "00-4bf92f3577b34da6a3ce929d0e0e473 -00f067aa0ba902b7-01",
                "Space in trace_id",
            ),
            (
                "00-4bf92f3577b34da6a3ce929d0e0e473\x00-00f067aa0ba902b7-01",
                "Null byte in trace_id",
            ),
            (
                "00-00000000000000000000000000000000-00f067aa0ba902b7-01",
                "All-zero trace_id",
            ),
            (
                "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b-01",
                "Span ID length 15",
            ),
            (
                "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b77-01",
                "Span ID length 17",
            ),
            (
                "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902bz-01",
                "Non-hex char in span_id",
            ),
            (
                "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",
                "All-zero span_id",
            ),
            (
                "00-00000000000000000000000000000000-0000000000000000-01",
                "Both trace_id and span_id all-zero",
            ),
            (
                "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-1",
                "Trace flags length 1",
            ),
            (
                "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-001",
                "Trace flags length 3",
            ),
            (
                "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-0z",
                "Non-hex trace flags",
            ),
            (
                "00-4bf92f3577b34da6a3ce929d0e0e47361-0f067aa0ba902b7-01",
                "Length 55 total but trace_id 33 and span_id 15",
            ),
        ],
    )
    def test_malformed_traceparent_rejected(self, malformed_header: str, description: str):
        """Verifies that all malformed traceparent headers are rejected."""
        with pytest.raises(TraceContextValidationError):
            validate_traceparent(malformed_header)

    @pytest.mark.parametrize(
        "future_header,exp_version,exp_trace_id,exp_span_id,exp_flags",
        [
            (
                "01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
                "01",
                "4bf92f3577b34da6a3ce929d0e0e4736",
                "00f067aa0ba902b7",
                "01",
            ),
            (
                "02-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00-future-metadata",
                "02",
                "4bf92f3577b34da6a3ce929d0e0e4736",
                "00f067aa0ba902b7",
                "00",
            ),
            (
                "fe-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-ext1-ext2-ext3",
                "fe",
                "4bf92f3577b34da6a3ce929d0e0e4736",
                "00f067aa0ba902b7",
                "01",
            ),
        ],
    )
    def test_future_version_forward_compatibility(
        self,
        future_header: str,
        exp_version: str,
        exp_trace_id: str,
        exp_span_id: str,
        exp_flags: str,
    ):
        """Verifies that future versions (>00) allow forward-compatible extension fields."""
        v, tid, sid, flags = validate_traceparent(future_header)
        assert v == exp_version
        assert tid == exp_trace_id
        assert sid == exp_span_id
        assert flags == exp_flags

    @pytest.mark.parametrize(
        "invalid_future_header,reason",
        [
            (
                "01-00000000000000000000000000000000-00f067aa0ba902b7-01-extra",
                "All-zero trace_id in future version",
            ),
            (
                "01-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01-extra",
                "All-zero span_id in future version",
            ),
            (
                "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra",
                "Version ff in future version with extra fields",
            ),
            (
                "01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7",
                "Fewer than 4 fields in future version",
            ),
            (
                "1-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
                "Single hex digit version",
            ),
            (
                "0g-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra",
                "Non-hex future version",
            ),
        ],
    )
    def test_future_version_core_field_validation(
        self, invalid_future_header: str, reason: str
    ):
        """Verifies that future versions still enforce core W3C invariant checks."""
        with pytest.raises(TraceContextValidationError):
            validate_traceparent(invalid_future_header)

    def test_traceparent_create_and_parse_adversarial(self):
        """Validates TraceParent model methods directly under invalid inputs."""
        with pytest.raises(TraceContextValidationError):
            TraceParent.create(version="ff")
        with pytest.raises(TraceContextValidationError):
            TraceParent.create(trace_id="0" * 32)
        with pytest.raises(TraceContextValidationError):
            TraceParent.parse("00-invalid-span-01")

        valid_tp = TraceParent.create(sampled=True)
        assert valid_tp.is_sampled is True
        child_tp = valid_tp.create_child()
        assert child_tp.trace_id == valid_tp.trace_id
        assert child_tp.span_id != valid_tp.span_id


# ==============================================================================
# Challenge Suite 2: W3C TraceState Header Limits & Spec Conformance
# ==============================================================================


class TestW3CTraceStateAdversarial:
    def test_tracestate_32_member_limit_on_set(self):
        """Verifies that adding more than 32 members prunes the oldest entries."""
        ts = TraceState()
        for i in range(40):
            ts.set(f"v{i}", f"val{i}")
        assert len(ts) == 32
        assert "v39" in ts
        assert "v8" in ts
        assert "v7" not in ts
        assert "v0" not in ts

    def test_tracestate_512_char_limit_on_set(self):
        """Verifies that adding entries exceeding 512 total characters prunes oldest entries."""
        ts = TraceState()
        # Add 10 entries of length 60 each (600+ chars)
        for i in range(10):
            ts.set(f"k{i}", "v" * 55)
        formatted = ts.format()
        assert len(formatted) <= 512
        assert len(ts) < 10
        assert "k9" in ts

    def test_tracestate_duplicate_keys_preserve_first(self):
        """W3C Section 3.3: First occurrence of duplicate key must be preserved."""
        raw = "alpha=first,beta=2,alpha=second,gamma=3,alpha=third"
        ts = TraceState.parse(raw)
        assert len(ts) == 3
        assert ts.get("alpha") == "first"
        assert ts.get("beta") == "2"
        assert ts.get("gamma") == "3"

    def test_tracestate_update_key_moves_to_front(self):
        """W3C Section 3.3: Setting an existing key moves it to position 0."""
        ts = TraceState.parse("first=1,second=2,third=3")
        ts.set("second", "updated")
        assert ts.format().startswith("second=updated")
        assert ts.get("second") == "updated"
        assert len(ts) == 3

    @pytest.mark.parametrize(
        "bad_key",
        [
            "UPPERCASE",
            "key with spaces",
            "key;semicolon",
            "key,comma",
            "key=equals",
            "@system_only",
            "tenant@system@extra",
            "",
        ],
    )
    def test_tracestate_invalid_key_rejected(self, bad_key: str):
        """Verifies invalid keys are rejected with TraceContextValidationError."""
        ts = TraceState()
        with pytest.raises(TraceContextValidationError):
            ts.set(bad_key, "valid_value")

    @pytest.mark.parametrize(
        "bad_val",
        [
            "value with space",
            "value,comma",
            "value=equal",
            "value\tcontrol",
            "value\nnewline",
            "",
            "x" * 257,
        ],
    )
    def test_tracestate_invalid_value_rejected(self, bad_val: str):
        """Verifies invalid values are rejected with TraceContextValidationError."""
        ts = TraceState()
        with pytest.raises(TraceContextValidationError):
            ts.set("validkey", bad_val)

    def test_tracestate_parse_32_member_limit(self):
        """Verifies that parsing an incoming header with > 32 entries truncates to 32."""
        raw = ",".join(f"k{i:02d}=v{i:02d}" for i in range(45))
        ts = TraceState.parse(raw)
        assert len(ts) == 32

    def test_tracestate_parse_512_char_limit(self):
        """W3C Section 3.3.1.2: tracestate MUST NOT exceed 512 characters upon parse."""
        raw = ",".join(f"k{i:02d}=" + "a" * 25 for i in range(25))
        assert len(raw) > 512, "Precondition: input header must exceed 512 chars"
        ts = TraceState.parse(raw)
        formatted = ts.format()
        assert (
            len(formatted) <= 512
        ), f"tracestate parsed length {len(formatted)} exceeds 512 character limit!"


# ==============================================================================
# Challenge Suite 3: Ambient Async Context Propagation Concurrency
# ==============================================================================


class TestAsyncContextConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_async_context_isolation(self):
        """Verifies 30 concurrent coroutines preserve isolated trace scopes."""
        contexts = [TraceContext(trace_id=secrets.token_hex(16)) for _ in range(30)]

        async def worker(expected_ctx: TraceContext) -> bool:
            with trace_scope(expected_ctx):
                for _ in range(5):
                    if get_current_trace_context() is not expected_ctx:
                        return False
                    await asyncio.sleep(0.002)
                    if get_current_trace_context() is not expected_ctx:
                        return False
            return True

        tasks = [worker(ctx) for ctx in contexts]
        results = await asyncio.gather(*tasks)
        assert all(results)
        assert get_current_trace_context() is None


# ==============================================================================
# Challenge Suite 4: Token Accounting Concurrency Stress & Race Conditions
# ==============================================================================


class TestTokenAccountingConcurrencyStress:
    def test_high_concurrency_token_accumulation_30_threads(self):
        """Simulates 30 concurrent threads performing 500 token updates each.

        Asserts exact atomic token counts without loss or double counting.
        """
        tracker = TokenTracker()
        num_threads = 30
        ops_per_thread = 500
        p_val = 11
        c_val = 19

        def worker(thread_idx: int) -> None:
            models = ["default", "gpt-4", "gpt-4o", "claude-3-5-sonnet"]
            m = models[thread_idx % len(models)]
            for _ in range(ops_per_thread):
                tracker.record(prompt_tokens=p_val, completion_tokens=c_val, model=m)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            concurrent.futures.wait(futures)

        expected_prompt = num_threads * ops_per_thread * p_val
        expected_completion = num_threads * ops_per_thread * c_val
        expected_total = expected_prompt + expected_completion

        assert tracker.prompt_tokens == expected_prompt
        assert tracker.completion_tokens == expected_completion
        assert tracker.total_tokens == expected_total

        metrics = tracker.get_metrics()
        breakdown_total = sum(v["total"] for v in metrics["model_breakdown"].values())
        assert breakdown_total == expected_total

    def test_atomic_quota_enforcement_and_rollback_25_threads(self):
        """Simulates 25 concurrent threads competing against a strict quota.

        Verifies exact atomic cutoff, zero budget leakage, and atomic rollback.
        """
        budget = 3000
        tracker = TokenTracker(budget_limit=budget)
        successful_tokens = 0
        successful_calls = 0
        quota_exceeded_count = 0
        lock = threading.Lock()

        def worker(thread_idx: int) -> None:
            nonlocal successful_tokens, successful_calls, quota_exceeded_count
            token_count = 15 + (thread_idx % 20)
            p = token_count // 2
            c = token_count - p
            try:
                tracker.record(prompt_tokens=p, completion_tokens=c)
                with lock:
                    successful_tokens += token_count
                    successful_calls += 1
            except QuotaExceededError:
                with lock:
                    quota_exceeded_count += 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
            # 250 total concurrent requests
            futures = [executor.submit(worker, i) for i in range(250)]
            concurrent.futures.wait(futures)

        # Assert no budget leakage
        assert (
            tracker.total_tokens <= budget
        ), f"Budget leakage detected: {tracker.total_tokens} > {budget}"
        # Assert exact match with tracked successful additions
        assert tracker.total_tokens == successful_tokens
        assert quota_exceeded_count > 0

    def test_concurrent_mixed_operations_and_invariants(self):
        """Tests concurrent records, metrics reads, and property accesses."""
        tracker = TokenTracker()
        stop_signal = threading.Event()
        inconsistencies: list[str] = []

        def recorder(idx: int) -> None:
            models = ["default", "gpt-4", "gpt-4o", "claude-3-5-sonnet"]
            for i in range(300):
                tracker.record(prompt_tokens=3, completion_tokens=7, model=models[i % 4])

        def reader(idx: int) -> None:
            while not stop_signal.is_set():
                m = tracker.get_metrics()
                if m["total_tokens"] != m["prompt_tokens"] + m["completion_tokens"]:
                    inconsistencies.append("Metric total != prompt + completion")
                breakdown_sum = sum(v["total"] for v in m["model_breakdown"].values())
                if m["total_tokens"] != breakdown_sum:
                    inconsistencies.append("Breakdown sum mismatch")

        with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
            read_futures = [executor.submit(reader, i) for i in range(5)]
            rec_futures = [executor.submit(recorder, i) for i in range(20)]
            concurrent.futures.wait(rec_futures)
            stop_signal.set()
            concurrent.futures.wait(read_futures)

        assert not inconsistencies, f"Inconsistencies detected: {inconsistencies}"
        assert tracker.total_tokens == 20 * 300 * 10


# ==============================================================================
# Challenge Suite 5: Structured AuditLogger Concurrency & State Separation
# ==============================================================================


class TestAuditLoggerConcurrencyAndStateSeparation:
    def test_concurrent_audit_logging_25_threads(self):
        """Simulates 25 concurrent threads appending structured audit events.

        Asserts exact record counts in memory and valid JSONL on disk.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit_stress.jsonl"
            logger = AuditLogger(log_file=log_path)
            num_threads = 25
            ops_per_thread = 40

            def worker(t_idx: int) -> None:
                for i in range(ops_per_thread):
                    logger.log_tool_call(
                        trace_id=f"trace_{t_idx}_{i}",
                        span_id=f"span_{t_idx}_{i}",
                        actor_id=f"actor_{t_idx}",
                        tool_name="sandbox_tool",
                        arguments={"thread": t_idx, "op": i},
                        status="SUCCESS",
                        result={"status": "OK"},
                    )

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker, i) for i in range(num_threads)]
                concurrent.futures.wait(futures)

            expected_records = num_threads * ops_per_thread
            records = logger.get_records()
            assert len(records) == expected_records

            # Verify on-disk JSONL integrity
            assert log_path.exists()
            lines = [line for line in log_path.read_text().split("\n") if line.strip()]
            assert len(lines) == expected_records
            for line in lines:
                obj = json.loads(line)
                assert obj["event_type"] == "TOOL_CALL"
                assert "timestamp" in obj
                assert "timestamp_ns" in obj
                assert obj["target"] == "sandbox_tool"

    def test_physical_state_separation_compliance(self):
        """Audits repository workspace for zero database binaries or runtime state."""
        repo_root = Path(__file__).resolve().parent.parent
        blocked_suffixes = (".db", ".sqlite", ".sqlite3", ".wal", ".shm", ".dump")

        offending_files: list[str] = []
        for path in repo_root.rglob("*"):
            if not path.is_file():
                continue
            # Ignore hidden cache directories like .mypy_cache, .pytest_cache, .git
            parts = path.parts
            if any(p.startswith(".") and p not in (".agents",) for p in parts):
                continue
            for sfx in blocked_suffixes:
                if path.name.endswith(sfx):
                    offending_files.append(str(path))

        assert (
            not offending_files
        ), f"Found database files violating physical state separation: {offending_files}"
