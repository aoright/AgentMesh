"""Dedicated Tier 5 Adversarial Coverage Hardening Test Suite.

Exhaustively stress-tests unexercised branches, edge cases, error conditions,
and boundary attacks across engine, mesh, security, and telemetry modules.
Complies with strict physical state separation and zero emojis.
"""

from __future__ import annotations

import json
import os
import time
from enum import Enum
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from agentmesh.engine.activity import (
    ActivityDefinition,
    ActivityExecutionContext,
    ActivityRecord,
    ActivityStatus,
    RetryPolicy,
    activity,
    current_activity_context,
    execute_activity,
)
from agentmesh.engine.checkpoint import CheckpointManager, StateSeparationError
from agentmesh.engine.graph import Graph
from agentmesh.engine.memoizer import (
    ActivityMemoizer,
    IdempotencyConflictError,
    canonical_json_dump,
)
from agentmesh.engine.replay import (
    CryptographicIntegrityError,
    EventReplayer,
    ResumptionError,
    ResumptionPlanner,
)
from agentmesh.engine.state import (
    Event,
    EventType,
    State,
)
from agentmesh.mesh.a2a import AgentCapability, AgentCard
from agentmesh.mesh.discovery import ServiceRegistry
from agentmesh.mesh.mcp_client import (
    HttpSseTransport,
    InMemoryTransport,
    MCPToolCallRequest,
    MCPToolClient,
    MCPTransportError,
    StdioTransport,
)
from agentmesh.mesh.mcp_protocol import (
    JSONRPCError,
    JSONRPCErrorCode,
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPMethod,
    MCPResource,
    MCPTextContent,
    ToolsCallResult,
)
from agentmesh.mesh.router import (
    AgentEndpoint,
    MeshRouter,
    SmoothWeightedRoundRobin,
)
from agentmesh.security.policy import CapabilityPolicy, CommandRule
from agentmesh.security.sandbox import (
    PermissionDeniedError,
    SecuritySandbox,
)
from agentmesh.telemetry.tracer import (
    AgentTracer,
    QuotaExceededError,
    TokenTracker,
    TraceContext,
    TraceContextValidationError,
)
from agentmesh.telemetry.w3c import (
    TraceParent,
    TraceState,
    extract_w3c_trace_context,
)


@pytest.fixture
def isolated_env(tmp_path: Path):
    """Provides isolated state directory outside repository tree."""
    state_dir = tmp_path / "tier5_state_isolated"
    state_dir.mkdir(parents=True, exist_ok=True)
    orig = os.environ.get("AGENTMESH_STATE_DIR")
    os.environ["AGENTMESH_STATE_DIR"] = str(state_dir)
    yield state_dir
    if orig is not None:
        os.environ["AGENTMESH_STATE_DIR"] = orig
    else:
        os.environ.pop("AGENTMESH_STATE_DIR", None)


# ---------------------------------------------------------------------------
# 1. Engine: Activity & Retry Policy Hardening
# ---------------------------------------------------------------------------


class TestTier5EngineActivityHardening:
    """Stress tests activity execution, retries, and edge cases."""

    def test_retry_policy_behavior(self):
        policy = RetryPolicy(
            max_attempts=3,
            initial_backoff_ms=50.0,
            backoff_multiplier=2.0,
            max_backoff_ms=150.0,
            retryable_exceptions=["TimeoutError", "ConnectionError"],
        )
        # Attempt exceeding max
        assert not policy.should_retry(3, TimeoutError("timed out"))
        assert not policy.should_retry(4, TimeoutError("timed out"))

        # Retryable exceptions
        assert policy.should_retry(1, TimeoutError("timed out"))
        assert policy.should_retry(2, ConnectionError("conn failed"))

        # Non-retryable exception
        assert not policy.should_retry(1, ValueError("bad value"))

        # Delays
        assert policy.get_delay_seconds(1) == 0.05
        assert policy.get_delay_seconds(2) == 0.10
        # Capped at max_backoff_ms (150ms -> 0.15s)
        assert policy.get_delay_seconds(5) == 0.15

    def test_activity_definition(self):
        def dummy():
            return 42

        policy = RetryPolicy(max_attempts=2)
        defn = ActivityDefinition("dummy_act", dummy, timeout_seconds=10.0, retry_policy=policy)
        assert defn.name == "dummy_act"
        assert defn.timeout_seconds == 10.0
        assert defn.retry_policy.max_attempts == 2

    @pytest.mark.asyncio
    async def test_sync_activity_execution_with_timeout(self, isolated_env: Path):
        memoizer = ActivityMemoizer(storage_dir=str(isolated_env))
        state = State(workflow_id="wf-sync-act")
        ctx = ActivityExecutionContext(
            workflow_id="wf-sync-act",
            node_id="node_sync",
            memoizer=memoizer,
            state=state,
        )

        def sync_worker(x: int, y: int) -> int:
            return x * y

        token = current_activity_context.set(ctx)
        try:
            res = await execute_activity(
                activity_name="sync_mul",
                fn=sync_worker,
                args=(6, 7),
                timeout_seconds=5.0,
                context=ctx,
            )
            assert res == 42
        finally:
            current_activity_context.reset(token)

    @pytest.mark.asyncio
    async def test_activity_retry_and_eventual_failure(self, isolated_env: Path):
        memoizer = ActivityMemoizer(storage_dir=str(isolated_env))
        state = State(workflow_id="wf-fail-retry")
        ctx = ActivityExecutionContext(
            workflow_id="wf-fail-retry",
            node_id="node_fail",
            memoizer=memoizer,
            state=state,
        )

        attempts_seen = 0

        async def failing_worker():
            nonlocal attempts_seen
            attempts_seen += 1
            raise RuntimeError(f"Simulated fault attempt {attempts_seen}")

        policy = RetryPolicy(
            max_attempts=2,
            initial_backoff_ms=5.0,
            backoff_multiplier=1.0,
            retryable_exceptions=["RuntimeError"],
        )

        token = current_activity_context.set(ctx)
        try:
            with pytest.raises(RuntimeError, match="Simulated fault attempt 2"):
                await execute_activity(
                    activity_name="failing_act",
                    fn=failing_worker,
                    retry_policy=policy,
                    context=ctx,
                )
        finally:
            current_activity_context.reset(token)

        assert attempts_seen == 2
        # Verify failure recorded in memoizer and state
        failed_events = [e for e in state.events if e.event_type == EventType.ACTIVITY_FAILED]
        assert len(failed_events) == 1
        assert failed_events[0].payload["attempts"] == 2

    @pytest.mark.asyncio
    async def test_activity_cached_failure_logging(self, isolated_env: Path):
        memoizer = ActivityMemoizer(storage_dir=str(isolated_env))
        state = State(workflow_id="wf-cached-fail")
        ctx = ActivityExecutionContext(
            workflow_id="wf-cached-fail",
            node_id="node_cfail",
            memoizer=memoizer,
            state=state,
        )

        key = memoizer.compute_key("wf-cached-fail", "node_cfail", "act_cached_fail", args=(1,))
        fail_rec = ActivityRecord(
            idempotency_key=key,
            activity_name="act_cached_fail",
            node_id="node_cfail",
            workflow_id="wf-cached-fail",
            status=ActivityStatus.FAILED,
            error="Previous fatal fault",
        )
        memoizer.put(fail_rec)

        # On cache miss for success, it logs warning and re-invokes
        async def worker(v: int) -> int:
            return v + 10

        token = current_activity_context.set(ctx)
        try:
            res = await execute_activity(
                activity_name="act_cached_fail",
                fn=worker,
                args=(1,),
                context=ctx,
            )
            assert res == 11
        finally:
            current_activity_context.reset(token)

    @pytest.mark.asyncio
    async def test_activity_token_accounting_formats(self, isolated_env: Path):
        memoizer = ActivityMemoizer(storage_dir=str(isolated_env))
        tracer = AgentTracer()
        state = State(workflow_id="wf-tokens")
        ctx = ActivityExecutionContext(
            workflow_id="wf-tokens",
            node_id="node_tok",
            memoizer=memoizer,
            state=state,
            tracer=tracer,
        )

        async def worker_with_tokens_spent():
            return {"result": "ok", "tokens_spent": 150}

        token = current_activity_context.set(ctx)
        try:
            res = await execute_activity(
                activity_name="tok_act_1",
                fn=worker_with_tokens_spent,
                context=ctx,
            )
            assert res["result"] == "ok"
        finally:
            current_activity_context.reset(token)

        completed_events = [e for e in state.events if e.event_type == EventType.ACTIVITY_COMPLETED]
        assert len(completed_events) == 1
        assert completed_events[0].payload["total_tokens"] == 150

    @pytest.mark.asyncio
    async def test_activity_decorator(self, isolated_env: Path):
        memoizer = ActivityMemoizer(storage_dir=str(isolated_env))
        state = State(workflow_id="wf-decor")
        ctx = ActivityExecutionContext(
            workflow_id="wf-decor",
            node_id="node_decor",
            memoizer=memoizer,
            state=state,
        )

        @activity(name="decorated_calc", timeout_seconds=5.0)
        async def decorated_calc(a: int, b: int) -> int:
            return a + b

        token = current_activity_context.set(ctx)
        try:
            res = await decorated_calc(10, 20)
            assert res == 30
            assert hasattr(decorated_calc, "__activity_name__")
            assert decorated_calc.__activity_name__ == "decorated_calc"
        finally:
            current_activity_context.reset(token)


# ---------------------------------------------------------------------------
# 2. Engine: Checkpoint & State Separation Hardening
# ---------------------------------------------------------------------------


class TestTier5EngineCheckpointHardening:
    """Stress tests checkpoint manager invariants and physical state separation."""

    def test_validate_state_dir_inside_repo_raises_error(self):
        repo_root = CheckpointManager._detect_repo_root()
        assert repo_root is not None
        inside_path = repo_root / "subdir_inside_repo"
        with pytest.raises(StateSeparationError, match="Physical State Separation Violation"):
            CheckpointManager.validate_state_dir(inside_path)

    def test_checkpoint_load_corrupted_manifest_invariants(self, isolated_env: Path):
        mgr = CheckpointManager(base_dir=str(isolated_env))
        wf_dir = isolated_env / "wf-corrupt"
        wf_dir.mkdir(parents=True, exist_ok=True)
        ckpt_file = wf_dir / "ckpt_000001_test.json"

        # Case 1: events_count negative
        ckpt_file.write_text(json.dumps({"events": [], "events_count": -1}))
        with pytest.raises(CryptographicIntegrityError, match="must be a non-negative integer"):
            mgr.load_from_file(str(ckpt_file))

        # Case 2: events_count string instead of int
        ckpt_file.write_text(json.dumps({"events": [], "events_count": "invalid"}))
        with pytest.raises(CryptographicIntegrityError, match="must be a non-negative integer"):
            mgr.load_from_file(str(ckpt_file))

        # Case 3: events_count mismatch (manifest says 5, actual is 0)
        ckpt_file.write_text(json.dumps({"events": [], "events_count": 5}))
        with pytest.raises(CryptographicIntegrityError, match="Checkpoint event count mismatch"):
            mgr.load_from_file(str(ckpt_file))

        # Case 4: last_hash present but events empty
        ckpt_file.write_text(json.dumps({"events": [], "last_hash": "deadbeef"}))
        with pytest.raises(CryptographicIntegrityError, match="checkpoint records last_hash"):
            mgr.load_from_file(str(ckpt_file))

        # Case 5: events non-empty but last_hash empty
        dummy_event = {
            "event_id": "evt-01",
            "event_type": "STATE_MUTATED",
            "timestamp_ns": 1000,
            "state_hash": "abc",
            "prev_hash": "",
        }
        ckpt_file.write_text(json.dumps({"events": [dummy_event], "last_hash": ""}))
        with pytest.raises(CryptographicIntegrityError, match="checkpoint contains 1 events but last_hash is empty"):
            mgr.load_from_file(str(ckpt_file))

    def test_checkpoint_pruning_and_error_handling(self, isolated_env: Path):
        mgr = CheckpointManager(base_dir=str(isolated_env), keep_last_n=2)
        state = State(workflow_id="wf-prune")

        # Save 4 checkpoints
        for i in range(1, 5):
            state.append_event(EventType.STATE_MUTATED, payload={"step": i})
            mgr.save_checkpoint(state, label=f"step{i}")

        wf_dir = isolated_env / "wf-prune"
        remaining = sorted(wf_dir.glob("ckpt_*.json"))
        assert len(remaining) == 2

        # Pruning when count <= keep_last returns 0
        assert mgr.prune_checkpoints("wf-prune", keep_last=5) == 0

        # Handle OSError during prune safely
        with patch.object(Path, "unlink", side_effect=OSError("Permission denied")):
            deleted = mgr.prune_checkpoints("wf-prune", keep_last=1)
            assert deleted == 0


# ---------------------------------------------------------------------------
# 3. Engine: Memoizer & State Invariants Hardening
# ---------------------------------------------------------------------------


class TestTier5EngineMemoizerAndStateHardening:
    """Stress tests canonical serialization, custom idempotency conflicts, and state."""

    def test_canonical_json_dump_comprehensive_types(self):
        class StatusEnum(str, Enum):
            ACTIVE = "ACTIVE"

        class PydanticModel(BaseModel):
            name: str
            val: int

        class DictConvertible:
            def to_dict(self):
                return {"custom": "dict_converted"}

        class PathObj:
            def __fspath__(self):
                return "/tmp/agentmesh/path"

        class ClassWithDict:
            def __init__(self):
                self.prop = "hello"

        class FallbackRepr:
            __slots__ = ()

            def __repr__(self):
                return "<FallbackReprObject>"

        payload = {
            "enum": StatusEnum.ACTIVE,
            "pydantic": PydanticModel(name="test", val=99),
            "dictable": DictConvertible(),
            "fspath": Path("/tmp/agentmesh/path"),
            "set": {3, 1, 2},
            "bytes": b"\x01\x02\x03",
            "with_dict": ClassWithDict(),
            "fallback": FallbackRepr(),
        }

        serialized = canonical_json_dump(payload)
        parsed = json.loads(serialized)
        assert parsed["enum"] == "ACTIVE"
        assert parsed["pydantic"] == {"name": "test", "val": 99}
        assert parsed["dictable"] == {"custom": "dict_converted"}
        assert parsed["fspath"] == "/tmp/agentmesh/path"
        assert parsed["set"] == [1, 2, 3]
        assert parsed["bytes"] == "010203"
        assert parsed["with_dict"] == {"prop": "hello"}
        assert parsed["fallback"] == "<FallbackReprObject>"

    def test_memoizer_dict_args_and_custom_key_conflict(self, isolated_env: Path):
        memoizer = ActivityMemoizer(storage_dir=str(isolated_env))

        # Dict args merged into kwargs
        key1 = memoizer.compute_key("wf-1", "node-1", "act-1", args={"k1": "v1"})
        key2 = memoizer.compute_key("wf-1", "node-1", "act-1", kwargs={"k1": "v1"})
        assert key1 == key2

        # Custom key with matching and conflicting args
        ck = memoizer.compute_key("wf-1", "node-1", "act-1", args=(10,), custom_key="custom_id_100")
        assert ck == "custom_id_100"

        # Matching re-invocation works
        assert memoizer.compute_key("wf-1", "node-1", "act-1", args=(10,), custom_key="custom_id_100") == "custom_id_100"

        # Conflicting re-invocation raises IdempotencyConflictError
        with pytest.raises(IdempotencyConflictError, match="Idempotency key conflict"):
            memoizer.compute_key("wf-1", "node-1", "act-1", args=(20,), custom_key="custom_id_100")

    def test_memoizer_has_clear_and_disk_io(self, isolated_env: Path):
        memoizer = ActivityMemoizer(storage_dir=str(isolated_env))
        rec = ActivityRecord(
            idempotency_key="key_123",
            activity_name="test_act",
            node_id="n1",
            workflow_id="wf-disk",
            status=ActivityStatus.COMPLETED,
            result={"status": "ok"},
        )
        memoizer.put(rec)
        assert memoizer.has("key_123")
        assert not memoizer.has("key_absent")

        # Save to disk and reload
        memoizer.save_to_disk("wf-disk")
        memoizer.clear()
        assert not memoizer.has("key_123")

        loaded = memoizer.load_from_disk("wf-disk")
        assert loaded == 1
        assert memoizer.has("key_123")

        # Load nonexistent returns 0
        assert memoizer.load_from_disk("wf-nonexistent") == 0

    def test_state_legacy_timestamp_and_monotonicity(self):
        # Legacy float timestamp handling
        evt = Event.model_validate({
            "event_id": "evt-legacy",
            "event_type": EventType.STATE_MUTATED,
            "timestamp": 1609459200.5,
        })
        assert evt.timestamp_ns == 1609459200500000000
        assert abs(evt.timestamp - 1609459200.5) < 1e-6

        # Monotonic timestamp resolution
        state = State(workflow_id="wf-mono")
        e1 = state.append_event(EventType.STATE_MUTATED, timestamp_ns=1000)
        e2 = state.append_event(EventType.STATE_MUTATED, timestamp_ns=500)
        assert e1.timestamp_ns == 1000
        assert e2.timestamp_ns == 1001

        # State snapshot
        snap = state.snapshot()
        assert snap["workflow_id"] == "wf-mono"
        assert snap["events_count"] == 2
        assert snap["last_sequence_num"] == 2


# ---------------------------------------------------------------------------
# 4. Engine: Graph & Replay Invariants Hardening
# ---------------------------------------------------------------------------


class TestTier5EngineGraphAndReplayHardening:
    """Stress tests graph setup validations, resumption planner, and replay."""

    def test_graph_validation_errors(self):
        g = Graph("fault_graph")
        with pytest.raises(ValueError, match="Entry node 'missing_entry' not found"):
            g.set_entry_point("missing_entry")

        with pytest.raises(ValueError, match="Finish node 'missing_finish' not found"):
            g.set_finish_point("missing_finish")

        g.add_node("node_1", lambda s: None)
        with pytest.raises(ValueError, match="Target node 'missing_target' not registered"):
            g.add_edge("node_1", "missing_target")

        with pytest.raises(ValueError, match="Source node 'missing_src' not registered"):
            g.add_edge("missing_src", "node_1")

    @pytest.mark.asyncio
    async def test_graph_sync_node_payload_and_finish(self, isolated_env: Path):
        g = Graph("sync_graph")

        def step_dict(state: State):
            return {"processed": 10, "status": "done"}

        def step_repr(state: State):
            return 9999

        g.add_node("step_dict", step_dict)
        g.add_node("step_repr", step_repr)
        g.add_edge("step_dict", "step_repr")
        g.set_entry_point("step_dict")
        g.set_finish_point("step_repr")

        state = State(workflow_id="wf-sync-graph")
        final_state = await g.run(state)

        node_events = [e for e in final_state.events if e.event_type == EventType.NODE_COMPLETE]
        assert len(node_events) == 2
        assert "result_keys" in node_events[0].payload
        assert node_events[1].payload.get("result_repr") == "9999"

    @pytest.mark.asyncio
    async def test_graph_resumption_already_completed(self, isolated_env: Path):
        g = Graph("completed_graph")
        executed = False

        def step_one(state: State):
            nonlocal executed
            executed = True

        g.add_node("step_one", step_one)
        g.set_entry_point("step_one")
        g.set_finish_point("step_one")

        state = State(workflow_id="wf-already-done")
        state.append_event(EventType.WORKFLOW_COMPLETE)

        res_state = await g.run(state)
        assert not executed
        assert res_state.workflow_id == "wf-already-done"

    def test_resumption_planner_terminal_conditions(self):
        g = Graph("resumption_graph")
        g.add_node("start", lambda s: None)
        g.add_node("finish", lambda s: None)
        g.add_edge("start", "finish", condition=lambda s: s.get("route") == "go")
        g.set_entry_point("start")
        g.set_finish_point("finish")

        # Finished node in finish_points
        state = State(workflow_id="wf-res-finish")
        state.append_event(EventType.NODE_COMPLETE, node_id="finish")
        plan = ResumptionPlanner.plan_resumption(g, state)
        assert plan.is_completed
        assert "declared finish point" in plan.reason

        # Completed node with no outgoing edges
        g_isolated = Graph("isolated_graph")
        g_isolated.add_node("lone_node", lambda s: None)
        g_isolated.set_entry_point("lone_node")
        state_isolated = State(workflow_id="wf-res-isolated")
        state_isolated.append_event(EventType.NODE_COMPLETE, node_id="lone_node")
        plan_isolated = ResumptionPlanner.plan_resumption(g_isolated, state_isolated)
        assert plan_isolated.is_completed
        assert "no outgoing edges" in plan_isolated.reason

        # No matching condition on outgoing edge raises ResumptionError
        state_blocked = State(workflow_id="wf-res-blocked")
        state_blocked.append_event(EventType.NODE_COMPLETE, node_id="start")
        state_blocked.set("route", "stop")
        with pytest.raises(ResumptionError, match="No valid outgoing edge condition matched"):
            ResumptionPlanner.plan_resumption(g, state_blocked)

    def test_event_replayer_verification_anomalies(self):
        # Empty events with expected last hash
        valid, err = EventReplayer.verify_event_chain([], expected_last_hash="hash123")
        assert not valid
        assert "events list is empty" in str(err)

        # Empty events with no expected last hash
        valid, err = EventReplayer.verify_event_chain([])
        assert valid
        assert err is None

        # State with events but empty last_hash
        state = State(workflow_id="wf-empty-last-hash")
        state.events.append(
            Event(
                event_id="evt-1",
                event_type=EventType.STATE_MUTATED,
                timestamp_ns=1000,
                state_hash="h1",
                prev_hash="",
            )
        )
        state.last_hash = ""
        with pytest.raises(CryptographicIntegrityError, match="last_hash is empty"):
            EventReplayer.enforce_integrity(state)

        # Nonexistent checkpoint returns None tuples
        mgr = CheckpointManager(base_dir="/tmp/agentmesh/isolated_replay_checkpoints")
        res_no_graph = EventReplayer.resume_from_crash("wf-nonexistent", mgr, graph=None)
        assert res_no_graph == (None, None)

        res_with_graph = EventReplayer.resume_from_crash("wf-nonexistent", mgr, graph=Graph("g"))
        assert res_with_graph == (None, None, None)


# ---------------------------------------------------------------------------
# 5. Mesh: MCP Protocol & Transport Hardening
# ---------------------------------------------------------------------------


class TestTier5MeshMcpProtocolHardening:
    """Stress tests MCP JSON-RPC 2.0 wire protocol, transports, and client."""

    def test_jsonrpc_error_factories(self):
        e1 = JSONRPCError.parse_error("invalid json payload")
        assert e1.code == JSONRPCErrorCode.PARSE_ERROR

        e2 = JSONRPCError.invalid_request("missing jsonrpc version")
        assert e2.code == JSONRPCErrorCode.INVALID_REQUEST

        e3 = JSONRPCError.internal_error("database failure")
        assert e3.code == JSONRPCErrorCode.INTERNAL_ERROR

        e4 = JSONRPCError.tool_execution_error("tool timed out")
        assert e4.code == JSONRPCErrorCode.TOOL_EXECUTION_ERROR

        e5 = JSONRPCError.resource_not_found("file:///missing.txt")
        assert e5.code == JSONRPCErrorCode.RESOURCE_NOT_FOUND

    def test_jsonrpc_response_validation_and_unwrap(self):
        # Both result and error present -> ValueError
        with pytest.raises(ValueError, match="cannot contain both 'result' and 'error'"):
            JSONRPCResponse.model_validate({
                "jsonrpc": "2.0",
                "id": "1",
                "result": {"ok": True},
                "error": {"code": -32600, "message": "err"},
            })

        # Neither result nor error present -> ValueError
        with pytest.raises(ValueError, match="must contain either 'result' or 'error'"):
            JSONRPCResponse.model_validate({"jsonrpc": "2.0", "id": "1"})

    def test_tools_call_result_get_text_formats(self):
        res1 = ToolsCallResult(content=[
            MCPTextContent(text="Hello"),
            {"type": "text", "text": "World"},
            "Raw string content",
        ])
        assert res1.get_text() == "Hello\nWorld\nRaw string content"

        res_empty = ToolsCallResult(content=[])
        assert res_empty.get_text() == ""

    @pytest.mark.asyncio
    async def test_in_memory_transport_resources_and_custom_handlers(self):
        transport = InMemoryTransport(server_name="test_inmem")

        # Register resource
        def resource_reader(uri: str) -> str:
            if "error" in uri:
                raise RuntimeError("Resource read fault")
            return f"Content of {uri}"

        res = MCPResource(uri="agentmesh://doc1", name="doc1")
        transport.register_resource_handler(res, resource_reader)

        # List resources
        list_resp = await transport.send_request(
            JSONRPCRequest(id="r1", method=MCPMethod.RESOURCES_LIST)
        )
        assert list_resp.is_success
        assert list_resp.result is not None
        assert len(list_resp.result["resources"]) == 1

        # Read resource: valid
        read_resp = await transport.send_request(
            JSONRPCRequest(id="r2", method=MCPMethod.RESOURCES_READ, params={"uri": "agentmesh://doc1"})
        )
        assert read_resp.is_success
        assert read_resp.result is not None
        assert read_resp.result["contents"][0]["text"] == "Content of agentmesh://doc1"

        # Read resource: missing uri param
        err_param = await transport.send_request(
            JSONRPCRequest(id="r3", method=MCPMethod.RESOURCES_READ, params={})
        )
        assert not err_param.is_success
        assert err_param.error is not None
        assert err_param.error.code == JSONRPCErrorCode.INVALID_PARAMS

        # Read resource: unknown uri
        err_unknown = await transport.send_request(
            JSONRPCRequest(id="r4", method=MCPMethod.RESOURCES_READ, params={"uri": "agentmesh://unknown"})
        )
        assert not err_unknown.is_success
        assert err_unknown.error is not None
        assert err_unknown.error.code == JSONRPCErrorCode.RESOURCE_NOT_FOUND

        # Read resource: handler exception
        transport.register_resource_handler(
            MCPResource(uri="agentmesh://error", name="err"), resource_reader
        )
        err_ex = await transport.send_request(
            JSONRPCRequest(id="r5", method=MCPMethod.RESOURCES_READ, params={"uri": "agentmesh://error"})
        )
        assert not err_ex.is_success
        assert err_ex.error is not None
        assert err_ex.error.code == JSONRPCErrorCode.INTERNAL_ERROR

        # Ping & Initialize
        ping_resp = await transport.send_request(JSONRPCRequest(id="r6", method=MCPMethod.PING))
        assert ping_resp.is_success

        init_resp = await transport.send_request(JSONRPCRequest(id="r7", method=MCPMethod.INITIALIZE))
        assert init_resp.is_success
        assert init_resp.result is not None
        assert init_resp.result["protocolVersion"] == "2024-11-05"

        # Custom handler
        transport.custom_handlers["custom/echo"] = lambda p: {"echoed": p}
        cust_resp = await transport.send_request(
            JSONRPCRequest(id="r8", method="custom/echo", params={"msg": "hi"})
        )
        assert cust_resp.result is not None
        assert cust_resp.result["echoed"]["msg"] == "hi"

        # Disconnected transport raises MCPTransportError
        await transport.close()
        with pytest.raises(MCPTransportError, match="InMemoryTransport is closed"):
            await transport.send_request(JSONRPCRequest(id="r9", method=MCPMethod.PING))

    @pytest.mark.asyncio
    async def test_stdio_and_http_sse_transports_disconnected_errors(self):
        # StdioTransport closed
        stdio = StdioTransport("echo", ["hi"])
        stdio._closed = True
        with pytest.raises(MCPTransportError, match="StdioTransport is not connected"):
            await stdio.send_request(JSONRPCRequest(id="s1", method="test"))
        with pytest.raises(MCPTransportError, match="StdioTransport is not connected"):
            await stdio.send_notification(JSONRPCNotification(method="test_notify"))

        # HttpSseTransport framing
        framed = HttpSseTransport.encode_sse_event("mcp_event", "line1\nline2", event_id="ev_01")
        assert "id: ev_01\n" in framed
        assert "event: mcp_event\n" in framed
        assert "data: line1\ndata: line2\n\n" in framed

        parsed = HttpSseTransport.parse_sse_events(framed)
        assert len(parsed) == 1
        assert parsed[0] == ("mcp_event", "line1\nline2")

        # HttpSseTransport disconnected / without handler
        sse = HttpSseTransport("http://127.0.0.1:8000/sse")
        with pytest.raises(NotImplementedError, match="requires an http_handler adapter"):
            await sse.send_request(JSONRPCRequest(id="h1", method="test"))

        await sse.close()
        with pytest.raises(MCPTransportError, match="HttpSseTransport is closed"):
            await sse.send_request(JSONRPCRequest(id="h2", method="test"))

    @pytest.mark.asyncio
    async def test_mcp_tool_client_adversarial_invocations(self):
        client = MCPToolClient(server_name="adv_mcp")

        # Register tool with dict parameter schemas
        client.register_tool(
            name="echo_tool",
            description="Echoes input",
            parameters={"data": {"type": "object"}, "count": "int"},
            handler=lambda data, count: {"data": data, "count": count},
        )

        resp = await client.invoke_tool(
            MCPToolCallRequest(tool_name="echo_tool", arguments={"data": {"k": "v"}, "count": 3})
        )
        assert resp.success
        assert resp.result["count"] == 3

        # Tool execution error
        client.register_tool(
            name="error_tool",
            description="Always raises",
            parameters={},
            handler=lambda: 1 / 0,
        )
        err_resp = await client.invoke_tool(MCPToolCallRequest(tool_name="error_tool"))
        assert not err_resp.success
        assert "division by zero" in str(err_resp.error)


# ---------------------------------------------------------------------------
# 6. Mesh: Router, Discovery & Dynamic Failover Hardening
# ---------------------------------------------------------------------------


class TestTier5MeshRouterAndDiscoveryHardening:
    """Stress tests router scheduling, circular fallback detection, and service registry."""

    def test_swrr_reset_and_active_properties(self):
        swrr = SmoothWeightedRoundRobin()
        ep1 = AgentEndpoint(agent_id="a1", weight=10, active=True)
        assert ep1.active
        ep1.active = False
        assert not ep1.is_active

        # SWRR reset
        swrr.reset("a1")
        swrr.reset()

    def test_router_select_endpoint_empty_and_inactive(self):
        router = MeshRouter()
        # Nonexistent pool
        with pytest.raises(ValueError, match="Endpoint pool 'missing_pool' is empty"):
            router.select_endpoint("missing_pool")

        # Pool with all inactive endpoints
        ep_inactive = AgentEndpoint(agent_id="inact_1", role="worker", weight=0, is_active=False)
        router.register_endpoint(ep_inactive, handler=lambda p: p)
        with pytest.raises(RuntimeError, match="No active or eligible endpoint available"):
            router.select_endpoint("worker", healthy_only=False)

    @pytest.mark.asyncio
    async def test_router_capability_routing_errors(self):
        # No discovery registry configured
        router = MeshRouter(discovery_registry=None)
        with pytest.raises(ValueError, match="No discovery registry configured"):
            await router.route_by_capability("ocr", {})

        # Registry present but no matching agents
        reg = ServiceRegistry()
        router_with_reg = MeshRouter(discovery_registry=reg)
        with pytest.raises(ValueError, match="No healthy agents found for capability 'ocr'"):
            await router_with_reg.route_by_capability("ocr", {})

    @pytest.mark.asyncio
    async def test_router_circular_fallback_detection(self):
        router = MeshRouter()
        ep_a = AgentEndpoint(agent_id="agent_a", role="worker")
        ep_b = AgentEndpoint(agent_id="agent_b", role="worker")
        ep_c = AgentEndpoint(agent_id="agent_c", role="worker")

        # Cyclic fallbacks: A -> B -> C -> A
        router.register_endpoint(ep_a, handler=lambda p: 1 / 0, fallback_agent_id="agent_b")
        router.register_endpoint(ep_b, handler=lambda p: 1 / 0, fallback_agent_id="agent_c")
        router.register_endpoint(ep_c, handler=lambda p: 1 / 0, fallback_agent_id="agent_a")

        with pytest.raises(RuntimeError, match="Fallback cycle detected"):
            await router.route_and_call("agent_a", {"task": "cycle_test"})

    @pytest.mark.asyncio
    async def test_router_sync_handler_awaitable_and_missing_handler(self):
        router = MeshRouter()
        ep = AgentEndpoint(agent_id="sync_agent", role="calc")

        async def inner_async():
            return "async_inner_ok"

        def sync_returning_future(payload):
            return inner_async()

        router.register_endpoint(ep, handler=sync_returning_future)
        res = await router.route_and_call("sync_agent", {})
        assert res == "async_inner_ok"

        # Missing handler
        ep_nohandler = AgentEndpoint(agent_id="no_handler_agent", role="calc")
        router.endpoints["no_handler_agent"] = ep_nohandler
        with pytest.raises(RuntimeError, match="has no registered handler"):
            await router.route_and_call("no_handler_agent", {})

    def test_service_registry_comprehensive(self):
        registry = ServiceRegistry()

        # Listener subscription and exception safety
        notifications = []

        def failing_listener(evt, reg):
            raise RuntimeError("listener boom")

        def normal_listener(evt, reg):
            notifications.append((evt, reg.agent_id))

        registry.subscribe(failing_listener)
        registry.subscribe(normal_listener)

        card = AgentCard(
            name="Agent Reg 1",
            agent_id="agent_reg_1",
            capabilities=[AgentCapability(name="search", tags=["fast", "web"])],
        )
        registry.register(card, ttl_sec=1.0)
        assert len(notifications) == 1
        assert notifications[0] == ("REGISTERED", "agent_reg_1")

        # Heartbeat and deregister unknown
        assert not registry.heartbeat("unknown_agent")
        assert not registry.deregister("unknown_agent")

        # Queries
        assert len(registry.find_by_capability("search")) == 1
        assert len(registry.find_by_tag("fast")) == 1
        assert len(registry.list_active_agents()) == 1

        # Eviction
        # Artificially set last heartbeat far in past
        reg = registry.get_agent("agent_reg_1")
        assert reg is not None
        reg.last_heartbeat_ns = time.time_ns() - 10_000_000_000  # 10s ago, ttl is 1s

        # With include_expired=False, get_agent returns None
        assert registry.get_agent("agent_reg_1", include_expired=False) is None
        assert registry.get_agent("agent_reg_1", include_expired=True) is not None

        evicted = registry.evict_expired()
        assert "agent_reg_1" in evicted
        assert len(registry.list_active_agents(include_expired=True)) == 0


# ---------------------------------------------------------------------------
# 7. Security: Capability Policy & Sandbox Adversarial Hardening
# ---------------------------------------------------------------------------


class TestTier5SecurityPolicyAndSandboxHardening:
    """Stress tests capability sandbox defenses, null bytes, command injection, and SSRF."""

    def test_policy_command_rules_with_execute_paths(self):
        policy = CapabilityPolicy(
            policy_name="exec_path_policy",
            allowed_execute_paths=["/usr/bin", "/usr/local/bin"],
            allowed_commands={"echo", "cat"},
            command_rules=[CommandRule(executable="ls")],
        )
        # Inside allowed_execute_paths
        assert policy.is_executable_whitelisted("/usr/bin/echo")
        assert policy.is_executable_whitelisted("/usr/bin/ls")

        # Outside allowed_execute_paths
        assert not policy.is_executable_whitelisted("/tmp/evil_echo")

        # Policy with NO allowed_execute_paths requires exact full path or bare command
        policy_bare = CapabilityPolicy(
            policy_name="bare_cmd_policy",
            allowed_commands={"/bin/date", "echo"},
        )
        assert policy_bare.is_executable_whitelisted("/bin/date")
        assert not policy_bare.is_executable_whitelisted("/bin/echo")

    def test_sandbox_file_access_modes_and_null_bytes(self, tmp_path: Path):
        safe_dir = tmp_path / "sandbox_storage"
        safe_dir.mkdir(parents=True, exist_ok=True)
        test_file = safe_dir / "target.txt"
        test_file.write_text("secure data")

        policy = CapabilityPolicy(
            policy_name="file_policy",
            allowed_read_paths=[str(safe_dir)],
            allowed_write_paths=[str(safe_dir)],
            allowed_execute_paths=[str(safe_dir)],
        )
        sandbox = SecuritySandbox(policy=policy)

        # Mode parsing
        assert sandbox.validate_file_access(str(test_file), mode="r")
        assert sandbox.validate_file_access(str(test_file), mode="w")
        assert sandbox.validate_file_access(str(test_file), mode="a")
        assert sandbox.validate_file_access(str(test_file), mode="x")
        assert sandbox.validate_file_access(str(test_file), mode="exec")

        # Null-byte variations
        with pytest.raises(PermissionDeniedError, match="Null-byte injection detected"):
            sandbox.validate_file_access(str(test_file) + "\x00.evil")

        with pytest.raises(PermissionDeniedError, match="Encoded null-byte injection detected"):
            sandbox.validate_file_access(str(test_file) + "%00.evil")

        with pytest.raises(PermissionDeniedError, match="Encoded null-byte injection detected"):
            sandbox.validate_file_access(str(test_file) + "%2500.evil")

        # Empty allowed roots
        empty_policy = CapabilityPolicy(policy_name="empty", allowed_read_paths=[])
        empty_sb = SecuritySandbox(policy=empty_policy)
        with pytest.raises(PermissionDeniedError, match="No allowed read roots configured"):
            empty_sb.validate_file_access(str(test_file), mode="r")

    def test_sandbox_command_validation_corner_cases(self):
        policy = CapabilityPolicy(
            policy_name="cmd_policy",
            allowed_commands={"echo", "grep"},
        )
        sandbox = SecuritySandbox(policy=policy)

        # None command
        with pytest.raises(PermissionDeniedError, match="Null command blocked"):
            sandbox.validate_command(None)  # type: ignore

        # Invalid command type
        with pytest.raises(PermissionDeniedError, match="Invalid command type"):
            sandbox.validate_command(12345)  # type: ignore

        # Empty vector or whitespace only
        with pytest.raises(PermissionDeniedError, match="Empty command vector blocked"):
            sandbox.validate_command(["", "  "])

        # Unclosed quote
        with pytest.raises(PermissionDeniedError, match="Syntax error or unclosed quote"):
            sandbox.validate_command('echo "unclosed string')

        # Null byte in arguments
        with pytest.raises(PermissionDeniedError, match="Forbidden shell metacharacter"):
            sandbox.validate_command("echo test\x00arg")

        with pytest.raises(PermissionDeniedError, match="Null byte detected in argument vector"):
            sandbox.validate_command(["echo", "test\x00arg"])

    def test_sandbox_network_adversarial_destinations(self):
        policy = CapabilityPolicy(
            policy_name="net_policy",
            allow_network=True,
            allowed_schemes={"https", "http"},
            allowed_ports={80, 443, 8080},
            allowed_domains=["api.trusted.com", "*.trusted.com"],
            custom_blocked_cidrs=["203.0.113.0/24"],
        )
        sandbox = SecuritySandbox(policy=policy)

        # Null-byte in URL
        with pytest.raises(PermissionDeniedError, match="Null-byte injection detected"):
            sandbox.validate_url("https://api.trusted.com\x00evil.com")

        # Missing scheme
        with pytest.raises(PermissionDeniedError, match="Missing URL scheme"):
            sandbox.validate_url("://api.trusted.com")

        # Blocked scheme
        with pytest.raises(PermissionDeniedError, match="Protocol scheme 'gopher' blocked"):
            sandbox.validate_url("gopher://api.trusted.com")

        # Port out of range
        with pytest.raises(PermissionDeniedError, match="Port out of range"):
            sandbox.validate_url("https://api.trusted.com:99999/path")

        # Port not in allowed_ports
        with pytest.raises(PermissionDeniedError, match="Port 8443 not in allowed ports"):
            sandbox.validate_url("https://api.trusted.com:8443/path")

        # Custom blocked CIDR
        with pytest.raises(PermissionDeniedError, match="prohibited by SSRF policy"):
            sandbox.validate_network("203.0.113.50")

        # IPv6 Loopback and Link-Local
        with pytest.raises(PermissionDeniedError, match="prohibited by SSRF policy"):
            sandbox.validate_network("[::1]:8080")

        with pytest.raises(PermissionDeniedError, match="prohibited by SSRF policy"):
            sandbox.validate_network("[fe80::1]:8080")

        # Valid domain wildcard matching
        assert sandbox.validate_url("https://sub.trusted.com/v1") == "https://sub.trusted.com/v1"


# ---------------------------------------------------------------------------
# 8. Telemetry: Tracer, Token Tracker & W3C TraceContext Hardening
# ---------------------------------------------------------------------------


class TestTier5TelemetryTracerAndW3CHardening:
    """Stress tests token tracking budgets, tracer registries, and W3C context."""

    def test_token_tracker_custom_rates_budget_and_reset(self):
        custom_rates = {"gpt-custom": {"prompt": 0.00001, "completion": 0.00003}}
        tracker = TokenTracker(budget_limit=100, custom_rates=custom_rates)

        assert tracker.budget_limit == 100
        tracker.set_budget(150)
        assert tracker.budget_limit == 150

        # Negative tokens rejected
        with pytest.raises(ValueError, match="Token counts cannot be negative"):
            tracker.record(prompt_tokens=-5, completion_tokens=10)

        # Record valid
        tracker.record(prompt_tokens=40, completion_tokens=60, model="gpt-custom")
        assert tracker.total_tokens == 100
        assert tracker.cost_estimate > 0.0

        # Budget exceeded
        with pytest.raises(QuotaExceededError, match="Token quota exceeded"):
            tracker.record(prompt_tokens=30, completion_tokens=30)

        # Reset
        tracker.reset()
        assert tracker.total_tokens == 0
        assert tracker.cost_estimate == 0.0

    def test_agent_tracer_registry_and_summary(self):
        tracer = AgentTracer()

        # Re-accessing existing context preserves it
        ctx1 = tracer.get_or_create_context(trace_id="4bf92f3577b34da6a3ce929d0e0e4736", budget_limit=500)
        ctx2 = tracer.get_or_create_context(trace_id="4bf92f3577b34da6a3ce929d0e0e4736")
        assert ctx1 is ctx2
        assert ctx1.token_tracker.budget_limit == 500

        # Record tokens on latest active context
        tracer.record_token_usage(prompt=20, completion=30)
        assert ctx1.total_tokens_consumed == 50

        # Export summary for unknown trace_id returns empty dict
        assert tracer.export_summary("unknown_trace_id") == {}

        # Export summary for valid trace_id
        summary = tracer.export_summary("4bf92f3577b34da6a3ce929d0e0e4736")
        assert summary["total_tokens"] == 50

        # Export audit logs
        logs = tracer.export_audit_logs()
        assert isinstance(logs, list)

    def test_w3c_traceparent_and_tracestate_edge_cases(self):
        # TraceParent string conversion and sampling
        tp = TraceParent.create(sampled=False)
        assert str(tp) == tp.format()
        assert not tp.is_sampled

        # TraceState mutations and boundaries
        ts = TraceState()
        ts.set("vendor1", "opaqueValue1")
        ts.set("vendor2", "opaqueValue2")
        assert "vendor1" in ts
        assert str(ts) == ts.format()
        assert len(ts) == 2

        # Delete key
        ts.delete("vendor1")
        assert "vendor1" not in ts
        assert len(ts) == 1

        # TraceState parsing malformed members
        parsed_ts = TraceState.parse("valid=123, invalid_no_equal, =no_key, key=, valid2=456")
        assert parsed_ts.get("valid") == "123"
        assert parsed_ts.get("valid2") == "456"

        # TraceContext from headers without traceparent raises validation error
        with pytest.raises(TraceContextValidationError, match="Missing required 'traceparent'"):
            TraceContext.from_w3c_headers({})

        # Case-insensitive W3C header extraction
        carrier = {"TRACEPARENT": tp.format(), "Tracestate": "foo=bar"}
        extracted_tp, extracted_ts = extract_w3c_trace_context(carrier)
        assert extracted_tp == tp.format()
        assert extracted_ts == "foo=bar"
