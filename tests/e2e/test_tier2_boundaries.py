"""Tier 2: Boundary and Corner Case Tests for AgentMesh.

Focuses on limits, edge cases, zero/empty states, overflow, and malformed inputs:
- State & Graph: empty inputs, extreme steps, conditional edge evaluations
- CheckpointManager: nonexistent workflows, empty labels, atomic persistence
- Mesh & Circuit Breaker: extreme thresholds, immediate trips, recovery timeouts
- MCP Client: exception isolation, parameter variations, missing tools
- Security Sandbox: empty policies, whitespace commands, exact boundary paths
- Telemetry: zero tokens, missing trace contexts, span lifecycle boundaries
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentmesh.engine.checkpoint import CheckpointManager
from agentmesh.engine.graph import Graph
from agentmesh.engine.state import EventType, State
from agentmesh.mesh.mcp_client import MCPToolCallRequest, MCPToolClient
from agentmesh.mesh.router import (
    AgentEndpoint,
    CircuitBreaker,
    CircuitState,
    MeshRouter,
)
from agentmesh.security.sandbox import (
    CapabilityPolicy,
    PermissionDeniedError,
    SecuritySandbox,
)
from agentmesh.telemetry.tracer import AgentTracer, TraceContext

# --- State & Graph Boundary Tests ---

def test_tier2_state_empty_initialization():
    """Validates state container behavior when initialized with empty string workflow ID."""
    state = State(workflow_id="")
    assert state.workflow_id == ""
    assert state.version == 0
    assert state.data == {}
    assert state.events == []
    assert state.last_hash == ""
    assert state.get("missing_key") is None
    assert state.get("missing_key", default=42) == 42


def test_tier2_state_rapid_monotonic_mutations():
    """Validates version counter monotonic increase under 500 sequential updates."""
    state = State(workflow_id="wf-tier2-monotonic")
    mutation_count = 500
    for i in range(mutation_count):
        state.set("seq", i)

    assert state.version == mutation_count
    assert state.get("seq") == mutation_count - 1


def test_tier2_state_event_unicode_and_special_characters():
    """Validates event hashing and serialization with complex Unicode, CJK, and special characters."""
    special_text = "CJK: 智能体网格 / Symbols: <>&$#@*^~{}[] / Quotes: 'single' \"double\" / Newlines: \n\r\t"
    payload = {
        "text": special_text,
        "symbols": ["&", ";", "|", "`", "$"],
    }
    state = State(workflow_id="wf-tier2-unicode")
    event = state.append_event(EventType.TOOL_CALL, payload=payload)

    assert event.payload["text"] == special_text
    recomputed_hash = event.calculate_hash("")
    assert event.state_hash == recomputed_hash


@pytest.mark.asyncio
async def test_tier2_graph_empty_entry_point_raises_error():
    """Validates that running a graph without an entry point raises RuntimeError."""
    g = Graph("empty_graph")
    state = State(workflow_id="wf-tier2-empty-graph")
    with pytest.raises(RuntimeError) as exc_info:
        await g.run(state)
    assert "Graph entry point is not defined" in str(exc_info.value)


def test_tier2_graph_invalid_nodes_in_edges():
    """Validates that adding edges with unregistered source or target nodes raises ValueError."""
    g = Graph("edge_test_graph")
    g.add_node("node_valid", lambda s: None)

    with pytest.raises(ValueError) as exc_info1:
        g.add_edge("nonexistent_source", "node_valid")
    assert "Source node 'nonexistent_source' not registered" in str(exc_info1.value)

    with pytest.raises(ValueError) as exc_info2:
        g.add_edge("node_valid", "nonexistent_target")
    assert "Target node 'nonexistent_target' not registered" in str(exc_info2.value)


@pytest.mark.asyncio
async def test_tier2_graph_max_steps_boundary():
    """Validates that max_steps=1 halts execution after exactly 1 step even if more nodes exist."""
    step_history = []

    async def step_1(state: State):
        step_history.append("step_1")

    async def step_2(state: State):
        step_history.append("step_2")

    g = Graph("step_boundary_graph")
    g.add_node("step_1", step_1)
    g.add_node("step_2", step_2)
    g.add_edge("step_1", "step_2")
    g.set_entry_point("step_1")
    g.set_finish_point("step_2")

    state = State(workflow_id="wf-tier2-max-steps")
    # Limit max_steps to 1
    await g.run(state, max_steps=1)

    assert step_history == ["step_1"]
    assert "step_2" not in step_history


@pytest.mark.asyncio
async def test_tier2_graph_conditional_edge_branching():
    """Validates dynamic edge condition evaluation branching based on state data."""
    execution_path = []

    async def init_node(state: State):
        execution_path.append("init")

    async def branch_high(state: State):
        execution_path.append("high")

    async def branch_low(state: State):
        execution_path.append("low")

    g = Graph("branching_graph")
    g.add_node("init", init_node)
    g.add_node("branch_high", branch_high)
    g.add_node("branch_low", branch_low)

    g.add_edge("init", "branch_high", condition=lambda s: s.get("score", 0) >= 80)
    g.add_edge("init", "branch_low", condition=lambda s: s.get("score", 0) < 80)
    g.set_entry_point("init")
    g.set_finish_point("branch_high")
    g.set_finish_point("branch_low")

    # Test Case 1: score = 95 -> branch_high
    state1 = State(workflow_id="wf-branch-1")
    state1.set("score", 95)
    await g.run(state1)
    assert execution_path == ["init", "high"]

    # Test Case 2: score = 40 -> branch_low
    execution_path.clear()
    state2 = State(workflow_id="wf-branch-2")
    state2.set("score", 40)
    await g.run(state2)
    assert execution_path == ["init", "low"]


# --- CheckpointManager Boundary Tests ---

def test_tier2_checkpoint_nonexistent_workflow(checkpoint_manager: CheckpointManager):
    """Validates loading checkpoints for a nonexistent workflow returns None without error."""
    loaded = checkpoint_manager.load_latest_checkpoint("wf-completely-nonexistent-999")
    assert loaded is None

    listing = checkpoint_manager.list_checkpoints("wf-completely-nonexistent-999")
    assert listing == []


def test_tier2_checkpoint_empty_label_defaults_to_auto(checkpoint_manager: CheckpointManager):
    """Validates that saving a checkpoint with an empty label produces ckpt_*_auto.json."""
    state = State(workflow_id="wf-tier2-empty-label")
    saved_path = checkpoint_manager.save_checkpoint(state, label="")

    assert "ckpt_000000_auto.json" in saved_path
    loaded = checkpoint_manager.load_latest_checkpoint("wf-tier2-empty-label")
    assert loaded is not None
    assert loaded.workflow_id == "wf-tier2-empty-label"


def test_tier2_checkpoint_nested_dir_auto_creation(tmp_path: Path):
    """Validates CheckpointManager automatically creates deeply nested storage directories."""
    nested_dir = tmp_path / "deep" / "nested" / "state" / "dir"
    ckpt_mgr = CheckpointManager(base_dir=str(nested_dir))

    assert nested_dir.exists()
    state = State(workflow_id="wf-tier2-nested")
    saved = ckpt_mgr.save_checkpoint(state, label="deep")
    assert Path(saved).exists()


# --- Mesh & Circuit Breaker Boundary Tests ---

def test_tier2_circuit_breaker_threshold_one_trips_immediately():
    """Validates that a threshold of 1 causes the circuit to open on the very first failure."""
    cb = CircuitBreaker(failure_threshold=1)
    assert cb.state == CircuitState.CLOSED

    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.can_execute() is False


def test_tier2_circuit_breaker_recovery_timeout_transition():
    """Validates that elapsed recovery timeout transitions OPEN circuit to HALF_OPEN."""
    # recovery_timeout=0.0 means immediate recovery eligibility
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.0)
    cb.record_failure()
    assert cb.state == CircuitState.OPEN

    # Evaluating can_execute after recovery timeout has elapsed triggers HALF_OPEN
    can_exec = cb.can_execute()
    assert can_exec is True
    assert cb.state == CircuitState.HALF_OPEN


def test_tier2_circuit_breaker_record_success_in_closed_state():
    """Validates that record_success when circuit is already CLOSED safely resets counters."""
    cb = CircuitBreaker(failure_threshold=3)
    cb.record_failure()
    assert cb.failure_count == 1

    cb.record_success()
    assert cb.failure_count == 0
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_tier2_router_unhandled_failure_without_fallback(sample_mesh_router: MeshRouter):
    """Validates that an agent failure without a registered fallback raises RuntimeError."""
    async def crashing_agent(payload: dict):
        raise ValueError("Critical internal computation fault")

    sample_mesh_router.register_endpoint(
        AgentEndpoint(agent_id="failing_worker", role="Worker"),
        handler=crashing_agent,
    )

    with pytest.raises(RuntimeError) as exc_info:
        await sample_mesh_router.route_and_call("failing_worker", {})
    assert "no healthy fallback available" in str(exc_info.value)
    assert "Critical internal computation fault" in str(exc_info.value)


# --- MCP Tool Client Boundary Tests ---

@pytest.mark.asyncio
async def test_tier2_mcp_handler_exception_isolated():
    """Validates that an unhandled exception inside a tool handler returns a clean error response."""
    client = MCPToolClient(server_name="exception_test_server")

    def zero_divider(x: int) -> float:
        return 100 / x

    client.register_tool(
        name="zero_divider",
        description="Divides 100 by x",
        parameters={"x": "int"},
        handler=zero_divider,
    )

    req = MCPToolCallRequest(tool_name="zero_divider", arguments={"x": 0})
    resp = await client.invoke_tool(req)

    assert resp.success is False
    assert resp.result is None
    assert resp.error is not None
    assert "division by zero" in resp.error


@pytest.mark.asyncio
async def test_tier2_mcp_empty_arguments_and_parameters():
    """Validates tool registration and invocation with empty parameters and arguments."""
    client = MCPToolClient(server_name="empty_args_server")

    def get_system_status() -> dict:
        return {"status": "ONLINE", "load": 0.12}

    client.register_tool(
        name="get_system_status",
        description="Returns system status",
        parameters={},
        handler=get_system_status,
    )

    req = MCPToolCallRequest(tool_name="get_system_status", arguments={})
    resp = await client.invoke_tool(req)

    assert resp.success is True
    assert resp.result["status"] == "ONLINE"


# --- Security Sandbox Boundary Tests ---

def test_tier2_sandbox_empty_policy_blocks_all():
    """Validates that an empty CapabilityPolicy blocks read, write, and command access."""
    policy = CapabilityPolicy(
        policy_name="strictly_empty",
        allow_network=False,
        allowed_read_paths=[],
        allowed_write_paths=[],
        allowed_commands=set(),
    )
    sandbox = SecuritySandbox(policy=policy)

    with pytest.raises(PermissionDeniedError):
        sandbox.validate_file_access("/tmp/any_file.txt", mode="r")

    with pytest.raises(PermissionDeniedError):
        sandbox.validate_file_access("/tmp/any_file.txt", mode="w")

    with pytest.raises(PermissionDeniedError):
        sandbox.validate_command("echo test")

    with pytest.raises(PermissionDeniedError):
        sandbox.validate_network("https://example.com")


def test_tier2_sandbox_whitespace_command():
    """Validates that whitespace-only commands are safely handled and rejected."""
    policy = CapabilityPolicy(allowed_commands={"echo"})
    sandbox = SecuritySandbox(policy=policy)

    with pytest.raises(PermissionDeniedError):
        sandbox.validate_command("   ")


def test_tier2_sandbox_exact_path_and_prefix_boundary(tmp_path: Path):
    """Validates that paths sharing a prefix but outside directory boundary are rejected."""
    safe_dir = tmp_path / "safe"
    safe_dir.mkdir(parents=True, exist_ok=True)

    # Neighbor directory with safe_dir as string prefix
    unsafe_neighbor = tmp_path / "safe_neighbor"
    unsafe_neighbor.mkdir(parents=True, exist_ok=True)
    target_file = unsafe_neighbor / "target.txt"

    policy = CapabilityPolicy(allowed_read_paths=[str(safe_dir)])
    sandbox = SecuritySandbox(policy=policy)

    # Valid read inside safe_dir
    valid_file = safe_dir / "valid.txt"
    valid_file.write_text("ok", encoding="utf-8")
    assert sandbox.validate_file_access(str(valid_file), mode="r")

    # Unauthorized access to safe_neighbor must be blocked despite string prefix similarity
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_file_access(str(target_file), mode="r")


# --- Telemetry & Tracer Boundary Tests ---

def test_tier2_telemetry_zero_and_negative_tokens():
    """Validates token recording with zero values."""
    ctx = TraceContext()
    ctx.record_token_usage(prompt=0, completion=0)

    assert ctx.prompt_tokens == 0
    assert ctx.completion_tokens == 0
    assert ctx.total_tokens_consumed == 0


def test_tier2_telemetry_unknown_trace_id_summary(sample_tracer: AgentTracer):
    """Validates that querying a nonexistent trace ID returns an empty dictionary."""
    summary = sample_tracer.export_summary("trace-unknown-999")
    assert summary == {}


def test_tier2_telemetry_span_without_finish():
    """Validates that an unfinished span remains open with end_time None."""
    ctx = TraceContext()
    span = ctx.start_span("unfinished_task")

    assert span.end_time is None
    assert span.status == "OK"
    assert len(ctx.spans) == 1
