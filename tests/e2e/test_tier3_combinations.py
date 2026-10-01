"""Tier 3: Cross-Feature Integration and Pairwise Interaction Tests.

Tests the intersection and cooperative workflows across architectural layers:
- Engine + Mesh: Multi-node DAG orchestrating mesh agent invocations with state propagation
- Mesh + Circuit Breaker + Telemetry: Tripped circuit with fallback tracking spans and tokens
- Engine + Security Sandbox: Sandboxed execution inside graph nodes with failure routing
- Engine + Checkpoint + Crash + Mesh: Replay skipping completed mesh calls without duplicate side effects
- Mesh + MCP Client + Security Sandbox: Guarding tool filesystem operations through policy checks
- Mesh + Telemetry: Distributed cross-agent trace context propagation and token accounting
- Full Stack: Engine + Mesh + Security + Telemetry executing an end-to-end coordinated task
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentmesh.engine.checkpoint import CheckpointManager
from agentmesh.engine.graph import Graph
from agentmesh.engine.replay import EventReplayer
from agentmesh.engine.state import EventType, State
from agentmesh.mesh.mcp_client import MCPToolCallRequest, MCPToolClient
from agentmesh.mesh.router import AgentEndpoint, CircuitState, MeshRouter
from agentmesh.security.sandbox import (
    PermissionDeniedError,
)
from agentmesh.telemetry.tracer import AgentTracer

# --- Combination 1: Engine + Mesh Integration ---

@pytest.mark.asyncio
async def test_tier3_engine_orchestrates_mesh_pipeline(sample_mesh_router: MeshRouter):
    """Engine + Mesh: 3-step DAG routes data through mesh agents and stores intermediate results in State."""
    # Register 2 specialized agents in Mesh
    async def normalization_agent(payload: dict) -> dict:
        raw_values = payload.get("data", [])
        return {"normalized": [v.strip().upper() for v in raw_values]}

    async def enrichment_agent(payload: dict) -> dict:
        items = payload.get("items", [])
        return {"enriched": [{"code": item, "valid": len(item) > 3} for item in items]}

    sample_mesh_router.register_endpoint(
        AgentEndpoint(agent_id="agent_norm", role="Normalizer"),
        handler=normalization_agent,
    )
    sample_mesh_router.register_endpoint(
        AgentEndpoint(agent_id="agent_enrich", role="Enricher"),
        handler=enrichment_agent,
    )

    # Build DAG orchestrator
    g = Graph("mesh_orchestration_graph")

    async def step_norm(state: State):
        raw = state.get("raw_input")
        res = await sample_mesh_router.route_and_call("agent_norm", {"data": raw})
        state.set("normalized_data", res["normalized"])

    async def step_enrich(state: State):
        norm = state.get("normalized_data")
        res = await sample_mesh_router.route_and_call("agent_enrich", {"items": norm})
        state.set("final_output", res["enriched"])

    g.add_node("step_norm", step_norm)
    g.add_node("step_enrich", step_enrich)
    g.add_edge("step_norm", "step_enrich")
    g.set_entry_point("step_norm")
    g.set_finish_point("step_enrich")

    state = State(workflow_id="wf-tier3-engine-mesh")
    state.set("raw_input", ["  apple ", "pear", "  banana "])

    final_state = await g.run(state)
    output = final_state.get("final_output")

    assert len(output) == 3
    assert output[0] == {"code": "APPLE", "valid": True}
    assert output[1] == {"code": "PEAR", "valid": True}
    assert output[2] == {"code": "BANANA", "valid": True}
    assert final_state.version == 3


# --- Combination 2: Mesh + Circuit Breaker + Telemetry ---

@pytest.mark.asyncio
async def test_tier3_circuit_breaker_failover_with_telemetry(sample_mesh_router: MeshRouter):
    """Mesh + Circuit Breaker + Telemetry: Trips circuit to OPEN, routes to fallback, tracks spans & tokens."""
    tracer = AgentTracer()
    ctx = tracer.get_or_create_context()
    root_span = ctx.start_span("mesh_resilience_test")

    async def unreliable_primary(payload: dict):
        # Consume prompt tokens before failing
        ctx.record_token_usage(prompt=200, completion=0)
        raise ConnectionResetError("Primary inference service timed out")

    async def reliable_backup(payload: dict):
        # Fallback computes answer with lightweight model
        ctx.record_token_usage(prompt=50, completion=20)
        return {"served_by": "backup_agent", "result": "HEURISTIC_PREDICTION"}

    sample_mesh_router.register_endpoint(
        AgentEndpoint(agent_id="primary_llm", role="Primary LLM"),
        handler=unreliable_primary,
        fallback_agent_id="backup_rules",
        failure_threshold=2,
    )
    sample_mesh_router.register_endpoint(
        AgentEndpoint(agent_id="backup_rules", role="Backup Rules"),
        handler=reliable_backup,
    )

    # First attempt: primary fails, falls back
    res1 = await sample_mesh_router.route_and_call("primary_llm", {"query": "test1"})
    assert res1["served_by"] == "backup_agent"

    # Second attempt: primary fails, trips breaker to OPEN, falls back
    res2 = await sample_mesh_router.route_and_call("primary_llm", {"query": "test2"})
    assert res2["served_by"] == "backup_agent"
    assert sample_mesh_router.circuit_breakers["primary_llm"].state == CircuitState.OPEN

    root_span.finish()
    summary = tracer.export_summary(ctx.trace_id)

    # 2 attempts: (200 + 50 + 20) * 2 = 540 tokens
    assert summary["total_tokens"] == 540
    assert summary["span_count"] == 1


# --- Combination 3: Engine + Security Sandbox ---

@pytest.mark.asyncio
async def test_tier3_engine_sandbox_policy_enforcement(sandbox_fixture):
    """Engine + Sandbox: Node unauthorized file access is blocked and routed to error handler branch."""
    sandbox = sandbox_fixture["sandbox"]
    read_dir = sandbox_fixture["read_dir"]

    g = Graph("sandbox_guarded_graph")

    async def secure_read_node(state: State):
        target = state.get("target_path")
        try:
            valid_path = sandbox.validate_file_access(target, mode="r")
            state.set("content", Path(valid_path).read_text(encoding="utf-8"))
            state.set("security_status", "ALLOWED")
        except PermissionDeniedError as e:
            state.set("security_status", "VIOLATION_BLOCKED")
            state.set("error_message", str(e))
            state.append_event(EventType.NODE_FAILED, node_id="secure_read_node", payload={"error": str(e)})

    async def log_security_alert_node(state: State):
        state.set("audit_logged", True)

    async def process_content_node(state: State):
        state.set("processed", True)

    g.add_node("secure_read_node", secure_read_node)
    g.add_node("log_security_alert_node", log_security_alert_node)
    g.add_node("process_content_node", process_content_node)

    g.add_edge(
        "secure_read_node",
        "log_security_alert_node",
        condition=lambda s: s.get("security_status") == "VIOLATION_BLOCKED",
    )
    g.add_edge(
        "secure_read_node",
        "process_content_node",
        condition=lambda s: s.get("security_status") == "ALLOWED",
    )

    g.set_entry_point("secure_read_node")
    g.set_finish_point("log_security_alert_node")
    g.set_finish_point("process_content_node")

    # Run with unauthorized traversal path
    malicious_state = State(workflow_id="wf-tier3-attack")
    malicious_state.set("target_path", "/etc/shadow")
    final_malicious = await g.run(malicious_state)

    assert final_malicious.get("security_status") == "VIOLATION_BLOCKED"
    assert final_malicious.get("audit_logged") is True
    assert final_malicious.get("processed") is None

    # Run with authorized safe path
    safe_state = State(workflow_id="wf-tier3-safe")
    safe_state.set("target_path", str(read_dir / "ledger_source.txt"))
    final_safe = await g.run(safe_state)

    assert final_safe.get("security_status") == "ALLOWED"
    assert "TX_001" in final_safe.get("content")
    assert final_safe.get("processed") is True
    assert final_safe.get("audit_logged") is None


# --- Combination 4: Engine + Checkpoint + Crash + Mesh ---

@pytest.mark.asyncio
async def test_tier3_crash_recovery_skips_mesh_call_without_duplicate(
    checkpoint_manager: CheckpointManager,
    sample_mesh_router: MeshRouter,
):
    """Engine + Checkpoint + Mesh: Crashed workflow skips already executed mesh calls on recovery."""
    mesh_call_tracker = {"agent_a_calls": 0, "agent_b_calls": 0}
    crash_control = {"should_crash": True}

    async def agent_a_handler(payload: dict):
        mesh_call_tracker["agent_a_calls"] += 1
        return {"token_a": "ACC_TOKEN_123"}

    async def agent_b_handler(payload: dict):
        mesh_call_tracker["agent_b_calls"] += 1
        return {"report": "FINAL_AUDIT_REPORT"}

    sample_mesh_router.register_endpoint(
        AgentEndpoint(agent_id="mesh_a", role="Authenticator"),
        handler=agent_a_handler,
    )
    sample_mesh_router.register_endpoint(
        AgentEndpoint(agent_id="mesh_b", role="Reporter"),
        handler=agent_b_handler,
    )

    g = Graph("resilient_mesh_graph")

    async def node_authenticate(state: State):
        res = await sample_mesh_router.route_and_call("mesh_a", {})
        state.set("auth_token", res["token_a"])

    async def node_process_and_report(state: State):
        if crash_control["should_crash"]:
            crash_control["should_crash"] = False
            raise RuntimeError("OOM_KILLED_DURING_REPORTING")
        res = await sample_mesh_router.route_and_call("mesh_b", {"token": state.get("auth_token")})
        state.set("final_report", res["report"])

    g.add_node("auth", node_authenticate)
    g.add_node("report", node_process_and_report)
    g.add_edge("auth", "report")
    g.set_entry_point("auth")
    g.set_finish_point("report")

    wf_id = "wf-tier3-crash-mesh"
    state = State(workflow_id=wf_id)

    # Run 1: auth succeeds, report crashes
    with pytest.raises(RuntimeError):
        await g.run(state, checkpoint_manager=checkpoint_manager)

    assert mesh_call_tracker["agent_a_calls"] == 1
    assert mesh_call_tracker["agent_b_calls"] == 0

    # Resume from checkpoint
    recovered_state, last_completed_node = EventReplayer.resume_from_crash(wf_id, checkpoint_manager)
    assert recovered_state is not None
    assert last_completed_node == "auth"
    assert recovered_state.get("auth_token") == "ACC_TOKEN_123"

    # Verify event chain
    is_valid, err = EventReplayer.verify_event_chain(recovered_state.events)
    assert is_valid is True
    assert err is None

    # Continue execution starting at interrupted node 'report'
    final_state = await g.run(
        recovered_state,
        checkpoint_manager=checkpoint_manager,
        start_from_node="report",
    )

    # Crucial assertion: agent_a was NOT called again during recovery!
    assert mesh_call_tracker["agent_a_calls"] == 1, "Duplicate call detected on already completed Agent A!"
    assert mesh_call_tracker["agent_b_calls"] == 1
    assert final_state.get("final_report") == "FINAL_AUDIT_REPORT"


# --- Combination 5: Mesh + MCP Client + Security Sandbox ---

@pytest.mark.asyncio
async def test_tier3_mcp_tool_guarded_by_sandbox(sandbox_fixture):
    """Mesh + MCP + Sandbox: MCP tool execution wrapped by security sandbox policy."""
    sandbox = sandbox_fixture["sandbox"]
    write_dir = sandbox_fixture["write_dir"]

    mcp_client = MCPToolClient(server_name="guarded_tool_server")

    # Tool that writes files on behalf of agent
    def export_data_tool(file_path: str, data: str) -> dict:
        # Enforce sandbox validation before executing write
        valid_path = sandbox.validate_file_access(file_path, mode="w")
        with open(valid_path, "w", encoding="utf-8") as f:
            f.write(data)
        return {"bytes_written": len(data), "path": valid_path}

    mcp_client.register_tool(
        name="export_data",
        description="Exports serialized agent data to designated file",
        parameters={"file_path": "str", "data": "str"},
        handler=export_data_tool,
    )

    # Valid export to write directory succeeds
    safe_target = str(write_dir / "clean_export.txt")
    req_valid = MCPToolCallRequest(
        tool_name="export_data",
        arguments={"file_path": safe_target, "data": "Audit record payload"},
    )
    resp_valid = await mcp_client.invoke_tool(req_valid)
    assert resp_valid.success is True
    assert resp_valid.result["bytes_written"] == 20

    # Malicious export attempting directory traversal to /etc/cron.d
    req_attack = MCPToolCallRequest(
        tool_name="export_data",
        arguments={"file_path": "/etc/cron.d/malicious_job", "data": "* * * * * root"},
    )
    resp_attack = await mcp_client.invoke_tool(req_attack)
    assert resp_attack.success is False
    assert resp_attack.error is not None
    assert "Unauthorized file write access blocked" in resp_attack.error


# --- Combination 6: Mesh + Telemetry Context Propagation ---

@pytest.mark.asyncio
async def test_tier3_mesh_distributed_tracing(sample_mesh_router: MeshRouter):
    """Mesh + Telemetry: Multi-agent interaction propagating trace context and recording child spans."""
    tracer = AgentTracer()
    parent_ctx = tracer.get_or_create_context()
    root_span = parent_ctx.start_span("orchestrator_root")

    async def worker_agent_handler(payload: dict) -> dict:
        trace_id = payload.get("trace_id")
        parent_span_id = payload.get("parent_span_id")

        # Worker creates child span under orchestrator's trace
        worker_ctx = tracer.active_contexts.get(trace_id) or tracer.get_or_create_context(trace_id=trace_id)
        child_span = worker_ctx.start_span("worker_compute", parent_span_id=parent_span_id)
        worker_ctx.record_token_usage(prompt=300, completion=120)
        child_span.finish()

        return {"status": "SUCCESS", "span_id": child_span.span_id}

    sample_mesh_router.register_endpoint(
        AgentEndpoint(agent_id="worker_tracer", role="Worker"),
        handler=worker_agent_handler,
    )

    res = await sample_mesh_router.route_and_call(
        "worker_tracer",
        {"trace_id": parent_ctx.trace_id, "parent_span_id": root_span.span_id},
    )
    root_span.finish()

    assert res["status"] == "SUCCESS"
    summary = tracer.export_summary(parent_ctx.trace_id)

    assert summary["trace_id"] == parent_ctx.trace_id
    assert summary["span_count"] == 2
    assert summary["total_tokens"] == 420
    assert summary["prompt_tokens"] == 300
    assert summary["completion_tokens"] == 120


# --- Combination 7: Full Stack Integration (All 4 Layers) ---

@pytest.mark.asyncio
async def test_tier3_full_stack_collaborative_workflow(
    checkpoint_manager: CheckpointManager,
    sandbox_fixture,
):
    """Full Stack: Engine (Graph, State, Checkpoint) + Mesh (Router, CircuitBreaker) + Security + Telemetry."""
    sandbox = sandbox_fixture["sandbox"]
    write_dir = sandbox_fixture["write_dir"]

    router = MeshRouter()
    tracer = AgentTracer()
    trace_ctx = tracer.get_or_create_context()
    root_span = trace_ctx.start_span("full_stack_job")

    # Service 1: Risk Evaluator
    async def risk_service(payload: dict):
        trace_ctx.record_token_usage(prompt=150, completion=50)
        score = payload.get("amount", 0) / 1000.0
        return {"risk_score": score, "flag": "ALERT" if score > 50 else "CLEAN"}

    router.register_endpoint(
        AgentEndpoint(agent_id="evaluator", role="RiskEvaluator"),
        handler=risk_service,
    )

    g = Graph("full_stack_graph")

    async def step_evaluate(state: State):
        amt = state.get("tx_amount")
        res = await router.route_and_call("evaluator", {"amount": amt})
        state.set("evaluation", res)

    async def step_persist_report(state: State):
        eval_res = state.get("evaluation")
        report_file = str(write_dir / "full_stack_report.json")
        sandbox.validate_file_access(report_file, mode="w")
        Path(report_file).write_text(json.dumps(eval_res), encoding="utf-8")
        state.set("report_saved_at", report_file)

    g.add_node("evaluate", step_evaluate)
    g.add_node("persist", step_persist_report)
    g.add_edge("evaluate", "persist")
    g.set_entry_point("evaluate")
    g.set_finish_point("persist")

    wf_id = "wf-tier3-full-stack"
    state = State(workflow_id=wf_id)
    state.set("tx_amount", 75000)

    final_state = await g.run(state, checkpoint_manager=checkpoint_manager)
    root_span.finish()

    # Assertions across all 4 layers
    # Layer 1: Engine & Checkpoint
    assert final_state.version == 3
    assert Path(final_state.get("report_saved_at")).exists()
    checkpoints = checkpoint_manager.list_checkpoints(wf_id)
    assert len(checkpoints) >= 2

    # Layer 2: Mesh & Router
    assert final_state.get("evaluation")["flag"] == "ALERT"
    assert final_state.get("evaluation")["risk_score"] == 75.0

    # Layer 3: Security Sandbox
    report_path = final_state.get("report_saved_at")
    assert str(write_dir) in report_path

    # Layer 4: Telemetry
    summary = tracer.export_summary(trace_ctx.trace_id)
    assert summary["total_tokens"] == 200
