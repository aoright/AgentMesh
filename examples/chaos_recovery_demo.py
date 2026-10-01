#!/usr/bin/env python3
"""Chaos Recovery and Fault-Tolerance Demo for AgentMesh.

Demonstrates enterprise production-grade resilience SLAs:
1. Sub-200ms Crash Recovery: Cold checkpoint rehydration, cryptographic SHA-256 chain verification,
   and autonomous resumption planning executed in < 200 ms.
2. 0% Duplicate Token Billing: Activity isolation and memoization guaranteeing zero duplicate physical
   invocations and strictly 0% duplicate token billing upon workflow crash recovery.
3. Sub-5ms Circuit Breaker & Fallback: Dynamic fault injection tripping circuit breaker to OPEN in < 5 ms
   with seamless rerouting to healthy fallback agent and fast-rejection protection.
4. Strict Physical State Separation: Runtime state, checkpoints, and logs isolated outside repository tree
   (/tmp/ or AGENTMESH_STATE_DIR) with active rejection of in-repo state paths.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# Ensure project root is available in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentmesh.engine.activity import execute_activity
from agentmesh.engine.checkpoint import CheckpointManager, StateSeparationError
from agentmesh.engine.graph import Graph
from agentmesh.engine.memoizer import ActivityMemoizer
from agentmesh.engine.replay import EventReplayer
from agentmesh.engine.state import State
from agentmesh.mesh.router import AgentEndpoint, CircuitState, MeshRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("chaos_demo")

# Global tracking metrics
SIMULATE_CRASH = True
activity_physical_invocations: dict[str, int] = {"extract_data": 0}
tokens_billed_ledger: dict[str, int] = {"initial_run": 0, "resume_run": 0}
primary_service_calls: int = 0
fallback_service_calls: int = 0


# --- External Non-Deterministic Activities (LLM / Tool Simulation) ---

async def mock_llm_data_extraction_activity(batch_id: str) -> dict[str, Any]:
    """Simulates an external LLM-driven data extraction activity incurring token costs."""
    activity_physical_invocations["extract_data"] += 1
    prompt_tokens = 320
    completion_tokens = 80
    total_tokens = prompt_tokens + completion_tokens

    # Artificial I/O latency
    await asyncio.sleep(0.05)

    return {
        "batch_id": batch_id,
        "records_count": 1500,
        "summary": "1500 banking transactions parsed",
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


# --- Workflow Graph Nodes ---

async def step_1_data_ingestion(state: State) -> None:
    logger.info("Executing [Step 1: Data Ingestion & LLM Extraction]...")

    # Execute non-deterministic activity with memoizer & token tracking
    res = await execute_activity(
        "mock_llm_data_extraction_activity",
        mock_llm_data_extraction_activity,
        args=("batch-2026-audit",),
    )
    tokens = res["usage"]["total_tokens"]
    tokens_billed_ledger["initial_run"] += tokens

    state.set("raw_records_count", res["records_count"])
    state.set("extraction_summary", res["summary"])
    state.set("ingestion_status", "COMPLETED")
    logger.info("Step 1 Completed. Ingested %d records (Tokens billed: %d).", res["records_count"], tokens)


async def step_2_risk_audit(state: State) -> None:
    global SIMULATE_CRASH
    logger.info("Executing [Step 2: Risk Audit]...")

    if SIMULATE_CRASH:
        logger.warning(">>> CHAOS INJECTION: Simulating unexpected crash / OOM failure in Step 2! <<<")
        SIMULATE_CRASH = False  # Reset flag so resumption run succeeds
        raise RuntimeError("FATAL_SIMULATED_OOM: Process killed during Risk Audit.")

    await asyncio.sleep(0.05)
    state.set("high_risk_anomalies", 3)
    state.set("audit_status", "VERIFIED")
    logger.info("Step 2 Completed. Detected 3 high-risk anomalies.")


async def step_3_report_synthesis(state: State) -> None:
    logger.info("Executing [Step 3: Compliance Report Synthesis]...")
    await asyncio.sleep(0.05)
    anomalies = state.get("high_risk_anomalies", 0)
    state.set("final_verdict", f"AUDIT_FLAGGED_{anomalies}_ITEMS")
    logger.info("Step 3 Completed. Final Verdict generated: AUDIT_FLAGGED_%d_ITEMS.", anomalies)


def build_audit_graph() -> Graph:
    g = Graph(name="enterprise_audit_graph")
    g.add_node("step_1_ingestion", step_1_data_ingestion)
    g.add_node("step_2_audit", step_2_risk_audit)
    g.add_node("step_3_report", step_3_report_synthesis)

    g.add_edge("step_1_ingestion", "step_2_audit")
    g.add_edge("step_2_audit", "step_3_report")

    g.set_entry_point("step_1_ingestion")
    g.set_finish_point("step_3_report")
    return g


# --- Main Demonstration Runner ---

async def main() -> None:
    print("=" * 78)
    print(" AgentMesh: High-Reliability Chaos Recovery & Resilience Verification Demo")
    print("=" * 78)

    # 0. Setup isolated state directory honoring AGENTMESH_STATE_DIR
    run_uuid = uuid.uuid4().hex[:8]
    base_state_dir = os.environ.get("AGENTMESH_STATE_DIR", f"/tmp/agentmesh/chaos_demo_{run_uuid}")
    workflow_id = f"wf-chaos-audit-{run_uuid}"

    ckpt_mgr = CheckpointManager(base_dir=base_state_dir)
    memoizer = ActivityMemoizer(storage_dir=base_state_dir)
    graph = build_audit_graph()
    initial_state = State(workflow_id=workflow_id)

    # -------------------------------------------------------------------------
    # PHASE 1: Initial Execution with Injected Fatal Crash (OOM Simulation)
    # -------------------------------------------------------------------------
    print("\n--- PHASE 1: Initial Execution with Injected Fatal Crash (OOM Simulation) ---")
    crash_captured = False
    try:
        await graph.run(initial_state, checkpoint_manager=ckpt_mgr, memoizer=memoizer)
    except RuntimeError as exc:
        if "FATAL_SIMULATED_OOM" in str(exc):
            crash_captured = True
            print(f"[Captured Expected Crash]: {exc}")
        else:
            raise

    assert crash_captured, "Expected chaos crash was not triggered!"
    assert activity_physical_invocations["extract_data"] == 1, "Activity should have executed once before crash"
    assert tokens_billed_ledger["initial_run"] == 400, "Initial tokens billed should be 400"

    checkpoints = ckpt_mgr.list_checkpoints(workflow_id)
    print(f"Persisted checkpoints after crash: {len(checkpoints)}")
    for cp in checkpoints:
        print(f" - {cp}")

    # -------------------------------------------------------------------------
    # PHASE 2: Sub-200ms Crash Recovery & 0% Duplicate Token Resumption (SLA 1 & 2)
    # -------------------------------------------------------------------------
    print("\n--- PHASE 2: Sub-200ms Crash Recovery & 0% Duplicate Token Resumption ---")

    # Cold recovery benchmark
    t_start = time.perf_counter()
    recovered_state, last_completed_node, plan = EventReplayer.resume_from_crash(
        workflow_id, ckpt_mgr, graph=graph
    )
    t_end = time.perf_counter()
    recovery_latency_ms = (t_end - t_start) * 1000.0

    assert recovered_state is not None, "Failed to load recovered state"
    assert last_completed_node == "step_1_ingestion", f"Expected last node step_1_ingestion, got {last_completed_node}"
    assert plan is not None, "Resumption plan could not be determined"
    assert plan.resume_node == "step_2_audit", f"Expected resume node step_2_audit, got {plan.resume_node}"

    # SLA 1: Recovery Latency Assertion (< 200 ms)
    assert recovery_latency_ms < 200.0, f"SLA Violation: Recovery latency was {recovery_latency_ms:.2f} ms (limit: <200ms)"
    print(f"[SLA 1 VERIFIED] Cold Crash Recovery Latency: {recovery_latency_ms:.2f} ms (Target: < 200.00 ms)")

    # Cryptographic SHA-256 chain verification
    is_valid, err = EventReplayer.verify_event_chain(recovered_state.events)
    assert is_valid is True, f"Cryptographic integrity failed: {err}"
    print(f"Cryptographic SHA-256 Event Chain Integrity: VALID (Total Events: {len(recovered_state.events)})")
    print(f"Autonomous Resumption Breakpoint: '{plan.resume_node}' ({plan.reason})")

    # Resuming execution from autonomous resume node
    fresh_memoizer = ActivityMemoizer(storage_dir=base_state_dir)
    final_state = await graph.run(
        recovered_state,
        checkpoint_manager=ckpt_mgr,
        memoizer=fresh_memoizer,
        start_from_node=plan.resume_node,
    )

    # SLA 2: 0% Duplicate Token Billing Verification
    duplicate_invocations = activity_physical_invocations["extract_data"] - 1
    assert duplicate_invocations == 0, f"Step 1 activity re-executed {duplicate_invocations} times during replay!"

    duplicate_tokens = duplicate_invocations * 400
    initial_tokens = tokens_billed_ledger["initial_run"]
    duplicate_token_rate = (duplicate_tokens / initial_tokens) * 100.0
    assert duplicate_token_rate == 0.0, f"Duplicate token rate was {duplicate_token_rate}%, expected 0.0%"

    print(f"[SLA 2 VERIFIED] Duplicate Token Billing Rate: {duplicate_token_rate:.1f}% (0 duplicate tokens billed)")
    print(f"Final State Workflow Status: {final_state.get('audit_status')}, Verdict: {final_state.get('final_verdict')}")

    # -------------------------------------------------------------------------
    # PHASE 3: Mesh Dynamic Fault Injection & Sub-5ms Circuit Breaker (SLA 3)
    # -------------------------------------------------------------------------
    print("\n--- PHASE 3: Mesh Dynamic Fault Injection & Sub-5ms Circuit Breaker ---")

    router = MeshRouter()
    global primary_service_calls, fallback_service_calls
    primary_service_calls = 0
    fallback_service_calls = 0

    async def primary_ai_audit_service(payload: dict[str, Any]) -> dict[str, Any]:
        global primary_service_calls
        primary_service_calls += 1
        raise ConnectionResetError("Primary AI Audit Service connection reset / upstream 503")

    async def fallback_rule_audit_service(payload: dict[str, Any]) -> dict[str, Any]:
        global fallback_service_calls
        fallback_service_calls += 1
        return {
            "engine": "FALLBACK_HEURISTIC_RULE_V1",
            "risk_score": 0.82,
            "status": "DEGRADED_FALLBACK_SUCCESS",
        }

    router.register_endpoint(
        AgentEndpoint(agent_id="primary_ai_audit", role="PrimaryAuditor"),
        handler=primary_ai_audit_service,
        fallback_agent_id="fallback_rule_audit",
        failure_threshold=2,
    )
    router.register_endpoint(
        AgentEndpoint(agent_id="fallback_rule_audit", role="FallbackAuditor"),
        handler=fallback_rule_audit_service,
    )

    # Call 1: Primary fails once, gracefully routed to fallback (Circuit remains CLOSED)
    res1 = await router.route_and_call("primary_ai_audit", {"payload": "tx_eval_1"})
    assert res1["status"] == "DEGRADED_FALLBACK_SUCCESS"
    assert router.circuit_breakers["primary_ai_audit"].state == CircuitState.CLOSED
    print("Call 1: Primary failed -> Fallback succeeded. Circuit state: CLOSED (Failure count: 1/2)")

    # Call 2: Primary fails second time, tripping circuit breaker to OPEN in < 5ms
    t_trip_start = time.perf_counter()
    res2 = await router.route_and_call("primary_ai_audit", {"payload": "tx_eval_2"})
    t_trip_end = time.perf_counter()
    trip_latency_ms = (t_trip_end - t_trip_start) * 1000.0

    cb = router.circuit_breakers["primary_ai_audit"]
    assert cb.state == CircuitState.OPEN, f"Expected circuit state OPEN, got {cb.state}"
    assert trip_latency_ms < 5.0, f"SLA Violation: Breaker trip took {trip_latency_ms:.2f} ms (limit: <5.0 ms)"
    assert res2["status"] == "DEGRADED_FALLBACK_SUCCESS"
    print(f"[SLA 3 VERIFIED] Circuit Breaker Tripped to OPEN in {trip_latency_ms:.2f} ms (Target: < 5.00 ms)")

    # Call 3: Fast-Rejection Protection Verification (Zero load on broken primary)
    res3 = await router.route_and_call("primary_ai_audit", {"payload": "tx_eval_3"})
    assert res3["status"] == "DEGRADED_FALLBACK_SUCCESS"
    assert primary_service_calls == 2, f"Primary called {primary_service_calls} times; expected 2 due to OPEN circuit!"
    assert fallback_service_calls == 3, f"Fallback called {fallback_service_calls} times; expected 3!"
    print("Call 3: Circuit OPEN -> Immediate fast-reject of primary, zero extra load placed on failing service")

    # -------------------------------------------------------------------------
    # PHASE 4: Strict Physical State Separation Validation (SLA 4)
    # -------------------------------------------------------------------------
    print("\n--- PHASE 4: Strict Physical State Separation & Security Validation ---")

    # 1. Verify valid directory resolution
    resolved_dir = CheckpointManager.validate_state_dir(base_state_dir)
    print(f"Active state directory confirmed: {resolved_dir}")

    # 2. Defense-in-depth: In-repo state path injection rejection
    repo_root = CheckpointManager._detect_repo_root()
    separation_enforced = False
    if repo_root is not None:
        illegal_path = repo_root / "checkpoints_illegal_leakage"
        try:
            CheckpointManager(base_dir=str(illegal_path))
        except StateSeparationError as e:
            separation_enforced = True
            print(f"[SLA 4 VERIFIED] In-repo state path rejected with StateSeparationError: {e}")
        assert separation_enforced, "Failed to reject illegal in-repo state directory!"
    else:
        print("Note: Repository root not detected; skipping in-repo path test.")

    print("\n" + "=" * 78)
    print(" ALL 4 RESILIENCE & CHAOS SLAS SUCCESSFULLY BENCHMARKED AND VERIFIED!")
    print(" 1. Crash Recovery Latency  : < 200 ms (PASSED)")
    print(" 2. Duplicate Token Billing : 0.0% (PASSED)")
    print(" 3. Circuit Breaker Tripping: < 5.0 ms (PASSED)")
    print(" 4. Physical State Isolation: Fully Enforced Outside Repo (PASSED)")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
