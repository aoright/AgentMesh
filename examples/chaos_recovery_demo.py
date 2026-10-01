"""Chaos Recovery and Fault-Tolerance Demo for AgentMesh.

Demonstrates:
1. Step-by-step deterministic DAG execution.
2. Simulated unexpected crash/failure in Node 2.
3. Checkpoint auto-recovery and event chain cryptographic verification.
4. Resuming from exact interrupted node with ZERO duplicate execution of Node 1.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentmesh.engine.state import State
from agentmesh.engine.graph import Graph
from agentmesh.engine.checkpoint import CheckpointManager
from agentmesh.engine.replay import EventReplayer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("demo")

# Flag to simulate a crash on the first run
SIMULATE_CRASH = True


async def step_1_data_ingestion(state: State):
    logger.info("Executing [Step 1: Data Ingestion]...")
    await asyncio.sleep(0.1)
    state.set("raw_records_count", 1500)
    state.set("ingestion_status", "COMPLETED")
    logger.info("Step 1 Completed. Ingested 1500 records.")


async def step_2_risk_audit(state: State):
    global SIMULATE_CRASH
    logger.info("Executing [Step 2: Risk Audit]...")
    if SIMULATE_CRASH:
        logger.warning(">>> CHAOS INJECTION: Simulating unexpected crash / OOM failure in Step 2! <<<")
        SIMULATE_CRASH = False  # Reset flag so recovery run passes
        raise RuntimeError("FATAL_SIMULATED_OOM: Process killed during Risk Audit.")

    await asyncio.sleep(0.1)
    state.set("high_risk_anomalies", 3)
    state.set("audit_status", "VERIFIED")
    logger.info("Step 2 Completed. Detected 3 high-risk anomalies.")


async def step_3_report_synthesis(state: State):
    logger.info("Executing [Step 3: Compliance Report Synthesis]...")
    await asyncio.sleep(0.1)
    anomalies = state.get("high_risk_anomalies", 0)
    state.set("final_verdict", f"AUDIT_FLAGGED_{anomalies}_ITEMS")
    logger.info("Step 3 Completed. Final Verdict generated.")


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


async def main():
    print("=" * 70)
    print(" AgentMesh: High-Reliability Chaos Recovery & Deterministic Resume Demo")
    print("=" * 70)

    workflow_id = "wf-enterprise-audit-1001"
    # Separate state directory outside source tree
    ckpt_mgr = CheckpointManager(base_dir="/tmp/agentmesh/checkpoints")

    graph = build_audit_graph()
    initial_state = State(workflow_id=workflow_id)

    # 1. First execution attempt (Expected to crash at Step 2)
    print("\n--- PHASE 1: Initial Execution (With Injected Chaos Failure) ---")
    try:
        await graph.run(initial_state, checkpoint_manager=ckpt_mgr)
    except Exception as e:
        print(f"[Captured Expected Crash]: {e}")

    # Inspect checkpoint status on disk
    checkpoints = ckpt_mgr.list_checkpoints(workflow_id)
    print(f"\nPersisted checkpoints after crash: {len(checkpoints)}")
    for cp in checkpoints:
        print(f" - {cp}")

    # 2. Recovery and Resume Execution
    print("\n--- PHASE 2: Autonomous Recovery & Resume from Interrupted Checkpoint ---")
    recovered_state, last_completed_node = EventReplayer.resume_from_crash(workflow_id, ckpt_mgr)

    if not recovered_state or not last_completed_node:
        print("ERROR: Failed to load recovery state!")
        return

    print(f"Restored state version: {recovered_state.version}")
    print(f"Last completed node: {last_completed_node}")
    print(f"Restored state data: {recovered_state.data}")

    # Cryptographic SHA256 chain verification
    is_valid, err = EventReplayer.verify_event_chain(recovered_state.events)
    print(f"SHA-256 Event Chain Integrity: {'VALID' if is_valid else 'CORRUPTED'}")

    # Resume from next node
    # Since step_1_ingestion was already completed, resume from step_2_audit directly
    print("\nResuming execution directly from 'step_2_audit' (Skipping Step 1 without duplicate cost)...")
    final_state = await graph.run(
        recovered_state,
        checkpoint_manager=ckpt_mgr,
        start_from_node="step_2_audit",
    )

    print("\n" + "=" * 70)
    print(" WORKFLOW SUCCESSFULLY RECOVERED AND COMPLETED!")
    print("=" * 70)
    print(f"Final State Version : {final_state.version}")
    print(f"Final State Data    : {final_state.data}")
    print(f"Total Event Records : {len(final_state.events)}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
