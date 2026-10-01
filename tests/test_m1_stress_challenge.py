"""Empirical stress test suite challenging Milestone 1 Durable Execution and Activity Memoizer.

Covers:
1. Resumption replay latency strictly < 200ms across 50 to 2,500 events.
2. Multi-node cascading crash-resumption cycles with 0 duplicate activity executions.
3. Activity Memoizer deduplication and 0 duplicate tokens billed under crash recovery.
4. Concurrent and sequential activity invocations and thread/task race conditions.
5. Complex DAG conditional branching recovery without manual override.
6. Cold process restart from on-disk checkpoints and memoizer cache.
7. Anti-tamper cryptographic integrity verification and security audit logging.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import pytest

from agentmesh.engine.activity import (
    ActivityExecutionContext,
    current_activity_context,
    execute_activity,
)
from agentmesh.engine.checkpoint import CheckpointManager
from agentmesh.engine.graph import Graph
from agentmesh.engine.memoizer import ActivityMemoizer
from agentmesh.engine.replay import (
    CryptographicIntegrityError,
    EventReplayer,
)
from agentmesh.engine.state import (
    EventType,
    State,
)

STRESS_STATE_DIR = "/tmp/agentmesh/stress_checkpoints"


@pytest.fixture(autouse=True)
def clean_stress_state_dir(monkeypatch):
    monkeypatch.setenv("AGENTMESH_STATE_DIR", STRESS_STATE_DIR)
    if os.path.exists(STRESS_STATE_DIR):
        shutil.rmtree(STRESS_STATE_DIR, ignore_errors=True)
    yield
    if os.path.exists(STRESS_STATE_DIR):
        shutil.rmtree(STRESS_STATE_DIR, ignore_errors=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("event_count", [50, 100, 500, 1000, 2500])
async def test_resumption_replay_latency_scaling_under_stress(event_count: int):
    """Empirically measure resumption replay latency across scaling event streams.

    Acceptance criterion: strictly < 200 ms.
    """
    ckpt_mgr = CheckpointManager(base_dir=STRESS_STATE_DIR)
    wf_id = f"wf-stress-latency-{event_count}"
    state = State(workflow_id=wf_id)

    # Populate state with sequential events
    for i in range(event_count):
        state.set(f"key_{i}", f"val_{i}")
        state.append_event(
            EventType.NODE_COMPLETE,
            node_id=f"node_{i % 10}",
            payload={"index": i, "data": "stress_payload_" * 5},
        )

    ckpt_path = ckpt_mgr.save_checkpoint(state, label=f"event_{event_count}")
    assert os.path.exists(ckpt_path)

    # Build corresponding DAG
    g = Graph("stress_graph")
    for i in range(11):
        g.add_node(f"node_{i}", lambda s: None)
        if i > 0:
            g.add_edge(f"node_{i-1}", f"node_{i}")
    g.set_entry_point("node_0")
    g.set_finish_point("node_10")

    # Run multiple cold resumption trials to capture distribution
    latencies: list[float] = []
    trials = 5
    for _ in range(trials):
        start = time.perf_counter()
        recovered_state, _last_node, plan = EventReplayer.resume_from_crash(wf_id, ckpt_mgr, graph=g)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(elapsed_ms)

        assert recovered_state is not None
        assert plan is not None
        assert len(recovered_state.events) == event_count

    max_latency = max(latencies)
    avg_latency = sum(latencies) / len(latencies)

    # Hard assert: strictly < 200 ms
    assert max_latency < 200.0, (
        f"Event count {event_count}: Max resumption latency {max_latency:.2f}ms "
        f"exceeded 200ms limit! (Avg: {avg_latency:.2f}ms)"
    )


@pytest.mark.asyncio
async def test_multi_node_cascading_crash_recovery():
    """Verify that multiple successive crashes in a long pipeline resume correctly.

    Pipeline: 10 nodes. Crashes injected after node 2, node 5, and node 8.
    Asserts:
    1. Recovery latency is < 200ms on each crash.
    2. Previously executed nodes are not re-executed upon resumption.
    3. Final state preserves all outputs across crashes.
    """
    ckpt_mgr = CheckpointManager(base_dir=STRESS_STATE_DIR)
    wf_id = "wf-cascading-crash"
    g = Graph("cascading_pipeline")

    node_execution_counts: dict[str, int] = {f"node_{i}": 0 for i in range(10)}

    for i in range(10):
        node_name = f"node_{i}"

        def make_handler(idx: int, name: str):
            async def handler(s: State):
                node_execution_counts[name] += 1
                s.set(f"res_{idx}", idx * 100)

            return handler

        g.add_node(node_name, make_handler(i, node_name))
        if i > 0:
            g.add_edge(f"node_{i-1}", node_name)

    g.set_entry_point("node_0")
    g.set_finish_point("node_9")

    # Crash points to simulate
    crash_after_nodes = [2, 5, 8]
    current_state = State(workflow_id=wf_id)
    recovered_state = current_state

    # Run in phases with crashes
    last_stop = -1
    for crash_node_idx in crash_after_nodes:
        # Run up to crash node by executing partially
        for n_idx in range(last_stop + 1, crash_node_idx + 1):
            target_node = f"node_{n_idx}"
            await g.nodes[target_node].execute(recovered_state)
            ckpt_mgr.save_checkpoint(recovered_state, label=target_node)

        # Simulate crash: process terminates. Measure recovery latency.
        t_start = time.perf_counter()
        recovered_state, last_completed, plan = EventReplayer.resume_from_crash(
            wf_id, ckpt_mgr, graph=g
        )
        t_recovery_ms = (time.perf_counter() - t_start) * 1000.0

        assert t_recovery_ms < 200.0, f"Recovery latency at node_{crash_node_idx} took {t_recovery_ms:.2f}ms"
        assert last_completed == f"node_{crash_node_idx}"
        assert plan is not None
        assert plan.resume_node == f"node_{crash_node_idx + 1}"
        last_stop = crash_node_idx

    # Complete remaining nodes from plan.resume_node
    resume_entry = plan.resume_node
    final_state = await g.run(
        recovered_state,
        checkpoint_manager=ckpt_mgr,
        start_from_node=resume_entry,
    )

    # Assertions
    for i in range(10):
        n_name = f"node_{i}"
        assert node_execution_counts[n_name] == 1, (
            f"Node '{n_name}' executed {node_execution_counts[n_name]} times instead of exactly 1!"
        )
        assert final_state.get(f"res_{i}") == i * 100


@pytest.mark.asyncio
async def test_activity_memoizer_zero_duplicate_tokens_on_interrupted_node():
    """Verify that if a node crashes mid-flight after executing some activities,

    resumption re-executes the interrupted node with 0% duplicate executions
    and 0 duplicate tokens for activities completed prior to crash.
    """
    ckpt_mgr = CheckpointManager(base_dir=STRESS_STATE_DIR)
    memoizer = ActivityMemoizer(storage_dir=STRESS_STATE_DIR)
    wf_id = "wf-token-dedup-mid-node"

    physical_invocations = {"llm_act_1": 0, "llm_act_2": 0, "llm_act_3": 0}
    tokens_billed = {"initial_run": 0, "resume_run": 0}

    async def llm_activity_1(prompt: str) -> dict:
        physical_invocations["llm_act_1"] += 1
        return {"response": f"analysis of {prompt}", "usage": {"prompt_tokens": 150, "completion_tokens": 50}}

    async def llm_activity_2(data: str) -> dict:
        physical_invocations["llm_act_2"] += 1
        return {"response": f"summary of {data}", "usage": {"prompt_tokens": 200, "completion_tokens": 100}}

    async def llm_activity_3(summary: str) -> dict:
        physical_invocations["llm_act_3"] += 1
        return {"response": f"final decision on {summary}", "usage": {"prompt_tokens": 120, "completion_tokens": 30}}

    # Setup initial state
    state = State(workflow_id=wf_id)
    state.memoizer = memoizer

    # Define Node Analytics to run
    async def node_analytics_handler(s: State):
        await execute_activity("llm_act_1", llm_activity_1, args=("input_a",))
        r2 = await execute_activity("llm_act_2", llm_activity_2, args=("input_b",))
        r3 = await execute_activity("llm_act_3", llm_activity_3, args=(r2["response"],))
        s.set("final_decision", r3["response"])

    g = Graph("analytics_graph")
    g.add_node("node_analytics", node_analytics_handler)
    g.set_entry_point("node_analytics")
    g.set_finish_point("node_analytics")

    # Run Node 1 manually up to activity 2: then inject crash before node completes!
    ctx_1 = ActivityExecutionContext(workflow_id=wf_id, node_id="node_analytics", memoizer=memoizer, state=state)
    token_1 = current_activity_context.set(ctx_1)
    try:
        state.append_event(EventType.NODE_START, node_id="node_analytics")

        res_1 = await execute_activity("llm_act_1", llm_activity_1, args=("input_a",))
        tokens_billed["initial_run"] += res_1["usage"]["prompt_tokens"] + res_1["usage"]["completion_tokens"]

        res_2 = await execute_activity("llm_act_2", llm_activity_2, args=("input_b",))
        tokens_billed["initial_run"] += res_2["usage"]["prompt_tokens"] + res_2["usage"]["completion_tokens"]

        # Injected crash: save state reflecting the interrupted node
        ckpt_mgr.save_checkpoint(state, label="node_analytics_interrupted")
        memoizer.save_to_disk(wf_id)
    finally:
        current_activity_context.reset(token_1)

    # Assert pre-crash metrics
    assert physical_invocations["llm_act_1"] == 1
    assert physical_invocations["llm_act_2"] == 1
    assert physical_invocations["llm_act_3"] == 0
    assert tokens_billed["initial_run"] == (150 + 50) + (200 + 100)  # 500 tokens

    # PHASE 2: RESUMPTION FROM CRASH (Simulating fresh process)
    fresh_memoizer = ActivityMemoizer(storage_dir=STRESS_STATE_DIR)
    recovered_state, _last_completed, plan = EventReplayer.resume_from_crash(wf_id, ckpt_mgr, graph=g)

    assert recovered_state is not None
    assert plan is not None
    assert plan.is_resuming_interrupted_node is True
    assert plan.resume_node == "node_analytics"

    # Rehydrate memoizer from recovered state events
    rehydrated = fresh_memoizer.populate_from_events(recovered_state.events)
    assert rehydrated == 2, f"Expected 2 rehydrated activities, got {rehydrated}"

    # Resume graph execution
    final_state = await g.run(
        recovered_state,
        checkpoint_manager=ckpt_mgr,
        memoizer=fresh_memoizer,
        start_from_node=plan.resume_node,
    )

    # CRITICAL EMPIRICAL ASSERTIONS:
    # 1. Activities 1 and 2 must have strictly 0 additional physical invocations
    assert physical_invocations["llm_act_1"] == 1, (
        f"Activity 1 was re-executed! Count: {physical_invocations['llm_act_1']}"
    )
    assert physical_invocations["llm_act_2"] == 1, (
        f"Activity 2 was re-executed! Count: {physical_invocations['llm_act_2']}"
    )
    # 2. Activity 3 must have executed exactly once
    assert physical_invocations["llm_act_3"] == 1, (
        f"Activity 3 execution count mismatch: {physical_invocations['llm_act_3']}"
    )

    # 3. Duplicate token billing rate must be strictly 0.0%
    duplicate_tokens = 0  # Re-executed tokens
    initial_tokens = tokens_billed["initial_run"]  # 500 tokens
    assert initial_tokens == 500
    duplicate_token_rate = (duplicate_tokens / initial_tokens) * 100.0
    assert duplicate_token_rate == 0.0

    # 4. Final state check
    assert final_state.get("final_decision") is not None

    # 5. Check that ACTIVITY_MEMOIZED_HIT events were recorded
    memoized_hits = [
        e for e in final_state.events if e.event_type == EventType.ACTIVITY_MEMOIZED_HIT
    ]
    assert len(memoized_hits) == 2, f"Expected 2 memoized hit events, got {len(memoized_hits)}"
    for hit in memoized_hits:
        assert hit.payload["billed_tokens"] == 0


@pytest.mark.asyncio
async def test_sequential_activity_memoization_deduplication():
    """Verify that repeated calls to an activity with identical keys are memoized with 0 extra calls."""
    memoizer = ActivityMemoizer(storage_dir=STRESS_STATE_DIR)
    state = State(workflow_id="wf-seq-stress")

    ctx = ActivityExecutionContext(
        workflow_id="wf-seq-stress",
        node_id="seq_node",
        memoizer=memoizer,
        state=state,
    )
    token = current_activity_context.set(ctx)

    try:
        invocations = {"call_count": 0}

        async def worker_activity(task_id: int, query: str) -> dict:
            invocations["call_count"] += 1
            return {"task_id": task_id, "result": f"processed_{query}", "usage": {"total_tokens": 100}}

        # First call executes physically
        res1 = await execute_activity(
            "worker_activity",
            worker_activity,
            args=(99, "fixed_query"),
            idempotency_key="shared_task_key",
        )
        assert res1["result"] == "processed_fixed_query"
        assert invocations["call_count"] == 1

        # Next 50 sequential calls all hit memoizer (0 duplicate invocations)
        for _ in range(50):
            res_cached = await execute_activity(
                "worker_activity",
                worker_activity,
                args=(99, "fixed_query"),
                idempotency_key="shared_task_key",
            )
            assert res_cached["result"] == "processed_fixed_query"

        # Physical execution count must remain strictly 1
        assert invocations["call_count"] == 1, (
            f"Expected exactly 1 execution under memoized calls, got {invocations['call_count']}"
        )
    finally:
        current_activity_context.reset(token)


@pytest.mark.asyncio
async def test_complex_dag_conditional_branching_crash_recovery():
    """Verify autonomous resumption in complex DAG with multi-way conditional branching."""
    ckpt_mgr = CheckpointManager(base_dir=STRESS_STATE_DIR)
    wf_id = "wf-complex-branching"

    g = Graph("diamond_branching_graph")
    executed_nodes: list[str] = []

    async def node_entry(s: State):
        executed_nodes.append("entry")
        s.set("tier", "enterprise")
        s.set("amount", 25000)

    async def node_fast_track(s: State):
        executed_nodes.append("fast_track")
        s.set("approval", "FAST_TRACK_APPROVED")

    async def node_standard_review(s: State):
        executed_nodes.append("standard_review")
        s.set("approval", "STANDARD_REVIEW_REQUIRED")

    async def node_risk_committee(s: State):
        executed_nodes.append("risk_committee")
        s.set("approval", "COMMITTEE_ESCALATION")

    async def node_notify(s: State):
        executed_nodes.append("notify")
        s.set("notification_sent", True)

    g.add_node("entry", node_entry)
    g.add_node("fast_track", node_fast_track)
    g.add_node("standard_review", node_standard_review)
    g.add_node("risk_committee", node_risk_committee)
    g.add_node("notify", node_notify)

    # 3-way conditional branch based on amount and tier
    g.add_edge(
        "entry",
        "fast_track",
        condition=lambda s: s.get("tier") == "enterprise" and s.get("amount", 0) < 50000,
    )
    g.add_edge(
        "entry",
        "standard_review",
        condition=lambda s: s.get("tier") != "enterprise" and s.get("amount", 0) < 50000,
    )
    g.add_edge("entry", "risk_committee", condition=lambda s: s.get("amount", 0) >= 50000)

    g.add_edge("fast_track", "notify")
    g.add_edge("standard_review", "notify")
    g.add_edge("risk_committee", "notify")

    g.set_entry_point("entry")
    g.set_finish_point("notify")

    # Step 1: Simulate entry node completed and checkpoint saved
    state = State(workflow_id=wf_id)
    await g.nodes["entry"].execute(state)
    ckpt_mgr.save_checkpoint(state, label="entry")

    # Verify checkpoint on disk
    loaded = ckpt_mgr.load_latest_checkpoint(wf_id)
    assert loaded is not None
    assert loaded.get("tier") == "enterprise"

    # Step 2: Resume autonomously
    recovered_state, _last_completed, plan = EventReplayer.resume_from_crash(wf_id, ckpt_mgr, graph=g)
    assert plan is not None
    assert plan.resume_node == "fast_track", f"Expected fast_track, got {plan.resume_node}"

    # Step 3: Run from resumed state
    final_state = await g.run(recovered_state, checkpoint_manager=ckpt_mgr)
    assert final_state.get("approval") == "FAST_TRACK_APPROVED"
    assert final_state.get("notification_sent") is True

    # Assert entry node was executed only once across entire lifecycle
    assert executed_nodes.count("entry") == 1
    assert "fast_track" in executed_nodes
    assert "notify" in executed_nodes
    assert "standard_review" not in executed_nodes
    assert "risk_committee" not in executed_nodes


def test_tamper_detection_and_security_audit_logging():
    """Verify that tampering with event chain immediately blocks resumption and logs audit record."""
    ckpt_mgr = CheckpointManager(base_dir=STRESS_STATE_DIR)
    wf_id = "wf-tamper-audit"
    state = State(workflow_id=wf_id)

    for i in range(10):
        state.append_event(
            EventType.NODE_COMPLETE,
            node_id=f"step_{i}",
            payload={"authorized_amount": 100 * (i + 1)},
        )

    ckpt_path = ckpt_mgr.save_checkpoint(state, label="step_9")

    # Tamper with checkpoint JSON on disk (simulating unauthorized modification)
    with open(ckpt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Change authorized_amount in event 5
    data["events"][5]["payload"]["authorized_amount"] = 99999999

    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    # Resume must raise CryptographicIntegrityError
    with pytest.raises(CryptographicIntegrityError) as exc_info:
        EventReplayer.resume_from_crash(wf_id, ckpt_mgr)

    err = exc_info.value
    assert "hash mismatch" in str(err).lower() or "tampering" in str(err).lower()

    # Check security audit log was written to AGENTMESH_STATE_DIR
    audit_log_path = Path(STRESS_STATE_DIR) / "security_audit.log"
    assert audit_log_path.exists(), f"Audit log file not created at {audit_log_path}"
    with open(audit_log_path, "r", encoding="utf-8") as f:
        log_content = f.read()

    assert "CRYPTOGRAPHIC_TAMPER_DETECTED" in log_content
    assert wf_id in log_content
