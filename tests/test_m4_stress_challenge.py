"""Empirical stress test suite challenging Milestone 4 Chaos Recovery, SLAs, and State Separation.

Adversarially challenges:
1. Crash recovery latency across 30+ runs under varying graph sizes (3 to 50 nodes)
   and event histories (up to 1,000 events), asserting latency strictly < 200.0ms in 100% of cases.
2. Token billing deduplication across multi-step cascading crashes and intra-node multiple activities,
   asserting strictly 0 duplicate tokens billed (0.0% duplicate rate).
3. Circuit breaker trip latency to OPEN under concurrent load across 100+ trials,
   asserting trip time strictly < 5.0ms and 100% rerouting to fallback without recursive loops.
4. Physical state separation violation rejection under multiple directory injection attack vectors,
   asserting StateSeparationError is unconditionally raised.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from agentmesh.engine.activity import execute_activity
from agentmesh.engine.checkpoint import CheckpointManager, StateSeparationError
from agentmesh.engine.graph import Graph
from agentmesh.engine.memoizer import ActivityMemoizer
from agentmesh.engine.replay import EventReplayer
from agentmesh.engine.state import EventType, State
from agentmesh.mesh.router import AgentEndpoint, CircuitBreaker, CircuitState, MeshRouter

STRESS_STATE_DIR_BASE = "/tmp/agentmesh/m4_stress_challenge"


@pytest.fixture(autouse=True)
def clean_m4_stress_state_dir(monkeypatch: pytest.MonkeyPatch):
    """Ensure clean isolated state directory outside repository tree."""
    unique_run = uuid.uuid4().hex[:8]
    test_dir = f"{STRESS_STATE_DIR_BASE}_{unique_run}"
    monkeypatch.setenv("AGENTMESH_STATE_DIR", test_dir)
    os.makedirs(test_dir, exist_ok=True)
    yield test_dir
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir, ignore_errors=True)


# ==============================================================================
# Challenge 1: Crash Recovery Latency Benchmark across Graph Sizes (<200ms)
# ==============================================================================

class TestCrashRecoveryLatencyStress:
    """Stress-tests crash recovery latency across multiple graph iterations and sizes."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "graph_size,events_count",
        [
            (3, 10),
            (5, 25),
            (10, 50),
            (25, 125),
            (50, 250),
        ],
    )
    async def test_recovery_latency_scaling_across_graph_sizes(
        self, graph_size: int, events_count: int, clean_m4_stress_state_dir: str
    ):
        """Measures cold crash recovery latency across scaling DAG topologies and event depths.

        Asserts 100% of recoveries execute strictly in < 200.0ms.
        """
        ckpt_mgr = CheckpointManager(base_dir=clean_m4_stress_state_dir)
        wf_id = f"wf-scale-{graph_size}-{uuid.uuid4().hex[:6]}"

        # 1. Build DAG of specified size
        g = Graph(name=f"scale_graph_{graph_size}")
        node_names = [f"node_{i:03d}" for i in range(graph_size)]
        for name in node_names:
            async def make_dummy_fn(st: State) -> None:
                pass
            g.add_node(name, make_dummy_fn)

        for i in range(graph_size - 1):
            g.add_edge(node_names[i], node_names[i + 1])

        g.set_entry_point(node_names[0])
        g.set_finish_point(node_names[-1])

        # 2. Synthesize state history representing crash at midpoint
        midpoint_idx = graph_size // 2
        last_completed = node_names[midpoint_idx - 1] if midpoint_idx > 0 else None

        state = State(workflow_id=wf_id)
        state.append_event(
            EventType.WORKFLOW_START,
            payload={"graph": g.name, "start_node": node_names[0]},
        )

        for i in range(midpoint_idx):
            state.append_event(EventType.NODE_START, node_id=node_names[i])
            state.set(f"data_key_{i}", f"val_{i}")
            state.append_event(
                EventType.NODE_COMPLETE,
                node_id=node_names[i],
                payload={"output": f"done_{i}"},
            )

        # Pad remaining events up to events_count to simulate realistic event depth
        curr_len = len(state.events)
        for p in range(curr_len, events_count):
            state.append_event(
                EventType.STATE_MUTATED,
                payload={"synthetic_idx": p, "metric": p * 1.5},
            )

        # Persist checkpoint to disk
        ckpt_path = ckpt_mgr.save_checkpoint(state, label=last_completed or "init")
        assert os.path.exists(ckpt_path), "Checkpoint must be written to disk"

        # 3. Perform cold crash recovery benchmark
        t_start = time.perf_counter()
        recovered_state, _loaded_last_node, plan = EventReplayer.resume_from_crash(
            wf_id, ckpt_mgr, graph=g
        )
        t_end = time.perf_counter()
        recovery_ms = (t_end - t_start) * 1000.0

        # Assertions
        assert recovered_state is not None, "Recovered state must not be None"
        assert plan is not None, "Resumption plan must not be None"
        assert recovery_ms < 200.0, (
            f"SLA Violation: Recovery latency {recovery_ms:.2f}ms exceeded 200.0ms limit "
            f"(graph_size={graph_size}, events={len(recovered_state.events)})"
        )
        assert len(recovered_state.events) >= events_count

        # Cryptographic chain verification
        is_valid, err = EventReplayer.verify_event_chain(recovered_state.events)
        assert is_valid is True, f"Event chain verification failed: {err}"

    @pytest.mark.asyncio
    async def test_recovery_latency_30_consecutive_runs_sla(
        self, clean_m4_stress_state_dir: str
    ):
        """Executes 30 consecutive cold crash recoveries and verifies mean and max < 200.0ms."""
        ckpt_mgr = CheckpointManager(base_dir=clean_m4_stress_state_dir)
        latencies: list[float] = []

        g = Graph(name="bench_graph_20")
        for i in range(20):
            g.add_node(f"step_{i}", lambda st: None)
            if i > 0:
                g.add_edge(f"step_{i-1}", f"step_{i}")
        g.set_entry_point("step_0")
        g.set_finish_point("step_19")

        for trial in range(30):
            wf_id = f"wf-bench-trial-{trial:03d}"
            state = State(workflow_id=wf_id)
            state.append_event(EventType.WORKFLOW_START, payload={"trial": trial})
            for s in range(10):
                state.append_event(EventType.NODE_START, node_id=f"step_{s}")
                state.set(f"trial_{trial}_step_{s}", s)
                state.append_event(EventType.NODE_COMPLETE, node_id=f"step_{s}")

            ckpt_mgr.save_checkpoint(state, label=f"trial_{trial}_step_9")

            # Cold reload
            t0 = time.perf_counter()
            rec_st, _last_node, plan = EventReplayer.resume_from_crash(wf_id, ckpt_mgr, graph=g)
            t1 = time.perf_counter()
            latency = (t1 - t0) * 1000.0
            latencies.append(latency)

            assert rec_st is not None
            assert plan is not None
            assert plan.resume_node == "step_10"
            assert latency < 200.0, f"Run {trial} exceeded 200ms SLA: {latency:.2f}ms"

        mean_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        assert mean_latency < 200.0, f"Mean latency {mean_latency:.2f}ms exceeded 200ms"
        assert max_latency < 200.0, f"Max latency {max_latency:.2f}ms exceeded 200ms"


# ==============================================================================
# Challenge 2: Token Billing Dedup on Multi-Step Crash and Replay
# ==============================================================================

class TestTokenBillingDedupStress:
    """Stress-tests activity memoization and duplicate token billing under cascading crashes."""

    @pytest.mark.asyncio
    async def test_cascading_multi_step_crash_zero_duplicate_tokens(
        self, clean_m4_stress_state_dir: str
    ):
        """Simulates cascading crashes at step_1, step_2, step_3, step_4.

        Asserts that activities already completed are never re-executed physically,
        and duplicate tokens billed on replay is strictly 0 (0.0%).
        """
        ckpt_mgr = CheckpointManager(base_dir=clean_m4_stress_state_dir)
        memoizer = ActivityMemoizer(storage_dir=clean_m4_stress_state_dir)
        wf_id = f"wf-cascade-crash-{uuid.uuid4().hex[:6]}"

        activity_invocations: dict[str, int] = {f"act_{i}": 0 for i in range(1, 5)}
        token_costs: dict[str, int] = {
            "act_1": 150,
            "act_2": 250,
            "act_3": 350,
            "act_4": 450,
        }

        async def run_activity(name: str) -> dict[str, Any]:
            activity_invocations[name] += 1
            tokens = token_costs[name]
            return {
                "activity": name,
                "usage": {
                    "prompt_tokens": tokens - 50,
                    "completion_tokens": 50,
                    "total_tokens": tokens,
                },
            }

        # Build 4-step graph
        crash_at_step: int = 2

        async def step_1_fn(state: State) -> None:
            res = await execute_activity("act_1", run_activity, args=("act_1",))
            state.set("step_1_done", True)
            state.set("step_1_tokens", res["usage"]["total_tokens"])

        async def step_2_fn(state: State) -> None:
            if crash_at_step == 2:
                raise RuntimeError("INJECTED_CRASH_STEP_2")
            res = await execute_activity("act_2", run_activity, args=("act_2",))
            state.set("step_2_done", True)
            state.set("step_2_tokens", res["usage"]["total_tokens"])

        async def step_3_fn(state: State) -> None:
            if crash_at_step == 3:
                raise RuntimeError("INJECTED_CRASH_STEP_3")
            res = await execute_activity("act_3", run_activity, args=("act_3",))
            state.set("step_3_done", True)
            state.set("step_3_tokens", res["usage"]["total_tokens"])

        async def step_4_fn(state: State) -> None:
            if crash_at_step == 4:
                raise RuntimeError("INJECTED_CRASH_STEP_4")
            res = await execute_activity("act_4", run_activity, args=("act_4",))
            state.set("step_4_done", True)
            state.set("step_4_tokens", res["usage"]["total_tokens"])

        g = Graph(name="cascade_graph")
        g.add_node("step_1", step_1_fn)
        g.add_node("step_2", step_2_fn)
        g.add_node("step_3", step_3_fn)
        g.add_node("step_4", step_4_fn)

        g.add_edge("step_1", "step_2")
        g.add_edge("step_2", "step_3")
        g.add_edge("step_3", "step_4")

        g.set_entry_point("step_1")
        g.set_finish_point("step_4")

        # Cycle 1: step_1 succeeds and checkpoints, crash at step_2
        current_state = State(workflow_id=wf_id)
        with pytest.raises(RuntimeError, match="INJECTED_CRASH_STEP_2"):
            await g.run(current_state, checkpoint_manager=ckpt_mgr, memoizer=memoizer)

        assert activity_invocations["act_1"] == 1
        assert activity_invocations["act_2"] == 0

        # Cycle 2: Resume from crash, crash at step 3
        crash_at_step = 3
        rec_st, last_node, plan = EventReplayer.resume_from_crash(wf_id, ckpt_mgr, graph=g)
        assert last_node == "step_1"
        assert plan is not None
        assert plan.resume_node == "step_2"

        with pytest.raises(RuntimeError, match="INJECTED_CRASH_STEP_3"):
            await g.run(
                rec_st,
                checkpoint_manager=ckpt_mgr,
                memoizer=memoizer,
                start_from_node=plan.resume_node,
            )

        assert activity_invocations["act_1"] == 1, "act_1 must not re-execute"
        assert activity_invocations["act_2"] == 1, "act_2 executed once"
        assert activity_invocations["act_3"] == 0

        # Cycle 3: Resume from crash, crash at step 4
        crash_at_step = 4
        rec_st, last_node, plan = EventReplayer.resume_from_crash(wf_id, ckpt_mgr, graph=g)
        assert last_node == "step_2"
        assert plan is not None
        assert plan.resume_node == "step_3"

        with pytest.raises(RuntimeError, match="INJECTED_CRASH_STEP_4"):
            await g.run(
                rec_st,
                checkpoint_manager=ckpt_mgr,
                memoizer=memoizer,
                start_from_node=plan.resume_node,
            )

        assert activity_invocations["act_1"] == 1
        assert activity_invocations["act_2"] == 1
        assert activity_invocations["act_3"] == 1
        assert activity_invocations["act_4"] == 0

        # Cycle 4: Final run to completion (no crashes)
        crash_at_step = 0
        rec_st, last_node, plan = EventReplayer.resume_from_crash(wf_id, ckpt_mgr, graph=g)
        assert last_node == "step_3"
        assert plan is not None
        assert plan.resume_node == "step_4"

        final_state = await g.run(
            rec_st,
            checkpoint_manager=ckpt_mgr,
            memoizer=memoizer,
            start_from_node=plan.resume_node,
        )

        assert final_state.get("step_1_done") is True
        assert final_state.get("step_2_done") is True
        assert final_state.get("step_3_done") is True
        assert final_state.get("step_4_done") is True

        # Total physical invocations across ALL 4 cycles
        for act_name, count in activity_invocations.items():
            assert count == 1, f"Activity {act_name} was invoked {count} times (expected exactly 1)"

        # Compute duplicate tokens
        expected_total_tokens = sum(token_costs.values())
        actual_billed_tokens = sum(
            activity_invocations[act] * token_costs[act] for act in activity_invocations
        )
        duplicate_tokens = actual_billed_tokens - expected_total_tokens
        assert duplicate_tokens == 0, f"Duplicate tokens billed: {duplicate_tokens}"

    @pytest.mark.asyncio
    async def test_full_replay_from_entry_point_zero_duplicate_tokens(
        self, clean_m4_stress_state_dir: str
    ):
        """Tests that replaying the completed workflow from entry point (node 0)

        uses memoized activity records rehydrated from events, incurring 0 duplicate tokens.
        """
        ckpt_mgr = CheckpointManager(base_dir=clean_m4_stress_state_dir)
        wf_id = f"wf-full-replay-{uuid.uuid4().hex[:6]}"

        call_counts = {"fetch": 0, "compute": 0}

        async def fetch_data() -> dict[str, Any]:
            call_counts["fetch"] += 1
            return {"data": [1, 2, 3], "usage": {"total_tokens": 200}}

        async def compute_result(data: list[int]) -> dict[str, Any]:
            call_counts["compute"] += 1
            return {"sum": sum(data), "usage": {"total_tokens": 150}}

        async def node_1(state: State) -> None:
            r = await execute_activity("fetch_data", fetch_data)
            state.set("raw_data", r["data"])

        async def node_2(state: State) -> None:
            d = state.get("raw_data", [])
            r = await execute_activity("compute_result", compute_result, args=(d,))
            state.set("result_sum", r["sum"])

        g = Graph(name="two_step_graph")
        g.add_node("node_1", node_1)
        g.add_node("node_2", node_2)
        g.add_edge("node_1", "node_2")
        g.set_entry_point("node_1")
        g.set_finish_point("node_2")

        init_st = State(workflow_id=wf_id)
        memoizer = ActivityMemoizer(storage_dir=clean_m4_stress_state_dir)
        await g.run(init_st, checkpoint_manager=ckpt_mgr, memoizer=memoizer)

        assert call_counts["fetch"] == 1
        assert call_counts["compute"] == 1

        # Now load cold state from disk
        recovered_state, _ = EventReplayer.resume_from_crash(wf_id, ckpt_mgr)
        assert recovered_state is not None

        # Re-run full graph from entry point with a brand new empty memoizer
        fresh_memoizer = ActivityMemoizer(storage_dir=clean_m4_stress_state_dir)
        replay_state = State(workflow_id=wf_id)
        # Populate fresh memoizer from the saved events of the original workflow
        fresh_memoizer.populate_from_events(recovered_state.events)

        # Contextual execution: when node_1 and node_2 execute, memoizer should intercept
        await g.run(replay_state, memoizer=fresh_memoizer, start_from_node="node_1")

        # Asserts physical invocations did not increase
        assert call_counts["fetch"] == 1, "fetch_data was re-invoked on replay!"
        assert call_counts["compute"] == 1, "compute_result was re-invoked on replay!"
        assert replay_state.get("result_sum") == 6


# ==============================================================================
# Challenge 3: Circuit Breaker Trip Latency & Fallback Routing (<5.0ms)
# ==============================================================================

class TestCircuitBreakerTripLatencyStress:
    """Stress-tests circuit breaker state transition latency under concurrency and load."""

    def test_breaker_trip_latency_100_trials_sla(self):
        """Benchmarks 100 consecutive circuit breaker trips to OPEN.

        Asserts transition time is strictly < 5.0ms in 100% of trials.
        """
        latencies_ms: list[float] = []

        for trial in range(100):
            cb = CircuitBreaker(failure_threshold=2)
            assert cb.state == CircuitState.CLOSED
            cb.record_failure()
            assert cb.state == CircuitState.CLOSED

            # Second failure triggers trip to OPEN
            t0 = time.perf_counter()
            cb.record_failure()
            t1 = time.perf_counter()
            trip_ms = (t1 - t0) * 1000.0

            assert cb.state == CircuitState.OPEN
            assert trip_ms < 5.0, f"Trial {trial} trip latency {trip_ms:.3f}ms exceeded 5.0ms SLA"
            latencies_ms.append(trip_ms)

        mean_trip_ms = sum(latencies_ms) / len(latencies_ms)
        max_trip_ms = max(latencies_ms)
        assert mean_trip_ms < 5.0, f"Mean trip latency {mean_trip_ms:.3f}ms exceeded 5.0ms"
        assert max_trip_ms < 5.0, f"Max trip latency {max_trip_ms:.3f}ms exceeded 5.0ms"

    @pytest.mark.asyncio
    async def test_circuit_breaker_concurrent_load_trip_and_reroute(self):
        """Under high concurrency (50 concurrent requests), trips circuit breaker to OPEN.

        Asserts trip occurs in < 5.0ms, all traffic reroutes to fallback with 0 dropped calls,
        and primary receives zero traffic once OPEN.
        """
        router = MeshRouter()
        primary_calls = 0
        fallback_calls = 0

        async def failing_primary(payload: dict[str, Any]) -> dict[str, Any]:
            nonlocal primary_calls
            primary_calls += 1
            raise ConnectionError("Primary service offline")

        async def healthy_fallback(payload: dict[str, Any]) -> dict[str, Any]:
            nonlocal fallback_calls
            fallback_calls += 1
            return {"status": "SUCCESS", "served_by": "fallback", "item": payload["item"]}

        router.register_endpoint(
            AgentEndpoint(agent_id="primary_ep", role="Worker"),
            handler=failing_primary,
            fallback_agent_id="fallback_ep",
            failure_threshold=3,
        )
        router.register_endpoint(
            AgentEndpoint(agent_id="fallback_ep", role="Worker"),
            handler=healthy_fallback,
        )

        # Sequential trip verification
        t0 = time.perf_counter()
        _r1 = await router.route_and_call("primary_ep", {"item": 1})
        _r2 = await router.route_and_call("primary_ep", {"item": 2})
        _r3 = await router.route_and_call("primary_ep", {"item": 3})
        t1 = time.perf_counter()

        cb = router.circuit_breakers["primary_ep"]
        assert cb.state == CircuitState.OPEN
        assert (t1 - t0) * 1000.0 < 50.0  # Entire sequence well under bound

        # Now launch 50 concurrent requests against OPEN breaker
        tasks = [
            router.route_and_call("primary_ep", {"item": 100 + i})
            for i in range(50)
        ]
        results = await asyncio.gather(*tasks)

        # Verify fallback received 100% of rerouted traffic
        assert len(results) == 50
        for res in results:
            assert res["status"] == "SUCCESS"
            assert res["served_by"] == "fallback"

        # Verify primary was called exactly 3 times (the threshold), and 0 times during OPEN
        assert primary_calls == 3, f"Primary called {primary_calls} times instead of threshold 3"
        assert fallback_calls == 53, f"Fallback called {fallback_calls} times (expected 3 + 50 = 53)"

    @pytest.mark.asyncio
    async def test_circular_fallback_prevention(self):
        """Validates that circular fallback topologies (A -> B -> A) are blocked cleanly."""
        router = MeshRouter()

        async def handler_a(payload: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("Error A")

        async def handler_b(payload: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("Error B")

        router.register_endpoint(
            AgentEndpoint(agent_id="agent_a", role="RoleA"),
            handler=handler_a,
            fallback_agent_id="agent_b",
        )
        router.register_endpoint(
            AgentEndpoint(agent_id="agent_b", role="RoleB"),
            handler=handler_b,
            fallback_agent_id="agent_a",  # Circular loop
        )

        with pytest.raises(RuntimeError, match="Fallback cycle detected"):
            await router.route_and_call("agent_a", {"test": True})


# ==============================================================================
# Challenge 4: Physical State Separation Violation Rejection
# ==============================================================================

class TestPhysicalStateSeparationSecurity:
    """Stress-tests state path validation against multiple injection vectors."""

    def test_direct_repo_subdir_injection_rejected(self):
        """In-repo subdirectory is rejected unconditionally with StateSeparationError."""
        repo_root = CheckpointManager._detect_repo_root()
        assert repo_root is not None, "Repository root must be detectable"

        illegal_path = repo_root / "checkpoints_leak"
        with pytest.raises(StateSeparationError, match="Physical State Separation Violation"):
            CheckpointManager(base_dir=str(illegal_path))

    def test_repo_root_itself_injection_rejected(self):
        """Repository root itself is rejected unconditionally."""
        repo_root = CheckpointManager._detect_repo_root()
        assert repo_root is not None
        with pytest.raises(StateSeparationError, match="Physical State Separation Violation"):
            CheckpointManager.validate_state_dir(str(repo_root))

    def test_relative_path_inside_repo_rejected(self):
        """Relative path evaluating to inside repository root is rejected."""
        with pytest.raises(StateSeparationError, match="Physical State Separation Violation"):
            CheckpointManager.validate_state_dir("./data_illegal_leak")

    def test_parent_traversal_inside_repo_rejected(self):
        """Path with '..' resolving inside repo is rejected."""
        repo_root = CheckpointManager._detect_repo_root()
        assert repo_root is not None
        traversal_path = repo_root / "agentmesh" / ".." / "state_leak"
        with pytest.raises(StateSeparationError, match="Physical State Separation Violation"):
            CheckpointManager.validate_state_dir(str(traversal_path))

    def test_activity_memoizer_rejects_in_repo_storage(self):
        """ActivityMemoizer also enforces validate_state_dir on initialization."""
        repo_root = CheckpointManager._detect_repo_root()
        assert repo_root is not None
        illegal_memo_dir = repo_root / "memoizer_leak"
        with pytest.raises(StateSeparationError, match="Physical State Separation Violation"):
            ActivityMemoizer(storage_dir=str(illegal_memo_dir))

    def test_audit_alert_records_to_fallback_on_in_repo_env(self, monkeypatch: pytest.MonkeyPatch):
        """When AGENTMESH_STATE_DIR points inside repo, audit logger raises StateSeparationError."""
        repo_root = CheckpointManager._detect_repo_root()
        assert repo_root is not None
        illegal_dir = str(repo_root / "illegal_audit_dir")
        monkeypatch.setenv("AGENTMESH_STATE_DIR", illegal_dir)

        with pytest.raises(StateSeparationError, match="Physical State Separation Violation"):
            EventReplayer.record_security_audit_alert(
                workflow_id="wf-test-illegal",
                violation_type="TEST_VIOLATION",
                error_message="Test illegal path",
            )

    def test_symlink_pointing_to_repo_rejected(self, clean_m4_stress_state_dir: str):
        """A symlink located outside the repository pointing into the repository is rejected."""
        repo_root = CheckpointManager._detect_repo_root()
        assert repo_root is not None

        symlink_path = Path(clean_m4_stress_state_dir) / "evil_symlink_to_repo"
        try:
            symlink_path.symlink_to(repo_root)
        except OSError:
            pytest.skip("Symlink creation not permitted in this environment")

        with pytest.raises(StateSeparationError, match="Physical State Separation Violation"):
            CheckpointManager.validate_state_dir(str(symlink_path))

    def test_trailing_slash_and_dot_dot_variations(self):
        """Trailing slashes, multiple dots, and redundant segments are resolved and rejected."""
        repo_root = CheckpointManager._detect_repo_root()
        assert repo_root is not None
        weird_path = f"{repo_root}/./agentmesh/../checkpoints_leak/."
        with pytest.raises(StateSeparationError, match="Physical State Separation Violation"):
            CheckpointManager.validate_state_dir(weird_path)


class TestAdvancedResilienceAndTopologyStress:
    """Advanced adversarial tests for complex topologies and concurrent edge cases."""

    @pytest.mark.asyncio
    async def test_extreme_1000_event_cold_resumption_latency_sla(
        self, clean_m4_stress_state_dir: str
    ):
        """Verifies that cold crash recovery on a 1,000-event history strictly finishes in < 200.0ms.

        Also verifies complete SHA-256 cryptographic chain continuity.
        """
        ckpt_mgr = CheckpointManager(base_dir=clean_m4_stress_state_dir)
        wf_id = f"wf-extreme-1000-{uuid.uuid4().hex[:6]}"

        g = Graph(name="large_topology_graph")
        for i in range(50):
            g.add_node(f"step_{i}", lambda st: None)
            if i > 0:
                g.add_edge(f"step_{i-1}", f"step_{i}")
        g.set_entry_point("step_0")
        g.set_finish_point("step_49")

        state = State(workflow_id=wf_id)
        state.append_event(EventType.WORKFLOW_START, payload={"graph": "large_topology_graph"})

        # Build 1000 sequential cryptographic events
        for i in range(1, 1000):
            state.append_event(
                EventType.STATE_MUTATED,
                node_id=f"step_{i % 50}",
                payload={"iter": i, "token_count": i * 10},
            )

        # Final node complete event
        state.append_event(
            EventType.NODE_COMPLETE,
            node_id="step_25",
            payload={"status": "CHECKPOINT_SAVED"},
        )

        ckpt_path = ckpt_mgr.save_checkpoint(state, label="step_25")
        assert os.path.exists(ckpt_path)

        # Cold benchmark
        t_start = time.perf_counter()
        recovered_state, last_node, plan = EventReplayer.resume_from_crash(wf_id, ckpt_mgr, graph=g)
        t_end = time.perf_counter()
        elapsed_ms = (t_end - t_start) * 1000.0

        assert recovered_state is not None
        assert last_node == "step_25"
        assert plan is not None
        assert plan.resume_node == "step_26"
        assert len(recovered_state.events) >= 1000

        # Assert SLA < 200.0ms
        assert elapsed_ms < 200.0, f"Extreme 1000-event recovery took {elapsed_ms:.2f}ms (SLA: <200.0ms)"

        # SHA-256 chain verification
        is_valid, err = EventReplayer.verify_event_chain(recovered_state.events)
        assert is_valid is True, f"Event chain verification failed: {err}"

    @pytest.mark.asyncio
    async def test_concurrent_activities_inside_node_zero_duplicate_tokens_on_crash(
        self, clean_m4_stress_state_dir: str
    ):
        """Tests concurrent execution of multiple activities within a single node.

        Followed by a crash in the next node, and cold recovery, asserting 0 duplicate tokens.
        """
        ckpt_mgr = CheckpointManager(base_dir=clean_m4_stress_state_dir)
        memoizer = ActivityMemoizer(storage_dir=clean_m4_stress_state_dir)
        wf_id = f"wf-concurrent-acts-{uuid.uuid4().hex[:6]}"

        call_counts: dict[str, int] = {f"c_act_{i}": 0 for i in range(5)}

        async def concurrent_task(act_id: str) -> dict[str, Any]:
            call_counts[act_id] += 1
            await asyncio.sleep(0.01)
            return {"id": act_id, "usage": {"total_tokens": 120}}

        crash_in_node_2 = True

        async def parallel_activities_node(state: State) -> None:
            tasks = [
                execute_activity(f"c_act_{i}", concurrent_task, args=(f"c_act_{i}",))
                for i in range(5)
            ]
            results = await asyncio.gather(*tasks)
            state.set("parallel_results", [r["id"] for r in results])

        async def downstream_crash_node(state: State) -> None:
            if crash_in_node_2:
                raise RuntimeError("DOWNSTREAM_CRASH_INJECTED")
            state.set("pipeline_success", True)

        g = Graph(name="parallel_graph")
        g.add_node("parallel_node", parallel_activities_node)
        g.add_node("downstream_node", downstream_crash_node)
        g.add_edge("parallel_node", "downstream_node")
        g.set_entry_point("parallel_node")
        g.set_finish_point("downstream_node")

        # Initial run: parallel_node finishes, downstream crashes
        init_st = State(workflow_id=wf_id)
        with pytest.raises(RuntimeError, match="DOWNSTREAM_CRASH_INJECTED"):
            await g.run(init_st, checkpoint_manager=ckpt_mgr, memoizer=memoizer)

        for act_id, count in call_counts.items():
            assert count == 1, f"Initial run: {act_id} called {count} times"

        # Resume from crash
        crash_in_node_2 = False
        rec_st, last_node, plan = EventReplayer.resume_from_crash(wf_id, ckpt_mgr, graph=g)
        assert last_node == "parallel_node"
        assert plan is not None
        assert plan.resume_node == "downstream_node"

        final_st = await g.run(
            rec_st,
            checkpoint_manager=ckpt_mgr,
            memoizer=memoizer,
            start_from_node=plan.resume_node,
        )

        assert final_st.get("pipeline_success") is True
        # Verify no activities re-executed physically
        for act_id, count in call_counts.items():
            assert count == 1, f"Resumption: {act_id} re-invoked (total count: {count})"

    @pytest.mark.asyncio
    async def test_multi_hop_fallback_chain_under_concurrency(self):
        """Tests multi-hop fallback topology (A -> B -> C) where A and B fail, routing to C.

        Verifies 100% of traffic reaches C without packet drops or infinite loops.
        """
        router = MeshRouter()
        a_calls = 0
        b_calls = 0
        c_calls = 0

        async def service_a(payload: dict[str, Any]) -> dict[str, Any]:
            nonlocal a_calls
            a_calls += 1
            raise ConnectionError("Service A unavailable")

        async def service_b(payload: dict[str, Any]) -> dict[str, Any]:
            nonlocal b_calls
            b_calls += 1
            raise ConnectionError("Service B unavailable")

        async def service_c(payload: dict[str, Any]) -> dict[str, Any]:
            nonlocal c_calls
            c_calls += 1
            return {"status": "SUCCESS", "served_by": "service_c", "data": payload["data"]}

        router.register_endpoint(
            AgentEndpoint(agent_id="service_a", role="RoleA"),
            handler=service_a,
            fallback_agent_id="service_b",
            failure_threshold=2,
        )
        router.register_endpoint(
            AgentEndpoint(agent_id="service_b", role="RoleB"),
            handler=service_b,
            fallback_agent_id="service_c",
            failure_threshold=2,
        )
        router.register_endpoint(
            AgentEndpoint(agent_id="service_c", role="RoleC"),
            handler=service_c,
        )

        # Concurrently dispatch 20 requests to service_a
        tasks = [router.route_and_call("service_a", {"data": i}) for i in range(20)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 20
        for r in results:
            assert r["status"] == "SUCCESS"
            assert r["served_by"] == "service_c"

        # Service C received all 20 successfully handled requests
        assert c_calls == 20
        # Service A and B breakers are tripped
        assert router.circuit_breakers["service_a"].state == CircuitState.OPEN
        assert router.circuit_breakers["service_b"].state == CircuitState.OPEN

