"""Unit tests for AgentMesh deterministic workflow engine, checkpointing, and durable execution."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import pytest

from agentmesh.engine.activity import (
    ActivityExecutionContext,
    ActivityRecord,
    ActivityStatus,
    current_activity_context,
    execute_activity,
)
from agentmesh.engine.checkpoint import CheckpointManager, StateSeparationError
from agentmesh.engine.graph import Graph
from agentmesh.engine.memoizer import ActivityMemoizer, IdempotencyConflictError
from agentmesh.engine.replay import (
    CryptographicIntegrityError,
    EventReplayer,
)
from agentmesh.engine.state import (
    Event,
    EventType,
    State,
    canonical_json_dumps,
)

TEST_STATE_DIR = "/tmp/agentmesh/test_checkpoints"


@pytest.fixture(autouse=True)
def cleanup():
    if os.path.exists(TEST_STATE_DIR):
        shutil.rmtree(TEST_STATE_DIR, ignore_errors=True)
    yield
    if os.path.exists(TEST_STATE_DIR):
        shutil.rmtree(TEST_STATE_DIR, ignore_errors=True)


@pytest.mark.asyncio
async def test_graph_deterministic_execution():
    g = Graph("test_graph")

    async def node_a(state: State):
        state.set("a", 10)

    async def node_b(state: State):
        state.set("b", state.get("a") * 2)

    g.add_node("node_a", node_a)
    g.add_node("node_b", node_b)
    g.add_edge("node_a", "node_b")
    g.set_entry_point("node_a")
    g.set_finish_point("node_b")

    state = State(workflow_id="wf-test-01")
    final_state = await g.run(state)

    assert final_state.get("a") == 10
    assert final_state.get("b") == 20
    assert final_state.version == 2
    assert len(final_state.events) > 0


@pytest.mark.asyncio
async def test_checkpoint_and_event_hash_chain():
    ckpt_mgr = CheckpointManager(base_dir=TEST_STATE_DIR)
    state = State(workflow_id="wf-test-chain")

    state.set("init", True)
    state.append_event(EventType.NODE_COMPLETE, node_id="node_init", payload={"ok": True})
    ckpt_path = ckpt_mgr.save_checkpoint(state, label="init")

    assert os.path.exists(ckpt_path)

    loaded = ckpt_mgr.load_latest_checkpoint("wf-test-chain")
    assert loaded is not None
    assert loaded.get("init") is True

    # Verify cryptographic integrity
    is_valid, err = EventReplayer.verify_event_chain(loaded.events)
    assert is_valid is True
    assert err is None


def test_canonical_json_rfc8785_determinism():
    """Verify key sorting and whitespace stripping in canonical JSON."""
    dict_a = {"b": 2, "a": 1, "nested": {"z": 26, "y": 25}}
    dict_b = {"nested": {"y": 25, "z": 26}, "a": 1, "b": 2}
    assert canonical_json_dumps(dict_a) == canonical_json_dumps(dict_b)
    assert canonical_json_dumps(dict_a) == '{"a":1,"b":2,"nested":{"y":25,"z":26}}'


def test_event_chain_valid_sequence():
    """Verify that normally appended events pass cryptographic validation."""
    state = State(workflow_id="wf-test-valid")
    state.append_event(EventType.WORKFLOW_START, payload={"user": "alice"})
    state.append_event(EventType.NODE_START, node_id="node_1")
    state.append_event(EventType.NODE_COMPLETE, node_id="node_1", payload={"out": 42})

    is_valid, err = EventReplayer.verify_event_chain(state.events)
    assert is_valid is True
    assert err is None


def test_tamper_payload_raises_cryptographic_error():
    """Verify that tampering with an event payload raises CryptographicIntegrityError."""
    state = State(workflow_id="wf-test-payload-tamper")
    state.append_event(EventType.WORKFLOW_START)
    state.append_event(EventType.NODE_COMPLETE, node_id="node_1", payload={"amount": 100})
    state.append_event(EventType.WORKFLOW_COMPLETE)

    # Tamper payload
    state.events[1].payload["amount"] = 999999

    is_valid, err = EventReplayer.verify_event_chain(state.events)
    assert is_valid is False
    assert "hash mismatch" in str(err).lower()

    with pytest.raises(CryptographicIntegrityError) as exc_info:
        EventReplayer.verify_and_enforce_chain(state.events, workflow_id="wf-test-payload-tamper")
    assert "hash mismatch" in str(exc_info.value).lower()


def test_tamper_sequence_break_raises_cryptographic_error():
    """Verify that altering sequence numbers raises CryptographicIntegrityError."""
    state = State(workflow_id="wf-test-seq-tamper")
    state.append_event(EventType.WORKFLOW_START)
    state.append_event(EventType.NODE_START, node_id="node_1")

    # Tamper sequence number
    state.events[1].sequence_num = 10

    with pytest.raises(CryptographicIntegrityError) as exc_info:
        EventReplayer.verify_and_enforce_chain(state.events, workflow_id="wf-test-seq-tamper")
    assert "sequence break" in str(exc_info.value).lower()


def test_tamper_event_deletion_detected():
    """Verify that deleting an intermediate event breaks both sequence and hash chain."""
    state = State(workflow_id="wf-test-del-tamper")
    state.append_event(EventType.WORKFLOW_START)
    state.append_event(EventType.NODE_START, node_id="node_1")
    state.append_event(EventType.NODE_COMPLETE, node_id="node_1")

    # Delete middle event
    del state.events[1]

    with pytest.raises(CryptographicIntegrityError) as exc_info:
        EventReplayer.verify_and_enforce_chain(state.events, workflow_id="wf-test-del-tamper")
    assert "sequence break" in str(exc_info.value).lower() or "hash mismatch" in str(exc_info.value).lower()


def test_resume_from_crash_blocks_tampered_disk_checkpoint(tmp_path: Path):
    """Verify that resume_from_crash halts immediately if on-disk checkpoint is tampered."""
    ckpt_mgr = CheckpointManager(base_dir=str(tmp_path))
    wf_id = "wf-tamper-disk"
    state = State(workflow_id=wf_id)
    state.append_event(EventType.WORKFLOW_START)
    state.append_event(EventType.NODE_COMPLETE, node_id="step_1", payload={"verified": True})
    ckpt_path = ckpt_mgr.save_checkpoint(state, label="step_1")

    # Manually tamper with checkpoint JSON file on disk
    with open(ckpt_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw["events"][1]["payload"]["verified"] = False
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(raw, f)

    # Resume must immediately raise CryptographicIntegrityError
    with pytest.raises(CryptographicIntegrityError):
        EventReplayer.resume_from_crash(wf_id, ckpt_mgr)


@pytest.mark.asyncio
async def test_resumption_latency_under_200ms():
    """Verify that cold checkpoint loading, event verification, and resumption planning take < 200ms."""
    ckpt_mgr = CheckpointManager(base_dir=TEST_STATE_DIR)
    wf_id = "wf-benchmark-latency"
    state = State(workflow_id=wf_id)

    # Populate state with 50 sequential events simulating a realistic workflow run
    for i in range(50):
        state.set(f"key_{i}", f"val_{i}")
        state.append_event(
            EventType.NODE_COMPLETE,
            node_id=f"node_{i}",
            payload={"metric": i * 10, "data": "benchmark_payload"},
        )
    ckpt_mgr.save_checkpoint(state, label="node_49")

    # Define DAG
    g = Graph("benchmark_graph")
    for i in range(51):
        g.add_node(f"node_{i}", lambda s: None)
        if i > 0:
            g.add_edge(f"node_{i-1}", f"node_{i}")
    g.set_entry_point("node_0")
    g.set_finish_point("node_50")

    # Measure cold resumption time
    start_time = time.perf_counter()
    recovered_state, _last_node, plan = EventReplayer.resume_from_crash(wf_id, ckpt_mgr, graph=g)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    assert recovered_state is not None
    assert plan is not None
    assert plan.resume_node == "node_50"
    assert elapsed_ms < 200.0, f"Resumption took {elapsed_ms:.2f}ms, exceeding 200ms limit"


@pytest.mark.asyncio
async def test_zero_duplicate_token_billing_on_crash_resumption():
    """Verify that restarting an interrupted workflow incurs 0 duplicate tokens for completed activities."""
    ckpt_mgr = CheckpointManager(base_dir=TEST_STATE_DIR)
    memoizer = ActivityMemoizer(storage_dir=TEST_STATE_DIR)
    wf_id = "wf-token-dedup-test"

    call_counters = {"activity_1": 0, "activity_2": 0}
    token_accounting = {"initial_tokens": 0, "resume_tokens": 0}

    def mock_llm_activity_1(prompt: str) -> dict:
        call_counters["activity_1"] += 1
        tokens = 350
        token_accounting["initial_tokens"] += tokens
        return {"output": "summary_generated", "tokens_spent": tokens}

    def mock_llm_activity_2(query: str) -> dict:
        call_counters["activity_2"] += 1
        tokens = 200
        token_accounting["resume_tokens"] += tokens
        return {"output": "report_ready", "tokens_spent": tokens}

    # Step 1: Pre-populate memoizer simulating Activity 1 completed prior to crash
    key_1 = memoizer.compute_key(wf_id, "node_ingest", "llm_summary", {"prompt": "audit data"})
    res_1 = mock_llm_activity_1("audit data")
    record_1 = ActivityRecord(
        idempotency_key=key_1,
        activity_name="llm_summary",
        node_id="node_ingest",
        workflow_id=wf_id,
        status=ActivityStatus.COMPLETED,
        result=res_1,
        prompt_tokens=250,
        completion_tokens=100,
        total_tokens=350,
        timestamp_ns=time.time_ns(),
    )
    memoizer.put(record_1)

    # State reflects crash happened after node_ingest
    state = State(workflow_id=wf_id)
    state.set("ingest_done", True)
    state.append_event(
        EventType.NODE_COMPLETE,
        node_id="node_ingest",
        payload={"idempotency_key": key_1, "result": res_1},
    )
    ckpt_mgr.save_checkpoint(state, label="node_ingest")

    # Step 2: Build graph with node_ingest -> node_report
    g = Graph("token_test_graph")

    async def run_ingest(s: State):
        k = memoizer.compute_key(wf_id, "node_ingest", "llm_summary", {"prompt": "audit data"})
        cached = memoizer.get(k)
        if cached:
            s.set("summary", cached.result)
            return
        res = mock_llm_activity_1("audit data")
        s.set("summary", res)

    async def run_report(s: State):
        res = mock_llm_activity_2("finalize")
        s.set("report", res)

    g.add_node("node_ingest", run_ingest)
    g.add_node("node_report", run_report)
    g.add_edge("node_ingest", "node_report")
    g.set_entry_point("node_ingest")
    g.set_finish_point("node_report")

    # Step 3: Resume execution autonomously
    recovered_state, _, plan = EventReplayer.resume_from_crash(wf_id, ckpt_mgr, graph=g)
    assert plan is not None
    assert plan.resume_node == "node_report"

    final_state = await g.run(recovered_state, checkpoint_manager=ckpt_mgr, memoizer=memoizer)

    # Assertions for 0% duplicate tokens
    assert call_counters["activity_1"] == 1, "Activity 1 was re-executed during resumption!"
    assert call_counters["activity_2"] == 1, "Activity 2 was not executed!"
    assert final_state.get("report") is not None

    duplicate_tokens = 0
    initial_tokens = token_accounting["initial_tokens"]
    duplicate_token_rate = (duplicate_tokens / initial_tokens) * 100.0
    assert duplicate_token_rate == 0.0


def test_state_separation_rejects_repository_internal_paths():
    """Verify CheckpointManager raises StateSeparationError if base_dir is inside source tree."""
    repo_root = CheckpointManager._detect_repo_root()
    assert repo_root is not None

    illegal_paths = [
        repo_root / "checkpoints",
        repo_root / "data" / "state",
        repo_root / "agentmesh" / "state",
    ]

    for illegal_path in illegal_paths:
        with pytest.raises(StateSeparationError) as exc_info:
            CheckpointManager(base_dir=str(illegal_path))
        assert "Physical State Separation Violation" in str(exc_info.value)

        with pytest.raises(StateSeparationError) as exc_info_m:
            ActivityMemoizer(storage_dir=str(illegal_path))
        assert "Physical State Separation Violation" in str(exc_info_m.value)

    # Valid external directories must succeed
    valid_mgr = CheckpointManager(base_dir="/tmp/agentmesh/valid_isolated_state")
    assert valid_mgr.base_dir == Path("/tmp/agentmesh/valid_isolated_state").resolve()


@pytest.mark.asyncio
async def test_autonomous_branching_resumption_without_manual_override():
    """Verify ResumptionPlanner evaluates conditional edge predicates using restored state."""
    ckpt_mgr = CheckpointManager(base_dir=TEST_STATE_DIR)
    wf_id = "wf-branch-resumption"
    g = Graph("branch_graph")

    execution_trail = []

    async def node_entry(s: State):
        execution_trail.append("entry")
        s.set("score", 85)

    async def node_high_priority(s: State):
        execution_trail.append("high_priority")
        s.set("verdict", "APPROVED_FAST_TRACK")

    async def node_standard(s: State):
        execution_trail.append("standard")
        s.set("verdict", "STANDARD_REVIEW")

    g.add_node("entry", node_entry)
    g.add_node("high_priority", node_high_priority)
    g.add_node("standard", node_standard)

    g.add_edge("entry", "high_priority", condition=lambda s: s.get("score", 0) >= 80)
    g.add_edge("entry", "standard", condition=lambda s: s.get("score", 0) < 80)

    g.set_entry_point("entry")
    g.set_finish_point("high_priority")
    g.set_finish_point("standard")

    # Step 1: Simulate entry node completed and state saved
    state = State(workflow_id=wf_id)
    state.set("score", 85)
    state.append_event(EventType.NODE_COMPLETE, node_id="entry", payload={"score": 85})
    ckpt_mgr.save_checkpoint(state, label="entry")

    # Step 2: Resume autonomously without passing start_from_node
    recovered_state, _, plan = EventReplayer.resume_from_crash(wf_id, ckpt_mgr, graph=g)

    assert plan is not None
    assert plan.resume_node == "high_priority", "Planner should have selected high_priority based on score=85"
    assert plan.is_resuming_interrupted_node is False

    final_state = await g.run(recovered_state, checkpoint_manager=ckpt_mgr)
    assert final_state.get("verdict") == "APPROVED_FAST_TRACK"
    assert "high_priority" in execution_trail
    assert "entry" not in execution_trail, "Entry node should have been skipped!"


@pytest.mark.asyncio
async def test_execute_activity_and_memoization_lifecycle():
    """Verify execute_activity with context, retries, and token accounting."""
    memoizer = ActivityMemoizer(storage_dir=TEST_STATE_DIR)
    state = State(workflow_id="wf-act-lifecycle")

    ctx = ActivityExecutionContext(
        workflow_id="wf-act-lifecycle",
        node_id="test_node",
        memoizer=memoizer,
        state=state,
    )
    token = current_activity_context.set(ctx)

    try:
        invocations = [0]

        async def dummy_llm(prompt: str) -> dict:
            invocations[0] += 1
            return {"text": f"echo: {prompt}", "usage": {"prompt_tokens": 10, "completion_tokens": 20}}

        # 1. Custom idempotency key memoization within same context
        res1 = await execute_activity(
            "dummy_llm",
            dummy_llm,
            args=("hello",),
            idempotency_key="custom_llm_key",
        )
        assert res1["text"] == "echo: hello"
        assert invocations[0] == 1

        # Second call with same idempotency key hits memoizer
        res2 = await execute_activity(
            "dummy_llm",
            dummy_llm,
            args=("hello",),
            idempotency_key="custom_llm_key",
        )
        assert res2["text"] == "echo: hello"
        assert invocations[0] == 1  # 0 additional invocations!

        # 2. Replay simulation (fresh context for same node upon recovery)
        ctx_replay = ActivityExecutionContext(
            workflow_id="wf-act-lifecycle",
            node_id="test_node_auto",
            memoizer=memoizer,
            state=state,
        )
        token_replay = current_activity_context.set(ctx_replay)
        try:
            # First execution under test_node_auto
            res3 = await execute_activity("dummy_llm", dummy_llm, args=("world",))
            assert res3["text"] == "echo: world"
            assert invocations[0] == 2

            # Replayed node execution with fresh context
            ctx_replayed_node = ActivityExecutionContext(
                workflow_id="wf-act-lifecycle",
                node_id="test_node_auto",
                memoizer=memoizer,
                state=state,
            )
            token_replayed_node = current_activity_context.set(ctx_replayed_node)
            try:
                res4 = await execute_activity("dummy_llm", dummy_llm, args=("world",))
                assert res4["text"] == "echo: world"
                assert invocations[0] == 2  # 0 additional invocations on replay!
            finally:
                current_activity_context.reset(token_replayed_node)
        finally:
            current_activity_context.reset(token_replay)

        # Check recorded events
        event_types = [e.event_type for e in state.events]
        assert EventType.ACTIVITY_STARTED in event_types
        assert EventType.ACTIVITY_COMPLETED in event_types
        assert EventType.ACTIVITY_MEMOIZED_HIT in event_types
    finally:
        current_activity_context.reset(token)


def test_memoizer_idempotency_conflict():
    """Verify IdempotencyConflictError is raised when an explicit key is reused with different args."""
    memoizer = ActivityMemoizer(storage_dir=TEST_STATE_DIR)
    key = "explicit_custom_key_123"

    memoizer.compute_key(
        workflow_id="wf-conflict",
        node_id="node_1",
        activity_name="query",
        args=("arg1",),
        custom_key=key,
    )

    with pytest.raises(IdempotencyConflictError) as exc_info:
        memoizer.compute_key(
            workflow_id="wf-conflict",
            node_id="node_1",
            activity_name="query",
            args=("arg2_different",),
            custom_key=key,
        )
    assert "conflict" in str(exc_info.value).lower()


def test_checkpoint_pruning():
    """Verify that keep_last_n prunes older checkpoints."""
    ckpt_mgr = CheckpointManager(base_dir=TEST_STATE_DIR, keep_last_n=3)
    wf_id = "wf-pruning"
    state = State(workflow_id=wf_id)

    for i in range(6):
        state.set("step", i)
        state.append_event(EventType.NODE_COMPLETE, node_id=f"node_{i}")
        ckpt_mgr.save_checkpoint(state, label=f"step_{i}")

    checkpoints = ckpt_mgr.list_checkpoints(wf_id)
    assert len(checkpoints) == 3
    # Check that latest checkpoint is preserved
    latest = ckpt_mgr.load_latest_checkpoint(wf_id)
    assert latest is not None
    assert latest.get("step") == 5


def test_checkpoint_tail_truncation_detection(tmp_path: Path):
    """Verify that deleting the tail event from on-disk checkpoint is blocked 100%."""
    ckpt_mgr = CheckpointManager(base_dir=str(tmp_path))
    wf_id = "wf-tail-truncation-test"
    state = State(workflow_id=wf_id)
    state.append_event(EventType.NODE_COMPLETE, node_id="node_1", payload={"step": 1})
    state.append_event(EventType.NODE_COMPLETE, node_id="node_2", payload={"step": 2})
    state.append_event(EventType.NODE_COMPLETE, node_id="node_3", payload={"step": 3})
    ckpt_path = ckpt_mgr.save_checkpoint(state, label="node_3")

    # Manually truncate tail event from checkpoint JSON on disk
    with open(ckpt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["events_count"] == 3
    data["events"].pop()  # remove event 3

    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    # Must raise CryptographicIntegrityError
    with pytest.raises(CryptographicIntegrityError) as exc_info:
        EventReplayer.resume_from_crash(wf_id, ckpt_mgr)
    assert (
        "tail truncation" in str(exc_info.value).lower()
        or "count mismatch" in str(exc_info.value).lower()
    )


def test_checkpoint_events_count_tamper_detection(tmp_path: Path):
    """Verify that modifying events_count in checkpoint JSON is blocked."""
    ckpt_mgr = CheckpointManager(base_dir=str(tmp_path))
    wf_id = "wf-count-tamper-test"
    state = State(workflow_id=wf_id)
    state.append_event(EventType.NODE_COMPLETE, node_id="node_1", payload={"step": 1})
    ckpt_path = ckpt_mgr.save_checkpoint(state, label="node_1")

    with open(ckpt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["events_count"] = 99

    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    with pytest.raises(CryptographicIntegrityError) as exc_info:
        EventReplayer.resume_from_crash(wf_id, ckpt_mgr)
    assert "count mismatch" in str(exc_info.value).lower()


def test_verify_and_enforce_chain_with_state_tail_truncation():
    """Verify in-memory state tail truncation raises CryptographicIntegrityError."""
    state = State(workflow_id="wf-inmemory-tail")
    state.append_event(EventType.NODE_COMPLETE, node_id="n1", payload={})
    state.append_event(EventType.NODE_COMPLETE, node_id="n2", payload={})

    # Delete tail event in memory without updating state.last_hash
    state.events.pop()

    with pytest.raises(CryptographicIntegrityError) as exc_info:
        EventReplayer.verify_and_enforce_chain(state)
    assert (
        "tail truncation" in str(exc_info.value).lower()
        or "mismatch" in str(exc_info.value).lower()
    )


def test_record_security_audit_alert_rejects_repository_internal_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Verify record_security_audit_alert raises StateSeparationError and creates no repo files."""
    repo_root = CheckpointManager._detect_repo_root()
    assert repo_root is not None

    illegal_dir = repo_root / "test_leak_audit_dir"
    assert not illegal_dir.exists()

    monkeypatch.setenv("AGENTMESH_STATE_DIR", str(illegal_dir))

    try:
        with pytest.raises(StateSeparationError) as exc_info:
            EventReplayer.record_security_audit_alert(
                workflow_id="wf-test-leak",
                violation_type="TEST_IN_REPO_AUDIT",
                error_message="Simulated security audit alert",
            )
        assert "Physical State Separation Violation" in str(exc_info.value)
        assert not illegal_dir.exists(), "Illegal audit dir inside repository must not be created"
    finally:
        if illegal_dir.exists():
            shutil.rmtree(illegal_dir, ignore_errors=True)


def test_record_security_audit_alert_writes_to_valid_external_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Verify record_security_audit_alert succeeds and writes audit log when path is external."""
    external_dir = tmp_path / "valid_audit_dir"
    monkeypatch.setenv("AGENTMESH_STATE_DIR", str(external_dir))

    EventReplayer.record_security_audit_alert(
        workflow_id="wf-external-audit",
        violation_type="TEST_VALID_EXTERNAL",
        error_message="External path test",
    )

    audit_file = external_dir / "security_audit.log"
    assert audit_file.exists()
    with open(audit_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "TEST_VALID_EXTERNAL" in content
    assert "wf-external-audit" in content


def test_event_hash_workflow_id_and_run_id_binding():
    """Verify that tampering with workflow_id or run_id causes cryptographic verification failure."""
    state = State(workflow_id="wf-original", run_id="run-original")
    evt = state.append_event(
        EventType.NODE_COMPLETE, node_id="node_auth", payload={"token": "secret"}
    )
    orig_hash = evt.state_hash

    # Tamper with workflow_id
    evt.workflow_id = "wf-unauthorized-replay"
    assert evt.calculate_hash(evt.prev_hash) != orig_hash

    is_valid, err = EventReplayer.verify_event_chain([evt])
    assert not is_valid
    assert err is not None
    assert (
        "cryptographic hash mismatch" in err.lower()
        or "workflow identity mismatch" in err.lower()
    )

    # Reset and tamper with run_id
    evt.workflow_id = "wf-original"
    evt.run_id = "run-attacker"
    assert evt.calculate_hash(evt.prev_hash) != orig_hash

    is_valid, err = EventReplayer.verify_event_chain([evt])
    assert not is_valid
    assert err is not None
    assert "cryptographic hash mismatch" in err.lower()


def test_event_hash_delimiter_collision_prevention():
    """Verify that delimiter characters in node_id vs activity_id cannot produce identical hashes."""
    e1 = Event(
        event_id="e1",
        sequence_num=1,
        workflow_id="wf",
        run_id="r",
        event_type=EventType.NODE_COMPLETE,
        timestamp_ns=1000,
        node_id="part1|part2",
        activity_id="part3",
        payload={},
        prev_hash="",
    )
    e2 = Event(
        event_id="e1",
        sequence_num=1,
        workflow_id="wf",
        run_id="r",
        event_type=EventType.NODE_COMPLETE,
        timestamp_ns=1000,
        node_id="part1",
        activity_id="part2|part3",
        payload={},
        prev_hash="",
    )
    assert e1.calculate_hash("") != e2.calculate_hash("")


def test_memoizer_key_delimiter_collision_prevention():
    """Verify that colon delimiter shifts across workflow_id, node_id, and activity cannot collide."""
    memoizer = ActivityMemoizer(storage_dir="/tmp/test_memoizer_probe")
    k1 = memoizer.compute_key("app:user", "task", "fetch", args=("x",))
    k2 = memoizer.compute_key("app", "user:task", "fetch", args=("x",))
    assert k1 != k2, "Memoizer keys must not collide on colons in workflow_id or node_id"

