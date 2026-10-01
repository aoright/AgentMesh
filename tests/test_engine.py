"""Unit tests for AgentMesh deterministic workflow engine and checkpointing."""

import pytest
import os
import shutil
from agentmesh.engine.state import State, EventType
from agentmesh.engine.graph import Graph
from agentmesh.engine.checkpoint import CheckpointManager
from agentmesh.engine.replay import EventReplayer

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
