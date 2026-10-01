"""Shared fixtures and configuration for AgentMesh E2E test suite."""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import pytest

from agentmesh.engine.checkpoint import CheckpointManager
from agentmesh.engine.state import State
from agentmesh.mesh.mcp_client import MCPToolClient
from agentmesh.mesh.router import MeshRouter
from agentmesh.security.sandbox import CapabilityPolicy, SecuritySandbox
from agentmesh.telemetry.tracer import AgentTracer


@pytest.fixture
def isolated_state_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Provides an isolated directory for checkpoint storage outside the repository."""
    state_dir = tmp_path / "agentmesh_state_isolated"
    state_dir.mkdir(parents=True, exist_ok=True)
    original_env = os.environ.get("AGENTMESH_STATE_DIR")
    os.environ["AGENTMESH_STATE_DIR"] = str(state_dir)

    yield state_dir

    if original_env is not None:
        os.environ["AGENTMESH_STATE_DIR"] = original_env
    else:
        os.environ.pop("AGENTMESH_STATE_DIR", None)


@pytest.fixture
def checkpoint_manager(isolated_state_dir: Path) -> CheckpointManager:
    """Provides a CheckpointManager backed by an isolated temporary directory."""
    return CheckpointManager(base_dir=str(isolated_state_dir))


@pytest.fixture
def sample_state() -> State:
    """Provides a fresh workflow state instance."""
    state = State(workflow_id="wf-e2e-test-001")
    state.set("initial_budget", 100000)
    state.set("processed_count", 0)
    return state


@pytest.fixture
def sandbox_fixture(tmp_path: Path):
    """Provides a configured SecuritySandbox with isolated read and write directories."""
    read_dir = tmp_path / "sandbox_safe_read"
    write_dir = tmp_path / "sandbox_safe_write"
    read_dir.mkdir(parents=True, exist_ok=True)
    write_dir.mkdir(parents=True, exist_ok=True)

    # Populate a sample read-only file
    sample_file = read_dir / "ledger_source.txt"
    sample_file.write_text("TX_001,1500,CNY\nTX_002,999999,CNY\n", encoding="utf-8")

    policy = CapabilityPolicy(
        policy_name="e2e_test_policy",
        allow_network=False,
        allowed_read_paths=[str(read_dir)],
        allowed_write_paths=[str(write_dir)],
        allowed_commands={"echo", "cat", "grep"},
    )
    sandbox = SecuritySandbox(policy=policy)
    return {
        "sandbox": sandbox,
        "policy": policy,
        "read_dir": read_dir,
        "write_dir": write_dir,
        "sample_file": sample_file,
    }


@pytest.fixture
def sample_mcp_client() -> MCPToolClient:
    """Provides an MCPToolClient populated with standard test tools."""
    client = MCPToolClient(server_name="e2e_mcp_server")

    def calculate_tax(amount: float, rate: float = 0.05) -> float:
        return round(amount * rate, 2)

    def query_account(account_id: str) -> dict:
        accounts = {
            "ACC-101": {"owner": "Alice", "balance": 50000.0, "status": "ACTIVE"},
            "ACC-102": {"owner": "Bob", "balance": 1250000.0, "status": "HIGH_NET_WORTH"},
            "ACC-103": {"owner": "Charlie", "balance": 350.0, "status": "RESTRICTED"},
        }
        return accounts.get(account_id, {"error": "ACCOUNT_NOT_FOUND"})

    client.register_tool(
        name="calculate_tax",
        description="Calculates tax for a specified transaction amount",
        parameters={"amount": "float", "rate": "float"},
        handler=calculate_tax,
    )
    client.register_tool(
        name="query_account",
        description="Retrieves account status from the database",
        parameters={"account_id": "string"},
        handler=query_account,
    )
    return client


@pytest.fixture
def sample_mesh_router() -> MeshRouter:
    """Provides a MeshRouter instance ready for endpoint registration."""
    return MeshRouter()


@pytest.fixture
def sample_tracer() -> AgentTracer:
    """Provides an AgentTracer instance."""
    return AgentTracer()
