"""Unit tests for AgentMesh router, circuit breaker, and MCP client."""

import pytest
from agentmesh.mesh.router import MeshRouter, AgentEndpoint, CircuitState
from agentmesh.mesh.mcp_client import MCPToolClient, MCPToolCallRequest


@pytest.mark.asyncio
async def test_mesh_router_normal_and_fallback():
    router = MeshRouter()

    async def healthy_agent(payload: dict):
        return {"status": "success", "echo": payload.get("msg")}

    router.register_endpoint(
        AgentEndpoint(agent_id="healthy_agent", role="Worker"),
        handler=healthy_agent,
    )

    res = await router.route_and_call("healthy_agent", {"msg": "hello"})
    assert res["status"] == "success"
    assert res["echo"] == "hello"


@pytest.mark.asyncio
async def test_circuit_breaker_tripping_and_fallback():
    router = MeshRouter()

    async def flaky_primary(payload: dict):
        raise ConnectionResetError("Remote agent unreachable")

    async def safe_fallback(payload: dict):
        return {"status": "fallback_success"}

    router.register_endpoint(
        AgentEndpoint(agent_id="flaky_primary", role="Primary"),
        handler=flaky_primary,
        fallback_agent_id="safe_fallback",
        failure_threshold=2,
    )
    router.register_endpoint(
        AgentEndpoint(agent_id="safe_fallback", role="Backup"),
        handler=safe_fallback,
    )

    # First call fails on primary, triggers fallback
    res1 = await router.route_and_call("flaky_primary", {})
    assert res1["status"] == "fallback_success"

    # Second call trips breaker to OPEN
    res2 = await router.route_and_call("flaky_primary", {})
    assert res2["status"] == "fallback_success"
    assert router.circuit_breakers["flaky_primary"].state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_mcp_client_tool_registration_and_invoke():
    client = MCPToolClient(server_name="test_server")

    def add_numbers(a: int, b: int) -> int:
        return a + b

    client.register_tool(
        name="add",
        description="Add two numbers",
        parameters={"a": "int", "b": "int"},
        handler=add_numbers,
    )

    tools = client.list_tools()
    assert len(tools) == 1
    assert tools[0].name == "add"

    req = MCPToolCallRequest(tool_name="add", arguments={"a": 40, "b": 2})
    resp = await client.invoke_tool(req)
    assert resp.success is True
    assert resp.result == 42
