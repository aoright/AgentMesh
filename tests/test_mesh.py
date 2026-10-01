"""Unit tests for AgentMesh router, circuit breaker, MCP client, A2A protocol, and service discovery."""

from __future__ import annotations

import json
import sys
import threading
import time
from typing import Any

import pytest

from agentmesh.mesh.a2a import (
    A2AMessage,
    A2AProtocolError,
    AgentCapability,
    AgentCard,
    InvalidStateTransitionError,
    NegotiationManager,
    NegotiationSession,
    NegotiationState,
    Performative,
)
from agentmesh.mesh.discovery import (
    ServiceRegistration,
    ServiceRegistry,
    ServiceStatus,
)
from agentmesh.mesh.mcp_client import (
    HttpSseTransport,
    InMemoryTransport,
    MCPClient,
    MCPToolCallRequest,
    MCPToolClient,
    StdioTransport,
)
from agentmesh.mesh.mcp_protocol import (
    JSONRPCError,
    JSONRPCErrorCode,
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPMethod,
    MCPProtocolError,
    parse_jsonrpc,
    serialize_jsonrpc,
)
from agentmesh.mesh.router import (
    AgentEndpoint,
    CircuitBreaker,
    CircuitState,
    MeshRouter,
    SmoothWeightedRoundRobin,
)

# --- Baseline Tests (Preserved) ---

@pytest.mark.asyncio
async def test_mesh_router_normal_and_fallback():
    router = MeshRouter()

    async def healthy_agent(payload: dict[str, Any]) -> dict[str, Any]:
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

    async def flaky_primary(payload: dict[str, Any]) -> dict[str, Any]:
        raise ConnectionResetError("Remote agent unreachable")

    async def safe_fallback(payload: dict[str, Any]) -> dict[str, Any]:
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


# --- Smooth Weighted Round-Robin (SWRR) Tests ---

def test_swrr_proportional_distribution():
    ep_a = AgentEndpoint(agent_id="agent_a", role="Worker", weight=50)
    ep_b = AgentEndpoint(agent_id="agent_b", role="Worker", weight=20)
    swrr = SmoothWeightedRoundRobin()

    selections = {"agent_a": 0, "agent_b": 0}
    for _ in range(700):
        chosen = swrr.select([ep_a, ep_b])
        assert chosen is not None
        selections[chosen.agent_id] += 1

    assert selections["agent_a"] == 500
    assert selections["agent_b"] == 200


def test_swrr_smoothness_anti_burst():
    ep_a = AgentEndpoint(agent_id="A", role="Compute", weight=5)
    ep_b = AgentEndpoint(agent_id="B", role="Compute", weight=1)
    swrr = SmoothWeightedRoundRobin()

    sequence = []
    for _ in range(6):
        chosen = swrr.select([ep_a, ep_b])
        assert chosen is not None
        sequence.append(chosen.agent_id)

    # NGINX SWRR produces ['A', 'A', 'A', 'B', 'A', 'A']
    assert sequence == ["A", "A", "A", "B", "A", "A"]


def test_swrr_inactive_and_zero_weight_exclusion():
    ep_active = AgentEndpoint(agent_id="active_agent", role="Worker", weight=100, is_active=True)
    ep_inactive = AgentEndpoint(agent_id="inactive_agent", role="Worker", weight=100, is_active=False)
    ep_zero_weight = AgentEndpoint(agent_id="zero_agent", role="Worker", weight=0, is_active=True)
    swrr = SmoothWeightedRoundRobin()

    for _ in range(20):
        chosen = swrr.select([ep_active, ep_inactive, ep_zero_weight])
        assert chosen is not None
        assert chosen.agent_id == "active_agent"

    # All inactive returns None
    none_chosen = swrr.select([ep_inactive, ep_zero_weight])
    assert none_chosen is None


def test_swrr_dynamic_weight_and_active_toggle():
    ep_a = AgentEndpoint(agent_id="worker_a", role="Worker", weight=10)
    ep_b = AgentEndpoint(agent_id="worker_b", role="Worker", weight=10)
    swrr = SmoothWeightedRoundRobin()

    # Both active
    first_choice = swrr.select([ep_a, ep_b])
    assert first_choice is not None

    # Toggle B inactive via active property alias
    ep_b.active = False
    for _ in range(10):
        chosen = swrr.select([ep_a, ep_b])
        assert chosen is not None
        assert chosen.agent_id == "worker_a"

    # Reactivate B
    ep_b.active = True
    b_chosen_count = 0
    for _ in range(20):
        chosen = swrr.select([ep_a, ep_b])
        if chosen is not None and chosen.agent_id == "worker_b":
            b_chosen_count += 1
    assert b_chosen_count > 0


def test_swrr_thread_safety():
    ep_a = AgentEndpoint(agent_id="A", role="Worker", weight=30)
    ep_b = AgentEndpoint(agent_id="B", role="Worker", weight=10)
    swrr = SmoothWeightedRoundRobin()

    results: list[str] = []
    lock = threading.Lock()

    def worker_job():
        local_picks = []
        for _ in range(100):
            sel = swrr.select([ep_a, ep_b])
            if sel is not None:
                local_picks.append(sel.agent_id)
        with lock:
            results.extend(local_picks)

    threads = [threading.Thread(target=worker_job) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 800
    a_count = results.count("A")
    b_count = results.count("B")
    assert a_count == 600
    assert b_count == 200


# --- Hardened Circuit Breaker Tests ---

def test_circuit_breaker_sub_5ms_transition_latency():
    cb = CircuitBreaker(failure_threshold=2)
    cb.record_failure()

    t_start = time.perf_counter()
    cb.record_failure()
    t_end = time.perf_counter()

    duration_ms = (t_end - t_start) * 1000
    assert cb.state == CircuitState.OPEN
    assert duration_ms < 5.0, f"Transition latency {duration_ms:.4f} ms exceeded 5.0 ms threshold"


def test_circuit_breaker_half_open_multi_probe_recovery():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.0, half_open_success_threshold=2)

    cb.record_failure()
    assert cb.state == CircuitState.OPEN

    # Evaluating can_execute after recovery timeout moves state to HALF_OPEN
    assert cb.can_execute() is True
    assert cb.state == CircuitState.HALF_OPEN

    # First probe success: remains HALF_OPEN
    cb.record_success()
    assert cb.state == CircuitState.HALF_OPEN

    # Second probe success: reaches threshold, transitions to CLOSED
    cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


def test_circuit_breaker_half_open_probe_failure_retrips_to_open():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.0, half_open_success_threshold=2)
    cb.record_failure()
    assert cb.state == CircuitState.OPEN

    assert cb.can_execute() is True
    assert cb.state == CircuitState.HALF_OPEN

    # Probe failure in HALF_OPEN immediately re-trips to OPEN
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.can_execute() is False or cb.recovery_timeout == 0.0


def test_circuit_breaker_reset_method():
    cb = CircuitBreaker(failure_threshold=1)
    cb.record_failure()
    assert cb.state == CircuitState.OPEN

    cb.reset()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0
    assert cb.consecutive_successes == 0
    assert cb.can_execute() is True


def test_circuit_breaker_thread_safe_concurrency():
    cb = CircuitBreaker(failure_threshold=100, half_open_success_threshold=10)

    def hammer():
        for _ in range(500):
            cb.record_failure()
            cb.record_success()
            _ = cb.can_execute()

    threads = [threading.Thread(target=hammer) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Must finish without deadlock or invalid state
    assert cb.state in {CircuitState.CLOSED, CircuitState.OPEN, CircuitState.HALF_OPEN}


# --- Mesh Router Fallback Cycle & Pool Routing Tests ---

@pytest.mark.asyncio
async def test_fallback_cycle_detection_mutual_recursion():
    router = MeshRouter()

    async def fail_a(payload: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("Service A failed")

    async def fail_b(payload: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("Service B failed")

    router.register_endpoint(
        AgentEndpoint(agent_id="agent_a", role="Primary"),
        handler=fail_a,
        fallback_agent_id="agent_b",
    )
    router.register_endpoint(
        AgentEndpoint(agent_id="agent_b", role="Secondary"),
        handler=fail_b,
        fallback_agent_id="agent_a",
    )

    with pytest.raises(RuntimeError) as exc_info:
        await router.route_and_call("agent_a", {})

    err = str(exc_info.value)
    assert "Fallback cycle detected" in err
    assert "agent_a -> agent_b -> agent_a" in err
    assert "No healthy fallback available" in err


@pytest.mark.asyncio
async def test_fallback_cycle_detection_three_hop():
    router = MeshRouter()

    async def fail_node(payload: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("Downstream fault")

    router.register_endpoint(
        AgentEndpoint(agent_id="node_1", role="R1"),
        handler=fail_node,
        fallback_agent_id="node_2",
    )
    router.register_endpoint(
        AgentEndpoint(agent_id="node_2", role="R2"),
        handler=fail_node,
        fallback_agent_id="node_3",
    )
    router.register_endpoint(
        AgentEndpoint(agent_id="node_3", role="R3"),
        handler=fail_node,
        fallback_agent_id="node_1",
    )

    with pytest.raises(RuntimeError) as exc_info:
        await router.route_and_call("node_1", {})

    err = str(exc_info.value)
    assert "Fallback cycle detected" in err
    assert "node_1 -> node_2 -> node_3 -> node_1" in err
    assert "No healthy fallback available" in err


@pytest.mark.asyncio
async def test_fallback_self_loop():
    router = MeshRouter()

    async def fail_self(payload: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("Self loop failure")

    router.register_endpoint(
        AgentEndpoint(agent_id="self_agent", role="Solo"),
        handler=fail_self,
        fallback_agent_id="self_agent",
    )

    with pytest.raises(RuntimeError) as exc_info:
        await router.route_and_call("self_agent", {})

    assert "Fallback cycle detected" in str(exc_info.value)
    assert "No healthy fallback available" in str(exc_info.value)


@pytest.mark.asyncio
async def test_router_pool_selection_and_call():
    router = MeshRouter()

    async def calc_fast(payload: dict[str, Any]) -> dict[str, Any]:
        return {"handler": "fast", "val": payload["val"] * 2}

    async def calc_deep(payload: dict[str, Any]) -> dict[str, Any]:
        return {"handler": "deep", "val": payload["val"] * 3}

    router.register_endpoint(
        AgentEndpoint(agent_id="calc_fast", role="Calculator", weight=50),
        handler=calc_fast,
    )
    router.register_endpoint(
        AgentEndpoint(agent_id="calc_deep", role="Calculator", weight=50),
        handler=calc_deep,
    )

    # Calling by role name should balance across pool
    res1 = await router.route_and_call("Calculator", {"val": 10})
    res2 = await router.route_and_call("Calculator", {"val": 10})
    handlers = {res1["handler"], res2["handler"]}
    assert handlers == {"fast", "deep"}


@pytest.mark.asyncio
async def test_router_service_name_pool_routing():
    router = MeshRouter()

    async def svc_h1(payload: dict[str, Any]) -> dict[str, Any]:
        return {"served_by": "h1"}

    async def svc_h2(payload: dict[str, Any]) -> dict[str, Any]:
        return {"served_by": "h2"}

    router.register_endpoint(
        AgentEndpoint(agent_id="inst_1", role="Worker", service_name="risk_svc", weight=100),
        handler=svc_h1,
    )
    router.register_endpoint(
        AgentEndpoint(agent_id="inst_2", role="Worker", service_name="risk_svc", weight=100),
        handler=svc_h2,
    )

    pool_eps = router.get_pool_endpoints("risk_svc")
    assert len(pool_eps) == 2

    res = await router.route_and_call("risk_svc", {})
    assert res["served_by"] in {"h1", "h2"}


# --- MCP 2.0 JSON-RPC Wire Protocol Tests ---

def test_jsonrpc_request_serialization():
    req = JSONRPCRequest(id=101, method=MCPMethod.TOOLS_LIST, params={"cursor": "cur_abc"})
    raw = serialize_jsonrpc(req)
    data = json.loads(raw)

    assert data["jsonrpc"] == "2.0"
    assert data["id"] == 101
    assert data["method"] == "tools/list"
    assert data["params"]["cursor"] == "cur_abc"


def test_jsonrpc_notification_serialization():
    notif = JSONRPCNotification(method="telemetry/event", params={"event": "startup"})
    raw = serialize_jsonrpc(notif)
    data = json.loads(raw)

    assert data["jsonrpc"] == "2.0"
    assert "id" not in data
    assert data["method"] == "telemetry/event"


def test_jsonrpc_response_validation():
    # Valid success response
    success = JSONRPCResponse(id="req-1", result={"items": [1, 2, 3]})
    assert success.is_success is True
    assert success.unwrap() == {"items": [1, 2, 3]}

    # Valid error response
    err = JSONRPCResponse(
        id="req-2",
        error=JSONRPCError.method_not_found("unknown/action"),
    )
    assert err.is_success is False
    with pytest.raises(MCPProtocolError) as excinfo:
        err.unwrap()
    assert "Method not found: 'unknown/action'" in str(excinfo.value)

    # Mutually exclusive result and error validation
    with pytest.raises(ValueError):
        JSONRPCResponse.model_validate(
            {"jsonrpc": "2.0", "id": 1, "result": "ok", "error": {"code": -32600, "message": "err"}}
        )

    # Missing both result and error validation
    with pytest.raises(ValueError):
        JSONRPCResponse.model_validate({"jsonrpc": "2.0", "id": 1})


def test_jsonrpc_standard_error_codes():
    assert JSONRPCErrorCode.PARSE_ERROR == -32700
    assert JSONRPCErrorCode.INVALID_REQUEST == -32600
    assert JSONRPCErrorCode.METHOD_NOT_FOUND == -32601
    assert JSONRPCErrorCode.INVALID_PARAMS == -32602
    assert JSONRPCErrorCode.INTERNAL_ERROR == -32603
    assert JSONRPCErrorCode.TOOL_EXECUTION_ERROR == -32001
    assert JSONRPCErrorCode.RESOURCE_NOT_FOUND == -32002


def test_parse_jsonrpc_discriminator():
    # Parse Request
    req = parse_jsonrpc(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}))
    assert isinstance(req, JSONRPCRequest)
    assert req.method == "ping"

    # Parse Notification
    notif = parse_jsonrpc(json.dumps({"jsonrpc": "2.0", "method": "notify"}))
    assert isinstance(notif, JSONRPCNotification)

    # Parse Response
    resp = parse_jsonrpc(json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}))
    assert isinstance(resp, JSONRPCResponse)
    assert resp.result == {"ok": True}

    # Parse Invalid JSON
    with pytest.raises(MCPProtocolError):
        parse_jsonrpc("{invalid_json")


@pytest.mark.asyncio
async def test_in_memory_transport_tools_and_resources():
    transport = InMemoryTransport(server_name="test_mesh_mcp")

    def square(x: int) -> int:
        return x * x

    client = MCPToolClient(server_name="test_mesh_mcp", transport=transport)
    client.register_tool(
        name="square",
        description="Squares a number",
        parameters={"x": "int"},
        handler=square,
    )
    client.register_resource(
        uri="mesh://version",
        name="Version",
        handler=lambda uri: "AgentMesh v2.0.0",
        description="Version string",
    )

    mcp = MCPClient(transport)

    # 1. tools/list
    tools_res = await mcp.list_tools()
    assert len(tools_res.tools) == 1
    assert tools_res.tools[0].name == "square"

    # 2. tools/call
    call_res = await mcp.call_tool("square", {"x": 9})
    assert call_res.isError is False
    assert call_res.structured_result == 81

    # 3. resources/list
    res_list = await mcp.list_resources()
    assert len(res_list.resources) == 1
    assert res_list.resources[0].uri == "mesh://version"

    # 4. resources/read
    res_read = await mcp.read_resource("mesh://version")
    assert len(res_read.contents) == 1
    assert res_read.contents[0].text == "AgentMesh v2.0.0"

    # 5. ping
    ping_ok = await mcp.ping()
    assert ping_ok is True


def test_sse_framing_and_parsing():
    stream = HttpSseTransport.encode_sse_event("endpoint", "/messages?id=sess_1", event_id="1")
    stream += HttpSseTransport.encode_sse_event(
        "message",
        json.dumps({"jsonrpc": "2.0", "id": 42, "result": "pong"}),
        event_id="2",
    )

    events = HttpSseTransport.parse_sse_events(stream)
    assert len(events) == 2
    assert events[0][0] == "endpoint"
    assert events[0][1] == "/messages?id=sess_1"
    assert events[1][0] == "message"

    msg = parse_jsonrpc(events[1][1])
    assert isinstance(msg, JSONRPCResponse)
    assert msg.id == 42


@pytest.mark.asyncio
async def test_stdio_transport_execution():
    # Spawns Python subprocess echoing JSON-RPC response
    echo_code = (
        "import sys, json; "
        "line = sys.stdin.readline(); "
        "req = json.loads(line); "
        "resp = {'jsonrpc': '2.0', 'id': req['id'], 'result': {'echo': req['method']}}; "
        "sys.stdout.write(json.dumps(resp) + '\\n'); "
        "sys.stdout.flush()"
    )
    transport = StdioTransport(command=sys.executable, args=["-c", echo_code], timeout=5.0)

    async with transport:
        req = JSONRPCRequest(id="sub-1", method="echo/test")
        resp = await transport.send_request(req)
        assert resp.id == "sub-1"
        assert resp.result == {"echo": "echo/test"}


# --- A2A Protocol & Negotiation Tests ---

def test_a2a_performatives():
    expected = {"REQUEST", "PROPOSE", "AGREE", "REFUSE", "INFORM", "FAILURE"}
    actual = {p.value for p in Performative}
    assert actual == expected


def test_agent_capability_and_card_semver():
    cap_v1 = AgentCapability(name="audit", version="1.2.0", description="Audit capability")
    cap_v2 = AgentCapability(name="tax", version="2.0.1", description="Tax computation")

    card = AgentCard(
        agent_id="financial_auditor",
        name="Financial Auditor",
        capabilities=[cap_v1, cap_v2],
    )

    # Capability matching with semver comparison
    assert card.has_capability("audit") is True
    assert card.has_capability("audit", min_version="1.0.0") is True
    assert card.has_capability("audit", min_version="1.2.0") is True
    assert card.has_capability("audit", min_version="1.3.0") is False
    assert card.has_capability("missing_cap") is False

    retrieved = card.get_capability("tax")
    assert retrieved is not None
    assert retrieved.version == "2.0.1"


def test_a2a_message_correlation_and_reply():
    msg = A2AMessage(
        sender_id="requester_agent",
        recipient_id="worker_agent",
        performative=Performative.REQUEST,
        payload={"task": "process_batch"},
        traceparent="00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
    )

    reply = msg.create_reply(
        performative=Performative.PROPOSE,
        payload={"cost": 100, "eta_sec": 5},
    )

    assert reply.sender_id == "worker_agent"
    assert reply.recipient_id == "requester_agent"
    assert reply.performative == Performative.PROPOSE
    assert reply.conversation_id == msg.conversation_id
    assert reply.reply_to_id == msg.message_id
    assert reply.traceparent == msg.traceparent


def test_negotiation_session_happy_path():
    session = NegotiationSession(
        conversation_id="conv_101",
        requester_id="client_agent",
        provider_id="provider_agent",
    )
    assert session.current_state == NegotiationState.INIT

    # 1. REQUEST
    req = A2AMessage(
        sender_id="client_agent",
        recipient_id="provider_agent",
        performative=Performative.REQUEST,
        conversation_id="conv_101",
    )
    state = session.process_message(req)
    assert state == NegotiationState.REQUESTED

    # 2. PROPOSE
    prop = req.create_reply(Performative.PROPOSE, {"terms": "standard"})
    state = session.process_message(prop)
    assert state == NegotiationState.PROPOSED

    # 3. AGREE
    agree = prop.create_reply(Performative.AGREE, {"agreed": True})
    state = session.process_message(agree)
    assert state == NegotiationState.AGREED

    # 4. mark_executing
    session.mark_executing()
    assert session.current_state == NegotiationState.EXECUTING

    # 5. INFORM (terminal)
    inform = agree.create_reply(Performative.INFORM, {"result": "task_completed"})
    state = session.process_message(inform)
    assert state == NegotiationState.INFORMED
    assert session.is_terminal() is True
    assert len(session.history) == 4


def test_negotiation_session_direct_agreement():
    session = NegotiationSession(
        conversation_id="conv_fast",
        requester_id="a1",
        provider_id="a2",
    )
    req = A2AMessage(sender_id="a1", recipient_id="a2", performative=Performative.REQUEST, conversation_id="conv_fast")
    session.process_message(req)

    # Provider accepts directly
    agree = req.create_reply(Performative.AGREE, {})
    session.process_message(agree)
    assert session.current_state == NegotiationState.AGREED

    inform = agree.create_reply(Performative.INFORM, {"data": "done"})
    session.process_message(inform)
    assert session.current_state == NegotiationState.INFORMED


def test_negotiation_session_refusal():
    session = NegotiationSession(
        conversation_id="conv_refuse",
        requester_id="a1",
        provider_id="a2",
    )
    req = A2AMessage(sender_id="a1", recipient_id="a2", performative=Performative.REQUEST, conversation_id="conv_refuse")
    session.process_message(req)

    refuse = req.create_reply(Performative.REFUSE, {"reason": "capacity_exceeded"})
    session.process_message(refuse)
    assert session.current_state == NegotiationState.REFUSED
    assert session.is_terminal() is True


def test_negotiation_session_invalid_transition():
    session = NegotiationSession(
        conversation_id="conv_invalid",
        requester_id="a1",
        provider_id="a2",
    )
    # Sending INFORM in INIT state is illegal
    bad_msg = A2AMessage(
        sender_id="a1",
        recipient_id="a2",
        performative=Performative.INFORM,
        conversation_id="conv_invalid",
    )
    with pytest.raises(InvalidStateTransitionError):
        session.process_message(bad_msg)


def test_negotiation_session_terminal_protection():
    session = NegotiationSession(
        conversation_id="conv_term",
        requester_id="a1",
        provider_id="a2",
    )
    req = A2AMessage(sender_id="a1", recipient_id="a2", performative=Performative.REQUEST, conversation_id="conv_term")
    session.process_message(req)
    refuse = req.create_reply(Performative.REFUSE, {})
    session.process_message(refuse)

    # Further message in terminal state must raise
    subsequent = refuse.create_reply(Performative.REQUEST, {})
    with pytest.raises(InvalidStateTransitionError):
        session.process_message(subsequent)


def test_negotiation_manager_multi_session():
    manager = NegotiationManager()

    # Create message starting session automatically
    msg1 = A2AMessage(
        sender_id="client_1",
        recipient_id="agent_1",
        performative=Performative.REQUEST,
        conversation_id="c1",
    )
    sess1 = manager.process_message(msg1)
    assert sess1.conversation_id == "c1"
    assert sess1.current_state == NegotiationState.REQUESTED

    # Unknown session for non-REQUEST raises A2AProtocolError
    msg_unknown = A2AMessage(
        sender_id="client_2",
        recipient_id="agent_2",
        performative=Performative.PROPOSE,
        conversation_id="c_nonexistent",
    )
    with pytest.raises(A2AProtocolError):
        manager.process_message(msg_unknown)


def test_negotiation_manager_eviction():
    manager = NegotiationManager()
    manager.create_session("a1", "a2", conversation_id="c_exp", timeout_sec=0.01)
    time.sleep(0.02)

    evicted = manager.evict_expired_sessions()
    assert "c_exp" in evicted
    assert manager.get_session("c_exp") is None


# --- Dynamic Service Discovery Tests ---

def test_service_registration_model():
    card = AgentCard(agent_id="test_ag", name="Tester", version="1.0.0")
    reg = ServiceRegistration(agent_id="test_ag", card=card, weight=80, ttl_sec=10.0)

    assert reg.status == ServiceStatus.HEALTHY
    assert reg.is_expired() is False

    endpoint = reg.to_agent_endpoint()
    assert endpoint.agent_id == "test_ag"
    assert endpoint.role == "Tester"
    assert endpoint.weight == 80
    assert endpoint.is_active is True


def test_service_registry_registration_and_lookup():
    registry = ServiceRegistry()
    card = AgentCard(
        agent_id="ag_audit",
        name="Auditor",
        capabilities=[
            AgentCapability(name="audit_logs", version="1.0.0", tags=["finance", "security"]),
        ],
    )
    reg = registry.register(card=card, endpoint_url="grpc://audit:50051", weight=70)
    assert reg.agent_id == "ag_audit"

    # Direct lookup
    found = registry.get_agent("ag_audit")
    assert found is not None
    assert found.endpoint_url == "grpc://audit:50051"

    # Capability lookup
    cap_matches = registry.find_by_capability("audit_logs")
    assert len(cap_matches) == 1
    assert cap_matches[0].agent_id == "ag_audit"

    # Tag lookup
    tag_matches = registry.find_by_tag("finance")
    assert len(tag_matches) == 1
    assert tag_matches[0].agent_id == "ag_audit"


def test_service_registry_deregistration():
    registry = ServiceRegistry()
    card = AgentCard(
        agent_id="ag_temp",
        name="TempAgent",
        capabilities=[AgentCapability(name="temp_task", tags=["temp"])],
    )
    registry.register(card=card)
    assert len(registry.find_by_capability("temp_task")) == 1

    ok = registry.deregister("ag_temp")
    assert ok is True
    assert registry.get_agent("ag_temp") is None
    assert len(registry.find_by_capability("temp_task")) == 0
    assert len(registry.find_by_tag("temp")) == 0


def test_service_registry_heartbeat_renewal():
    registry = ServiceRegistry()
    card = AgentCard(agent_id="ag_heartbeat", name="HeartbeatAgent")
    registry.register(card=card, ttl_sec=5.0)

    # Renew heartbeat
    now = time.time_ns()
    ok = registry.heartbeat("ag_heartbeat", timestamp_ns=now)
    assert ok is True

    reg = registry.get_agent("ag_heartbeat")
    assert reg is not None
    assert reg.last_heartbeat_ns == now
    assert reg.status == ServiceStatus.HEALTHY

    # Heartbeat on unknown agent returns False
    assert registry.heartbeat("unknown_ag") is False


def test_service_registry_ttl_eviction():
    registry = ServiceRegistry()
    card = AgentCard(agent_id="ag_stale", name="StaleAgent")
    # 0.01s TTL
    registry.register(card=card, ttl_sec=0.01)
    time.sleep(0.02)

    evicted = registry.evict_expired()
    assert "ag_stale" in evicted
    assert registry.get_agent("ag_stale") is None


def test_service_registry_pubsub_listeners():
    registry = ServiceRegistry()
    events: list[tuple[str, str]] = []

    def on_event(event_type: str, reg: ServiceRegistration):
        events.append((event_type, reg.agent_id))

    registry.subscribe(on_event)

    card = AgentCard(agent_id="ag_listener", name="ListenerAgent")
    registry.register(card=card, ttl_sec=0.01)
    registry.heartbeat("ag_listener")
    time.sleep(0.02)
    registry.evict_expired()

    event_types = [e[0] for e in events]
    assert "REGISTERED" in event_types
    assert "HEARTBEAT" in event_types
    assert "EXPIRED" in event_types


@pytest.mark.asyncio
async def test_service_registry_sync_to_router():
    registry = ServiceRegistry()
    card = AgentCard(
        agent_id="ag_discovered",
        name="DiscoveredWorker",
        capabilities=[AgentCapability(name="nlp_extract")],
    )
    registry.register(card=card, weight=90)

    router = MeshRouter(discovery_registry=registry)

    async def nlp_handler(payload: dict[str, Any]) -> dict[str, Any]:
        return {"entities": ["ORG", "LOC"]}

    registry.sync_to_router(router, handler_map={"ag_discovered": nlp_handler})

    assert "ag_discovered" in router.endpoints
    assert router.endpoints["ag_discovered"].weight == 90

    res = await router.route_and_call("ag_discovered", {"text": "hello"})
    assert res["entities"] == ["ORG", "LOC"]


@pytest.mark.asyncio
async def test_mesh_router_route_by_capability():
    registry = ServiceRegistry()
    card = AgentCard(
        agent_id="ag_worker_cap",
        name="CapWorker",
        capabilities=[AgentCapability(name="financial_rating", version="1.0.0")],
    )
    registry.register(card=card, weight=100)

    router = MeshRouter(discovery_registry=registry)

    async def rating_handler(payload: dict[str, Any]) -> dict[str, Any]:
        return {"score": 95}

    router.register_endpoint(
        AgentEndpoint(agent_id="ag_worker_cap", role="CapWorker"),
        handler=rating_handler,
    )

    res = await router.route_by_capability("financial_rating", {"client_id": "c123"})
    assert res["score"] == 95

    with pytest.raises(ValueError) as exc:
        await router.route_by_capability("non_existent_capability", {})
    assert "No healthy agents found" in str(exc.value)
