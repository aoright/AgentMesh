"""Empirical stress test suite challenging Milestone 2 (MCP 2.0 & A2A Gateway).

Adversarially challenges:
1. MCP 2.0 JSON-RPC wire protocol under adversarial and malformed inputs.
2. A2A state machine under invalid transitions and terminal state immutability.
3. Semver matching with edge cases (prereleases, wildcards, malformed versions).
4. Concurrent ServiceRegistry registrations and TTL lease eviction under race conditions.
5. Circuit breaker sub-5ms tripping and cycle-protected fallback topologies.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import pytest
from pydantic import ValidationError

from agentmesh.mesh.a2a import (
    A2AMessage,
    AgentCapability,
    AgentCard,
    InvalidStateTransitionError,
    NegotiationManager,
    NegotiationSession,
    NegotiationState,
    Performative,
)
from agentmesh.mesh.discovery import (
    ServiceRegistry,
)
from agentmesh.mesh.mcp_client import (
    HttpSseTransport,
    InMemoryTransport,
    MCPToolCallRequest,
    MCPToolClient,
)
from agentmesh.mesh.mcp_protocol import (
    JSONRPCErrorCode,
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPProtocolError,
    MCPTool,
    MCPToolInputSchema,
    parse_jsonrpc,
)
from agentmesh.mesh.router import (
    AgentEndpoint,
    CircuitBreaker,
    CircuitState,
    MeshRouter,
    SmoothWeightedRoundRobin,
)

logger = logging.getLogger("agentmesh.stress.m2")


# ============================================================================
# 1. MCP 2.0 JSON-RPC Wire Protocol Stress and Adversarial Inputs
# ============================================================================


@pytest.mark.parametrize(
    "malformed_payload",
    [
        '{"jsonrpc": "2.0", "id": 1, "method"',  # Truncated JSON
        '{"jsonrpc": "2.0", "id": 1, "method": "\\uZZZZ"}',  # Invalid unicode escape
        "Hello World Plaintext",  # Non-JSON string
        "<xml><jsonrpc>2.0</jsonrpc></xml>",  # XML payload
        "",  # Empty string
        "   \n\t  ",  # Whitespace only
        "true",  # JSON boolean primitive
        "12345",  # JSON integer primitive
        '"quoted string"',  # JSON string primitive
        "[1, 2, 3]",  # JSON list instead of dict
    ],
)
def test_mcp_wire_malformed_syntax_rejection(malformed_payload: str):
    """Empirically verify that malformed JSON payloads are rejected with MCPProtocolError."""
    with pytest.raises(MCPProtocolError) as exc_info:
        parse_jsonrpc(malformed_payload)
    assert exc_info.value.code in (
        JSONRPCErrorCode.PARSE_ERROR,
        JSONRPCErrorCode.INVALID_REQUEST,
    )


@pytest.mark.parametrize(
    "invalid_dict",
    [
        {"id": 1, "method": "ping"},  # Missing jsonrpc field
        {"jsonrpc": "1.0", "id": 1, "method": "ping"},  # Unsupported jsonrpc version 1.0
        {"jsonrpc": "3.0", "id": 1, "method": "ping"},  # Future jsonrpc version 3.0
        {"jsonrpc": 2.0, "id": 1, "method": "ping"},  # Numeric float jsonrpc instead of "2.0"
        {"jsonrpc": "", "id": 1, "method": "ping"},  # Empty jsonrpc string
        {},  # Completely empty dictionary
        {"jsonrpc": "2.0"},  # Cannot identify message type
        {"jsonrpc": "2.0", "id": "req-1"},  # Has ID but neither method nor result/error
    ],
)
def test_mcp_wire_invalid_envelope_structures(invalid_dict: dict[str, Any]):
    """Empirically verify that invalid envelope dicts are rejected as INVALID_REQUEST."""
    with pytest.raises(MCPProtocolError) as exc_info:
        parse_jsonrpc(invalid_dict)
    assert exc_info.value.code == JSONRPCErrorCode.INVALID_REQUEST


def test_mcp_wire_response_result_error_mutual_exclusivity():
    """Verify that JSONRPCResponse enforces strict mutual exclusivity between result and error."""
    # Both result and error present and non-null -> must raise ValidationError
    conflicting = {
        "jsonrpc": "2.0",
        "id": 100,
        "result": {"status": "ok"},
        "error": {"code": -32600, "message": "fail"},
    }
    with pytest.raises(ValidationError):
        parse_jsonrpc(conflicting)

    # Neither result nor error present -> cannot identify message type or validation error
    empty_resp = {"jsonrpc": "2.0", "id": 100}
    with pytest.raises(MCPProtocolError):
        parse_jsonrpc(empty_resp)

    # Valid response with explicit null result (JSON-RPC 2.0 void return)
    null_result = {"jsonrpc": "2.0", "id": 101, "result": None}
    parsed = parse_jsonrpc(null_result)
    assert isinstance(parsed, JSONRPCResponse)
    assert parsed.id == 101
    assert parsed.result is None
    assert parsed.error is None
    assert parsed.is_success is True


def test_mcp_wire_id_types_and_discrimination():
    """Verify polymorphic ID types (string, integer) and request vs notification discrimination."""
    # Integer ID
    req_int = parse_jsonrpc({"jsonrpc": "2.0", "id": 42, "method": "test"})
    assert isinstance(req_int, JSONRPCRequest)
    assert req_int.id == 42

    # Negative integer ID
    req_neg = parse_jsonrpc({"jsonrpc": "2.0", "id": -99, "method": "test"})
    assert isinstance(req_neg, JSONRPCRequest)
    assert req_neg.id == -99

    # String UUID ID
    req_str = parse_jsonrpc({"jsonrpc": "2.0", "id": "req-abc-123", "method": "test"})
    assert isinstance(req_str, JSONRPCRequest)
    assert req_str.id == "req-abc-123"

    # Notification has no ID
    notif = parse_jsonrpc({"jsonrpc": "2.0", "method": "telemetry/heartbeat", "params": {"rate": 10}})
    assert isinstance(notif, JSONRPCNotification)
    assert notif.method == "telemetry/heartbeat"

    # Invalid ID types (list, dict) should fail schema validation
    with pytest.raises(ValidationError):
        parse_jsonrpc({"jsonrpc": "2.0", "id": [1, 2], "method": "test"})

    with pytest.raises(ValidationError):
        parse_jsonrpc({"jsonrpc": "2.0", "id": {"nested": "id"}, "method": "test"})


@pytest.mark.asyncio
async def test_mcp_tool_invocation_adversarial_exceptions():
    """Empirically challenge tool call handlers that raise varied exceptions."""
    transport = InMemoryTransport(server_name="adversarial_mcp")

    def crashing_tool(message: str) -> None:
        if message == "runtime":
            raise RuntimeError("Underlying subsystem crashed")
        if message == "value":
            raise ValueError("Invalid calculation argument")
        if message == "custom":
            raise PermissionError("Access to resource forbidden")
        raise RuntimeError("Generic failure")

    tool_def = MCPTool(
        name="crashing_tool",
        description="Fails on demand",
        inputSchema=MCPToolInputSchema(
            type="object",
            properties={"message": {"type": "string"}},
            required=["message"],
        ),
    )
    transport.register_tool_handler(tool_def, crashing_tool)
    client = MCPToolClient(server_name="adversarial_mcp", transport=transport)

    for error_type in ["runtime", "value", "custom", "unknown"]:
        req = MCPToolCallRequest(tool_name="crashing_tool", arguments={"message": error_type})
        resp = await client.invoke_tool(req)
        assert resp.success is False
        assert resp.result is None
        assert resp.error is not None
        assert len(resp.error) > 0


@pytest.mark.asyncio
async def test_mcp_tool_nonexistent_and_schema_mismatch():
    """Verify appropriate error responses for nonexistent tools and missing parameters."""
    transport = InMemoryTransport(server_name="strict_mcp")
    client = MCPToolClient(server_name="strict_mcp", transport=transport)

    # Calling nonexistent tool
    req_missing = MCPToolCallRequest(tool_name="phantom_tool", arguments={})
    resp_missing = await client.invoke_tool(req_missing)
    assert resp_missing.success is False
    assert resp_missing.error is not None
    assert "not found" in resp_missing.error.lower()

    # Calling tool with invalid wire method request via raw transport
    bad_req = JSONRPCRequest(id=1, method="tools/call", params={})  # Missing 'name'
    raw_resp = await transport.send_request(bad_req)
    assert raw_resp.error is not None
    assert raw_resp.error.code == JSONRPCErrorCode.INVALID_PARAMS


def test_http_sse_framing_parser_adversarial_streams():
    """Stress-test SSE parser with multiline data, comments, trailing whitespace, and unicode."""
    raw_stream = (
        ": comment to ignore\n"
        "event: message\n"
        "data: first line of payload\n"
        "data: second line of payload\n\n"
        ": another comment\n"
        "event: tool_progress\n"
        "id: 42\n"
        "data: {\"step\": 1, \"status\": \"processing\"}\n\n"
        "\n\n\n"  # Consecutive empty lines
        "data: standalone un-named event data\n\n"
        "event: finished\n"
        "data: final status ok\n\n"
    )
    events = HttpSseTransport.parse_sse_events(raw_stream)
    assert len(events) == 4

    assert events[0] == ("message", "first line of payload\nsecond line of payload")
    assert events[1] == ("tool_progress", '{"step": 1, "status": "processing"}')
    assert events[2] == ("message", "standalone un-named event data")
    assert events[3] == ("finished", "final status ok")

    # Empty and whitespace-only streams
    assert HttpSseTransport.parse_sse_events("") == []
    assert HttpSseTransport.parse_sse_events("   \n\n\t\n\n  ") == []


# ============================================================================
# 2. A2A State Machine: Invalid Transitions and Terminal Immutability
# ============================================================================


def test_a2a_all_invalid_state_transitions_matrix():
    """Exhaustively verify that illegal performatives raise InvalidStateTransitionError."""
    # Matrix of valid transitions from NegotiationSession.VALID_TRANSITIONS:
    # INIT: {REQUEST}
    # REQUESTED: {PROPOSE, AGREE, REFUSE, FAILURE}
    # PROPOSED: {AGREE, REFUSE, PROPOSE, FAILURE}
    # AGREED: {INFORM, FAILURE}
    # EXECUTING: {INFORM, FAILURE}
    # REFUSED: set()
    # INFORMED: set()
    # FAILED: set()

    all_performatives = list(Performative)

    # 1. Test from INIT
    for p in all_performatives:
        if p != Performative.REQUEST:
            session = NegotiationSession("c_init", "a1", "a2")
            msg = A2AMessage(sender_id="a1", recipient_id="a2", performative=p, conversation_id="c_init")
            with pytest.raises(InvalidStateTransitionError):
                session.process_message(msg)

    # 2. Test from REQUESTED
    for p in all_performatives:
        if p in {Performative.REQUEST, Performative.INFORM}:
            session = NegotiationSession("c_req", "a1", "a2")
            req = A2AMessage(sender_id="a1", recipient_id="a2", performative=Performative.REQUEST, conversation_id="c_req")
            session.process_message(req)
            bad_msg = req.create_reply(p, {})
            with pytest.raises(InvalidStateTransitionError):
                session.process_message(bad_msg)

    # 3. Test from PROPOSED
    for p in all_performatives:
        if p in {Performative.REQUEST, Performative.INFORM}:
            session = NegotiationSession("c_prop", "a1", "a2")
            req = A2AMessage(sender_id="a1", recipient_id="a2", performative=Performative.REQUEST, conversation_id="c_prop")
            session.process_message(req)
            prop = req.create_reply(Performative.PROPOSE, {})
            session.process_message(prop)
            bad_msg = prop.create_reply(p, {})
            with pytest.raises(InvalidStateTransitionError):
                session.process_message(bad_msg)

    # 4. Test from AGREED
    for p in all_performatives:
        if p in {Performative.REQUEST, Performative.PROPOSE, Performative.AGREE, Performative.REFUSE}:
            session = NegotiationSession("c_agr", "a1", "a2")
            req = A2AMessage(sender_id="a1", recipient_id="a2", performative=Performative.REQUEST, conversation_id="c_agr")
            session.process_message(req)
            agree = req.create_reply(Performative.AGREE, {})
            session.process_message(agree)
            bad_msg = agree.create_reply(p, {})
            with pytest.raises(InvalidStateTransitionError):
                session.process_message(bad_msg)

    # 5. Test from EXECUTING
    for p in all_performatives:
        if p in {Performative.REQUEST, Performative.PROPOSE, Performative.AGREE, Performative.REFUSE}:
            session = NegotiationSession("c_exec", "a1", "a2")
            req = A2AMessage(sender_id="a1", recipient_id="a2", performative=Performative.REQUEST, conversation_id="c_exec")
            session.process_message(req)
            agree = req.create_reply(Performative.AGREE, {})
            session.process_message(agree)
            session.mark_executing()
            bad_msg = agree.create_reply(p, {})
            with pytest.raises(InvalidStateTransitionError):
                session.process_message(bad_msg)


@pytest.mark.parametrize(
    "terminal_terminal_sequence",
    [
        # Path reaching REFUSED
        [Performative.REQUEST, Performative.REFUSE],
        # Path reaching INFORMED (via direct agree)
        [Performative.REQUEST, Performative.AGREE, Performative.INFORM],
        # Path reaching INFORMED (via proposal and execution)
        [Performative.REQUEST, Performative.PROPOSE, Performative.AGREE, "mark_executing", Performative.INFORM],
        # Path reaching FAILED (via failure performative)
        [Performative.REQUEST, Performative.FAILURE],
        # Path reaching FAILED from AGREED
        [Performative.REQUEST, Performative.AGREE, Performative.FAILURE],
    ],
)
def test_a2a_terminal_state_strict_immutability(terminal_terminal_sequence: list[Any]):
    """Empirically assert that once terminal (REFUSED, INFORMED, FAILED), NO further messages are accepted."""
    session = NegotiationSession("c_term_immut", "a1", "a2")
    last_msg = None

    for step in terminal_terminal_sequence:
        if step == "mark_executing":
            session.mark_executing()
        elif last_msg is None:
            last_msg = A2AMessage(sender_id="a1", recipient_id="a2", performative=step, conversation_id="c_term_immut")
            session.process_message(last_msg)
        else:
            last_msg = last_msg.create_reply(step, {})
            session.process_message(last_msg)

    assert session.is_terminal() is True
    terminal_state = session.current_state

    # Attempt EVERY possible performative once terminal
    for p in Performative:
        post_term_msg = A2AMessage(
            sender_id="a1",
            recipient_id="a2",
            performative=p,
            conversation_id="c_term_immut",
        )
        with pytest.raises(InvalidStateTransitionError) as exc_info:
            session.process_message(post_term_msg)
        assert f"terminal state '{terminal_state}'" in str(exc_info.value)

    # Calling mark_executing() on terminal session must also raise
    with pytest.raises(InvalidStateTransitionError):
        session.mark_executing()


def test_a2a_session_timeout_enforcement():
    """Verify that an expired session transitions to FAILED and rejects further messages."""
    session = NegotiationSession("c_timeout", "a1", "a2", timeout_sec=0.01)
    req = A2AMessage(sender_id="a1", recipient_id="a2", performative=Performative.REQUEST, conversation_id="c_timeout")
    session.process_message(req)

    # Sleep past timeout
    time.sleep(0.02)
    assert session.is_expired() is True

    # Processing another message on timed-out session must raise and set state to FAILED
    agree = req.create_reply(Performative.AGREE, {})
    with pytest.raises(InvalidStateTransitionError) as exc_info:
        session.process_message(agree)
    assert "timed out" in str(exc_info.value)
    assert session.current_state == NegotiationState.FAILED
    assert session.is_terminal() is True


def test_a2a_concurrent_negotiation_sessions_isolation():
    """Empirically stress-test concurrent multi-agent negotiation sessions without cross-talk."""
    manager = NegotiationManager()
    num_threads = 40
    errors: list[Exception] = []

    def run_session(thread_idx: int):
        cid = f"conv_thread_{thread_idx}"
        try:
            req = A2AMessage(
                sender_id=f"client_{thread_idx}",
                recipient_id=f"worker_{thread_idx}",
                performative=Performative.REQUEST,
                conversation_id=cid,
                payload={"task_id": thread_idx},
            )
            s = manager.process_message(req)
            assert s.current_state == NegotiationState.REQUESTED

            # Provider proposes counter terms
            prop = req.create_reply(Performative.PROPOSE, {"cost": 50})
            s = manager.process_message(prop)
            assert s.current_state == NegotiationState.PROPOSED

            # Requester agrees
            agree = prop.create_reply(Performative.AGREE, {"agreed": True})
            s = manager.process_message(agree)
            assert s.current_state == NegotiationState.AGREED

            # Provider marks executing and informs
            s.mark_executing()
            inform = agree.create_reply(Performative.INFORM, {"status": "done"})
            s = manager.process_message(inform)
            assert s.current_state == NegotiationState.INFORMED
            assert len(s.history) == 4
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=run_session, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors in concurrent A2A sessions: {errors}"


# ============================================================================
# 3. Semver Matching Edge Cases and Boundary Logic
# ============================================================================


def test_semver_parsing_edge_cases():
    """Empirically challenge AgentCard semver parsing and capability matching against edge cases."""
    card = AgentCard(
        agent_id="test_agent",
        name="SemverTester",
        capabilities=[
            AgentCapability(name="cap_exact", version="1.0.0"),
            AgentCapability(name="cap_minor", version="1.2.3"),
            AgentCapability(name="cap_short", version="1.2"),
            AgentCapability(name="cap_single", version="2"),
            AgentCapability(name="cap_prerelease", version="1.0.0-alpha.1"),
            AgentCapability(name="cap_buildmeta", version="1.0.0+20261001"),
            AgentCapability(name="cap_pre_complex", version="2.0.0-rc.3+build.42"),
        ],
    )

    # 1. Exact and standard comparisons
    assert card.has_capability("cap_exact", "1.0.0") is True
    assert card.has_capability("cap_exact", "0.9.9") is True
    assert card.has_capability("cap_exact", "1.0.1") is False

    # 2. Minor and patch comparisons
    assert card.has_capability("cap_minor", "1.2.0") is True
    assert card.has_capability("cap_minor", "1.2.3") is True
    assert card.has_capability("cap_minor", "1.2.4") is False
    assert card.has_capability("cap_minor", "1.3.0") is False

    # 3. None requirement matches anything
    assert card.has_capability("cap_exact", None) is True
    assert card.has_capability("cap_short", None) is True

    # 4. Prerelease tags and build metadata stripping
    # Clean version of "1.0.0-alpha.1" is parsed as (1, 0, 0)
    assert card.has_capability("cap_prerelease", "1.0.0") is True
    assert card.has_capability("cap_buildmeta", "1.0.0") is True
    assert card.has_capability("cap_pre_complex", "2.0.0") is True
    assert card.has_capability("cap_pre_complex", "2.0.1") is False

    # 5. Wildcards and malformed min_version query resilience
    # "*" parses to (0,), matching any normal positive version
    assert card.has_capability("cap_exact", "*") is True
    assert card.has_capability("cap_exact", "1.*") is True
    assert card.has_capability("cap_exact", "") is True

    # 6. Tuple length mismatch nuance:
    # "cap_short" is version "1.2" -> (1, 2)
    # Python tuple comparison (1, 2) < (1, 2, 0). Thus min_version "1.2.0" returns False
    assert card.has_capability("cap_short", "1.2") is True
    assert card.has_capability("cap_short", "1.1.9") is True
    assert card.has_capability("cap_short", "1.2.0") is False  # Shorter tuple is smaller


# ============================================================================
# 4. Concurrent ServiceRegistry & TTL Lease Eviction under Race Conditions
# ============================================================================


def test_concurrent_serviceregistry_heavy_stress():
    """Stress-test ServiceRegistry under massive concurrent registration, query, heartbeat, and eviction."""
    registry = ServiceRegistry()
    num_agents = 60
    operations_per_thread = 40
    exceptions: list[Exception] = []

    # Register initial batch of agents
    for i in range(num_agents):
        card = AgentCard(
            agent_id=f"agent_{i:03d}",
            name=f"Worker_{i}",
            capabilities=[
                AgentCapability(
                    name=f"cap_{(i % 5)}",
                    version=f"1.{(i % 3)}.0",
                    tags=[f"tag_{i % 4}", "common"],
                )
            ],
        )
        registry.register(card=card, ttl_sec=0.2)

    def worker_register(thread_id: int):
        for j in range(operations_per_thread):
            aid = f"dyn_{thread_id}_{j}"
            card = AgentCard(
                agent_id=aid,
                name=f"Dynamic_{thread_id}",
                capabilities=[
                    AgentCapability(
                        name="dynamic_cap",
                        version="1.0.0",
                        tags=["dynamic", "volatile"],
                    )
                ],
            )
            try:
                registry.register(card=card, ttl_sec=0.1)
                time.sleep(0.001)
                registry.deregister(aid)
            except Exception as e:  # noqa: BLE001
                exceptions.append(e)

    def worker_heartbeat():
        for _ in range(operations_per_thread):
            for i in range(0, num_agents, 3):
                try:
                    registry.heartbeat(f"agent_{i:03d}")
                except Exception as e:  # noqa: BLE001
                    exceptions.append(e)
            time.sleep(0.002)

    def worker_query():
        for _ in range(operations_per_thread):
            try:
                # Query by capability
                caps = registry.find_by_capability("cap_1", min_version="1.0.0")
                assert isinstance(caps, list)

                # Query by tag
                tags = registry.find_by_tag("common")
                assert isinstance(tags, list)

                # Snapshot active
                active = registry.list_active_agents()
                assert isinstance(active, list)
            except Exception as e:  # noqa: BLE001
                exceptions.append(e)
            time.sleep(0.001)

    def worker_evict():
        for _ in range(operations_per_thread):
            try:
                evicted = registry.evict_expired()
                assert isinstance(evicted, list)
            except Exception as e:  # noqa: BLE001
                exceptions.append(e)
            time.sleep(0.005)

    threads: list[threading.Thread] = []
    # 5 registration threads
    for tid in range(5):
        threads.append(threading.Thread(target=worker_register, args=(tid,)))
    # 5 heartbeat threads
    for _ in range(5):
        threads.append(threading.Thread(target=worker_heartbeat))
    # 10 query threads
    for _ in range(10):
        threads.append(threading.Thread(target=worker_query))
    # 2 eviction threads
    for _ in range(2):
        threads.append(threading.Thread(target=worker_evict))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not exceptions, f"Concurrent ServiceRegistry exceptions detected: {exceptions}"

    # Verify index integrity: No orphaned agent IDs in inverted indexes
    for cap_name, id_set in registry._capability_index.items():
        for aid in id_set:
            assert aid in registry._instances, f"Dangling agent '{aid}' in capability index '{cap_name}'"

    for tag_name, id_set in registry._tag_index.items():
        for aid in id_set:
            assert aid in registry._instances, f"Dangling agent '{aid}' in tag index '{tag_name}'"


def test_race_between_heartbeat_and_eviction():
    """Verify clean behavior during micro-race between TTL expiration eviction and heartbeat renewal."""
    registry = ServiceRegistry()
    card = AgentCard(agent_id="racing_agent", name="Racer")
    # Extremely short TTL (10ms)
    registry.register(card=card, ttl_sec=0.01)

    time.sleep(0.02)  # Allow lease to expire

    evicted_result = []
    heartbeat_result = []

    def do_evict():
        evicted = registry.evict_expired()
        evicted_result.extend(evicted)

    def do_heartbeat():
        hb = registry.heartbeat("racing_agent")
        heartbeat_result.append(hb)

    t1 = threading.Thread(target=do_evict)
    t2 = threading.Thread(target=do_heartbeat)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Either eviction won and heartbeat was False, or heartbeat won and renewal succeeded
    if "racing_agent" in evicted_result:
        # Eviction occurred
        assert registry.get_agent("racing_agent") is None
    else:
        # Heartbeat renewed before eviction
        assert heartbeat_result == [True]
        assert registry.get_agent("racing_agent") is not None


# ============================================================================
# 5. Circuit Breaker & Fallback Topology Stress
# ============================================================================


def test_circuit_breaker_sub_5ms_transition_latency():
    """Benchmark CircuitBreaker transition latency to OPEN state. Must be strictly < 5.0 ms."""
    cb = CircuitBreaker(failure_threshold=2)
    assert cb.state == CircuitState.CLOSED

    # Record first failure
    cb.record_failure()
    assert cb.state == CircuitState.CLOSED

    # Measure exact time taken to trip into OPEN
    start_ns = time.perf_counter_ns()
    cb.record_failure()
    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

    assert cb.state == CircuitState.OPEN
    assert cb.can_execute() is False
    assert elapsed_ms < 5.0, f"Circuit transition exceeded 5ms SLA: {elapsed_ms:.4f} ms"


@pytest.mark.asyncio
async def test_mesh_router_deep_fallback_cycle_detection():
    """Verify that circular fallback chains (A -> B -> C -> A) are blocked with RuntimeError."""
    router = MeshRouter()

    async def failure_handler(payload: dict[str, Any]) -> Any:
        raise ConnectionResetError("Node offline")

    # Cycle: node_a -> node_b -> node_c -> node_a
    router.register_endpoint(
        AgentEndpoint(agent_id="node_a", role="WorkerA"),
        handler=failure_handler,
        fallback_agent_id="node_b",
        failure_threshold=1,
    )
    router.register_endpoint(
        AgentEndpoint(agent_id="node_b", role="WorkerB"),
        handler=failure_handler,
        fallback_agent_id="node_c",
        failure_threshold=1,
    )
    router.register_endpoint(
        AgentEndpoint(agent_id="node_c", role="WorkerC"),
        handler=failure_handler,
        fallback_agent_id="node_a",
        failure_threshold=1,
    )

    with pytest.raises(RuntimeError) as exc_info:
        await router.route_and_call("node_a", {"task": "cycle_test"})

    error_str = str(exc_info.value)
    assert "Fallback cycle detected" in error_str
    assert "node_a -> node_b -> node_c -> node_a" in error_str


def test_smooth_weighted_round_robin_distribution_accuracy():
    """Empirically verify SWRR interleaved distribution over 700 calls."""
    swrr = SmoothWeightedRoundRobin()
    ep_a = AgentEndpoint(agent_id="ep_a", role="Worker", weight=5)
    ep_b = AgentEndpoint(agent_id="ep_b", role="Worker", weight=1)
    ep_c = AgentEndpoint(agent_id="ep_c", role="Worker", weight=1)

    endpoints = [ep_a, ep_b, ep_c]
    counts = {"ep_a": 0, "ep_b": 0, "ep_c": 0}

    for _ in range(700):
        selected = swrr.select(endpoints)
        assert selected is not None
        counts[selected.agent_id] += 1

    # In 700 selections with weights 5:1:1 (total 7 parts):
    # ep_a = 500, ep_b = 100, ep_c = 100
    assert counts["ep_a"] == 500
    assert counts["ep_b"] == 100
    assert counts["ep_c"] == 100
