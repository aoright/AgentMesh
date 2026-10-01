# Project: AgentMesh

## Architecture
AgentMesh is an enterprise production-grade resilient agent mesh and runtime system designed for durable deterministic execution, dual-protocol tool and cross-agent communication, capability security sandboxing, and full-stack telemetry.

### Core Architectural Layers:
1. Durable Execution Layer (agentmesh/engine):
   - Deterministic Graph Orchestrator decoupled from non-deterministic external activities.
   - Event sourcing with monotonic sequences, integer nanosecond timestamps, and RFC 8785 canonical JSON hashing.
   - Immutable SHA-256 cryptographic hash chain verifying event continuity.
   - ActivityMemoizer with deterministic idempotency keys ensuring zero duplicate external side-effects and zero duplicate token billing on crash recovery.
   - Autonomous DAG breakpoint resolution and crash resumption replaying state in <200ms.
   - Complete physical separation of state storage (/tmp/agentmesh/checkpoints or AGENTMESH_STATE_DIR) from repository code.

2. Agent Mesh & Dual-Protocol Gateway (agentmesh/mesh):
   - Model Context Protocol (MCP) 2.0 gateway with JSON-RPC 2.0 wire protocol envelope (tools/list, tools/call).
   - Agent-to-Agent (A2A) cross-agent protocol message schema (A2AMessage, AgentCard, performative semantics).
   - Dynamic service discovery and endpoint registry.
   - Weighted traffic scheduling across endpoint pools with smooth weighted selection.
   - Three-state circuit breaker (CLOSED, OPEN, HALF_OPEN) with sub-5ms transition to OPEN and circular fallback protection.

3. Capability Security Sandbox & Telemetry (agentmesh/security, agentmesh/telemetry):
   - Principle of least privilege capability access controller.
   - Filesystem security: canonical path resolution, prefix containment, and complete mode checking (including mode 'x').
   - Command execution security: strict command whitelisting with shell metacharacter banning (no chaining via ';', '&&', '|', '$()') and argument-vector execution.
   - Network outbound isolation with SSRF protection against loopback and cloud metadata (169.254.169.254).
   - W3C TraceContext standards-compliant distributed tracing (traceparent, tracestate, inject/extract).
   - Precise token consumption tracking and auditing across agent lifecycles.

4. System Audit & Verification Suite (tests, examples):
   - Four-dimensional automated verification: Chaos Recovery, Sandbox Security, Code Quality / State Separation, Protocol Interoperability.
   - Verified reference examples: chaos_recovery_demo.py and financial_audit_workflow.py.

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Activity Abstraction & ActivityRecord | Decouple non-deterministic external calls from graph orchestration | M1 | Survey / R1 |
| 2 | Activity Memoizer & Idempotency Keys | Cache activity results keyed by SHA-256(node+activity+args) to eliminate duplicate calls | M1 | Survey / R1 |
| 3 | Cryptographic Event Hash Chain & Anti-Tamper | Enforce SHA-256 event chaining; raise CryptographicIntegrityError immediately on tamper | M1 | Survey / R1 |
| 4 | Autonomous Breakpoint Resumption | Auto-resolve resume entry point from DAG topology and event history in <200ms | M1 | Survey / R1 |
| 5 | Physical State Separation Engine | Atomic snapshots and event logs isolated to dedicated data directory; zero DBs in repo | M1 | Survey / R1 |
| 6 | MCP 2.0 JSON-RPC Wire Protocol | Standard JSON-RPC 2.0 wire protocol envelope (tools/list, tools/call, error handling) | M2 | Survey / R2 |
| 7 | A2A Cross-Agent Protocol | Formal Agent-to-Agent message envelope, AgentCard, performatives, and negotiation | M2 | Survey / R2 |
| 8 | Dynamic Service Discovery & Registry | Dynamic agent registration, heartbeat tracking, and capability lookup | M2 | Survey / R2 |
| 9 | Weighted Traffic Routing | Smooth weighted round-robin and endpoint pool selection | M2 | Survey / R2 |
| 10 | Circuit Breaker & Fallback Topology | Sub-5ms OPEN tripping with cycle-free fallback routing | M2 | Survey / R2 |
| 11 | Filesystem Sandbox & Mode 'x' Defense | Canonical path resolution, prefix containment, 100% path traversal block | M3 | Survey / R3 |
| 12 | Command Whitelist & Anti-Chaining | Banning shell metacharacters, argument vector execution, 100% injection block | M3 | Survey / R3 |
| 13 | Network Sandbox & SSRF Isolation | Domain whitelisting, loopback blocking, cloud metadata IP protection | M3 | Survey / R3 |
| 14 | W3C TraceContext Distributed Tracing | Standard traceparent/tracestate serialization, context propagation | M3 | Survey / R3 |
| 15 | Token Accounting & Audit Logging | Accurate token usage tracking, audit trail logging, and budget enforcement | M3 | Survey / R3 |
| 16 | Chaos Recovery Verification Demo | Verification of crash recovery <200ms and 0% duplicate token billing in demo | M4 | Survey / R4 |
| 17 | Financial Audit Workflow Demo | Real-world scenario exercising Sandbox, MCP, A2A, and Telemetry | M4 | Survey / R4 |
| 18 | Static Typing & PEP 8 Compliance | Fix mypy type error in graph.py:126, resolve ruff warnings, enforce clean code | M4 | Survey / R4 |
| 19 | Comprehensive E2E Test Suite (Tiers 1-4) | Opaque-box requirement-driven 4-tier test suite passing 100% | M5 | Survey / R4 |
| 20 | Adversarial Coverage Hardening (Tier 5) | White-box adversarial testing, edge case stress testing, and forensic audit | M5 | Survey / R4 |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Durable Execution & Event Sourcing Engine | agentmesh/engine (activity.py, memoizer.py, state.py, checkpoint.py, replay.py, graph.py) | none | DONE |
| M2 | Agent Mesh & Dual-Protocol Gateway | agentmesh/mesh (mcp_protocol.py, mcp_client.py, a2a.py, router.py, discovery.py) | none | DONE |
| M3 | Capability Security Sandbox & W3C Telemetry | agentmesh/security (sandbox.py, policy.py), agentmesh/telemetry (tracer.py, w3c.py) | none | DONE |
| M4 | System Integration, Chaos Demo & Financial Workflow | examples/chaos_recovery_demo.py, examples/financial_audit_workflow.py, mypy fix | M1, M2, M3 | DONE |
| M5 | Final E2E Test Suite (Tiers 1-4) & Adversarial Hardening (Tier 5) | Full project E2E tests, TEST_READY.md validation, adversarial audit | M1, M2, M3, M4 | DONE |



---

## Interface Contracts

### Engine ↔ Mesh (Activity Execution)
- Function: `execute_activity(activity_name: str, fn: Callable, args: tuple, kwargs: dict, idempotency_key: str | None = None) -> Any`
- Semantics: Looks up `ActivityMemoizer`. If key exists in event history/memoizer, returns cached result without invoking `fn`. If not, invokes `fn`, records `ACTIVITY_COMPLETED` event with result payload, and stores in memoizer.

### Mesh ↔ Security Sandbox (Tool & Agent Call Protection)
- Function: `SecuritySandbox.validate_tool_call(tool_name: str, arguments: dict) -> None`
- Function: `SecuritySandbox.validate_file_access(target_path: str, mode: str) -> Path`
- Function: `SecuritySandbox.validate_command(cmd: str | list[str]) -> list[str]`
- Function: `SecuritySandbox.validate_network(url_or_host: str) -> None`
- Error handling: Raises `PermissionDeniedError` on any policy violation.

### Mesh ↔ Telemetry (W3C Trace Context Propagation)
- Class: `TraceContext`
  - Method: `to_w3c_headers() -> dict[str, str]` (generates `traceparent`, `tracestate`)
  - Method: `from_w3c_headers(headers: dict[str, str]) -> TraceContext`
  - Method: `record_token_usage(prompt_tokens: int, completion_tokens: int, model: str) -> None`

### Gateway Protocols (MCP 2.0 & A2A)
- MCP 2.0:
  - Request: `{"jsonrpc": "2.0", "id": str | int, "method": str, "params": dict}`
  - Response: `{"jsonrpc": "2.0", "id": str | int, "result": dict}` or `{"jsonrpc": "2.0", "id": str | int, "error": {"code": int, "message": str}}`
- A2A Message:
  - Envelope: `A2AMessage(message_id: str, sender_id: str, recipient_id: str, performative: str, payload: dict, traceparent: str | None = None)`

---

## Code Layout & Write Boundaries
To ensure concurrency safety and prevent merge conflicts across parallel subagents:
- Milestone 1 (M1) exclusively owns:
  - `agentmesh/engine/activity.py`
  - `agentmesh/engine/memoizer.py`
  - `agentmesh/engine/state.py`
  - `agentmesh/engine/checkpoint.py`
  - `agentmesh/engine/replay.py`
  - `agentmesh/engine/graph.py`
  - `tests/test_engine.py`
- Milestone 2 (M2) exclusively owns:
  - `agentmesh/mesh/mcp_protocol.py`
  - `agentmesh/mesh/mcp_client.py`
  - `agentmesh/mesh/a2a.py`
  - `agentmesh/mesh/router.py`
  - `agentmesh/mesh/discovery.py`
  - `tests/test_mesh.py`
- Milestone 3 (M3) exclusively owns:
  - `agentmesh/security/sandbox.py`
  - `agentmesh/security/policy.py`
  - `agentmesh/telemetry/tracer.py`
  - `agentmesh/telemetry/w3c.py`
  - `tests/test_security.py`
  - `tests/test_telemetry.py`
- Milestone 4 (M4) exclusively owns:
  - `examples/chaos_recovery_demo.py`
  - `examples/financial_audit_workflow.py`
  - Integration sanity verification
- E2E Testing Track exclusively owns:
  - `tests/e2e/`
  - `TEST_INFRA.md`
  - `TEST_READY.md`
