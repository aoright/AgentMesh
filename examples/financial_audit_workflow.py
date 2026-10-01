#!/usr/bin/env python3
"""Production-grade Financial Audit Workflow with Mesh Routing, MCP Tools, A2A, and Sandbox.

Demonstrates:
1. Durable Execution: Deterministic DAG orchestration, Activity isolation, and SHA-256 event chaining.
2. Dual-Protocol Gateway: MCP 2.0 tool calls (query_ledger, currency_exchange) and A2A negotiation.
3. Capability Security Sandbox: Tri-vector least-privilege defense (paths, commands, SSRF).
4. W3C Distributed Tracing: Ambient context propagation, token quota enforcement, and audit logs.
5. Invariants & Autonomous Execution: 100% automated execution, self-validating state consistency.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any

# Ensure project root is available on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentmesh.engine.checkpoint import CheckpointManager
from agentmesh.engine.graph import Graph
from agentmesh.engine.replay import EventReplayer
from agentmesh.engine.state import CryptographicIntegrityError, State
from agentmesh.mesh.a2a import (
    A2AMessage,
    AgentCapability,
    AgentCard,
    NegotiationManager,
    NegotiationState,
    Performative,
)
from agentmesh.mesh.mcp_client import MCPToolCallRequest, MCPToolClient
from agentmesh.mesh.router import AgentEndpoint, MeshRouter
from agentmesh.security.policy import CapabilityPolicy
from agentmesh.security.sandbox import PermissionDeniedError, SecuritySandbox
from agentmesh.telemetry.tracer import AgentTracer, QuotaExceededError
from agentmesh.telemetry.w3c import validate_traceparent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("financial_audit")


async def main() -> None:
    print("=" * 80)
    print(" AgentMesh: Financial Compliance Multi-Agent Workflow (Feature 17)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 0: State Separation Directories Setup
    # -------------------------------------------------------------------------
    run_uuid = uuid.uuid4().hex[:8]
    base_state_dir = os.environ.get("AGENTMESH_STATE_DIR", f"/tmp/agentmesh/checkpoints_{run_uuid}")
    state_dir = Path(base_state_dir).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    CheckpointManager.validate_state_dir(state_dir)

    audit_out_dir = Path(f"/tmp/agentmesh/audit_reports_{run_uuid}").resolve()
    audit_out_dir.mkdir(parents=True, exist_ok=True)

    telemetry_dir = Path(f"/tmp/agentmesh/audit_{run_uuid}").resolve()
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    telemetry_log_file = telemetry_dir / "financial_audit_telemetry.log"

    ckpt_mgr = CheckpointManager(base_dir=str(state_dir))

    # -------------------------------------------------------------------------
    # STEP 1: W3C Distributed Tracing & Token Accounting Setup
    # -------------------------------------------------------------------------
    tracer = AgentTracer(audit_log_file=telemetry_log_file)
    trace_ctx = tracer.get_or_create_context(budget_limit=3000)
    root_span = trace_ctx.start_span(
        "financial_audit_orchestration",
        attributes={"workflow": "financial_audit", "target": "ACC-668899"},
    )
    logger.info("Initialized W3C TraceContext trace_id=%s", trace_ctx.trace_id)

    # -------------------------------------------------------------------------
    # STEP 2: MCP 2.0 Tool Gateway Setup
    # -------------------------------------------------------------------------
    mcp_client = MCPToolClient(server_name="finance_gateway_v2")

    def query_ledger(account_id: str, limit: int = 10) -> list[dict[str, Any]]:
        return [
            {
                "tx_id": "TX-2026-001",
                "amount": 45000.0,
                "currency": "CNY",
                "counterparty": "Supplier_Domestic_A",
                "flag": "NORMAL",
            },
            {
                "tx_id": "TX-2026-002",
                "amount": 1800000.0,
                "currency": "CNY",
                "counterparty": "Offshore_Entity_BVI",
                "flag": "SUSPICIOUS",
            },
            {
                "tx_id": "TX-2026-003",
                "amount": 85000.0,
                "currency": "USD",
                "counterparty": "Global_Tech_Services",
                "flag": "FOREIGN_CURRENCY",
            },
        ]

    def currency_exchange(amount: float, from_currency: str, to_currency: str = "CNY") -> dict[str, Any]:
        fx_rates = {"USD": 7.25, "EUR": 7.85, "HKD": 0.93, "CNY": 1.0}
        rate = fx_rates.get(from_currency.upper(), 1.0)
        converted_amount = round(amount * rate, 2)
        return {
            "original_amount": amount,
            "from_currency": from_currency,
            "to_currency": to_currency,
            "exchange_rate": rate,
            "converted_amount": converted_amount,
        }

    mcp_client.register_tool(
        name="query_ledger",
        description="Query banking ledger transactions for a target account.",
        parameters={"account_id": "string", "limit": "integer"},
        handler=query_ledger,
    )

    mcp_client.register_tool(
        name="currency_exchange",
        description="Convert transaction amounts between currencies based on central bank FX rates.",
        parameters={"amount": "number", "from_currency": "string", "to_currency": "string"},
        handler=currency_exchange,
    )

    # -------------------------------------------------------------------------
    # STEP 3: Capability Security Sandbox Setup
    # -------------------------------------------------------------------------
    sandbox_policy = CapabilityPolicy(
        policy_name="financial_compliance_sandbox",
        allowed_read_paths=[str(audit_out_dir)],
        allowed_write_paths=[str(audit_out_dir)],
        allowed_commands={"echo", "sha256sum", "cat"},
        allow_network=True,
        allowed_domains=["api.swift-compliance.com", "fx.centralbank.org"],
        allowed_schemes={"https"},
        block_loopback=True,
        block_cloud_metadata=True,
        block_private_ips=True,
    )
    sandbox = SecuritySandbox(policy=sandbox_policy)

    # -------------------------------------------------------------------------
    # STEP 4: A2A Multi-Agent Setup & Protocol Definition
    # -------------------------------------------------------------------------
    auditor_card = AgentCard(
        agent_id="agent_auditor_lead",
        name="Lead Financial Auditor",
        description="Coordinates transaction ingestion and initiates regulatory audit",
        version="2.0.0",
    )

    compliance_card = AgentCard(
        agent_id="agent_compliance_aml",
        name="Regulatory Compliance Officer",
        description="Evaluates anti-money laundering indicators and determines sanctions",
        version="2.0.0",
        capabilities=[
            AgentCapability(
                name="aml_risk_assessment",
                version="2.0.0",
                description="Performs heuristic and threshold-based AML risk evaluation",
            )
        ],
    )

    negotiation_mgr = NegotiationManager()
    router = MeshRouter()

    # Ingestion Agent Handler (Uses MCP Tools)
    async def data_ingestion_agent(payload: dict[str, Any]) -> dict[str, Any]:
        acc = payload.get("account_id", "ACC-668899")
        span_ingest = trace_ctx.start_span("data_ingestion", parent_span_id=root_span.span_id)

        # Call MCP Tool: query_ledger
        req_ledger = MCPToolCallRequest(tool_name="query_ledger", arguments={"account_id": acc})
        res_ledger = await mcp_client.invoke_tool(req_ledger)
        raw_txs: list[dict[str, Any]] = res_ledger.result or []

        # Convert foreign currency transactions using MCP Tool: currency_exchange
        standardized_txs: list[dict[str, Any]] = []
        for tx in raw_txs:
            if tx.get("currency") != "CNY":
                fx_req = MCPToolCallRequest(
                    tool_name="currency_exchange",
                    arguments={
                        "amount": tx["amount"],
                        "from_currency": tx["currency"],
                        "to_currency": "CNY",
                    },
                )
                fx_res = await mcp_client.invoke_tool(fx_req)
                tx["amount_cny"] = fx_res.result["converted_amount"]
                tx["fx_applied"] = fx_res.result["exchange_rate"]
            else:
                tx["amount_cny"] = tx["amount"]
                tx["fx_applied"] = 1.0
            standardized_txs.append(tx)

        trace_ctx.record_token_usage(prompt=180, completion=60, model="gpt-3.5-turbo")
        tracer.audit_logger.log_tool_call(
            trace_id=trace_ctx.trace_id,
            span_id=span_ingest.span_id,
            actor_id="agent_ingestion",
            tool_name="query_ledger",
            arguments={"account_id": acc},
            status="SUCCESS",
        )
        span_ingest.finish()
        return {"account_id": acc, "transactions": standardized_txs}

    # Compliance Agent Handler (Engages in A2A Negotiation)
    async def compliance_officer_agent(payload: dict[str, Any]) -> dict[str, Any]:
        txs: list[dict[str, Any]] = payload.get("transactions", [])
        span_compliance = trace_ctx.start_span("compliance_officer_review", parent_span_id=root_span.span_id)

        # Step 1: Auditor sends REQUEST
        conv_id = f"conv-audit-{payload.get('account_id', 'ACC')}"
        tp_header = trace_ctx.to_traceparent(span_compliance.span_id)
        msg_req = A2AMessage(
            sender_id=auditor_card.agent_id,
            recipient_id=compliance_card.agent_id,
            performative=Performative.REQUEST,
            payload={"intent": "aml_review_request", "record_count": len(txs)},
            conversation_id=conv_id,
            traceparent=tp_header,
        )
        session = negotiation_mgr.process_message(msg_req)

        # Step 2: Compliance Agent returns PROPOSE
        msg_prop = msg_req.create_reply(
            performative=Performative.PROPOSE,
            payload={"service_fee_tokens": 120, "policy_ruleset": "AML_PBOC_2026_STRICT"},
        )
        negotiation_mgr.process_message(msg_prop)

        # Step 3: Auditor accepts via AGREE
        msg_agree = msg_prop.create_reply(
            performative=Performative.AGREE,
            payload={"accepted": True, "priority": "HIGH"},
        )
        negotiation_mgr.process_message(msg_agree)

        # Step 4: Compliance Agent executes analysis
        session.mark_executing()
        suspicious = [
            t for t in txs if t.get("amount_cny", 0) > 500000.0 or t.get("flag") == "SUSPICIOUS"
        ]

        # Step 5: Compliance Agent returns final assessment via INFORM
        verdict = "CRITICAL" if suspicious else "CLEAN"
        msg_inform = msg_agree.create_reply(
            performative=Performative.INFORM,
            payload={
                "risk_level": verdict,
                "suspicious_count": len(suspicious),
                "suspicious_records": suspicious,
                "regulatory_action": "REPORT_TO_REGULATOR" if suspicious else "ARCHIVE",
            },
        )
        negotiation_mgr.process_message(msg_inform)

        trace_ctx.record_token_usage(prompt=420, completion=110, model="gpt-4o")
        tracer.audit_logger.log_agent_interaction(
            trace_id=trace_ctx.trace_id,
            span_id=span_compliance.span_id,
            sender_id=auditor_card.agent_id,
            recipient_id=compliance_card.agent_id,
            performative=Performative.INFORM.value,
            payload=msg_inform.payload,
        )
        span_compliance.finish()
        return msg_inform.payload

    router.register_endpoint(
        AgentEndpoint(agent_id="agent_ingestion", role="Data Ingestion"),
        handler=data_ingestion_agent,
    )
    router.register_endpoint(
        AgentEndpoint(agent_id="agent_compliance_aml", role="AML Compliance Officer"),
        handler=compliance_officer_agent,
    )

    # -------------------------------------------------------------------------
    # STEP 5: Construct Deterministic DAG Workflow
    # -------------------------------------------------------------------------
    workflow_id = f"wf-financial-audit-{run_uuid}"
    workflow_state = State(workflow_id=workflow_id)
    workflow_state.set("target_account", "ACC-668899")

    async def step_fetch(state: State) -> None:
        acc = state.get("target_account")
        # Isolate via State.execute_activity with deterministic idempotency key
        res = await state.execute_activity(
            activity_name="fetch_transactions_activity",
            fn=router.route_and_call,
            args=("agent_ingestion", {"account_id": acc}),
            idempotency_key=f"idemp-fetch-{acc}",
        )
        state.set("transactions", res["transactions"])

    async def step_assess(state: State) -> None:
        txs = state.get("transactions")
        # Isolate compliance evaluation with memoizer support
        assessment = await state.execute_activity(
            activity_name="compliance_assessment_activity",
            fn=router.route_and_call,
            args=("agent_compliance_aml", {"transactions": txs}),
            idempotency_key=f"idemp-aml-{state.get('target_account')}",
        )
        state.set("assessment", assessment)

    async def step_sandbox_and_conclude(state: State) -> None:
        acc = state.get("target_account")
        assessment = state.get("assessment", {})

        # 1. Authorized File Write
        report_file = audit_out_dir / f"audit_verdict_{acc}.json"
        safe_path = sandbox.validate_file_access(str(report_file), mode="w")
        with open(safe_path, "w", encoding="utf-8") as f:
            json.dump(assessment, f, indent=2)
        state.set("report_file", str(report_file))

        # 2. Authorized Command Validation
        cmd_vector = sandbox.validate_command(["sha256sum", str(report_file)])
        state.set("validated_command", str(cmd_vector))

        # 3. Authorized Network Call Validation
        safe_url = sandbox.validate_network("https://fx.centralbank.org/rates")
        state.set("validated_network_url", safe_url)

        # 4. Tri-Vector Sandbox Defense Tests (100% Intercept Rate across 8 vectors)
        interceptions: list[str] = []

        # Vector 1: Path Traversal
        try:
            sandbox.validate_file_access("/etc/passwd", mode="w")
        except PermissionDeniedError:
            interceptions.append("BLOCKED_ETC_PASSWD")

        try:
            sandbox.validate_file_access(f"{audit_out_dir}/../../etc/shadow", mode="r")
        except PermissionDeniedError:
            interceptions.append("BLOCKED_TRAVERSAL")

        # Vector 2: Command Injection & Anti-Chaining
        try:
            sandbox.validate_command(f"cat {report_file}; rm -rf /")
        except PermissionDeniedError:
            interceptions.append("BLOCKED_COMMAND_CHAINING")

        try:
            sandbox.validate_command("echo ok && curl http://evil.com")
        except PermissionDeniedError:
            interceptions.append("BLOCKED_COMMAND_AMPERSAND")

        # Vector 3: Network SSRF Defense
        try:
            sandbox.validate_network("http://127.0.0.1:8080/admin")
        except PermissionDeniedError:
            interceptions.append("BLOCKED_SSRF_LOOPBACK")

        try:
            sandbox.validate_network("http://169.254.169.254/latest/meta-data")
        except PermissionDeniedError:
            interceptions.append("BLOCKED_SSRF_CLOUD_METADATA")

        try:
            sandbox.validate_network("http://10.0.0.1/core-banking")
        except PermissionDeniedError:
            interceptions.append("BLOCKED_SSRF_PRIVATE_IP")

        try:
            sandbox.validate_network("https://unauthorized-attacker-site.com/leak")
        except PermissionDeniedError:
            interceptions.append("BLOCKED_UNAUTHORIZED_DOMAIN")

        state.set("sandbox_interceptions", interceptions)
        state.set("audit_status", f"COMPLETED_{assessment.get('risk_level')}")

        tracer.audit_logger.log_security_check(
            trace_id=trace_ctx.trace_id,
            span_id=root_span.span_id,
            actor_id="sandbox_engine",
            check_type="tri_vector_defense",
            target="system_security_boundaries",
            status="SUCCESS",
            details={"interceptions_count": len(interceptions)},
        )

    graph = Graph(name="enterprise_financial_audit_dag")
    graph.add_node("fetch_records", step_fetch)
    graph.add_node("risk_assessment", step_assess)
    graph.add_node("sandbox_and_conclude", step_sandbox_and_conclude)

    graph.add_edge("fetch_records", "risk_assessment")
    graph.add_edge("risk_assessment", "sandbox_and_conclude")

    graph.set_entry_point("fetch_records")
    graph.set_finish_point("sandbox_and_conclude")

    # -------------------------------------------------------------------------
    # STEP 6: Execute Workflow with Checkpointing & Memoization
    # -------------------------------------------------------------------------
    final_state = await graph.run(workflow_state, checkpoint_manager=ckpt_mgr)
    root_span.finish()
    summary = tracer.export_summary(trace_ctx.trace_id)

    # Demonstrate QuotaExceededError defense
    quota_exceeded_caught = False
    try:
        trace_ctx.record_token_usage(prompt=50000, completion=50000)
    except QuotaExceededError:
        quota_exceeded_caught = True
    assert quota_exceeded_caught is True

    # -------------------------------------------------------------------------
    # STEP 7: Assert Invariants & Autonomous Verification
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" INVARIANT VERIFICATION & DEFENSE AUDIT")
    print("=" * 80)

    # Invariant 1: State Separation
    assert state_dir.exists()
    assert len(list(state_dir.glob(f"{workflow_id}/ckpt_*.json"))) >= 3
    print(" [INVARIANT 1: State Separation] PASS - State persisted outside code repository.")

    # Invariant 2: Cryptographic Event Chain Integrity
    is_valid, err = EventReplayer.verify_event_chain(final_state.events)
    assert is_valid is True, f"Event chain corrupt: {err}"
    EventReplayer.verify_and_enforce_chain(final_state)
    print(f" [INVARIANT 2: Hash Chain Integrity] PASS - Verified {len(final_state.events)} events.")

    # Invariant 3: Anti-Tamper Exception Enforcement
    tampered_state = State(
        workflow_id=final_state.workflow_id,
        run_id=final_state.run_id,
        version=final_state.version,
        data=final_state.data.copy(),
        events=[e.model_copy(deep=True) for e in final_state.events],
        last_hash=final_state.last_hash,
    )
    tampered_state.events[2].payload["tampered_amount"] = 999999999
    tamper_blocked = False
    try:
        EventReplayer.enforce_integrity(tampered_state)
    except CryptographicIntegrityError:
        tamper_blocked = True
    assert tamper_blocked is True
    print(" [INVARIANT 3: Anti-Tamper Enforcement] PASS - Tampering detected and blocked.")

    # Invariant 4: Sandbox Tri-Vector Defense
    interceptions = final_state.get("sandbox_interceptions", [])
    assert len(interceptions) == 8, f"Expected 8 blocked vectors, got {len(interceptions)}"
    print(f" [INVARIANT 4: Sandbox Defense] PASS - 100% intercept rate ({len(interceptions)}/8 attacks blocked).")

    # Invariant 5: A2A Protocol Negotiation Completion
    sessions = negotiation_mgr._sessions
    assert len(sessions) > 0
    active_session = next(iter(sessions.values()))
    assert active_session.current_state == NegotiationState.INFORMED
    assert len(active_session.history) == 4
    print(f" [INVARIANT 5: A2A Negotiation] PASS - Full FIPA ACL lifecycle completed ({active_session.current_state}).")

    # Invariant 6: Telemetry Token Accounting & W3C Tracing
    assert summary["total_tokens"] > 0
    assert summary["prompt_tokens"] > 0
    assert summary["completion_tokens"] > 0
    assert summary["cost_estimate"] > 0.0
    validate_traceparent(trace_ctx.to_traceparent())
    assert quota_exceeded_caught is True
    print(f" [INVARIANT 6: Telemetry & Tracing] PASS - Total Tokens={summary['total_tokens']}, Cost=${summary['cost_estimate']:.6f}.")

    # Invariant 7: Idempotency Replay (Zero duplicate tokens on re-execution)
    replayed_state = await graph.run(
        final_state,
        checkpoint_manager=ckpt_mgr,
        start_from_node="fetch_records",
    )
    memoized_hits = [
        e for e in replayed_state.events if e.event_type.value == "ACTIVITY_MEMOIZED_HIT"
    ]
    assert len(memoized_hits) >= 2
    print(f" [INVARIANT 7: Memoizer Replay] PASS - {len(memoized_hits)} activities memoized with 0 extra tokens.")

    print("=" * 80)
    print(" ALL PRODUCTION CRITERIA & FEATURE 17 INVARIANTS SATISFIED")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
