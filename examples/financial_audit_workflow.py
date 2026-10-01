"""Production-grade Financial Audit Workflow with Mesh Routing, MCP Tools, and Security Sandbox."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentmesh.engine.state import State
from agentmesh.engine.graph import Graph
from agentmesh.mesh.router import MeshRouter, AgentEndpoint
from agentmesh.mesh.mcp_client import MCPToolClient, MCPToolCallRequest
from agentmesh.security.sandbox import SecuritySandbox, CapabilityPolicy, PermissionDeniedError
from agentmesh.telemetry.tracer import AgentTracer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("financial_audit")


async def main():
    print("=" * 75)
    print(" AgentMesh: Financial Compliance Multi-Agent Workflow with MCP & Sandbox")
    print("=" * 75)

    tracer = AgentTracer()
    trace_ctx = tracer.get_or_create_context()
    span_root = trace_ctx.start_span("financial_audit_root")

    # 1. Setup MCP Tools
    mcp_client = MCPToolClient(server_name="finance_tools_v1")

    # Mock tool to query ledgers
    def query_ledger(account_id: str, limit: int = 5):
        return [
            {"tx_id": "TX-901", "amount": 49999, "currency": "CNY", "counterparty": "Company_Alpha", "flag": "NORMAL"},
            {"tx_id": "TX-902", "amount": 1200000, "currency": "CNY", "counterparty": "Offshore_Beta", "flag": "SUSPICIOUS"},
            {"tx_id": "TX-903", "amount": 2500, "currency": "CNY", "counterparty": "Supplier_Gamma", "flag": "NORMAL"},
        ]

    mcp_client.register_tool(
        name="query_ledger",
        description="Query banking ledger transactions for a target account.",
        parameters={"account_id": "string", "limit": "integer"},
        handler=query_ledger,
    )

    # 2. Setup Security Sandbox for Compliance Agent
    sandbox_policy = CapabilityPolicy(
        policy_name="finance_compliance_restricted",
        allow_network=False,
        allowed_read_paths=["/tmp/audit_reports"],
        allowed_write_paths=["/tmp/audit_reports"],
        allowed_commands={"echo", "cat"},
    )
    sandbox = SecuritySandbox(policy=sandbox_policy)

    # 3. Setup Mesh Router for Multi-Agent Communication
    router = MeshRouter()

    # Agent A: Ingestion Agent
    async def data_ingestion_agent(payload: dict) -> dict:
        account_id = payload.get("account_id", "ACC-001")
        req = MCPToolCallRequest(tool_name="query_ledger", arguments={"account_id": account_id})
        res = await mcp_client.invoke_tool(req)
        trace_ctx.record_token_usage(prompt=120, completion=45)
        return {"transactions": res.result, "status": "FETCHED"}

    # Agent B: Primary Compliance Agent
    async def primary_compliance_agent(payload: dict) -> dict:
        txs = payload.get("transactions", [])
        suspicious = [t for t in txs if t.get("amount", 0) > 500000 or t.get("flag") == "SUSPICIOUS"]
        trace_ctx.record_token_usage(prompt=350, completion=80)
        return {
            "suspicious_count": len(suspicious),
            "suspicious_records": suspicious,
            "risk_level": "CRITICAL" if suspicious else "LOW",
        }

    # Agent C: Fallback Compliance Agent (in case primary experiences issues)
    async def fallback_compliance_agent(payload: dict) -> dict:
        logger.info("Fallback rule-based compliance triggered.")
        return {"suspicious_count": 1, "risk_level": "HIGH", "note": "FALLBACK_RULE_EVALUATED"}

    router.register_endpoint(
        AgentEndpoint(agent_id="agent_ingestion", role="Data Ingestion"),
        handler=data_ingestion_agent,
    )
    router.register_endpoint(
        AgentEndpoint(agent_id="agent_compliance_primary", role="AI Risk Assessor"),
        handler=primary_compliance_agent,
        fallback_agent_id="agent_compliance_fallback",
        failure_threshold=2,
    )
    router.register_endpoint(
        AgentEndpoint(agent_id="agent_compliance_fallback", role="Rule-based Fallback Assessor"),
        handler=fallback_compliance_agent,
    )

    # 4. Construct Graph Workflow
    workflow_state = State(workflow_id="wf-financial-audit-001")
    workflow_state.set("target_account", "ACC-668899")

    async def step_fetch(state: State):
        acc = state.get("target_account")
        res = await router.route_and_call("agent_ingestion", {"account_id": acc})
        state.set("transactions", res["transactions"])

    async def step_assess(state: State):
        txs = state.get("transactions")
        assessment = await router.route_and_call("agent_compliance_primary", {"transactions": txs})
        state.set("assessment", assessment)

    async def step_verify_sandbox_and_conclude(state: State):
        # Demonstrate Sandbox protection: attempting to write outside allowed directory
        try:
            sandbox.validate_file_access("/etc/passwd", mode="w")
        except PermissionDeniedError as e:
            logger.info("Sandbox successfully blocked unauthorized system file write: %s", e)
            state.set("sandbox_verification", "PASSED_PROTECTED")

        assessment = state.get("assessment", {})
        state.set("report_status", f"AUDIT_COMPLETE_STATUS_{assessment.get('risk_level')}")

    graph = Graph(name="financial_compliance_flow")
    graph.add_node("fetch_records", step_fetch)
    graph.add_node("risk_assessment", step_assess)
    graph.add_node("sandbox_and_conclude", step_verify_sandbox_and_conclude)

    graph.add_edge("fetch_records", "risk_assessment")
    graph.add_edge("risk_assessment", "sandbox_and_conclude")

    graph.set_entry_point("fetch_records")
    graph.set_finish_point("sandbox_and_conclude")

    # Run workflow
    final_state = await graph.run(workflow_state)

    span_root.finish()
    summary = tracer.export_summary(trace_ctx.trace_id)

    print("\n" + "=" * 75)
    print(" FINANCIAL AUDIT EXECUTION RESULT SUMMARY")
    print("=" * 75)
    print(f"Target Account     : {final_state.get('target_account')}")
    print(f"Suspicious Records : {final_state.get('assessment', {}).get('suspicious_count')}")
    print(f"Assessed Risk Level: {final_state.get('assessment', {}).get('risk_level')}")
    print(f"Sandbox Defense    : {final_state.get('sandbox_verification')}")
    print(f"Telemetry Tokens   : Total={summary.get('total_tokens')} (Prompt={summary.get('prompt_tokens')}, Completion={summary.get('completion_tokens')})")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(main())
