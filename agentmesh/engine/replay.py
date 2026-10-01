"""Deterministic Event Replay, Resumption Planning, and Crash Recovery Engine."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from agentmesh.engine.checkpoint import CheckpointManager, StateSeparationError
from agentmesh.engine.state import (
    GENESIS_HASH,
    CryptographicIntegrityError,
    Event,
    EventType,
    State,
)

if TYPE_CHECKING:
    from agentmesh.engine.graph import Graph

logger = logging.getLogger("agentmesh.engine.replay")
audit_logger = logging.getLogger("agentmesh.security.audit")

__all__ = [
    "CryptographicIntegrityError",
    "EventReplayer",
    "ResumptionError",
    "ResumptionPlan",
    "ResumptionPlanner",
]


class ResumptionError(Exception):
    """Raised when autonomous resumption planning encounters an unresolvable graph state."""




class ResumptionPlan(BaseModel):
    """Execution plan determined autonomously by ResumptionPlanner."""

    workflow_id: str
    resume_node: str | None = None
    last_completed_node: str | None = None
    is_completed: bool = False
    is_resuming_interrupted_node: bool = False
    evaluated_edge_target: str | None = None
    memoized_activity_count: int = 0
    reason: str = ""


class ResumptionPlanner:
    """Autonomously resolves DAG resumption breakpoint from event stream and topology."""

    @staticmethod
    def plan_resumption(graph: Graph, state: State) -> ResumptionPlan:
        """Determines the exact node to resume from based on graph topology and state history."""
        # 1. Terminal State Check
        for event in reversed(state.events):
            if event.event_type == EventType.WORKFLOW_COMPLETE:
                return ResumptionPlan(
                    workflow_id=state.workflow_id,
                    resume_node=None,
                    last_completed_node=EventReplayer.find_last_completed_node(state),
                    is_completed=True,
                    reason="Workflow was already marked completed",
                )

        # 2. Last Node Event Identification
        last_node_event: Event | None = None
        for event in reversed(state.events):
            if event.event_type in (
                EventType.NODE_COMPLETE,
                EventType.NODE_START,
                EventType.NODE_FAILED,
            ):
                last_node_event = event
                break

        if not last_node_event or not last_node_event.node_id:
            return ResumptionPlan(
                workflow_id=state.workflow_id,
                resume_node=graph.entry_point,
                reason="No prior node executions found, starting from entry point",
            )

        # 3. Interrupted node handling (crash occurred during node execution)
        if last_node_event.event_type in (EventType.NODE_START, EventType.NODE_FAILED):
            interrupted_node = last_node_event.node_id
            return ResumptionPlan(
                workflow_id=state.workflow_id,
                resume_node=interrupted_node,
                last_completed_node=EventReplayer.find_last_completed_node(state),
                is_resuming_interrupted_node=True,
                reason=f"Node '{interrupted_node}' was interrupted mid-flight; resuming execution of node '{interrupted_node}'",
            )

        # 4. Node completed cleanly: determine next node via DAG edges
        completed_node = last_node_event.node_id
        if completed_node in graph.finish_points:
            return ResumptionPlan(
                workflow_id=state.workflow_id,
                resume_node=None,
                last_completed_node=completed_node,
                is_completed=True,
                reason=f"Last completed node '{completed_node}' is a declared finish point",
            )

        matching_edges = [edge for edge in graph.edges if edge.source == completed_node]
        if not matching_edges:
            return ResumptionPlan(
                workflow_id=state.workflow_id,
                resume_node=None,
                last_completed_node=completed_node,
                is_completed=True,
                reason=f"Node '{completed_node}' has no outgoing edges",
            )

        for edge in matching_edges:
            if edge.evaluate(state):
                return ResumptionPlan(
                    workflow_id=state.workflow_id,
                    resume_node=edge.target,
                    last_completed_node=completed_node,
                    evaluated_edge_target=edge.target,
                    reason=f"Evaluated outgoing edge from '{completed_node}' to '{edge.target}' as True using recovered state",
                )

        raise ResumptionError(
            f"No valid outgoing edge condition matched from completed node '{completed_node}'. State: {state.data}"
        )


class EventReplayer:
    """Verifies cryptographic hash chains, enforces anti-tamper blocking, and replays state."""

    @staticmethod
    def verify_event_chain(
        events: list[Event],
        expected_last_hash: str | None = None,
        expected_workflow_id: str | None = None,
    ) -> tuple[bool, str | None]:
        """Verifies cryptographic SHA-256 chain integrity across all events in sequence."""
        if expected_last_hash:
            if not events:
                err_msg = (
                    f"Event hash mismatch / Tail truncation: events list is empty "
                    f"but expected_last_hash is '{expected_last_hash}'"
                )
                logger.error(err_msg)
                return False, err_msg
            if events[-1].state_hash != expected_last_hash:
                err_msg = (
                    f"Event hash mismatch / Tail truncation: events[-1].state_hash '{events[-1].state_hash}' "
                    f"does not match expected_last_hash '{expected_last_hash}'"
                )
                logger.error(err_msg)
                return False, err_msg

        if not events:
            return True, None

        expected_prev_hash = ""

        for i, event in enumerate(events):
            expected_seq = i + 1

            # 0. Workflow identity binding check
            if expected_workflow_id and event.workflow_id and event.workflow_id != expected_workflow_id:
                err_msg = (
                    f"Event hash mismatch / Workflow identity mismatch at index {i} (event_id={event.event_id}): "
                    f"expected workflow_id '{expected_workflow_id}', got '{event.workflow_id}'"
                )
                logger.error(err_msg)
                return False, err_msg

            # 1. Monotonic sequence number check
            if event.sequence_num != expected_seq:
                err_msg = (
                    f"Event hash mismatch / Event sequence break at index {i} (event_id={event.event_id}): "
                    f"expected sequence_num {expected_seq}, got {event.sequence_num}"
                )
                logger.error(err_msg)
                return False, err_msg

            # 2. Predecessor hash linkage check
            if i == 0:
                if event.prev_hash != GENESIS_HASH and event.prev_hash != "":
                    err_msg = (
                        f"Event hash mismatch / Genesis predecessor hash mismatch at index 0 (event_id={event.event_id}): "
                        f"expected '{GENESIS_HASH}' or '', got '{event.prev_hash}'"
                    )
                    logger.error(err_msg)
                    return False, err_msg
                expected_prev_hash = event.prev_hash or ""
            else:
                prev_event = events[i - 1]
                if event.prev_hash != prev_event.state_hash:
                    err_msg = (
                        f"Event hash mismatch / Predecessor hash break at index {i} (event_id={event.event_id}): "
                        f"prev_hash '{event.prev_hash}' does not match previous state_hash '{prev_event.state_hash}'"
                    )
                    logger.error(err_msg)
                    return False, err_msg
                expected_prev_hash = prev_event.state_hash or ""

                # 3. Monotonic nanosecond timestamp check
                if event.timestamp_ns < prev_event.timestamp_ns:
                    err_msg = (
                        f"Event hash mismatch / Timestamp regression anomaly at index {i} (event_id={event.event_id}): "
                        f"timestamp_ns {event.timestamp_ns} < previous {prev_event.timestamp_ns}"
                    )
                    logger.error(err_msg)
                    return False, err_msg

            # 4. Canonical hash recomputation
            computed_hash = event.calculate_hash(expected_prev_hash)
            if event.state_hash != computed_hash:
                err_msg = (
                    f"Event hash mismatch / Cryptographic hash mismatch at index {i} (event_id={event.event_id}, seq={event.sequence_num}): "
                    f"Expected {computed_hash}, Actual {event.state_hash}"
                )
                logger.error(err_msg)
                return False, err_msg

        return True, None

    @staticmethod
    def record_security_audit_alert(
        workflow_id: str,
        violation_type: str,
        error_message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Logs a critical security audit record and appends to the physical state audit file."""
        record = {
            "timestamp_ns": time.time_ns(),
            "audit_event": "INTEGRITY_VIOLATION_DETECTED",
            "workflow_id": workflow_id,
            "severity": "CRITICAL",
            "violation_type": violation_type,
            "error_message": error_message,
            "component": "EventReplayer",
            "action_taken": "EXECUTION_HALTED",
            "details": details or {},
        }

        audit_logger.critical("SECURITY_AUDIT: %s", json.dumps(record, ensure_ascii=False))

        state_dir_path = os.environ.get("AGENTMESH_STATE_DIR", "/tmp/agentmesh/checkpoints")
        target_path = Path(state_dir_path).expanduser().resolve()
        repo_root = CheckpointManager._detect_repo_root()
        is_inside_repo = (
            repo_root is not None and (target_path == repo_root or repo_root in target_path.parents)
        )

        if is_inside_repo:
            # Persist to emergency fallback directory outside repo so forensics are retained
            fallback_dir = Path("/tmp/agentmesh/checkpoints").resolve()
            if repo_root is None or (fallback_dir != repo_root and repo_root not in fallback_dir.parents):
                try:
                    fallback_dir.mkdir(parents=True, exist_ok=True)
                    audit_file = fallback_dir / "security_audit.log"
                    with open(audit_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f.flush()
                except OSError as io_err:
                    audit_logger.error("Failed to append fallback security audit file: %s", io_err)

            raise StateSeparationError(
                f"Physical State Separation Violation: Security audit directory '{target_path}' is inside "
                f"the repository root '{repo_root}'. State and audit logs must be stored outside (e.g., /tmp/agentmesh/checkpoints)."
            )

        try:
            target_path.mkdir(parents=True, exist_ok=True)
            audit_file = target_path / "security_audit.log"
            with open(audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
        except OSError as io_err:
            audit_logger.error("Failed to append security audit file: %s", io_err)

    @staticmethod
    def enforce_integrity(
        events: list[Event] | State,
        workflow_id: str = "",
        expected_last_hash: str | None = None,
    ) -> None:
        """Verifies event chain and immediately raises CryptographicIntegrityError on failure."""
        wf_id: str
        target_events: list[Event]
        target_expected_hash: str | None

        if isinstance(events, State):
            wf_id = workflow_id or events.workflow_id
            target_events = events.events
            target_expected_hash = (
                expected_last_hash if expected_last_hash is not None else events.last_hash
            )
            # Inverted anomaly check on State
            if target_events and not target_expected_hash:
                err_msg = (
                    f"State integrity anomaly for workflow '{wf_id}': "
                    f"state contains {len(target_events)} events but last_hash is empty"
                )
                EventReplayer.record_security_audit_alert(
                    workflow_id=wf_id,
                    violation_type="CRYPTOGRAPHIC_TAMPER_DETECTED",
                    error_message=err_msg,
                    details={"workflow_id": wf_id, "events_count": len(target_events)},
                )
                raise CryptographicIntegrityError(
                    message=err_msg,
                    violation_type="CRYPTOGRAPHIC_TAMPER_DETECTED",
                    details={"workflow_id": wf_id, "events_count": len(target_events)},
                )
        else:
            wf_id = workflow_id
            target_events = events
            target_expected_hash = expected_last_hash

        is_valid, err = EventReplayer.verify_event_chain(
            target_events,
            expected_last_hash=target_expected_hash,
            expected_workflow_id=wf_id or None,
        )
        if not is_valid:
            EventReplayer.record_security_audit_alert(
                workflow_id=wf_id,
                violation_type="CRYPTOGRAPHIC_TAMPER_DETECTED",
                error_message=err or "Unknown integrity violation",
                details={
                    "workflow_id": wf_id,
                    "expected_last_hash": target_expected_hash,
                    "events_count": len(target_events),
                },
            )
            raise CryptographicIntegrityError(
                message=f"Event stream tampering or integrity violation detected: {err}",
                violation_type="CRYPTOGRAPHIC_TAMPER_DETECTED",
                expected_hash=target_expected_hash,
                actual_hash=target_events[-1].state_hash if target_events else None,
                details={"workflow_id": wf_id, "error": err},
            )

    @staticmethod
    def verify_and_enforce_chain(
        events: list[Event] | State,
        workflow_id: str = "",
        expected_last_hash: str | None = None,
    ) -> None:
        """Alias for enforce_integrity."""
        EventReplayer.enforce_integrity(
            events, workflow_id=workflow_id, expected_last_hash=expected_last_hash
        )

    @staticmethod
    def find_last_completed_node(state: State) -> str | None:
        """Finds the last successfully executed node name from event history."""
        for event in reversed(state.events):
            if event.event_type == EventType.NODE_COMPLETE and event.node_id:
                return event.node_id
        return None

    @staticmethod
    def resume_from_crash(
        workflow_id: str,
        checkpoint_manager: CheckpointManager,
        graph: Graph | None = None,
    ) -> Any:
        """Loads latest valid checkpoint, strictly enforces cryptographic integrity, and identifies next node.

        Raises:
            CryptographicIntegrityError: Immediately upon detecting any hash mismatch, tail truncation, or broken sequence.
        """
        try:
            state = checkpoint_manager.load_latest_checkpoint(workflow_id)
        except CryptographicIntegrityError as cie:
            EventReplayer.record_security_audit_alert(
                workflow_id=workflow_id,
                violation_type=cie.violation_type,
                error_message=str(cie),
                details=cie.to_audit_dict() if hasattr(cie, "to_audit_dict") else None,
            )
            raise

        if not state:
            if graph is not None:
                return None, None, None
            return None, None

        EventReplayer.verify_and_enforce_chain(state, workflow_id=workflow_id)

        last_node = EventReplayer.find_last_completed_node(state)
        if graph is not None:
            plan = ResumptionPlanner.plan_resumption(graph, state)
            return state, last_node, plan

        return state, last_node
