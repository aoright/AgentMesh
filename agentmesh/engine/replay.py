"""Deterministic Event Replay and Crash Recovery Engine for AgentMesh."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
from agentmesh.engine.state import State, Event, EventType
from agentmesh.engine.checkpoint import CheckpointManager

logger = logging.getLogger("agentmesh.engine.replay")


class EventReplayer:
    """Verifies hash chains and replays event logs for deterministic state reconstruction."""

    @staticmethod
    def verify_event_chain(events: List[Event]) -> Tuple[bool, Optional[str]]:
        """Verifies cryptographic SHA256 chain integrity of the event stream."""
        prev_hash = ""
        for i, event in enumerate(events):
            expected_hash = event.calculate_hash(prev_hash)
            if event.state_hash != expected_hash:
                err_msg = f"Event hash mismatch at index {i} (id={event.event_id}). Expected: {expected_hash}, Actual: {event.state_hash}"
                logger.error(err_msg)
                return False, err_msg
            prev_hash = event.state_hash
        return True, None

    @staticmethod
    def find_last_completed_node(state: State) -> Optional[str]:
        """Finds the last successfully executed node name from event history."""
        for event in reversed(state.events):
            if event.event_type == EventType.NODE_COMPLETE and event.node_id:
                return event.node_id
        return None

    @staticmethod
    def resume_from_crash(
        workflow_id: str,
        checkpoint_manager: CheckpointManager,
    ) -> Tuple[Optional[State], Optional[str]]:
        """Loads latest valid checkpoint and identifies the next node to resume from."""
        state = checkpoint_manager.load_latest_checkpoint(workflow_id)
        if not state:
            return None, None

        is_valid, err = EventReplayer.verify_event_chain(state.events)
        if not is_valid:
            logger.warning("Event chain validation failed during resume: %s", err)

        last_node = EventReplayer.find_last_completed_node(state)
        return state, last_node
