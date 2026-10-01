"""Pluggable snapshot and checkpoint manager for AgentMesh.

Complies with enterprise state separation: state snapshots and WAL are
strictly stored in a dedicated runtime state directory, isolated from code.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from agentmesh.engine.state import State, Event


class CheckpointManager:
    """Manages saving, loading, and rolling back state checkpoints."""

    def __init__(self, base_dir: Optional[str] = None):
        # Default to /tmp/agentmesh/checkpoints or custom state directory
        self.base_dir = Path(base_dir or os.environ.get("AGENTMESH_STATE_DIR", "/tmp/agentmesh/checkpoints"))
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_workflow_dir(self, workflow_id: str) -> Path:
        wf_dir = self.base_dir / workflow_id
        wf_dir.mkdir(parents=True, exist_ok=True)
        return wf_dir

    def save_checkpoint(self, state: State, label: str = "") -> str:
        """Persist a full snapshot of the state to disk."""
        wf_dir = self._get_workflow_dir(state.workflow_id)
        ckpt_filename = f"ckpt_{state.version:06d}_{label or 'auto'}.json"
        ckpt_path = wf_dir / ckpt_filename

        serialized_events = [e.model_dump() for e in state.events]
        payload = {
            "workflow_id": state.workflow_id,
            "version": state.version,
            "data": state.data,
            "last_hash": state.last_hash,
            "events": serialized_events,
            "label": label,
        }

        temp_path = ckpt_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        temp_path.replace(ckpt_path)

        return str(ckpt_path)

    def load_latest_checkpoint(self, workflow_id: str) -> Optional[State]:
        """Load the latest checkpoint for a given workflow ID."""
        wf_dir = self._get_workflow_dir(workflow_id)
        checkpoints = sorted(wf_dir.glob("ckpt_*.json"))
        if not checkpoints:
            return None

        latest_path = checkpoints[-1]
        return self.load_from_file(str(latest_path))

    def load_from_file(self, file_path: str) -> State:
        """Load state from a specific checkpoint file."""
        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        events = [Event(**e) for e in raw.get("events", [])]
        state = State(
            workflow_id=raw["workflow_id"],
            version=raw["version"],
            data=raw["data"],
            events=events,
            last_hash=raw.get("last_hash", ""),
        )
        return state

    def list_checkpoints(self, workflow_id: str) -> List[str]:
        """List all checkpoint files for a workflow."""
        wf_dir = self._get_workflow_dir(workflow_id)
        return [str(p) for p in sorted(wf_dir.glob("ckpt_*.json"))]
