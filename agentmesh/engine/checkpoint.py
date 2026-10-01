"""Pluggable snapshot and checkpoint manager for AgentMesh.

Complies with enterprise state separation: state snapshots and WAL are
strictly stored in a dedicated runtime state directory, isolated from code.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from agentmesh.engine.state import CryptographicIntegrityError, Event, State

logger = logging.getLogger("agentmesh.engine.checkpoint")


class StateSeparationError(Exception):
    """Raised when runtime state storage violates physical isolation from source code."""



class CheckpointManager:
    """Manages saving, loading, and pruning state checkpoints with atomic durability."""

    def __init__(self, base_dir: str | None = None, keep_last_n: int | None = None):
        target_dir = base_dir or os.environ.get(
            "AGENTMESH_STATE_DIR", "/tmp/agentmesh/checkpoints"
        )
        self.base_dir = self._validate_and_resolve_dir(Path(target_dir))
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n

    @staticmethod
    def _detect_repo_root() -> Path | None:
        """Locates the repository root by finding pyproject.toml or .git relative to this file."""
        current = Path(__file__).resolve().parent
        for parent in [current, *current.parents]:
            if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
                return parent
        return None

    @classmethod
    def validate_state_dir(cls, path: Path | str) -> Path:
        """Enforces that the state directory resides strictly outside the repository root.

        Resolves symlinks and relative components. Raises StateSeparationError if the target
        is equal to or contained within the detected repository root.
        """
        resolved = Path(path).expanduser().resolve()
        repo_root = cls._detect_repo_root()
        if repo_root is not None and (resolved == repo_root or repo_root in resolved.parents):
            raise StateSeparationError(
                f"Physical State Separation Violation: State directory '{resolved}' is inside "
                f"the repository root '{repo_root}'. State must be stored outside (e.g., /tmp/agentmesh/checkpoints)."
            )
        return resolved

    def _validate_and_resolve_dir(self, path: Path) -> Path:
        """Backward-compatible instance wrapper for validate_state_dir."""
        return self.validate_state_dir(path)

    def _get_workflow_dir(self, workflow_id: str) -> Path:
        wf_dir = self.base_dir / workflow_id
        wf_dir.mkdir(parents=True, exist_ok=True)
        return wf_dir

    def save_checkpoint(self, state: State, label: str = "") -> str:
        """Persist a full snapshot of the state to disk with atomic durability and fsync."""
        wf_dir = self._get_workflow_dir(state.workflow_id)
        ckpt_filename = f"ckpt_{state.version:06d}_{label or 'auto'}.json"
        ckpt_path = wf_dir / ckpt_filename

        serialized_events = [e.model_dump() for e in state.events]
        payload: dict[str, Any] = {
            "workflow_id": state.workflow_id,
            "run_id": state.run_id,
            "version": state.version,
            "data": state.data,
            "last_hash": state.last_hash,
            "events_count": len(state.events),
            "events": serialized_events,
            "label": label,
        }

        temp_path = ckpt_path.with_suffix(f".tmp.{os.getpid()}_{time.time_ns()}")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_path, ckpt_path)
        try:
            os.chmod(ckpt_path, 0o600)
        except OSError:
            pass

        if self.keep_last_n is not None and self.keep_last_n > 0:
            self.prune_checkpoints(state.workflow_id, keep_last=self.keep_last_n)

        return str(ckpt_path)

    def prune_checkpoints(self, workflow_id: str, keep_last: int = 5) -> int:
        """Retains only the latest N checkpoints for a workflow, safely deleting older files."""
        wf_dir = self._get_workflow_dir(workflow_id)
        checkpoints = sorted(wf_dir.glob("ckpt_*.json"))
        if len(checkpoints) <= keep_last:
            return 0

        to_delete = checkpoints[:-keep_last]
        deleted_count = 0
        for cp in to_delete:
            try:
                cp.unlink()
                deleted_count += 1
            except OSError as e:
                logger.warning("Failed to prune checkpoint %s: %s", cp, e)
        return deleted_count

    def load_latest_checkpoint(self, workflow_id: str) -> State | None:
        """Load the latest checkpoint for a given workflow ID."""
        wf_dir = self._get_workflow_dir(workflow_id)
        checkpoints = sorted(wf_dir.glob("ckpt_*.json"))
        if not checkpoints:
            return None

        latest_path = checkpoints[-1]
        return self.load_from_file(str(latest_path))

    def load_from_file(self, file_path: str) -> State:
        """Load state from a specific checkpoint file and enforce manifest invariants."""
        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        events = [Event(**e) for e in raw.get("events", [])]
        expected_count = raw.get("events_count")
        if expected_count is not None:
            if not isinstance(expected_count, int) or expected_count < 0:
                raise CryptographicIntegrityError(
                    f"Malformed checkpoint '{file_path}': 'events_count' must be a non-negative integer, got {expected_count}",
                    violation_type="CRYPTOGRAPHIC_TAMPER_DETECTED",
                    details={"file_path": file_path, "events_count": expected_count},
                )
            if len(events) != expected_count:
                raise CryptographicIntegrityError(
                    f"Checkpoint event count mismatch in '{file_path}': "
                    f"manifest expects {expected_count} events, but found {len(events)} events (tail truncation detected)",
                    violation_type="CRYPTOGRAPHIC_TAMPER_DETECTED",
                    details={
                        "file_path": file_path,
                        "expected_events_count": expected_count,
                        "actual_events_count": len(events),
                    },
                )

        saved_last_hash = raw.get("last_hash", "")
        if saved_last_hash:
            if not events:
                raise CryptographicIntegrityError(
                    f"Checkpoint tail truncation detected in '{file_path}': "
                    f"checkpoint records last_hash '{saved_last_hash}' but events list is empty",
                    violation_type="CRYPTOGRAPHIC_TAMPER_DETECTED",
                    expected_hash=saved_last_hash,
                    actual_hash="",
                    details={"file_path": file_path, "events_count": 0},
                )
            if events[-1].state_hash != saved_last_hash:
                raise CryptographicIntegrityError(
                    f"Checkpoint tail event hash mismatch in '{file_path}': "
                    f"events[-1].state_hash '{events[-1].state_hash}' does not match last_hash '{saved_last_hash}' (tail truncation detected)",
                    violation_type="CRYPTOGRAPHIC_TAMPER_DETECTED",
                    expected_hash=saved_last_hash,
                    actual_hash=events[-1].state_hash,
                    details={
                        "file_path": file_path,
                        "events_count": len(events),
                        "last_event_id": events[-1].event_id,
                    },
                )
        elif len(events) > 0:
            raise CryptographicIntegrityError(
                f"Checkpoint integrity anomaly in '{file_path}': "
                f"checkpoint contains {len(events)} events but last_hash is empty",
                violation_type="CRYPTOGRAPHIC_TAMPER_DETECTED",
                details={"file_path": file_path, "events_count": len(events)},
            )

        state = State(
            workflow_id=raw["workflow_id"],
            run_id=raw.get("run_id", ""),
            version=raw["version"],
            data=raw["data"],
            events=events,
            last_hash=saved_last_hash,
        )
        return state


    def list_checkpoints(self, workflow_id: str) -> list[str]:
        """List all checkpoint files for a workflow."""
        wf_dir = self._get_workflow_dir(workflow_id)
        return [str(p) for p in sorted(wf_dir.glob("ckpt_*.json"))]
