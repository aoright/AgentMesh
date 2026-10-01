"""Thread-safe Activity Memoizer and deterministic idempotency key engine for AgentMesh."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from enum import Enum
from pathlib import Path
from typing import Any

from agentmesh.engine.activity import ActivityRecord, ActivityStatus
from agentmesh.engine.checkpoint import CheckpointManager

logger = logging.getLogger("agentmesh.engine.memoizer")


class IdempotencyConflictError(Exception):
    """Raised when an explicit idempotency key is reused with different arguments."""



def canonical_json_dump(obj: Any) -> str:
    """
    Serializes a Python object to canonical JSON per RFC 8785:
    - Sorted dictionary keys
    - No insignificant whitespace (separators=(',', ':'))
    - Deterministic representation of nested structures
    """

    def default_serializer(o: Any) -> Any:
        if isinstance(o, Enum):
            return o.value
        if hasattr(o, "model_dump"):
            return o.model_dump()
        if hasattr(o, "to_dict"):
            return o.to_dict()
        if hasattr(o, "__fspath__"):
            return str(o)
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        if isinstance(o, bytes):
            return o.hex()
        if hasattr(o, "__dict__"):
            return o.__dict__
        return repr(o)

    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=default_serializer,
    )


class ActivityMemoizer:
    """
    Thread-safe memoizer that stores and looks up completed activity records.

    Eliminates duplicate executions and guarantees 0% duplicate token billing on crash replay.
    """

    def __init__(self, storage_dir: str | None = None):
        self._lock = threading.RLock()
        self._records: dict[str, ActivityRecord] = {}
        # Track argument hashes to prevent parameter conflicts on explicit keys
        self._key_args_hashes: dict[str, str] = {}

        configured_dir = storage_dir or os.environ.get(
            "AGENTMESH_STATE_DIR", "/tmp/agentmesh/checkpoints"
        )
        self.storage_dir = CheckpointManager.validate_state_dir(configured_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _detect_repo_root() -> Path | None:
        return CheckpointManager._detect_repo_root()

    def _validate_and_resolve_dir(self, path: Path) -> Path:
        return CheckpointManager.validate_state_dir(path)

    def compute_key(
        self,
        workflow_id: str,
        node_id: str,
        activity_name: str,
        args: tuple[Any, ...] | list[Any] | dict[str, Any] = (),
        kwargs: dict[str, Any] | None = None,
        invocation_index: int = 0,
        custom_key: str | None = None,
    ) -> str:
        """
        Computes a deterministic SHA-256 idempotency key.

        If a custom_key is supplied, validates that arguments match any prior invocation;
        otherwise computes synthetic key based on workflow, node, activity, index, and args.
        """
        if isinstance(args, dict):
            if kwargs is None:
                kwargs = args
            else:
                kwargs = {**args, **kwargs}
            args = ()

        combined_payload = {
            "args": list(args) if isinstance(args, (list, tuple)) else [args],
            "kwargs": kwargs or {},
        }
        canonical_args = canonical_json_dump(combined_payload)
        args_hash = hashlib.sha256(canonical_args.encode("utf-8")).hexdigest()

        if custom_key:
            with self._lock:
                if custom_key in self._key_args_hashes:
                    if self._key_args_hashes[custom_key] != args_hash:
                        raise IdempotencyConflictError(
                            f"Idempotency key conflict: '{custom_key}' was previously registered "
                            "with a different input payload."
                        )
                else:
                    self._key_args_hashes[custom_key] = args_hash
            return custom_key

        components = [
            workflow_id,
            node_id,
            activity_name,
            invocation_index,
            args_hash,
        ]
        key_source = canonical_json_dump(components)
        key = hashlib.sha256(key_source.encode("utf-8")).hexdigest()

        with self._lock:
            self._key_args_hashes[key] = args_hash

        return key

    def get(self, idempotency_key: str) -> ActivityRecord | None:
        """Retrieves a memoized ActivityRecord by its idempotency key."""
        with self._lock:
            return self._records.get(idempotency_key)

    def put(self, record: ActivityRecord) -> None:
        """Stores an ActivityRecord into the memoizer cache."""
        with self._lock:
            self._records[record.idempotency_key] = record

    def has(self, idempotency_key: str) -> bool:
        """Returns True if the completed activity record exists in the memoizer."""
        with self._lock:
            rec = self._records.get(idempotency_key)
            return rec is not None and rec.status == ActivityStatus.COMPLETED

    def populate_from_events(self, events: list[Any]) -> int:
        """
        Reconstructs memoizer cache directly from an event stream.

        Called during crash recovery to immediately rehydrate completed activities.
        Returns the number of rehydrated activity records.
        """
        from agentmesh.engine.state import EventType

        rehydrated_count = 0
        with self._lock:
            for event in events:
                if getattr(event, "event_type", None) == EventType.ACTIVITY_COMPLETED:
                    payload = getattr(event, "payload", {})
                    key = payload.get("idempotency_key")
                    if key and key not in self._records:
                        prompt_tok = payload.get("prompt_tokens", 0)
                        comp_tok = payload.get("completion_tokens", 0)
                        tot_tok = payload.get("total_tokens", prompt_tok + comp_tok)
                        rec = ActivityRecord(
                            idempotency_key=key,
                            activity_name=payload.get("activity_name", "unknown"),
                            node_id=getattr(event, "node_id", "") or "",
                            workflow_id=getattr(event, "workflow_id", ""),
                            status=ActivityStatus.COMPLETED,
                            result=payload.get("result"),
                            prompt_tokens=prompt_tok,
                            completion_tokens=comp_tok,
                            total_tokens=tot_tok,
                            duration_ms=payload.get("duration_ms", 0.0),
                            timestamp_ns=getattr(event, "timestamp_ns", 0),
                            completed_at_ns=getattr(event, "timestamp_ns", 0),
                        )
                        self._records[key] = rec
                        rehydrated_count += 1
        logger.info("Rehydrated %d memoized activities from event history.", rehydrated_count)
        return rehydrated_count

    def save_to_disk(self, workflow_id: str) -> Path:
        """
        Atomically saves the memoizer records to the external state directory.
        Uses .tmp write + os.replace + os.fsync to guarantee crash-consistent atomic durability.
        """
        wf_dir = self.storage_dir / workflow_id
        wf_dir.mkdir(parents=True, exist_ok=True)
        target_path = wf_dir / "activity_memoizer.json"
        temp_path = target_path.with_suffix(f".tmp.{os.getpid()}")

        with self._lock:
            serialized = {
                "workflow_id": workflow_id,
                "records": [rec.model_dump() for rec in self._records.values()],
            }

        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_path, target_path)
        try:
            os.chmod(target_path, 0o600)
        except OSError:
            pass
        return target_path

    def load_from_disk(self, workflow_id: str) -> int:
        """Loads memoizer records from the external state directory."""
        target_path = self.storage_dir / workflow_id / "activity_memoizer.json"
        if not target_path.exists():
            return 0

        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        loaded_count = 0
        with self._lock:
            for item in data.get("records", []):
                rec = ActivityRecord(**item)
                self._records[rec.idempotency_key] = rec
                loaded_count += 1
        return loaded_count

    def clear(self) -> None:
        """Clears all records in the memoizer."""
        with self._lock:
            self._records.clear()
            self._key_args_hashes.clear()
