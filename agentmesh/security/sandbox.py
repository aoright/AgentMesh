"""Capability-based security sandbox for enterprise Agent runtime."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Set
from pydantic import BaseModel, Field

logger = logging.getLogger("agentmesh.security.sandbox")


class PermissionDeniedError(Exception):
    """Raised when an Agent attempts an unauthorized operation violating sandbox policies."""
    pass


class CapabilityPolicy(BaseModel):
    """Fine-grained capability permissions assigned to an Agent."""

    policy_name: str = "default_restricted"
    allow_network: bool = False
    allowed_read_paths: List[str] = Field(default_factory=list)
    allowed_write_paths: List[str] = Field(default_factory=list)
    allowed_commands: Set[str] = Field(default_factory=set)
    max_memory_mb: int = 512
    timeout_seconds: float = 30.0


class SecuritySandbox:
    """Enforces capability policies on tool calls and agent actions."""

    def __init__(self, policy: Optional[CapabilityPolicy] = None):
        self.policy = policy or CapabilityPolicy()

    def validate_file_access(self, target_path: str, mode: str = "r") -> str:
        """Validates that a path is within allowed read/write boundaries."""
        resolved = str(Path(target_path).resolve())
        is_write = "w" in mode or "a" in mode or "+" in mode

        allowed_list = self.policy.allowed_write_paths if is_write else self.policy.allowed_read_paths

        # Check path containment
        is_allowed = any(
            resolved == str(Path(p).resolve()) or resolved.startswith(str(Path(p).resolve()) + os.sep)
            for p in allowed_list
        )

        if not is_allowed:
            op = "write" if is_write else "read"
            msg = f"SecuritySandbox: Unauthorized file {op} access blocked for '{resolved}'. Policy: {self.policy.policy_name}"
            logger.warning(msg)
            raise PermissionDeniedError(msg)

        return resolved

    def validate_command(self, cmd: str) -> str:
        """Validates shell/CLI command against the allowed whitelist."""
        base_cmd = cmd.strip().split()[0] if cmd.strip() else ""
        if base_cmd not in self.policy.allowed_commands:
            msg = f"SecuritySandbox: Execution of command '{base_cmd}' blocked by policy '{self.policy.policy_name}'."
            logger.warning(msg)
            raise PermissionDeniedError(msg)
        return cmd

    def validate_network(self, url: str) -> str:
        """Validates external network access requests."""
        if not self.policy.allow_network:
            msg = f"SecuritySandbox: Outbound network request to '{url}' blocked. Network access disabled."
            logger.warning(msg)
            raise PermissionDeniedError(msg)
        return url
