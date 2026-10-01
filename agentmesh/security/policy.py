"""Capability policy models and fine-grained permissions for AgentMesh."""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger("agentmesh.security.policy")


class FileAccessMode(str, Enum):
    """File access mode permissions."""

    READ = "r"
    WRITE = "w"
    APPEND = "a"
    EXECUTE = "x"
    READ_WRITE = "rw"


class PathRule(BaseModel):
    """Fine-grained permission rule for a specific path prefix."""

    path: str
    allow_read: bool = True
    allow_write: bool = False
    allow_append: bool = False
    allow_execute: bool = False
    recursive: bool = True


class CommandRule(BaseModel):
    """Fine-grained constraint for an allowed executable."""

    executable: str
    allowed_subcommands: set[str] = Field(default_factory=set)
    allowed_flags: set[str] = Field(default_factory=set)
    blocked_flags: set[str] = Field(default_factory=set)
    max_args: int = 100


class CapabilityPolicy(BaseModel):
    """Fine-grained capability permissions assigned to an Agent or Tool execution context."""

    policy_name: str = "default_restricted"

    # Path isolation boundaries
    allowed_read_paths: list[str] = Field(default_factory=list)
    allowed_write_paths: list[str] = Field(default_factory=list)
    allowed_execute_paths: list[str] = Field(default_factory=list)

    # Command whitelist
    allowed_commands: set[str] = Field(default_factory=set)

    # Network capabilities (M3-2)
    allow_network: bool = False
    allowed_domains: list[str] = Field(
        default_factory=list,
        description="Allowed domain rules (exact e.g. 'api.github.com' or wildcard '*.example.com'). If empty and allow_network=True, all public non-private domains are permitted.",
    )
    allowed_schemes: set[str] = Field(
        default_factory=lambda: {"http", "https"},
        description="Allowed URL protocol schemes. Strictly http and https by default.",
    )
    allowed_ports: set[int] | None = Field(
        default=None,
        description="Allowed destination ports. None allows standard ports 1-65535.",
    )

    # SSRF & IP Protection Switches
    block_loopback: bool = True
    block_private_ips: bool = True
    block_cloud_metadata: bool = True
    block_carrier_grade_nat: bool = True
    resolve_dns: bool = True
    allow_unresolved_domains: bool = False
    custom_blocked_cidrs: list[str] = Field(
        default_factory=list,
        description="Additional custom CIDR ranges to block (e.g. ['198.18.0.0/15']).",
    )

    # Resource quotas
    max_memory_mb: int = 512
    timeout_seconds: float = 30.0

    # Advanced security controls
    strict_mode: bool = True
    allow_symlinks_within_root: bool = True
    path_rules: list[PathRule] = Field(default_factory=list)
    command_rules: list[CommandRule] = Field(default_factory=list)

    # Shell metacharacters strictly banned in commands
    banned_shell_metachars: set[str] = Field(
        default_factory=lambda: {
            ";",
            "&",
            "|",
            "`",
            "$",
            "(",
            ")",
            "<",
            ">",
            "\n",
            "\r",
            "\x00",
        }
    )

    def is_executable_whitelisted(
        self, executable_name: str, canonical_path: Path | None = None
    ) -> bool:
        """Checks if executable base name or full path is permitted.

        Security constraints:
        - If executable contains path separators ('/' or '\\'), bare command name
          matching in allowed_commands is strictly rejected.
        - If allowed_execute_paths is defined, canonical path must reside within
          allowed_execute_paths.
        - If allowed_execute_paths is not defined, the executable must match
          an exact path in allowed_commands or command_rules.
        """
        if not self.allowed_commands and not self.command_rules:
            return False

        clean_name = executable_name.strip()
        has_sep = "/" in clean_name or "\\" in clean_name
        base_name = Path(clean_name.replace("\\", "/")).name

        # Resolve canonical path if not already provided and path contains separators
        resolved: Path | None = canonical_path
        if has_sep and resolved is None:
            try:
                resolved = Path(clean_name.replace("\\", "/")).expanduser().resolve()
            except (OSError, RuntimeError):
                return False

        resolved_str = str(resolved) if resolved is not None else None

        if not has_sep:
            # Bare command name (e.g. 'echo')
            if clean_name in self.allowed_commands:
                return True
            for rule in self.command_rules:
                if rule.executable == clean_name:
                    return True
            return False

        # Has path separators ('/' or '\\')
        if self.allowed_execute_paths:
            if resolved is None:
                return False
            # Verify containment in allowed_execute_paths
            is_contained = False
            for root in self.allowed_execute_paths:
                try:
                    root_path = Path(root).expanduser().resolve()
                    if resolved == root_path or resolved.is_relative_to(root_path):
                        is_contained = True
                        break
                except OSError:
                    continue
            if not is_contained:
                return False

            # If contained in allowed_execute_paths, check allowed_commands or command_rules
            if (
                clean_name in self.allowed_commands
                or (resolved_str and resolved_str in self.allowed_commands)
                or base_name in self.allowed_commands
            ):
                return True

            for rule in self.command_rules:
                if (
                    rule.executable == clean_name
                    or (resolved_str and rule.executable == resolved_str)
                    or rule.executable == base_name
                ):
                    return True
            return False
        else:
            # allowed_execute_paths is NOT configured
            # Reject matching bare name in allowed_commands
            # Require exact full path in allowed_commands or command_rules
            if clean_name in self.allowed_commands or (
                resolved_str and resolved_str in self.allowed_commands
            ):
                return True

            for rule in self.command_rules:
                if rule.executable == clean_name or (
                    resolved_str and rule.executable == resolved_str
                ):
                    return True

            return False

    def get_matching_command_rules(
        self, executable_name: str, canonical_path: Path | None = None
    ) -> list[CommandRule]:
        """Finds all CommandRule configurations matching the executable."""
        if not self.command_rules:
            return []

        clean_name = executable_name.strip()
        has_sep = "/" in clean_name or "\\" in clean_name
        base_name = Path(clean_name.replace("\\", "/")).name
        resolved_str = str(canonical_path) if canonical_path is not None else None

        matching: list[CommandRule] = []
        for rule in self.command_rules:
            rule_exe = rule.executable.strip()
            # Exact path, exact bare name match, or bare command matching base_name
            if (
                rule_exe == clean_name
                or (resolved_str and rule_exe == resolved_str)
                or (not has_sep and rule_exe == base_name)
            ):
                matching.append(rule)
            elif has_sep and self.allowed_execute_paths and canonical_path is not None:
                # Contained in allowed_execute_paths -> can match base_name
                for root in self.allowed_execute_paths:
                    try:
                        root_path = Path(root).expanduser().resolve()
                        if canonical_path == root_path or canonical_path.is_relative_to(
                            root_path
                        ):
                            if rule_exe == base_name:
                                matching.append(rule)
                            break
                    except OSError:
                        continue
        return matching
