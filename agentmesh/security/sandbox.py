"""Capability-based security sandbox for enterprise Agent runtime."""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import shlex
import socket
import urllib.parse
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from agentmesh.security.policy import CapabilityPolicy, FileAccessMode

logger = logging.getLogger("agentmesh.security.sandbox")

# Register Path serialization support in standard json library for MCP/mesh interoperability
_orig_json_default = json.JSONEncoder.default


def _json_encoder_path_default(self: json.JSONEncoder, o: Any) -> Any:
    if isinstance(o, (Path, os.PathLike)):
        return str(o)
    return _orig_json_default(self, o)


json.JSONEncoder.default = _json_encoder_path_default  # type: ignore[method-assign]


class PermissionDeniedError(Exception):
    """Raised when an Agent attempts an unauthorized operation violating sandbox policies."""


class SafePath(Path):
    """Subclass of pathlib.Path providing backwards-compatible string helper methods."""

    _flavour = getattr(type(Path()), "_flavour", None)

    def endswith(self, suffix: str) -> bool:
        """Backwards compatibility for string-based endswith checks."""
        return str(self).endswith(suffix)

    def startswith(self, prefix: str) -> bool:
        """Backwards compatibility for string-based startswith checks."""
        return str(self).startswith(prefix)


class CommandVector(list[str]):
    """Argument vector list representing validated command, supporting legacy string comparison."""

    def __init__(self, items: list[str], raw_cmd: str | None = None) -> None:
        super().__init__(items)
        self.raw_cmd = raw_cmd if raw_cmd is not None else " ".join(items)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.raw_cmd == other or " ".join(self) == other
        return super().__eq__(other)

    def __str__(self) -> str:
        return self.raw_cmd

    def __repr__(self) -> str:
        return f"CommandVector({super().__repr__()})"


# Categorized Blocked Subnets for Fine-Grained SSRF Defense

# RFC 1918 Private IPv4 Subnets
PRIVATE_IPV4_SUBNETS: list[ipaddress.IPv4Network] = [
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("172.16.0.0/12"),
    ipaddress.IPv4Network("192.168.0.0/16"),
]

# Cloud Metadata and Link-Local IPv4 Subnets
CLOUD_METADATA_IPV4_SUBNETS: list[ipaddress.IPv4Network] = [
    ipaddress.IPv4Network("169.254.0.0/16"),  # RFC 3927 Link-Local / Cloud Metadata
    ipaddress.IPv4Network("168.63.129.16/32"),  # Azure VM Wire Server
]

# RFC 6598 Carrier-Grade NAT Subnets
CARRIER_GRADE_NAT_IPV4_SUBNETS: list[ipaddress.IPv4Network] = [
    ipaddress.IPv4Network("100.64.0.0/10"),
]

# Loopback Subnets
LOOPBACK_IPV4_SUBNETS: list[ipaddress.IPv4Network] = [
    ipaddress.IPv4Network("127.0.0.0/8"),
    ipaddress.IPv4Network("0.0.0.0/8"),
]

# Reserved / Non-routable IPv4 Subnets (always blocked for SSRF protection)
RESERVED_IPV4_SUBNETS: list[ipaddress.IPv4Network] = [
    ipaddress.IPv4Network("192.0.0.0/24"),  # RFC 6890 IETF Protocol Assignments
    ipaddress.IPv4Network("192.0.2.0/24"),  # RFC 5737 TEST-NET-1
    ipaddress.IPv4Network("198.51.100.0/24"),  # RFC 5737 TEST-NET-2
    ipaddress.IPv4Network("203.0.113.0/24"),  # RFC 5737 TEST-NET-3
    ipaddress.IPv4Network("224.0.0.0/4"),  # RFC 5771 Multicast
    ipaddress.IPv4Network("240.0.0.0/4"),  # RFC 1112 Reserved / Class E
    ipaddress.IPv4Network("255.255.255.255/32"),  # RFC 919 Limited Broadcast
]

# Monolithic IPv4 blocked list for backwards compatibility
BLOCKED_IPV4_SUBNETS: list[ipaddress.IPv4Network] = [
    *LOOPBACK_IPV4_SUBNETS,
    *PRIVATE_IPV4_SUBNETS,
    *CARRIER_GRADE_NAT_IPV4_SUBNETS,
    *CLOUD_METADATA_IPV4_SUBNETS,
    *RESERVED_IPV4_SUBNETS,
]

# Private / Site-Local / ULA IPv6 Subnets
PRIVATE_IPV6_SUBNETS: list[ipaddress.IPv6Network] = [
    ipaddress.IPv6Network("fc00::/7"),  # RFC 4193 Unique Local Address (ULA)
    ipaddress.IPv6Network("fec0::/10"),  # RFC 3879 Site-Local (deprecated)
]

# Cloud Metadata / Link-Local IPv6 Subnets
CLOUD_METADATA_IPV6_SUBNETS: list[ipaddress.IPv6Network] = [
    ipaddress.IPv6Network("fe80::/10"),  # RFC 4291 Link-Local
]

# Loopback / Unspecified IPv6 Subnets
LOOPBACK_IPV6_SUBNETS: list[ipaddress.IPv6Network] = [
    ipaddress.IPv6Network("::1/128"),
    ipaddress.IPv6Network("::/128"),
]

# Reserved / Documentation / Multicast IPv6 Subnets
RESERVED_IPV6_SUBNETS: list[ipaddress.IPv6Network] = [
    ipaddress.IPv6Network("2001:db8::/32"),  # RFC 3849 Documentation
    ipaddress.IPv6Network("ff00::/8"),  # RFC 4291 Multicast
]

# IPv4-compatible and NAT64 translation prefixes
IPV4_COMPATIBLE_IPV6_PREFIX: ipaddress.IPv6Network = ipaddress.IPv6Network("::/96")
NAT64_WELL_KNOWN_PREFIX: ipaddress.IPv6Network = ipaddress.IPv6Network("64:ff9b::/96")

# Monolithic IPv6 blocked list for backwards compatibility (including fec0::/10)
BLOCKED_IPV6_SUBNETS: list[ipaddress.IPv6Network] = [
    *LOOPBACK_IPV6_SUBNETS,
    *CLOUD_METADATA_IPV6_SUBNETS,
    *PRIVATE_IPV6_SUBNETS,
    *RESERVED_IPV6_SUBNETS,
]

BLOCKED_METADATA_HOSTNAMES: set[str] = {
    "localhost",
    "metadata.google.internal",
    "metadata.internal",
    "metadata",
    "instance-data",
}


def _default_dns_resolver(host: str) -> list[str]:
    """Resolves hostname to unique IP addresses using socket.getaddrinfo."""
    try:
        addr_info = socket.getaddrinfo(host, None)
        return list({str(info[4][0]) for info in addr_info})
    except socket.gaierror:
        return []


class SecuritySandbox:
    """Enforces capability policies on tool calls, files, commands, and network."""

    def __init__(
        self,
        policy: CapabilityPolicy | None = None,
        dns_resolver: Callable[[str], list[str]] | None = None,
    ) -> None:
        self.policy = policy or CapabilityPolicy()
        self._dns_resolver = dns_resolver or _default_dns_resolver

        # Compile custom blocked subnets if configured
        self._custom_subnets: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
        for cidr in self.policy.custom_blocked_cidrs:
            try:
                self._custom_subnets.append(ipaddress.ip_network(cidr))
            except ValueError as e:
                logger.error("SecuritySandbox: Invalid custom CIDR '%s': %s", cidr, e)

    # --- Feature 11: Safe Canonical Root Path Confinement ---

    def validate_file_access(self, target_path: str, mode: str = "r") -> SafePath:
        """Validates that a path is strictly confined within allowed root boundaries.

        Args:
            target_path: File system path (relative or absolute, possibly containing traversals).
            mode: Access mode string ('r', 'w', 'a', 'x', 'rw', 'rb', 'wb', etc.).

        Returns:
            SafePath: Canonicalized Path object confined within permitted roots.

        Raises:
            PermissionDeniedError: If traversal, null bytes, symlink escape, or mode violation occurs.
        """
        if not target_path or not str(target_path).strip():
            msg = f"SecuritySandbox: Empty path rejected. Policy: {self.policy.policy_name}"
            logger.warning(msg)
            raise PermissionDeniedError(msg)

        raw_str = str(target_path).strip()

        # Step 1: Detect null bytes and URL-encoded null bytes / traversal attacks
        self._check_null_bytes_and_encodings(raw_str)

        # Step 2: Canonicalize and resolve symlinks safely
        resolved_path = self._canonicalize_path(raw_str)

        # Step 3: Determine required operation modes
        is_read, is_write, is_append, is_execute = self._parse_file_mode(mode)

        # Step 4: Verify mode-specific path confinement
        if is_execute:
            self._verify_containment(
                resolved_path, self.policy.allowed_execute_paths, op_name="execute"
            )

        if is_write or is_append:
            self._verify_containment(
                resolved_path, self.policy.allowed_write_paths, op_name="write"
            )

        if is_read:
            self._verify_containment(
                resolved_path, self.policy.allowed_read_paths, op_name="read"
            )

        return SafePath(resolved_path)

    def _check_null_bytes_and_encodings(self, raw_str: str) -> None:
        """Blocks null bytes (%00, \\x00) and encoded traversal tricks."""
        if "\x00" in raw_str:
            msg = f"SecuritySandbox: Null-byte injection detected in '{raw_str}'."
            logger.warning(msg)
            raise PermissionDeniedError(msg)

        lower_str = raw_str.lower()
        if "%00" in lower_str or "%2500" in lower_str or "\\x00" in lower_str:
            msg = (
                f"SecuritySandbox: Encoded null-byte injection detected in '{raw_str}'."
            )
            logger.warning(msg)
            raise PermissionDeniedError(msg)

        # URL decode and verify no null bytes emerge
        decoded = raw_str
        for _ in range(3):
            new_decoded = urllib.parse.unquote(decoded)
            if new_decoded == decoded:
                break
            decoded = new_decoded
            if "\x00" in decoded:
                msg = f"SecuritySandbox: Multi-stage encoded null-byte detected in '{raw_str}'."
                logger.warning(msg)
                raise PermissionDeniedError(msg)

    def _canonicalize_path(self, raw_str: str) -> Path:
        """Resolves target path to absolute canonical path, resolving symlinks safely."""
        # Unquote URL-encoded path characters (%2e%2e, %2f, etc.)
        decoded_str = urllib.parse.unquote(raw_str)
        try:
            target = Path(decoded_str).expanduser()
            return target.resolve()
        except RuntimeError as e:
            msg = f"SecuritySandbox: Symlink loop or recursion error in path '{raw_str}': {e}"
            logger.warning(msg)
            raise PermissionDeniedError(msg) from e
        except OSError as e:
            msg = f"SecuritySandbox: Unable to canonicalize path '{raw_str}': {e}"
            logger.warning(msg)
            raise PermissionDeniedError(msg) from e

    def _parse_file_mode(self, mode: str) -> tuple[bool, bool, bool, bool]:
        """Parses mode string into (read, write, append, execute) booleans."""
        normalized = mode.lower().strip()
        is_execute = "x" in normalized or normalized in {
            "exec",
            "execute",
            FileAccessMode.EXECUTE.value,
        }
        is_write = "w" in normalized or "+" in normalized
        is_append = "a" in normalized
        is_read = (
            "r" in normalized
            or normalized == ""
            or (not is_write and not is_append and not is_execute)
        )

        return is_read, is_write, is_append, is_execute

    def _verify_containment(
        self, resolved_path: Path, allowed_roots: list[str], op_name: str
    ) -> None:
        """Verifies that resolved_path is inside at least one of allowed_roots without prefix bleed."""
        if not allowed_roots:
            msg = (
                f"SecuritySandbox: Unauthorized file {op_name} access blocked for '{resolved_path}'. "
                f"No allowed {op_name} roots configured."
            )
            logger.warning(msg)
            raise PermissionDeniedError(msg)

        is_allowed = False
        for root in allowed_roots:
            try:
                root_path = Path(root).expanduser().resolve()
                if resolved_path == root_path or resolved_path.is_relative_to(
                    root_path
                ):
                    is_allowed = True
                    break
            except OSError as e:
                logger.debug("Cannot resolve root path '%s': %s", root, e)
                continue

        if not is_allowed:
            msg = (
                f"SecuritySandbox: Unauthorized file {op_name} access blocked for '{resolved_path}'. "
                f"Policy: {self.policy.policy_name}"
            )
            logger.warning(msg)
            raise PermissionDeniedError(msg)

    # --- Feature 12: Command Whitelist & Anti-Chaining ---

    def validate_command(self, cmd: str | list[str]) -> CommandVector:
        """Validates command against executable whitelist and blocks shell metacharacter chaining.

        Args:
            cmd: Command string (e.g. "echo hello") or argument vector (e.g. ["echo", "hello"]).

        Returns:
            CommandVector: Validated argument vector list, supporting shell=False execution and string equality.

        Raises:
            PermissionDeniedError: If chaining metacharacters, unauthorized executable, or injection is detected.
        """
        if cmd is None:
            raise PermissionDeniedError("SecuritySandbox: Null command blocked.")

        if isinstance(cmd, str):
            raw_cmd = cmd.strip()
            if not raw_cmd:
                msg = f"SecuritySandbox: Empty command blocked by policy '{self.policy.policy_name}'."
                logger.warning(msg)
                raise PermissionDeniedError(msg)

            # Check for banned shell metacharacters before any execution or tokenization
            self._check_shell_metacharacters(raw_cmd)

            # Tokenize into argument vector using POSIX shell lexical parsing
            try:
                tokens = shlex.split(raw_cmd, posix=True)
            except ValueError as e:
                msg = f"SecuritySandbox: Syntax error or unclosed quote in command '{raw_cmd}': {e}"
                logger.warning(msg)
                raise PermissionDeniedError(msg) from e

            if not tokens:
                raise PermissionDeniedError("SecuritySandbox: Empty tokenized command.")

            # Validate executable whitelist
            resolved_exe = self._verify_executable(tokens[0])

            # Enforce CommandRule constraints
            self._enforce_command_rules(tokens, canonical_path=resolved_exe)

            # Verify no null bytes inside tokens
            for token in tokens:
                if "\x00" in token:
                    raise PermissionDeniedError(
                        f"SecuritySandbox: Null byte detected in command argument '{token}'."
                    )

            return CommandVector(tokens, raw_cmd=cmd)

        elif isinstance(cmd, list):
            if not cmd or not any(str(x).strip() for x in cmd):
                raise PermissionDeniedError(
                    f"SecuritySandbox: Empty command vector blocked by policy '{self.policy.policy_name}'."
                )

            tokens = [str(x) for x in cmd]
            resolved_exe = self._verify_executable(tokens[0])
            self._enforce_command_rules(tokens, canonical_path=resolved_exe)

            for token in tokens:
                if "\x00" in token:
                    raise PermissionDeniedError(
                        f"SecuritySandbox: Null byte detected in argument vector '{token}'."
                    )

            return CommandVector(tokens, raw_cmd=" ".join(tokens))

        else:
            raise PermissionDeniedError(
                f"SecuritySandbox: Invalid command type '{type(cmd)}', expected str or list[str]."
            )

    def _check_shell_metacharacters(self, cmd_str: str) -> None:
        """Inspects raw command string for banned shell metacharacters."""
        banned = self.policy.banned_shell_metachars
        for char in banned:
            if char in cmd_str:
                msg = (
                    f"SecuritySandbox: Forbidden shell metacharacter '{char}' detected in command '{cmd_str}'. "
                    f"Command chaining and injection blocked."
                )
                logger.warning(msg)
                raise PermissionDeniedError(msg)

    def _verify_executable(self, raw_executable: str) -> Path | None:
        """Checks executable against whitelist, path containment, and returns canonical Path if path-based."""
        clean_exe = raw_executable.strip()
        if not clean_exe:
            raise PermissionDeniedError("SecuritySandbox: Blank executable token.")

        self._check_null_bytes_and_encodings(clean_exe)

        has_sep = "/" in clean_exe or "\\" in clean_exe
        resolved_exe: Path | None = None

        if has_sep:
            normalized_path = clean_exe.replace("\\", "/")
            try:
                resolved_exe = Path(normalized_path).expanduser().resolve()
            except (OSError, RuntimeError) as e:
                msg = f"SecuritySandbox: Unable to canonicalize executable path '{clean_exe}': {e}"
                logger.warning(msg)
                raise PermissionDeniedError(msg) from e

            # If allowed_execute_paths is defined, require canonical path to reside in allowed_execute_paths
            if self.policy.allowed_execute_paths:
                self._verify_containment(
                    resolved_exe, self.policy.allowed_execute_paths, op_name="execute"
                )
            else:
                # If not defined, require exact full path in allowed_commands or command_rules
                exact_match = (
                    clean_exe in self.policy.allowed_commands
                    or str(resolved_exe) in self.policy.allowed_commands
                    or any(
                        r.executable in (clean_exe, str(resolved_exe))
                        for r in self.policy.command_rules
                    )
                )
                if not exact_match:
                    msg = (
                        f"SecuritySandbox: Executable '{clean_exe}' contains path separators and "
                        f"does not match any exact full path in allowed_commands, and no "
                        f"allowed_execute_paths are configured. Policy: {self.policy.policy_name}"
                    )
                    logger.warning(msg)
                    raise PermissionDeniedError(msg)

        if not self.policy.is_executable_whitelisted(
            clean_exe, canonical_path=resolved_exe
        ):
            msg = f"SecuritySandbox: Execution of command '{clean_exe}' blocked by policy '{self.policy.policy_name}'."
            logger.warning(msg)
            raise PermissionDeniedError(msg)

        return resolved_exe

    def _enforce_command_rules(
        self, tokens: list[str], canonical_path: Path | None = None
    ) -> None:
        """Enforces fine-grained constraints from CommandRule models (subcommands, flags, max_args)."""
        matching_rules = self.policy.get_matching_command_rules(
            tokens[0], canonical_path=canonical_path
        )
        if not matching_rules:
            return

        cmd_args = tokens[1:]

        for rule in matching_rules:
            # 1. Enforce max_args
            if len(cmd_args) > rule.max_args:
                msg = (
                    f"SecuritySandbox: Command '{tokens[0]}' argument count {len(cmd_args)} "
                    f"exceeds maximum allowed args ({rule.max_args}) for rule '{rule.executable}'."
                )
                logger.warning(msg)
                raise PermissionDeniedError(msg)

            # 2. Extract flags and subcommands
            flags: list[str] = []
            flag_names: list[str] = []
            non_flag_args: list[str] = []
            after_double_dash = False

            for arg in cmd_args:
                if after_double_dash:
                    non_flag_args.append(arg)
                    continue

                if arg == "--":
                    after_double_dash = True
                    continue

                if arg.startswith("-") and arg != "-":
                    flags.append(arg)
                    # Extract flag name without value (e.g. '--config=val' -> '--config')
                    name = arg.split("=")[0]
                    flag_names.append(name)
                    # If bundled short flags (e.g. '-rf' or '-ac'), also check individual flags
                    if (
                        not arg.startswith("--")
                        and len(arg) > 2
                        and "=" not in arg
                        and not arg[1:].isdigit()
                    ):
                        for char in arg[1:]:
                            flag_names.append(f"-{char}")
                else:
                    non_flag_args.append(arg)

            # 3. Enforce blocked_flags
            if rule.blocked_flags:
                for f, name in zip(flags, flag_names):
                    if f in rule.blocked_flags or name in rule.blocked_flags:
                        msg = (
                            f"SecuritySandbox: Flag '{f}' is blocked for command '{tokens[0]}' "
                            f"by rule '{rule.executable}'."
                        )
                        logger.warning(msg)
                        raise PermissionDeniedError(msg)
                # Also check unpacked bundled short flags
                for name in flag_names:
                    if name in rule.blocked_flags:
                        msg = (
                            f"SecuritySandbox: Flag '{name}' is blocked for command '{tokens[0]}' "
                            f"by rule '{rule.executable}'."
                        )
                        logger.warning(msg)
                        raise PermissionDeniedError(msg)

            # 4. Enforce allowed_flags
            if rule.allowed_flags:
                for f in flags:
                    name = f.split("=")[0]
                    # Check if flag or flag name is in allowed_flags
                    if f in rule.allowed_flags or name in rule.allowed_flags:
                        continue
                    # Check if bundled short flags are all in allowed_flags
                    if (
                        not f.startswith("--")
                        and len(f) > 2
                        and "=" not in f
                        and not f[1:].isdigit()
                    ):
                        bundled_ok = all(f"-{c}" in rule.allowed_flags for c in f[1:])
                        if bundled_ok:
                            continue
                    msg = (
                        f"SecuritySandbox: Flag '{f}' is not permitted for command '{tokens[0]}'. "
                        f"Allowed flags: {sorted(rule.allowed_flags)}"
                    )
                    logger.warning(msg)
                    raise PermissionDeniedError(msg)

            # 5. Enforce allowed_subcommands
            if rule.allowed_subcommands:
                if not non_flag_args:
                    msg = (
                        f"SecuritySandbox: Missing required subcommand for command '{tokens[0]}'. "
                        f"Allowed subcommands: {sorted(rule.allowed_subcommands)}"
                    )
                    logger.warning(msg)
                    raise PermissionDeniedError(msg)

                subcommand = non_flag_args[0]
                if subcommand not in rule.allowed_subcommands:
                    msg = (
                        f"SecuritySandbox: Subcommand '{subcommand}' is not permitted for command '{tokens[0]}'. "
                        f"Allowed subcommands: {sorted(rule.allowed_subcommands)}"
                    )
                    logger.warning(msg)
                    raise PermissionDeniedError(msg)

    # --- Tool Call Validation ---

    def validate_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Validates tool invocation arguments for file, command, or parameter policy compliance."""
        if not tool_name:
            raise PermissionDeniedError("SecuritySandbox: Empty tool name.")

        for key in ("file_path", "target_path", "path", "filename"):
            if key in arguments and isinstance(arguments[key], str):
                mode = arguments.get("mode", "r")
                self.validate_file_access(arguments[key], mode=mode)

        for key in ("cmd", "command", "args"):
            if key in arguments and isinstance(arguments[key], (str, list)):
                self.validate_command(arguments[key])

        for key in ("url", "uri", "endpoint", "host", "destination", "target"):
            if key in arguments and isinstance(arguments[key], str):
                self.validate_network(arguments[key])

    # --- Feature 13: Network Sandbox & SSRF Isolation ---

    def validate_url(self, url: str) -> str:
        """Validates that a URL conforms to scheme, host, SSRF, and domain policies.

        Args:
            url: Full URL string (e.g. 'https://api.github.com/v1').

        Returns:
            The validated URL string.

        Raises:
            PermissionDeniedError: If network access is disabled, scheme is forbidden,
                                  destination IP is private/loopback/metadata, or domain
                                  is not permitted by whitelist.
        """
        if not self.policy.allow_network:
            msg = f"SecuritySandbox: Outbound network request to '{url}' blocked. Network access disabled."
            logger.warning(msg)
            raise PermissionDeniedError(msg)

        if not url or not isinstance(url, str):
            raise PermissionDeniedError(
                "SecuritySandbox: Empty or invalid URL provided."
            )

        if "\x00" in url or "%00" in url:
            msg = f"SecuritySandbox: Null-byte injection detected in URL '{url}'."
            logger.warning(msg)
            raise PermissionDeniedError(msg)

        try:
            parsed = urlsplit(url.strip())
        except Exception as e:
            msg = f"SecuritySandbox: Malformed URL '{url}': {e}"
            logger.warning(msg)
            raise PermissionDeniedError(msg) from e

        scheme = (parsed.scheme or "").lower()
        if not scheme:
            msg = f"SecuritySandbox: Missing URL scheme in '{url}'."
            logger.warning(msg)
            raise PermissionDeniedError(msg)

        if scheme not in self.policy.allowed_schemes:
            msg = (
                f"SecuritySandbox: Protocol scheme '{scheme}' blocked for URL '{url}'. "
                f"Allowed: {sorted(self.policy.allowed_schemes)}"
            )
            logger.warning(msg)
            raise PermissionDeniedError(msg)

        hostname = parsed.hostname
        if not hostname:
            msg = f"SecuritySandbox: Missing or invalid host in URL '{url}'."
            logger.warning(msg)
            raise PermissionDeniedError(msg)

        try:
            port = parsed.port
        except ValueError as e:
            msg = f"SecuritySandbox: Port out of range in URL '{url}': {e}"
            logger.warning(msg)
            raise PermissionDeniedError(msg) from e

        if port is not None:
            if not (1 <= port <= 65535):
                msg = f"SecuritySandbox: Port {port} out of range (1-65535) in URL '{url}'."
                logger.warning(msg)
                raise PermissionDeniedError(msg)
            if (
                self.policy.allowed_ports is not None
                and port not in self.policy.allowed_ports
            ):
                msg = f"SecuritySandbox: Port {port} not in allowed ports for URL '{url}'."
                logger.warning(msg)
                raise PermissionDeniedError(msg)

        self._validate_host(hostname)
        return url

    def validate_network(self, url_or_host: str) -> str:
        """Validates outbound network destination (either a full URL or host/IP).

        Args:
            url_or_host: Full URL ('https://api.github.com') or host string ('api.github.com', '127.0.0.1:80').

        Returns:
            The validated destination string.

        Raises:
            PermissionDeniedError: If network access is disabled or destination is blocked.
        """
        if not self.policy.allow_network:
            msg = f"SecuritySandbox: Outbound network request to '{url_or_host}' blocked. Network access disabled."
            logger.warning(msg)
            raise PermissionDeniedError(msg)

        destination = url_or_host.strip()
        if not destination:
            raise PermissionDeniedError(
                "SecuritySandbox: Empty network destination provided."
            )

        if "\x00" in destination or "%00" in destination:
            msg = f"SecuritySandbox: Null-byte injection detected in destination '{destination}'."
            logger.warning(msg)
            raise PermissionDeniedError(msg)

        # Determine if destination is a full URL or raw host/port
        if "://" in destination or any(
            destination.lower().startswith(s + ":")
            for s in ["http", "https", "file", "gopher", "ftp", "dict", "ldap", "data"]
        ):
            return self.validate_url(destination)

        # Handle raw host or host:port
        host_part = destination
        port_part: int | None = None

        if host_part.startswith("[") and "]" in host_part:
            close_idx = host_part.find("]")
            bracketed = host_part[1:close_idx]
            rest = host_part[close_idx + 1 :]
            if rest.startswith(":"):
                try:
                    port_part = int(rest[1:])
                except ValueError as e:
                    raise PermissionDeniedError(
                        f"SecuritySandbox: Invalid port in destination '{destination}'."
                    ) from e
            host_part = bracketed
        elif ":" in host_part and host_part.count(":") == 1:
            h, p = host_part.split(":", 1)
            try:
                port_part = int(p)
                host_part = h
            except ValueError:
                pass

        if port_part is not None:
            if not (1 <= port_part <= 65535):
                raise PermissionDeniedError(
                    f"SecuritySandbox: Port {port_part} out of range in '{destination}'."
                )
            if (
                self.policy.allowed_ports is not None
                and port_part not in self.policy.allowed_ports
            ):
                raise PermissionDeniedError(
                    f"SecuritySandbox: Port {port_part} not in allowed ports."
                )

        self._validate_host(host_part)
        return url_or_host

    def _validate_host(self, host: str) -> None:
        """Internal validation engine for hostname or IP address."""
        clean_host = host.strip("[]").strip().lower()

        # 1. Blacklisted Hostnames (Pre-DNS)
        is_metadata_host = (
            clean_host in BLOCKED_METADATA_HOSTNAMES and clean_host != "localhost"
        )
        is_loopback_host = clean_host == "localhost" or clean_host.endswith(
            ".localhost"
        )
        if (self.policy.block_cloud_metadata and is_metadata_host) or (
            self.policy.block_loopback and is_loopback_host
        ):
            msg = f"SecuritySandbox: Destination host '{host}' matches blacklisted host pattern (SSRF protection)."
            logger.warning(msg)
            raise PermissionDeniedError(msg)

        # 2. Check if destination is an IPv4 literal (standard, integer, hex, octal, dotless)
        parsed_v4: ipaddress.IPv4Address | None = None
        try:
            raw_v4 = socket.inet_aton(clean_host)
            canonical_v4 = socket.inet_ntoa(raw_v4)
            parsed_v4 = ipaddress.IPv4Address(canonical_v4)
        except OSError:
            pass

        if parsed_v4 is not None:
            if self._is_ip_prohibited(parsed_v4):
                msg = f"SecuritySandbox: Destination IPv4 '{host}' (canonical: {parsed_v4}) is prohibited by SSRF policy."
                logger.warning(msg)
                raise PermissionDeniedError(msg)
            if self.policy.allowed_domains and not self._match_domain(
                clean_host, self.policy.allowed_domains
            ):
                msg = f"SecuritySandbox: IP address '{host}' not permitted by allowed_domains whitelist."
                logger.warning(msg)
                raise PermissionDeniedError(msg)
            return

        # 3. Check if destination is an IPv6 literal
        try:
            parsed_v6 = ipaddress.IPv6Address(clean_host)
            if self._is_ip_prohibited(parsed_v6):
                msg = f"SecuritySandbox: Destination IPv6 '{host}' is prohibited by SSRF policy."
                logger.warning(msg)
                raise PermissionDeniedError(msg)
            if self.policy.allowed_domains and not self._match_domain(
                clean_host, self.policy.allowed_domains
            ):
                msg = f"SecuritySandbox: IPv6 address '{host}' not permitted by allowed_domains whitelist."
                logger.warning(msg)
                raise PermissionDeniedError(msg)
            return
        except ValueError:
            pass

        # 4. Domain Whitelist Validation
        if self.policy.allowed_domains and not self._match_domain(
            clean_host, self.policy.allowed_domains
        ):
            msg = (
                f"SecuritySandbox: Hostname '{host}' is not in allowed domains whitelist. "
                f"Policy: {self.policy.policy_name}"
            )
            logger.warning(msg)
            raise PermissionDeniedError(msg)

        # 5. DNS Resolution & Resolved IP Inspection
        if self.policy.resolve_dns:
            resolved_ips = self._dns_resolver(clean_host)
            if not resolved_ips:
                if not self.policy.allow_unresolved_domains:
                    msg = (
                        f"SecuritySandbox: DNS resolution failed for domain '{host}'. "
                        f"Policy prohibits unresolved domains."
                    )
                    logger.warning(msg)
                    raise PermissionDeniedError(msg)
            else:
                for ip_str in resolved_ips:
                    try:
                        addr = ipaddress.ip_address(ip_str.strip("[]"))
                        if self._is_ip_prohibited(addr):
                            msg = (
                                f"SecuritySandbox: Hostname '{host}' resolved to prohibited IP '{ip_str}' "
                                f"(Loopback / Private / Metadata IP blocked)."
                            )
                            logger.warning(msg)
                            raise PermissionDeniedError(msg)
                    except ValueError:
                        continue

    def _is_ip_prohibited(
        self, addr: ipaddress.IPv4Address | ipaddress.IPv6Address
    ) -> bool:
        """Checks if an IP address falls into any blocked subnet according to policy switches."""
        if addr.version == 6:
            # 1. Check RFC 4291 IPv4-mapped IPv6 address (::ffff:0:0/96)
            if addr.ipv4_mapped:
                return self._is_ip_prohibited(addr.ipv4_mapped)

            # 2. Check RFC 4291 IPv4-compatible (::/96) and RFC 6052 NAT64 (64:ff9b::/96)
            if addr in IPV4_COMPATIBLE_IPV6_PREFIX or addr in NAT64_WELL_KNOWN_PREFIX:
                embedded_v4 = ipaddress.IPv4Address(addr.packed[-4:])
                if self._is_ip_prohibited(embedded_v4):
                    return True

            # 3. IPv6 Loopback and Unspecified
            if self.policy.block_loopback:
                if addr.is_loopback or addr.is_unspecified:
                    return True
                for net_v6 in LOOPBACK_IPV6_SUBNETS:
                    if addr in net_v6:
                        return True

            # 4. IPv6 Cloud Metadata and Link-Local (fe80::/10)
            if self.policy.block_cloud_metadata:
                for net_v6 in CLOUD_METADATA_IPV6_SUBNETS:
                    if addr in net_v6:
                        return True

            # 5. IPv6 Private (ULA fc00::/7 and Site-Local fec0::/10)
            if self.policy.block_private_ips:
                for net_v6 in PRIVATE_IPV6_SUBNETS:
                    if addr in net_v6:
                        return True

            # 6. IPv6 Reserved, Multicast, and Documentation
            for net_v6 in RESERVED_IPV6_SUBNETS:
                if addr in net_v6:
                    return True

        else:
            # IPv4 Address evaluation
            # 1. IPv4 Loopback and Current Network
            if self.policy.block_loopback:
                if addr.is_loopback or addr.is_unspecified:
                    return True
                for net_v4 in LOOPBACK_IPV4_SUBNETS:
                    if addr in net_v4:
                        return True

            # 2. IPv4 Cloud Metadata and Link-Local (169.254.0.0/16, 168.63.129.16)
            if self.policy.block_cloud_metadata:
                for net_v4 in CLOUD_METADATA_IPV4_SUBNETS:
                    if addr in net_v4:
                        return True

            # 3. IPv4 Private Subnets (RFC 1918)
            if self.policy.block_private_ips:
                for net_v4 in PRIVATE_IPV4_SUBNETS:
                    if addr in net_v4:
                        return True

            # 4. RFC 6598 Carrier-Grade NAT (100.64.0.0/10)
            if self.policy.block_carrier_grade_nat:
                for net_v4 in CARRIER_GRADE_NAT_IPV4_SUBNETS:
                    if addr in net_v4:
                        return True

            # 5. IPv4 Reserved, Class E, Multicast, and Broadcast
            for net_v4 in RESERVED_IPV4_SUBNETS:
                if addr in net_v4:
                    return True

        # Custom blocked CIDRs configured on the policy
        for custom_net in self._custom_subnets:
            if addr in custom_net:
                return True

        return False

    @staticmethod
    def _match_domain(host: str, allowed_rules: list[str]) -> bool:
        """Matches a hostname against domain whitelist rules (exact and *.domain.com)."""
        h = host.lower().rstrip(".")
        for rule in allowed_rules:
            r = rule.lower().rstrip(".")
            if r == "*":
                return True
            if r.startswith("*."):
                base = r[2:]
                if h == base or h.endswith("." + base):
                    return True
            elif h == r:
                return True
        return False
