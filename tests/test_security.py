"""Comprehensive Unit and Security Penetration Tests for AgentMesh Sandbox."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agentmesh.security.policy import CapabilityPolicy, CommandRule
from agentmesh.security.sandbox import (
    PermissionDeniedError,
    SafePath,
    SecuritySandbox,
)

# ==============================================================================
# 0. Baseline Backward Compatibility Tests
# ==============================================================================


def test_sandbox_path_validation():
    policy = CapabilityPolicy(
        policy_name="test_policy",
        allowed_read_paths=["/tmp/safe_dir"],
        allowed_write_paths=["/tmp/safe_dir/out"],
    )
    sandbox = SecuritySandbox(policy)

    # Allowed read
    assert sandbox.validate_file_access("/tmp/safe_dir/test.txt", mode="r")

    # Blocked read outside path
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_file_access("/etc/hosts", mode="r")

    # Blocked write to read-only directory
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_file_access("/tmp/safe_dir/test.txt", mode="w")


def test_sandbox_command_and_network_validation():
    policy = CapabilityPolicy(
        allowed_commands={"echo", "grep"},
        allow_network=False,
    )
    sandbox = SecuritySandbox(policy)

    # Allowed command
    assert sandbox.validate_command("echo 'hello'") == "echo 'hello'"

    # Blocked command
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_command("rm -rf /")

    # Blocked network
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_network("https://external-api.com")


# ==============================================================================
# Fixtures for Comprehensive Penetration Testing
# ==============================================================================


@pytest.fixture
def sandbox_env(tmp_path: Path):
    """Sets up an isolated directory hierarchy for path and symlink security tests."""
    read_dir = tmp_path / "sandbox_read"
    write_dir = tmp_path / "sandbox_write"
    exec_dir = tmp_path / "sandbox_exec"
    external_dir = tmp_path / "external_system"

    read_dir.mkdir(parents=True, exist_ok=True)
    write_dir.mkdir(parents=True, exist_ok=True)
    exec_dir.mkdir(parents=True, exist_ok=True)
    external_dir.mkdir(parents=True, exist_ok=True)

    # Seed files
    (read_dir / "readable.txt").write_text("READABLE DATA", encoding="utf-8")
    (external_dir / "secret.key").write_text("SUPER_SECRET", encoding="utf-8")
    (exec_dir / "tool.sh").write_text("#!/bin/sh\necho ok\n", encoding="utf-8")

    policy = CapabilityPolicy(
        policy_name="comprehensive_test_policy",
        allow_network=False,
        allowed_read_paths=[str(read_dir)],
        allowed_write_paths=[str(write_dir)],
        allowed_execute_paths=[str(exec_dir)],
        allowed_commands={"echo", "cat", "grep", "python3"},
    )
    sandbox = SecuritySandbox(policy=policy)

    return {
        "sandbox": sandbox,
        "policy": policy,
        "read_dir": read_dir,
        "write_dir": write_dir,
        "exec_dir": exec_dir,
        "external_dir": external_dir,
    }


# ==============================================================================
# 1. Directory Traversal & Boundary Tests (Feature 11)
# ==============================================================================


def test_path_confinement_valid_read(sandbox_env):
    """Validates that reading legitimate files inside the read root succeeds."""
    sandbox = sandbox_env["sandbox"]
    read_dir = sandbox_env["read_dir"]
    target = read_dir / "readable.txt"

    res = sandbox.validate_file_access(str(target), mode="r")
    assert isinstance(res, Path)
    assert isinstance(res, SafePath)
    assert res.endswith("readable.txt")
    assert res.exists()


def test_path_confinement_traversal_relative_blocked(sandbox_env):
    """Validates that dot-dot path traversal attempts are blocked."""
    sandbox = sandbox_env["sandbox"]
    read_dir = sandbox_env["read_dir"]

    traversals = [
        str(read_dir / ".." / "external_system" / "secret.key"),
        str(read_dir / "subdir" / ".." / ".." / "external_system" / "secret.key"),
        "../../../../etc/passwd",
        "/etc/shadow",
    ]
    for attack in traversals:
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_file_access(attack, mode="r")


def test_path_confinement_prefix_bleed_defense(sandbox_env, tmp_path: Path):
    """Validates that sibling directories sharing a common prefix string are rejected."""
    sandbox = sandbox_env["sandbox"]
    read_dir = sandbox_env["read_dir"]

    bleed_dir = tmp_path / f"{read_dir.name}_extra"
    bleed_dir.mkdir(parents=True, exist_ok=True)
    bleed_file = bleed_dir / "stolen.txt"
    bleed_file.write_text("leak", encoding="utf-8")

    with pytest.raises(PermissionDeniedError):
        sandbox.validate_file_access(str(bleed_file), mode="r")


def test_path_confinement_empty_or_whitespace_path(sandbox_env):
    """Validates that empty or whitespace-only paths raise PermissionDeniedError."""
    sandbox = sandbox_env["sandbox"]
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_file_access("", mode="r")
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_file_access("   ", mode="r")


# ==============================================================================
# 2. Null-Byte & Encoded Traversal Tests (Feature 11)
# ==============================================================================


def test_null_byte_injection_raw_blocked(sandbox_env):
    """Validates that raw null bytes in path strings are rejected immediately."""
    sandbox = sandbox_env["sandbox"]
    read_dir = sandbox_env["read_dir"]
    attack_path = str(read_dir / "readable.txt") + "\x00.evil"

    with pytest.raises(PermissionDeniedError) as exc_info:
        sandbox.validate_file_access(attack_path, mode="r")
    assert "null-byte" in str(exc_info.value).lower()


def test_null_byte_injection_url_encoded_blocked(sandbox_env):
    """Validates that URL-encoded null bytes (%00) are blocked."""
    sandbox = sandbox_env["sandbox"]
    read_dir = sandbox_env["read_dir"]
    attack_path = str(read_dir / "readable.txt%00.evil")

    with pytest.raises(PermissionDeniedError) as exc_info:
        sandbox.validate_file_access(attack_path, mode="r")
    assert "null-byte" in str(exc_info.value).lower()


def test_null_byte_injection_double_encoded_blocked(sandbox_env):
    """Validates that double URL-encoded null bytes (%2500) are blocked."""
    sandbox = sandbox_env["sandbox"]
    read_dir = sandbox_env["read_dir"]
    attack_path = str(read_dir / "readable%2500.txt")

    with pytest.raises(PermissionDeniedError) as exc_info:
        sandbox.validate_file_access(attack_path, mode="r")
    assert "null-byte" in str(exc_info.value).lower()


def test_encoded_path_traversal_url_blocked(sandbox_env):
    """Validates that URL-encoded traversal (%2e%2e%2f) is intercepted."""
    sandbox = sandbox_env["sandbox"]
    read_dir = sandbox_env["read_dir"]
    attack_path = str(read_dir) + "/%2e%2e/%2e%2e/etc/passwd"

    with pytest.raises(PermissionDeniedError):
        sandbox.validate_file_access(attack_path, mode="r")


# ==============================================================================
# 3. Symlink Resolution & Confinement Tests (Feature 11)
# ==============================================================================


def test_symlink_pointing_outside_root_blocked(sandbox_env):
    """Validates that a symlink located inside the sandbox pointing outside is blocked."""
    sandbox = sandbox_env["sandbox"]
    read_dir = sandbox_env["read_dir"]
    external_dir = sandbox_env["external_dir"]

    symlink_file = read_dir / "leak_symlink.txt"
    try:
        os.symlink(external_dir / "secret.key", symlink_file)
    except OSError:
        pytest.skip("Symlink creation not permitted in test environment")

    with pytest.raises(PermissionDeniedError):
        sandbox.validate_file_access(str(symlink_file), mode="r")


def test_symlink_directory_pointing_outside_blocked(sandbox_env):
    """Validates that accessing files through a symlinked directory pointing outside is blocked."""
    sandbox = sandbox_env["sandbox"]
    read_dir = sandbox_env["read_dir"]
    external_dir = sandbox_env["external_dir"]

    symlink_dir = read_dir / "sym_external_dir"
    try:
        os.symlink(external_dir, symlink_dir)
    except OSError:
        pytest.skip("Symlink creation not permitted in test environment")

    with pytest.raises(PermissionDeniedError):
        sandbox.validate_file_access(str(symlink_dir / "secret.key"), mode="r")


def test_symlink_internal_allowed(sandbox_env):
    """Validates that a symlink pointing to another location inside the same sandbox root is allowed."""
    sandbox = sandbox_env["sandbox"]
    read_dir = sandbox_env["read_dir"]

    internal_target = read_dir / "readable.txt"
    internal_symlink = read_dir / "link_to_readable.txt"
    try:
        os.symlink(internal_target, internal_symlink)
    except OSError:
        pytest.skip("Symlink creation not permitted in test environment")

    res = sandbox.validate_file_access(str(internal_symlink), mode="r")
    assert res.endswith("readable.txt")


def test_symlink_circular_loop_handling(sandbox_env):
    """Validates that circular symlinks are handled gracefully without an uncaught crash."""
    sandbox = sandbox_env["sandbox"]
    read_dir = sandbox_env["read_dir"]

    l1 = read_dir / "loop1"
    l2 = read_dir / "loop2"
    try:
        os.symlink(l2, l1)
        os.symlink(l1, l2)
    except OSError:
        pytest.skip("Symlink creation not permitted")

    with pytest.raises(PermissionDeniedError) as exc_info:
        sandbox.validate_file_access(str(l1 / "file.txt"), mode="r")
    assert (
        "loop" in str(exc_info.value).lower()
        or "canonicalize" in str(exc_info.value).lower()
    )


# ==============================================================================
# 4. Mode Isolation Tests (Feature 11)
# ==============================================================================


def test_mode_isolation_write_to_read_path_blocked(sandbox_env):
    """Validates that attempting to write or append to a read-only root raises PermissionDeniedError."""
    sandbox = sandbox_env["sandbox"]
    read_dir = sandbox_env["read_dir"]

    target = str(read_dir / "readable.txt")
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_file_access(target, mode="w")
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_file_access(target, mode="a")
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_file_access(target, mode="r+")


def test_mode_isolation_read_from_write_only_path_blocked(sandbox_env):
    """Validates that reading from a write-only root raises PermissionDeniedError."""
    sandbox = sandbox_env["sandbox"]
    write_dir = sandbox_env["write_dir"]

    target = str(write_dir / "out.txt")
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_file_access(target, mode="r")


def test_mode_isolation_execute_allowed_in_exec_dir(sandbox_env):
    """Validates that mode 'x' is permitted in allowed_execute_paths."""
    sandbox = sandbox_env["sandbox"]
    exec_dir = sandbox_env["exec_dir"]

    target = str(exec_dir / "tool.sh")
    res = sandbox.validate_file_access(target, mode="x")
    assert res.endswith("tool.sh")


def test_mode_isolation_execute_blocked_in_read_or_write_dir(sandbox_env):
    """Validates that mode 'x' is rejected in read_dir or write_dir."""
    sandbox = sandbox_env["sandbox"]
    read_dir = sandbox_env["read_dir"]
    write_dir = sandbox_env["write_dir"]

    with pytest.raises(PermissionDeniedError):
        sandbox.validate_file_access(str(read_dir / "readable.txt"), mode="x")
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_file_access(str(write_dir / "script.sh"), mode="x")


# ==============================================================================
# 5. Command Whitelist & Anti-Chaining Tests (Feature 12)
# ==============================================================================


def test_command_whitelist_allowed_commands_pass(sandbox_env):
    """Validates that whitelisted commands with clean arguments pass validation."""
    sandbox = sandbox_env["sandbox"]

    assert sandbox.validate_command("echo hello") == "echo hello"
    assert sandbox.validate_command("cat /tmp/test") == "cat /tmp/test"
    assert sandbox.validate_command("grep -rn pattern .") == "grep -rn pattern ."
    assert sandbox.validate_command(["python3", "script.py"]) == [
        "python3",
        "script.py",
    ]


def test_command_whitelist_unauthorized_binary_blocked(sandbox_env):
    """Validates that unlisted binaries are strictly blocked."""
    sandbox = sandbox_env["sandbox"]
    blocked = [
        "rm -rf /",
        "wget http://evil.com",
        "curl http://leak.com",
        "bash -i",
        "sh",
    ]
    for cmd in blocked:
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_command(cmd)


def test_command_chaining_semicolon_blocked(sandbox_env):
    """Validates that semicolon command chaining is 100% blocked."""
    sandbox = sandbox_env["sandbox"]
    with pytest.raises(PermissionDeniedError) as exc_info:
        sandbox.validate_command("echo hello; rm -rf /")
    assert "metacharacter ';'" in str(exc_info.value)


def test_command_chaining_ampersand_blocked(sandbox_env):
    """Validates that ampersand backgrounding and logical AND are blocked."""
    sandbox = sandbox_env["sandbox"]
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_command("echo hello && rm -rf /")
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_command("echo hello & evil_bg")


def test_command_chaining_pipe_blocked(sandbox_env):
    """Validates that pipeline redirection is 100% blocked."""
    sandbox = sandbox_env["sandbox"]
    with pytest.raises(PermissionDeniedError) as exc_info:
        sandbox.validate_command("echo hello | bash")
    assert "metacharacter '|'" in str(exc_info.value)


def test_command_substitution_backticks_blocked(sandbox_env):
    """Validates that backtick command substitution is blocked."""
    sandbox = sandbox_env["sandbox"]
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_command("echo `whoami`")


def test_command_substitution_dollar_subshell_blocked(sandbox_env):
    """Validates that dollar subshell $(...) and variable expansion are blocked."""
    sandbox = sandbox_env["sandbox"]
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_command("echo $(cat /etc/passwd)")
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_command("echo $USER")


def test_command_redirection_operators_blocked(sandbox_env):
    """Validates that shell I/O redirection (<, >) is blocked."""
    sandbox = sandbox_env["sandbox"]
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_command("echo evil > /tmp/target")
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_command("cat < /etc/shadow")


def test_command_newline_and_cr_injection_blocked(sandbox_env):
    """Validates that newline and carriage return injections are blocked."""
    sandbox = sandbox_env["sandbox"]
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_command("echo hello\nrm -rf /")
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_command("echo hello\r\ncat /etc/passwd")


def test_command_null_byte_injection_blocked(sandbox_env):
    """Validates that null bytes in commands or argument vectors are blocked."""
    sandbox = sandbox_env["sandbox"]
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_command("echo hello\x00rm -rf /")
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_command(["echo", "hello\x00evil"])


def test_command_vector_list_execution_structure(sandbox_env):
    """Validates that argument vector lists are preserved and returned as CommandVector."""
    sandbox = sandbox_env["sandbox"]
    vec = ["echo", "item 1", "item 2"]
    res = sandbox.validate_command(vec)

    assert isinstance(res, list)
    assert res[0] == "echo"
    assert res[1] == "item 1"
    assert res[2] == "item 2"
    assert res == vec


def test_command_vector_empty_list_blocked(sandbox_env):
    """Validates that empty list or list of blank strings is blocked."""
    sandbox = sandbox_env["sandbox"]
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_command([])
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_command(["   ", ""])


def test_command_vector_unclosed_quotes_blocked(sandbox_env):
    """Validates that syntax errors such as unclosed quotes raise PermissionDeniedError."""
    sandbox = sandbox_env["sandbox"]
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_command("echo 'unclosed string")


def test_tool_call_validation_integration(sandbox_env):
    """Validates that validate_tool_call checks paths and commands embedded in arguments."""
    sandbox = sandbox_env["sandbox"]
    read_dir = sandbox_env["read_dir"]

    # Legitimate tool call passes
    sandbox.validate_tool_call(
        "read_file", {"path": str(read_dir / "readable.txt"), "mode": "r"}
    )

    # Traversal in tool call raises PermissionDeniedError
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_tool_call("read_file", {"path": "/etc/passwd", "mode": "r"})

    # Injected command in tool call raises PermissionDeniedError
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_tool_call("run_cli", {"cmd": "echo test; rm -rf /"})


# ==============================================================================
# 6. Feature 13: Network Sandbox & SSRF Isolation Tests
# ==============================================================================


def test_f13_network_disabled_blocks_all():
    """Verify that allow_network=False blocks all URLs and hosts with exact message."""
    policy = CapabilityPolicy(allow_network=False)
    sandbox = SecuritySandbox(policy=policy)

    targets = [
        "https://api.github.com/v1",
        "http://169.254.169.254/latest/meta-data/",
        "http://localhost:8080",
        "http://127.0.0.1:9000",
        "api.partner.com",
    ]
    for target in targets:
        with pytest.raises(PermissionDeniedError) as exc_info:
            sandbox.validate_network(target)
        assert "Network access disabled" in str(exc_info.value)


def test_f13_loopback_ipv4_blocked():
    """Verify all IPv4 loopback variants (127.0.0.1, 127.0.0.2, 0.0.0.0, 127.0.0.0/8) are blocked."""
    policy = CapabilityPolicy(allow_network=True)
    sandbox = SecuritySandbox(policy=policy)

    loopbacks = [
        "http://127.0.0.1:8080/admin",
        "http://127.0.0.2/",
        "http://127.255.255.254/secret",
        "http://0.0.0.0:8000/",
        "127.0.0.1",
        "127.1.2.3:80",
        "0.0.0.0",
    ]
    for destination in loopbacks:
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_network(destination)


def test_f13_loopback_ipv6_blocked():
    """Verify IPv6 loopback and unspecified addresses are blocked."""
    policy = CapabilityPolicy(allow_network=True)
    sandbox = SecuritySandbox(policy=policy)

    ipv6_loopbacks = [
        "http://[::1]:8080/metrics",
        "http://[::]:80/",
        "::1",
        "[::1]",
        "::",
        "[::]:8080",
    ]
    for destination in ipv6_loopbacks:
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_network(destination)


def test_f13_localhost_names_blocked():
    """Verify localhost and *.localhost hostnames are blocked prior to DNS resolution."""
    policy = CapabilityPolicy(allow_network=True)
    sandbox = SecuritySandbox(policy=policy)

    local_hosts = [
        "http://localhost/api",
        "http://localhost:3000/",
        "http://sub.localhost:8080",
        "localhost",
        "service.localhost:5000",
    ]
    for destination in local_hosts:
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_network(destination)


def test_f13_cloud_metadata_ip_blocked():
    """Verify AWS/GCP/Azure link-local metadata IP (169.254.169.254) and subnets are blocked."""
    policy = CapabilityPolicy(allow_network=True)
    sandbox = SecuritySandbox(policy=policy)

    metadata_targets = [
        "http://169.254.169.254/latest/meta-data/",
        "http://169.254.169.254:80/computeMetadata/v1/",
        "http://169.254.0.2/opc/v1/instance/",
        "http://169.254.169.250:80/metadata",
        "169.254.169.254",
        "169.254.169.254:80",
    ]
    for destination in metadata_targets:
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_network(destination)


def test_f13_cloud_metadata_hostnames_blocked():
    """Verify GCP and AWS metadata hostnames are blocked."""
    policy = CapabilityPolicy(allow_network=True)
    sandbox = SecuritySandbox(policy=policy)

    hostnames = [
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://metadata.internal/",
        "http://instance-data/latest/meta-data/",
        "metadata.google.internal",
        "instance-data",
    ]
    for destination in hostnames:
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_network(destination)


def test_f13_azure_wire_server_blocked():
    """Verify Azure VM Wire Server (168.63.129.16) is blocked."""
    policy = CapabilityPolicy(allow_network=True)
    sandbox = SecuritySandbox(policy=policy)

    with pytest.raises(PermissionDeniedError):
        sandbox.validate_network("http://168.63.129.16/metadata")
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_network("168.63.129.16:80")


def test_f13_rfc1918_private_ips_blocked():
    """Verify RFC 1918 private IPv4 subnets (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16) are blocked."""
    policy = CapabilityPolicy(allow_network=True)
    sandbox = SecuritySandbox(policy=policy)

    private_ips = [
        "http://10.0.0.1:8080/internal",
        "http://10.255.255.254/",
        "http://172.16.0.1/",
        "http://172.31.255.254:9000",
        "http://192.168.1.1/admin",
        "http://192.168.254.254/",
        "10.10.10.10",
        "172.20.0.1:443",
        "192.168.0.100",
    ]
    for destination in private_ips:
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_network(destination)


def test_f13_rfc6598_carrier_grade_nat_blocked():
    """Verify RFC 6598 Carrier-Grade NAT (100.64.0.0/10) is blocked."""
    policy = CapabilityPolicy(allow_network=True)
    sandbox = SecuritySandbox(policy=policy)

    cgnat_targets = [
        "http://100.64.0.1/status",
        "http://100.127.255.254:8080",
        "100.64.0.1",
        "100.100.100.100:80",
    ]
    for destination in cgnat_targets:
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_network(destination)


def test_f13_ipv6_ula_and_link_local_blocked():
    """Verify IPv6 ULA (fc00::/7) and Link-Local (fe80::/10) addresses are blocked."""
    policy = CapabilityPolicy(allow_network=True)
    sandbox = SecuritySandbox(policy=policy)

    targets = [
        "http://[fc00::1]:8080/",
        "http://[fd12:3456:789a::1]/",
        "http://[fe80::1]/",
        "http://[fe80::215:5dff:fe12:3456]:8080",
        "fc00::1",
        "fe80::1",
    ]
    for destination in targets:
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_network(destination)


def test_f13_ipv4_mapped_ipv6_evasion_blocked():
    """Verify IPv4-mapped IPv6 bypass attempts (::ffff:127.0.0.1, ::ffff:169.254.169.254) are blocked."""
    policy = CapabilityPolicy(allow_network=True)
    sandbox = SecuritySandbox(policy=policy)

    mapped_evasions = [
        "http://[::ffff:127.0.0.1]/",
        "http://[::ffff:169.254.169.254]/latest/meta-data/",
        "http://[::ffff:10.0.0.1]:8080",
        "http://[::ffff:192.168.1.1]/",
        "::ffff:127.0.0.1",
        "::ffff:169.254.169.254",
    ]
    for destination in mapped_evasions:
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_network(destination)


def test_f13_ip_obfuscation_evasions_blocked():
    """Verify decimal integer, hex, octal, and short-form dotless IP representations are blocked."""
    policy = CapabilityPolicy(allow_network=True)
    sandbox = SecuritySandbox(policy=policy)

    obfuscated = [
        "http://2130706433/",  # Decimal 127.0.0.1
        "http://2852039166/",  # Decimal 169.254.169.254
        "http://0x7f000001/",  # Hex 127.0.0.1
        "http://0x7f.0.0.1/",  # Hex mixed
        "http://017700000001/",  # Octal 127.0.0.1
        "http://127.1/",  # Dotless short form
        "http://0/",  # Short form 0.0.0.0
        "2130706433",
        "0x7f000001",
        "127.1",
    ]
    for destination in obfuscated:
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_network(destination)


def test_f13_protocol_schemes_restricted():
    """Verify that non-HTTP/HTTPS protocols (file, gopher, ftp, dict, ldap, data) are blocked."""
    policy = CapabilityPolicy(allow_network=True)
    sandbox = SecuritySandbox(policy=policy)

    forbidden_schemes = [
        "file:///etc/passwd",
        "gopher://127.0.0.1:6379/_flushall",
        "ftp://ftp.example.com/file.txt",
        "dict://127.0.0.1:11211/stat",
        "ldap://corp.internal/dc=example",
        "data:text/plain;base64,SGVsbG8=",
    ]
    for url in forbidden_schemes:
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_url(url)
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_network(url)


def test_f13_userinfo_credentials_handling():
    """Verify userinfo credential syntax is handled correctly and cannot smuggle blocked targets."""
    policy = CapabilityPolicy(allow_network=True)
    sandbox = SecuritySandbox(policy=policy)

    with pytest.raises(PermissionDeniedError):
        sandbox.validate_url("http://google.com:password@127.0.0.1:8080/admin")

    with pytest.raises(PermissionDeniedError):
        sandbox.validate_url("http://admin@169.254.169.254/")


def test_f13_null_byte_injection_blocked():
    """Verify null byte injections in URL or host are rejected."""
    policy = CapabilityPolicy(allow_network=True)
    sandbox = SecuritySandbox(policy=policy)

    null_injections = [
        "https://api.partner.com%00.attacker.com",
        "https://api.partner.com\x00/admin",
        "api.partner.com%00",
    ]
    for target in null_injections:
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_network(target)


def test_f13_strict_domain_whitelisting_exact_match():
    """Verify exact domain matching in allowed_domains policy."""
    policy = CapabilityPolicy(
        allow_network=True,
        allowed_domains=["api.github.com", "service.internal.corp"],
        allow_unresolved_domains=True,
    )
    sandbox = SecuritySandbox(policy=policy)

    # Allowed exact domains
    assert (
        sandbox.validate_network("https://api.github.com/repos")
        == "https://api.github.com/repos"
    )
    assert (
        sandbox.validate_network("https://service.internal.corp/status")
        == "https://service.internal.corp/status"
    )
    assert sandbox.validate_network("api.github.com") == "api.github.com"

    # Disallowed domains
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_network("https://evil.github.com/repos")

    with pytest.raises(PermissionDeniedError):
        sandbox.validate_network("https://github.com/repos")

    with pytest.raises(PermissionDeniedError):
        sandbox.validate_network("https://other-service.com")


def test_f13_strict_domain_whitelisting_wildcard_match():
    """Verify wildcard matching (*.domain.com) matches subdomains and apex, while stopping suffix hijacking."""
    policy = CapabilityPolicy(
        allow_network=True,
        allowed_domains=["*.trusted.org", "*.sub.example.com"],
        allow_unresolved_domains=True,
    )
    sandbox = SecuritySandbox(policy=policy)

    # Allowed wildcard matches
    assert (
        sandbox.validate_network("https://api.trusted.org/feed")
        == "https://api.trusted.org/feed"
    )
    assert (
        sandbox.validate_network("https://deep.sub.trusted.org")
        == "https://deep.sub.trusted.org"
    )
    assert sandbox.validate_network("https://trusted.org") == "https://trusted.org"
    assert (
        sandbox.validate_network("https://v1.sub.example.com")
        == "https://v1.sub.example.com"
    )

    # Suffix hijacking attempts must fail
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_network("https://evil-trusted.org")

    with pytest.raises(PermissionDeniedError):
        sandbox.validate_network("https://trusted.org.attacker.com")

    with pytest.raises(PermissionDeniedError):
        sandbox.validate_network("https://nottrusted.org")


def test_f13_dns_resolution_rebinding_defense():
    """Verify DNS resolution inspection rejects hostnames that resolve to private or loopback IPs."""

    def mock_dns(host: str) -> list[str]:
        if host == "rebind.attacker.com":
            return ["127.0.0.1"]
        elif host == "cloud-probe.attacker.com":
            return ["169.254.169.254"]
        elif host == "split-brain.attacker.com":
            return ["93.184.216.34", "10.0.0.5"]
        elif host == "legit.example.com":
            return ["93.184.216.34"]
        return []

    policy = CapabilityPolicy(allow_network=True)
    sandbox = SecuritySandbox(policy=policy, dns_resolver=mock_dns)

    # Rebind to loopback
    with pytest.raises(PermissionDeniedError) as exc_info:
        sandbox.validate_network("http://rebind.attacker.com/data")
    assert "prohibited IP '127.0.0.1'" in str(exc_info.value)

    # Rebind to metadata
    with pytest.raises(PermissionDeniedError) as exc_info:
        sandbox.validate_network("http://cloud-probe.attacker.com/secret")
    assert "prohibited IP '169.254.169.254'" in str(exc_info.value)

    # Split-brain multi-homed resolution containing private IP
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_network("http://split-brain.attacker.com")

    # Legitimate external host passes
    assert (
        sandbox.validate_network("https://legit.example.com/api")
        == "https://legit.example.com/api"
    )


def test_f13_port_validation():
    """Verify out-of-range ports and unpermitted ports are rejected."""
    policy = CapabilityPolicy(
        allow_network=True,
        allowed_ports={80, 443, 8443},
    )
    sandbox = SecuritySandbox(policy=policy)

    # Allowed ports
    assert (
        sandbox.validate_network("https://api.github.com:443")
        == "https://api.github.com:443"
    )
    assert (
        sandbox.validate_network("https://api.github.com:8443")
        == "https://api.github.com:8443"
    )

    # Disallowed port
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_network("https://api.github.com:22")

    # Invalid port range (0 or > 65535)
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_network("https://api.github.com:0")

    with pytest.raises(PermissionDeniedError):
        sandbox.validate_network("https://api.github.com:70000")


# ==============================================================================
# 9. Command Whitelist Basename Spoofing & Traversal Hardening Tests
# ==============================================================================


def test_command_whitelist_basename_spoofing_blocked():
    """Validates that path-based commands cannot match bare names in allowed_commands."""
    policy = CapabilityPolicy(allowed_commands={"echo"})
    sandbox = SecuritySandbox(policy)

    spoofed_commands = [
        "../../tmp/echo hello",
        "./echo hello",
        "/tmp/evil/echo hello",
        "/usr/bin/echo hello",
        r"..\..\tools\echo.exe test",
    ]
    for cmd in spoofed_commands:
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_command(cmd)


def test_command_whitelist_vector_basename_spoofing_blocked():
    """Validates that argument vector path-based commands cannot spoof bare names."""
    policy = CapabilityPolicy(allowed_commands={"python3"})
    sandbox = SecuritySandbox(policy)

    with pytest.raises(PermissionDeniedError):
        sandbox.validate_command(["/tmp/evil/python3", "exploit.py"])
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_command(["./python3", "exploit.py"])


def test_command_whitelist_canonical_path_in_allowed_execute_paths(tmp_path: Path):
    """Validates that path-based commands are permitted when canonical path is in allowed_execute_paths."""
    exec_dir = tmp_path / "bin"
    exec_dir.mkdir(parents=True, exist_ok=True)
    tool_bin = exec_dir / "tool"
    tool_bin.write_text("#!/bin/sh\necho tool", encoding="utf-8")

    policy = CapabilityPolicy(
        allowed_execute_paths=[str(exec_dir)],
        allowed_commands={"tool"},
    )
    sandbox = SecuritySandbox(policy)

    # Allowed because canonical path is within allowed_execute_paths
    cmd = str(tool_bin)
    res = sandbox.validate_command(cmd)
    assert res[0] == cmd

    # Blocked because /tmp/evil/tool is outside allowed_execute_paths
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_command("/tmp/evil/tool")


def test_command_whitelist_exact_full_path_permitted_without_execute_paths():
    """Validates that exact full path in allowed_commands is permitted even without allowed_execute_paths."""
    policy = CapabilityPolicy(
        allowed_commands={"/usr/bin/echo"},
        allowed_execute_paths=[],
    )
    sandbox = SecuritySandbox(policy)

    assert sandbox.validate_command("/usr/bin/echo hello")[0] == "/usr/bin/echo"

    # Other paths blocked
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_command("/bin/echo hello")


# ==============================================================================
# 10. Composite Mode 'x' Execution Containment Tests
# ==============================================================================


def test_composite_mode_rx_execution_containment_blocked(tmp_path: Path):
    """Validates that composite mode 'rx' enforces execute containment."""
    read_dir = tmp_path / "read_dir"
    read_dir.mkdir(parents=True, exist_ok=True)
    file_path = read_dir / "binary.bin"
    file_path.write_bytes(b"DATA")

    # read permitted, execute not permitted
    policy = CapabilityPolicy(
        allowed_read_paths=[str(read_dir)],
        allowed_execute_paths=[],
    )
    sandbox = SecuritySandbox(policy)

    with pytest.raises(PermissionDeniedError) as exc_info:
        sandbox.validate_file_access(str(file_path), mode="rx")
    assert "execute" in str(exc_info.value).lower()


def test_composite_mode_wx_execution_containment_blocked(tmp_path: Path):
    """Validates that composite mode 'wx' enforces execute containment."""
    write_dir = tmp_path / "write_dir"
    write_dir.mkdir(parents=True, exist_ok=True)
    file_path = write_dir / "binary.bin"
    file_path.write_bytes(b"DATA")

    policy = CapabilityPolicy(
        allowed_write_paths=[str(write_dir)],
        allowed_execute_paths=[],
    )
    sandbox = SecuritySandbox(policy)

    with pytest.raises(PermissionDeniedError):
        sandbox.validate_file_access(str(file_path), mode="wx")


def test_composite_mode_rx_permitted_when_both_roots_allowed(tmp_path: Path):
    """Validates that composite mode 'rx' succeeds when target is in both read and execute paths."""
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    file_path = shared_dir / "tool.sh"
    file_path.write_text("#!/bin/sh\n", encoding="utf-8")

    policy = CapabilityPolicy(
        allowed_read_paths=[str(shared_dir)],
        allowed_execute_paths=[str(shared_dir)],
    )
    sandbox = SecuritySandbox(policy)

    res = sandbox.validate_file_access(str(file_path), mode="rx")
    assert res.endswith("tool.sh")


# ==============================================================================
# 11. CommandRule Constraints Enforcement Tests
# ==============================================================================


def test_command_rule_blocked_flags_enforced():
    """Validates that blocked_flags are strictly intercepted."""
    policy = CapabilityPolicy(
        command_rules=[
            CommandRule(executable="git", blocked_flags={"-c", "--config"}),
        ]
    )
    sandbox = SecuritySandbox(policy)

    # Clean git command passes
    assert sandbox.validate_command("git status") == "git status"

    # Blocked flag -c is intercepted
    with pytest.raises(PermissionDeniedError) as exc_info:
        sandbox.validate_command("git -c core.fsmonitor=evil commit")
    assert "blocked" in str(exc_info.value).lower()

    # Blocked flag with equals sign
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_command("git --config=evil commit")

    # Blocked bundled short flag
    with pytest.raises(PermissionDeniedError):
        sandbox.validate_command("git -ac commit")


def test_command_rule_allowed_flags_enforced():
    """Validates that allowed_flags restricts available flags."""
    policy = CapabilityPolicy(
        command_rules=[
            CommandRule(executable="git", allowed_flags={"-v", "--version", "-m"}),
        ]
    )
    sandbox = SecuritySandbox(policy)

    # Allowed flag passes
    assert sandbox.validate_command("git -v") == "git -v"
    assert sandbox.validate_command("git --version") == "git --version"

    # Disallowed flag blocked
    with pytest.raises(PermissionDeniedError) as exc_info:
        sandbox.validate_command("git --exec-path")
    assert "not permitted" in str(exc_info.value).lower()


def test_command_rule_allowed_subcommands_enforced():
    """Validates that allowed_subcommands strictly confines operations."""
    policy = CapabilityPolicy(
        command_rules=[
            CommandRule(
                executable="git",
                allowed_subcommands={"status", "commit"},
            )
        ]
    )
    sandbox = SecuritySandbox(policy)

    # Allowed subcommands pass
    assert sandbox.validate_command("git status") == "git status"
    assert sandbox.validate_command("git commit -m initial") == "git commit -m initial"

    # Disallowed subcommand blocked
    with pytest.raises(PermissionDeniedError) as exc_info:
        sandbox.validate_command("git push origin main")
    assert "subcommand 'push' is not permitted" in str(exc_info.value).lower()

    # Missing subcommand blocked
    with pytest.raises(PermissionDeniedError) as exc_info:
        sandbox.validate_command("git")
    assert "missing required subcommand" in str(exc_info.value).lower()


def test_command_rule_max_args_enforced():
    """Validates that max_args restricts argument count."""
    policy = CapabilityPolicy(
        command_rules=[
            CommandRule(executable="echo", max_args=2),
        ]
    )
    sandbox = SecuritySandbox(policy)

    # <= 2 args pass
    assert sandbox.validate_command("echo 1 2") == "echo 1 2"

    # > 2 args blocked
    with pytest.raises(PermissionDeniedError) as exc_info:
        sandbox.validate_command("echo 1 2 3")
    assert "exceeds maximum allowed args" in str(exc_info.value).lower()


# ==============================================================================
# 12. SSRF, NAT64, DNS & Dynamic Policy Hardening Tests
# ==============================================================================


def test_f13_ipv4_compatible_and_nat64_ssrf_blocked():
    """Verify IPv4-compatible (::/96) and NAT64 (64:ff9b::/96) SSRF attempts are blocked."""
    policy = CapabilityPolicy(allow_network=True)
    sandbox = SecuritySandbox(policy=policy)

    evasion_targets = [
        "http://[::169.254.169.254]/latest/meta-data/",
        "http://[64:ff9b::169.254.169.254]/latest/meta-data/",
        "http://[::127.0.0.1]:8080/admin",
        "http://[64:ff9b::127.0.0.1]/",
        "http://[::10.0.0.1]:9000/",
        "http://[64:ff9b::10.0.0.1]/",
        "::169.254.169.254",
        "64:ff9b::169.254.169.254",
        "::127.0.0.1",
        "64:ff9b::127.0.0.1",
        "::10.0.0.1",
        "64:ff9b::10.0.0.1",
    ]
    for target in evasion_targets:
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_network(target)


def test_f13_ipv6_site_local_fec0_blocked():
    """Verify IPv6 Site-Local addresses (fec0::/10) are blocked."""
    policy = CapabilityPolicy(allow_network=True)
    sandbox = SecuritySandbox(policy=policy)

    targets = [
        "http://[fec0::1]/",
        "http://[fec0::215:5dff:fe12:3456]:8080",
        "http://[fedf:ffff:ffff:ffff:ffff:ffff:ffff:ffff]/",
        "fec0::1",
        "fedf:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
    ]
    for target in targets:
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_network(target)


def test_f13_nat64_public_ip_permitted():
    """Verify NAT64 translation with public IPv4 destination is permitted."""
    policy = CapabilityPolicy(allow_network=True)
    sandbox = SecuritySandbox(policy=policy)

    # 93.184.216.34 is public (example.com)
    assert (
        sandbox.validate_network("http://[64:ff9b::93.184.216.34]/")
        == "http://[64:ff9b::93.184.216.34]/"
    )


def test_f13_dns_resolution_failure_enforcement():
    """Verify DNS resolution failure handling under allow_unresolved_domains policy."""
    # When allow_unresolved_domains=False (default), unresolved domain must raise PermissionDeniedError
    strict_policy = CapabilityPolicy(allow_network=True, allow_unresolved_domains=False)
    strict_sandbox = SecuritySandbox(policy=strict_policy, dns_resolver=lambda h: [])

    with pytest.raises(PermissionDeniedError) as exc_info:
        strict_sandbox.validate_network("https://unresolvable-domain-xyz.internal")
    assert "DNS resolution failed" in str(exc_info.value)

    # When allow_unresolved_domains=True, unresolved domain passes DNS check
    lenient_policy = CapabilityPolicy(allow_network=True, allow_unresolved_domains=True)
    lenient_sandbox = SecuritySandbox(policy=lenient_policy, dns_resolver=lambda h: [])

    assert (
        lenient_sandbox.validate_network("https://unresolvable-domain-xyz.internal")
        == "https://unresolvable-domain-xyz.internal"
    )


def test_tool_call_network_parameter_validation_all_keys():
    """Verify validate_tool_call inspects url, uri, endpoint, host, destination, and target."""
    policy = CapabilityPolicy(allow_network=False)
    sandbox = SecuritySandbox(policy=policy)

    network_keys = ["url", "uri", "endpoint", "host", "destination", "target"]
    for key in network_keys:
        args = {key: "http://169.254.169.254/latest/meta-data/"}
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_tool_call("http_fetch", args)

    # Verify when allow_network=True and allowed domain, valid calls pass
    allowed_policy = CapabilityPolicy(
        allow_network=True,
        allowed_domains=["api.github.com"],
        allow_unresolved_domains=True,
    )
    allowed_sandbox = SecuritySandbox(policy=allowed_policy)
    for key in network_keys:
        args = {key: "https://api.github.com/v1"}
        allowed_sandbox.validate_tool_call("api_client", args)


def test_dynamic_policy_switches():
    """Verify dynamic policy switches for private IPs, metadata, and CGNAT."""
    # 1. Private IPs allowed for internal mesh
    mesh_policy = CapabilityPolicy(allow_network=True, block_private_ips=False)
    mesh_sb = SecuritySandbox(policy=mesh_policy)
    assert mesh_sb.validate_network("http://10.0.0.1:8080") == "http://10.0.0.1:8080"
    assert mesh_sb.validate_network("http://192.168.1.1/") == "http://192.168.1.1/"
    assert mesh_sb.validate_network("http://[fc00::1]/") == "http://[fc00::1]/"
    assert mesh_sb.validate_network("http://[fec0::1]/") == "http://[fec0::1]/"
    # Metadata and loopback must remain blocked
    with pytest.raises(PermissionDeniedError):
        mesh_sb.validate_network("http://169.254.169.254/")
    with pytest.raises(PermissionDeniedError):
        mesh_sb.validate_network("http://127.0.0.1:9000/")

    # 2. Carrier-Grade NAT allowed
    cgnat_policy = CapabilityPolicy(allow_network=True, block_carrier_grade_nat=False)
    cgnat_sb = SecuritySandbox(policy=cgnat_policy)
    assert cgnat_sb.validate_network("http://100.64.0.1/") == "http://100.64.0.1/"
    with pytest.raises(PermissionDeniedError):
        cgnat_sb.validate_network("http://10.0.0.1/")

    # 3. Cloud metadata explicitly enabled in controlled environment
    meta_policy = CapabilityPolicy(allow_network=True, block_cloud_metadata=False)
    meta_sb = SecuritySandbox(policy=meta_policy)
    assert (
        meta_sb.validate_network("http://169.254.169.254/") == "http://169.254.169.254/"
    )
    assert (
        meta_sb.validate_network("http://[::169.254.169.254]/")
        == "http://[::169.254.169.254]/"
    )
    with pytest.raises(PermissionDeniedError):
        meta_sb.validate_network("http://10.0.0.1/")
