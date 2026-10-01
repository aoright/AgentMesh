"""Unit tests for AgentMesh security sandbox."""

import pytest
from agentmesh.security.sandbox import SecuritySandbox, CapabilityPolicy, PermissionDeniedError


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
