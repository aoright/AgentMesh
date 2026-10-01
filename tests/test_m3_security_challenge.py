"""Empirical security penetration challenge suite for Milestone 3 Iteration 2.

Adversarially challenges:
1. Command Whitelist Basename Spoofing & Path Containment:
   - Relative traversals (../../tmp/echo, ./echo, bin/echo) when only bare command is allowed.
   - Absolute path spoofing (/bin/echo, /tmp/echo) without exact match or allowed_execute_paths.
   - allowed_execute_paths containment and prefix bleed resistance.
   - Symlink escape in executable paths.
   - List vector argument validation.
2. Composite File Mode 'x' Containment:
   - Composite modes ('rx', 'wx', 'rwx', 'r+x', 'w+x').
   - Strict containment enforcement against allowed_execute_paths.
   - Rejection when allowed_execute_paths is not configured.
3. CommandRule Constraints Enforcement:
   - Blocked flags (-c, --eval, bundled short flags like -xf, assignment flags like -c=val).
   - Allowed flags whitelist and bundled short flag checking.
   - Allowed subcommands enforcement and flag position independence.
   - Max argument bounds for string and list vectors.
   - POSIX double-dash (--) option parsing separation.
4. Dual-Stack IPv6 & Advanced SSRF Vectors:
   - IPv4-compatible IPv6 (::/96: ::169.254.169.254, ::127.0.0.1, ::10.0.0.1).
   - NAT64 translation (64:ff9b::/96: 64:ff9b::169.254.169.254, 64:ff9b::127.0.0.1).
   - Deprecated site-local IPv6 (fec0::/10: fec0::1, fedf:ffff:...:ffff).
   - Dynamic policy switches for private IPs, metadata, and loopback.
   - IPv4-mapped IPv6 (::ffff:0:0/96).
5. Tool Call Network Parameter Validation & DNS Policy Enforcement:
   - Inspection of all destination keys (url, uri, endpoint, host, destination, target).
   - SSRF payload blocking via tool call arguments.
   - DNS resolution failure enforcement under allow_unresolved_domains=False.
   - Multi-parameter and mixed-mode tool call validations.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from agentmesh.security.policy import CapabilityPolicy, CommandRule
from agentmesh.security.sandbox import PermissionDeniedError, SecuritySandbox

# ==============================================================================
# Challenge Suite 1: Command Whitelist Basename Spoofing & Path Containment
# ==============================================================================


class TestCommandBasenameSpoofingAndPathContainment:
    """Stress tests verifying that path-based commands cannot spoof bare command names."""

    @pytest.mark.parametrize(
        "traversal_cmd",
        [
            "../../tmp/echo",
            "./echo",
            "../echo",
            "bin/echo",
            "sub/dir/echo",
            "../../../../tmp/echo",
            "..\\..\\tmp\\echo",
            ".\\echo",
            "bin\\echo",
            "./echo hello",
            "../../tmp/echo --arg",
        ],
    )
    def test_relative_traversal_spoofing_blocked_bare_whitelist(
        self, traversal_cmd: str
    ):
        """Relative traversals must not match bare command whitelist entry 'echo'."""
        policy = CapabilityPolicy(allowed_commands={"echo"})
        sandbox = SecuritySandbox(policy=policy)

        with pytest.raises(PermissionDeniedError) as exc_info:
            sandbox.validate_command(traversal_cmd)
        err = str(exc_info.value).lower()
        assert (
            "contains path separators" in err
            or "blocked by policy" in err
            or "not match any exact full path" in err
        )

    @pytest.mark.parametrize(
        "abs_cmd",
        [
            "/bin/echo",
            "/usr/bin/echo",
            "/tmp/echo",
            "/tmp/evil/echo",
            "/var/tmp/echo",
            "/usr/local/bin/echo",
        ],
    )
    def test_absolute_path_spoofing_blocked_without_allowed_execute_paths(
        self, abs_cmd: str
    ):
        """Absolute path must be blocked if only bare command is in allowed_commands."""
        policy = CapabilityPolicy(allowed_commands={"echo"})
        sandbox = SecuritySandbox(policy=policy)

        with pytest.raises(PermissionDeniedError) as exc_info:
            sandbox.validate_command(f"{abs_cmd} payload")
        err = str(exc_info.value).lower()
        assert "not match any exact full path" in err or "blocked by policy" in err

    def test_allowed_execute_paths_confinement_and_rejection(self):
        """Commands with path separators must be confined within allowed_execute_paths."""
        with tempfile.TemporaryDirectory() as safe_dir, tempfile.TemporaryDirectory() as evil_dir:
            safe_echo = Path(safe_dir) / "echo"
            safe_echo.write_text("#!/bin/sh\n")

            evil_echo = Path(evil_dir) / "echo"
            evil_echo.write_text("#!/bin/sh\n")

            policy = CapabilityPolicy(
                allowed_execute_paths=[safe_dir],
                allowed_commands={"echo"},
            )
            sandbox = SecuritySandbox(policy=policy)

            # Executable inside safe_dir is permitted
            res = sandbox.validate_command(f"{safe_echo} success")
            assert str(safe_echo) in res[0]

            # Executable in evil_dir is rejected
            with pytest.raises(PermissionDeniedError) as exc_info:
                sandbox.validate_command(f"{evil_echo} evil")
            assert (
                "unauthorized file execute access blocked"
                in str(exc_info.value).lower()
            )

            # Traversal escape attempting to leave safe_dir is rejected
            traversal_attempt = f"{safe_dir}/../{Path(evil_dir).name}/echo"
            with pytest.raises(PermissionDeniedError):
                sandbox.validate_command(traversal_attempt)

    def test_prefix_bleed_resistance_in_execute_paths(self):
        """Prefix bleed (e.g. /opt/safe_evil vs /opt/safe) must be rejected."""
        with tempfile.TemporaryDirectory() as base_temp:
            safe_root = Path(base_temp) / "safe"
            safe_root.mkdir()
            safe_cmd = safe_root / "mytool"
            safe_cmd.write_text("#!/bin/sh\n")

            bleed_root = Path(base_temp) / "safe_evil"
            bleed_root.mkdir()
            bleed_cmd = bleed_root / "mytool"
            bleed_cmd.write_text("#!/bin/sh\n")

            policy = CapabilityPolicy(
                allowed_execute_paths=[str(safe_root)],
                allowed_commands={"mytool"},
            )
            sandbox = SecuritySandbox(policy=policy)

            # Legitimate path passes
            assert sandbox.validate_command(str(safe_cmd))

            # Prefix-bleed path fails
            with pytest.raises(PermissionDeniedError):
                sandbox.validate_command(str(bleed_cmd))

    def test_symlink_escape_in_executable_paths(self):
        """Symlinks inside allowed_execute_paths pointing outside must be rejected."""
        with tempfile.TemporaryDirectory() as safe_dir, tempfile.TemporaryDirectory() as external_dir:
            ext_target = Path(external_dir) / "target_tool"
            ext_target.write_text("#!/bin/sh\n")

            symlink_cmd = Path(safe_dir) / "tool_link"
            try:
                symlink_cmd.symlink_to(ext_target)
            except OSError:
                pytest.skip("Symlink creation not supported in this environment")

            policy = CapabilityPolicy(
                allowed_execute_paths=[safe_dir],
                allowed_commands={"tool_link", "target_tool"},
            )
            sandbox = SecuritySandbox(policy=policy)

            # Resolving canonical path reveals target is in external_dir -> blocked
            with pytest.raises(PermissionDeniedError) as exc_info:
                sandbox.validate_command(str(symlink_cmd))
            assert (
                "unauthorized file execute access blocked"
                in str(exc_info.value).lower()
            )

    @pytest.mark.parametrize(
        "list_vector",
        [
            ["../../tmp/echo", "arg1", "arg2"],
            ["./echo", "test"],
            ["/tmp/evil/echo"],
            ["bin/echo", "-n"],
        ],
    )
    def test_list_vector_traversal_spoofing_blocked(
        self, list_vector: list[str]
    ):
        """Argument vectors passed as list must enforce the same path containment checks."""
        policy = CapabilityPolicy(allowed_commands={"echo"})
        sandbox = SecuritySandbox(policy=policy)

        with pytest.raises(PermissionDeniedError):
            sandbox.validate_command(list_vector)

    def test_exact_full_path_matching_allowed_without_execute_paths(self):
        """Exact full path in allowed_commands should be accepted even if allowed_execute_paths is empty."""
        with tempfile.TemporaryDirectory() as temp_dir:
            full_path = str(Path(temp_dir) / "specific_binary")
            Path(full_path).write_text("#!/bin/sh\n")

            policy = CapabilityPolicy(
                allowed_commands={full_path},
                allowed_execute_paths=[],
            )
            sandbox = SecuritySandbox(policy=policy)

            # Exact full path is allowed
            assert sandbox.validate_command(f"{full_path} --test")

            # Bare command name is blocked
            with pytest.raises(PermissionDeniedError):
                sandbox.validate_command("specific_binary")

            # Different path with same basename is blocked
            other_path = f"/tmp/{Path(full_path).name}"
            with pytest.raises(PermissionDeniedError):
                sandbox.validate_command(other_path)


# ==============================================================================
# Challenge Suite 2: Composite Mode 'x' Containment
# ==============================================================================


class TestCompositeFileModeContainment:
    """Stress tests verifying that composite file modes containing 'x' enforce execute containment."""

    @pytest.mark.parametrize("composite_mode", ["rx", "wx", "rwx", "r+x", "w+x", "ax", "RX", "RWX"])
    def test_composite_mode_x_rejects_unauthorized_execute_path(
        self, composite_mode: str
    ):
        """When target is in allowed_read/write paths but not allowed_execute_paths, 'x' must reject."""
        with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as exec_dir:
            test_file = Path(data_dir) / "script.sh"
            test_file.write_text("#!/bin/sh\n")

            policy = CapabilityPolicy(
                allowed_read_paths=[data_dir],
                allowed_write_paths=[data_dir],
                allowed_execute_paths=[exec_dir],
            )
            sandbox = SecuritySandbox(policy=policy)

            # Pure read or write works fine
            sandbox.validate_file_access(str(test_file), mode="r")
            sandbox.validate_file_access(str(test_file), mode="w")

            # Any composite mode containing 'x' must be rejected due to execute containment
            with pytest.raises(PermissionDeniedError) as exc_info:
                sandbox.validate_file_access(str(test_file), mode=composite_mode)
            assert (
                "unauthorized file execute access blocked"
                in str(exc_info.value).lower()
            )

    @pytest.mark.parametrize("composite_mode", ["rx", "wx", "rwx"])
    def test_composite_mode_x_rejected_when_no_execute_paths_configured(
        self, composite_mode: str
    ):
        """When allowed_execute_paths is empty, any composite mode with 'x' must fail."""
        with tempfile.TemporaryDirectory() as data_dir:
            test_file = Path(data_dir) / "app.py"
            test_file.write_text("print('hello')\n")

            policy = CapabilityPolicy(
                allowed_read_paths=[data_dir],
                allowed_write_paths=[data_dir],
                allowed_execute_paths=[],  # Empty execute paths
            )
            sandbox = SecuritySandbox(policy=policy)

            with pytest.raises(PermissionDeniedError) as exc_info:
                sandbox.validate_file_access(str(test_file), mode=composite_mode)
            assert "no allowed execute roots configured" in str(exc_info.value).lower()

    def test_composite_mode_rwx_succeeds_when_all_roots_permit(self):
        """Composite mode rwx must pass when target is contained in all three allowed roots."""
        with tempfile.TemporaryDirectory() as shared_dir:
            shared_file = Path(shared_dir) / "shared_binary"
            shared_file.write_text("#!/bin/sh\n")

            policy = CapabilityPolicy(
                allowed_read_paths=[shared_dir],
                allowed_write_paths=[shared_dir],
                allowed_execute_paths=[shared_dir],
            )
            sandbox = SecuritySandbox(policy=policy)

            # Must succeed without error
            safe_path = sandbox.validate_file_access(str(shared_file), mode="rwx")
            assert str(safe_path) == str(shared_file.resolve())


# ==============================================================================
# Challenge Suite 3: CommandRule Constraints Enforcement
# ==============================================================================


class TestCommandRuleConstraintsAdversarial:
    """Stress tests verifying CommandRule flag, subcommand, and argument restrictions."""

    def test_blocked_flags_various_formats(self):
        """Blocked flags must be intercepted across standalone, assignment, and bundled formats."""
        rule = CommandRule(
            executable="git",
            blocked_flags={"-c", "--upload-pack", "--exec-path"},
            max_args=10,
        )
        policy = CapabilityPolicy(allowed_commands={"git"}, command_rules=[rule])
        sandbox = SecuritySandbox(policy=policy)

        # Standalone blocked flag
        with pytest.raises(PermissionDeniedError) as exc:
            sandbox.validate_command("git -c core.fsmonitor=evil status")
        assert "flag '-c' is blocked" in str(exc.value).lower()

        # Assignment blocked flag
        with pytest.raises(PermissionDeniedError) as exc:
            sandbox.validate_command("git -c=foo status")
        assert "is blocked" in str(exc.value).lower()

        # Long blocked flag with value
        with pytest.raises(PermissionDeniedError) as exc:
            sandbox.validate_command("git --upload-pack=/bin/evil fetch")
        assert "is blocked" in str(exc.value).lower()

    def test_bundled_short_flags_blocked(self):
        """Bundled short flags (e.g. tar -xf) must detect embedded blocked flags."""
        rule = CommandRule(
            executable="tar",
            blocked_flags={"-f"},
            max_args=10,
        )
        policy = CapabilityPolicy(allowed_commands={"tar"}, command_rules=[rule])
        sandbox = SecuritySandbox(policy=policy)

        # Unbundled -f
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_command("tar -x -f archive.tar")

        # Bundled -xf contains -f
        with pytest.raises(PermissionDeniedError) as exc:
            sandbox.validate_command("tar -xf archive.tar")
        assert "flag '-f' is blocked" in str(exc.value).lower()

        # Bundled -zxvf contains -f
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_command("tar -zxvf archive.tar.gz")

    def test_allowed_flags_whitelist_enforcement(self):
        """Only explicitly allowed flags must be accepted when allowed_flags is specified."""
        rule = CommandRule(
            executable="ls",
            allowed_flags={"-l", "-a", "-h", "--color"},
            max_args=10,
        )
        policy = CapabilityPolicy(allowed_commands={"ls"}, command_rules=[rule])
        sandbox = SecuritySandbox(policy=policy)

        # Single allowed flag
        assert sandbox.validate_command("ls -l")

        # Bundled allowed flags
        assert sandbox.validate_command("ls -la")
        assert sandbox.validate_command("ls -lah")

        # Long allowed flag
        assert sandbox.validate_command("ls --color=always")

        # Unallowed flag
        with pytest.raises(PermissionDeniedError) as exc:
            sandbox.validate_command("ls -R")
        assert "flag '-r' is not permitted" in str(exc.value).lower()

        # Bundled containing unallowed flag
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_command("ls -laR")

    def test_allowed_subcommands_enforcement(self):
        """Commands requiring subcommands must enforce the allowed_subcommands whitelist."""
        rule = CommandRule(
            executable="git",
            allowed_subcommands={"status", "diff", "log"},
            max_args=10,
        )
        policy = CapabilityPolicy(allowed_commands={"git"}, command_rules=[rule])
        sandbox = SecuritySandbox(policy=policy)

        # Allowed subcommands
        assert sandbox.validate_command("git status")
        assert sandbox.validate_command("git diff HEAD~1")
        assert sandbox.validate_command("git log -n 5")

        # Disallowed subcommands
        with pytest.raises(PermissionDeniedError) as exc:
            sandbox.validate_command("git commit -m evil")
        assert "subcommand 'commit' is not permitted" in str(exc.value).lower()

        with pytest.raises(PermissionDeniedError) as exc:
            sandbox.validate_command("git push origin main")
        assert "subcommand 'push' is not permitted" in str(exc.value).lower()

        # Missing subcommand entirely
        with pytest.raises(PermissionDeniedError) as exc:
            sandbox.validate_command("git")
        assert "missing required subcommand" in str(exc.value).lower()

    def test_max_args_bounds_enforcement(self):
        """Commands exceeding max_args bound must be rejected."""
        rule = CommandRule(
            executable="echo",
            max_args=2,
        )
        policy = CapabilityPolicy(allowed_commands={"echo"}, command_rules=[rule])
        sandbox = SecuritySandbox(policy=policy)

        assert sandbox.validate_command("echo")
        assert sandbox.validate_command("echo 1")
        assert sandbox.validate_command("echo 1 2")

        with pytest.raises(PermissionDeniedError) as exc:
            sandbox.validate_command("echo 1 2 3")
        assert "exceeds maximum allowed args (2)" in str(exc.value).lower()

        with pytest.raises(PermissionDeniedError):
            sandbox.validate_command(["echo", "1", "2", "3"])

    def test_posix_double_dash_option_parsing(self):
        """Positional arguments after -- must not be parsed as flags."""
        rule = CommandRule(
            executable="git",
            allowed_subcommands={"status"},
            blocked_flags={"-c"},
            max_args=10,
        )
        policy = CapabilityPolicy(allowed_commands={"git"}, command_rules=[rule])
        sandbox = SecuritySandbox(policy=policy)

        # -c before -- is parsed as flag -> blocked
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_command("git -c foo=bar status")

        # -c after -- is a path parameter, not a flag -> allowed
        res = sandbox.validate_command("git status -- -c")
        assert res == ["git", "status", "--", "-c"]


# ==============================================================================
# Challenge Suite 4: Dual-Stack IPv6 & Advanced SSRF Vectors
# ==============================================================================


class TestDualStackAndAdvancedSSRFVectors:
    """Stress tests for IPv4-compatible (::/96), NAT64 (64:ff9b::/96), and site-local (fec0::/10)."""

    @pytest.mark.parametrize(
        "target",
        [
            "http://[::169.254.169.254]/latest/meta-data/",
            "http://[::169.254.169.254]:8080/computeMetadata/v1/",
            "http://[::127.0.0.1]/",
            "http://[::127.0.0.1]:9000/admin",
            "http://[::10.0.0.1]:8080/internal",
            "http://[::172.16.0.1]/",
            "http://[::192.168.1.1]/",
            "http://[::100.64.0.1]/",
            "::169.254.169.254",
            "::127.0.0.1",
            "::10.0.0.1",
            "::192.168.1.1",
            "::100.64.0.1",
        ],
    )
    def test_ipv4_compatible_ipv6_ssrf_blocked(self, target: str):
        """IPv4-compatible IPv6 addresses (::/96) embedding private/metadata IPs must be blocked."""
        policy = CapabilityPolicy(allow_network=True)
        sandbox = SecuritySandbox(policy=policy)

        with pytest.raises(PermissionDeniedError) as exc:
            sandbox.validate_network(target)
        assert "prohibited by ssrf policy" in str(exc.value).lower()

    @pytest.mark.parametrize(
        "target",
        [
            "http://[64:ff9b::169.254.169.254]/latest/meta-data/",
            "http://[64:ff9b::169.254.0.1]/",
            "http://[64:ff9b::127.0.0.1]/",
            "http://[64:ff9b::127.0.0.1]:8080/",
            "http://[64:ff9b::10.0.0.1]/",
            "http://[64:ff9b::172.31.255.254]/",
            "http://[64:ff9b::192.168.100.1]/",
            "http://[64:ff9b::100.64.0.1]/",
            "64:ff9b::169.254.169.254",
            "64:ff9b::127.0.0.1",
            "64:ff9b::10.0.0.1",
            "64:ff9b::192.168.1.1",
        ],
    )
    def test_nat64_ssrf_blocked(self, target: str):
        """RFC 6052 NAT64 addresses (64:ff9b::/96) embedding private/metadata IPs must be blocked."""
        policy = CapabilityPolicy(allow_network=True)
        sandbox = SecuritySandbox(policy=policy)

        with pytest.raises(PermissionDeniedError) as exc:
            sandbox.validate_network(target)
        assert "prohibited by ssrf policy" in str(exc.value).lower()

    @pytest.mark.parametrize(
        "target",
        [
            "http://[fec0::1]/",
            "http://[fec0::1234:5678]:8080/",
            "http://[fedf:ffff:ffff:ffff:ffff:ffff:ffff:ffff]/",
            "fec0::1",
            "fec0::dead:beef",
            "fedf:ffff:ffff:ffff::1",
        ],
    )
    def test_ipv6_site_local_fec0_blocked(self, target: str):
        """Deprecated IPv6 site-local addresses (fec0::/10) must be blocked as private IPs."""
        policy = CapabilityPolicy(allow_network=True)
        sandbox = SecuritySandbox(policy=policy)

        with pytest.raises(PermissionDeniedError) as exc:
            sandbox.validate_network(target)
        assert "prohibited by ssrf policy" in str(exc.value).lower()

    def test_nat64_public_destination_allowed(self):
        """NAT64 translation to a public IPv4 address (e.g. 93.184.216.34) should be permitted."""
        policy = CapabilityPolicy(allow_network=True)
        sandbox = SecuritySandbox(policy=policy)

        url = "http://[64:ff9b::93.184.216.34]/"
        assert sandbox.validate_network(url) == url

    def test_dynamic_switches_with_dual_stack(self):
        """Dynamic switches must properly govern dual-stack and translated addresses."""
        # 1. Allow private IPs -> IPv4-compatible private IP permitted, metadata still blocked
        private_policy = CapabilityPolicy(allow_network=True, block_private_ips=False)
        private_sb = SecuritySandbox(policy=private_policy)

        assert (
            private_sb.validate_network("http://[::10.0.0.1]/")
            == "http://[::10.0.0.1]/"
        )
        assert (
            private_sb.validate_network("http://[64:ff9b::192.168.1.1]/")
            == "http://[64:ff9b::192.168.1.1]/"
        )
        assert (
            private_sb.validate_network("http://[fec0::1]/")
            == "http://[fec0::1]/"
        )

        with pytest.raises(PermissionDeniedError):
            private_sb.validate_network("http://[::169.254.169.254]/")

        # 2. Allow metadata -> IPv4-compatible metadata IP permitted, private IP still blocked
        meta_policy = CapabilityPolicy(allow_network=True, block_cloud_metadata=False)
        meta_sb = SecuritySandbox(policy=meta_policy)

        assert (
            meta_sb.validate_network("http://[::169.254.169.254]/")
            == "http://[::169.254.169.254]/"
        )
        assert (
            meta_sb.validate_network("http://[64:ff9b::169.254.169.254]/")
            == "http://[64:ff9b::169.254.169.254]/"
        )

        with pytest.raises(PermissionDeniedError):
            meta_sb.validate_network("http://[::10.0.0.1]/")


# ==============================================================================
# Challenge Suite 5: Tool Call Network Parameter Validation & DNS Enforcement
# ==============================================================================


class TestToolCallAndDNSValidationAdversarial:
    """Stress tests for tool call network validation and DNS resolution enforcement."""

    @pytest.mark.parametrize(
        "dest_key", ["url", "uri", "endpoint", "host", "destination", "target"]
    )
    @pytest.mark.parametrize(
        "payload",
        [
            "http://169.254.169.254/latest/meta-data/",
            "http://[::169.254.169.254]/",
            "http://[64:ff9b::169.254.169.254]/",
            "http://127.0.0.1:8080/",
            "http://[::1]:9090/",
            "http://10.0.0.1:3000/",
            "http://[fec0::1]/",
            "169.254.169.254:80",
            "127.0.0.1:8080",
            "[::1]:8000",
            "file:///etc/shadow",
            "http://metadata.google.internal/",
            "http://localhost:8080",
        ],
    )
    def test_tool_call_all_network_keys_intercepted(
        self, dest_key: str, payload: str
    ):
        """validate_tool_call must intercept prohibited network destinations across all standard keys."""
        policy = CapabilityPolicy(allow_network=True)
        sandbox = SecuritySandbox(policy=policy)

        tool_args = {dest_key: payload}
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_tool_call("network_tool", tool_args)

    def test_tool_call_with_allowed_public_domain(self):
        """validate_tool_call must succeed when destination conforms to domain whitelist."""
        policy = CapabilityPolicy(
            allow_network=True,
            allowed_domains=["api.github.com"],
            allow_unresolved_domains=True,
        )
        sandbox = SecuritySandbox(policy=policy)

        for key in ["url", "uri", "endpoint", "host", "destination", "target"]:
            tool_args = {key: "https://api.github.com/v1"}
            sandbox.validate_tool_call("git_tool", tool_args)

    def test_tool_call_multi_parameter_mixed_modes(self):
        """Tool call containing valid file but malicious network destination must fail."""
        with tempfile.TemporaryDirectory() as temp_dir:
            safe_file = Path(temp_dir) / "data.json"
            safe_file.write_text("{}")

            policy = CapabilityPolicy(
                allow_network=True,
                allowed_read_paths=[temp_dir],
            )
            sandbox = SecuritySandbox(policy=policy)

            # File is safe, but URL is SSRF payload -> blocked
            mixed_args = {
                "file_path": str(safe_file),
                "url": "http://169.254.169.254/latest/meta-data/",
            }
            with pytest.raises(PermissionDeniedError):
                sandbox.validate_tool_call("fetch_and_read", mixed_args)

    def test_tool_call_multiple_network_keys_partial_malicious(self):
        """If one network parameter is safe and another is malicious, call must fail."""
        policy = CapabilityPolicy(
            allow_network=True,
            allowed_domains=["api.github.com"],
            allow_unresolved_domains=True,
        )
        sandbox = SecuritySandbox(policy=policy)

        args = {
            "endpoint": "https://api.github.com/v1",
            "destination": "http://127.0.0.1:8080/admin",
        }
        with pytest.raises(PermissionDeniedError):
            sandbox.validate_tool_call("proxy_tool", args)

    def test_dns_resolution_failure_policy_enforcement(self):
        """When allow_unresolved_domains=False, DNS resolution failure must raise PermissionDeniedError."""
        # Resolver returning empty list (resolution failure)
        strict_policy = CapabilityPolicy(
            allow_network=True,
            allow_unresolved_domains=False,
            allowed_domains=["safe.external.com"],
        )
        strict_sb = SecuritySandbox(policy=strict_policy, dns_resolver=lambda h: [])

        with pytest.raises(PermissionDeniedError) as exc_info:
            strict_sb.validate_network("https://safe.external.com/data")
        assert "dns resolution failed" in str(exc_info.value).lower()

        # Lenient policy: allow_unresolved_domains=True
        lenient_policy = CapabilityPolicy(
            allow_network=True,
            allow_unresolved_domains=True,
            allowed_domains=["safe.external.com"],
        )
        lenient_sb = SecuritySandbox(policy=lenient_policy, dns_resolver=lambda h: [])
        assert (
            lenient_sb.validate_network("https://safe.external.com/data")
            == "https://safe.external.com/data"
        )

    def test_dns_rebinding_prohibited_ip_enforcement(self):
        """Even if domain is in allowed_domains, if DNS resolves to private/loopback IP, it must be blocked."""
        policy = CapabilityPolicy(
            allow_network=True,
            allowed_domains=["trusted-partner.com"],
        )

        rebinding_scenarios = [
            ["127.0.0.1"],
            ["169.254.169.254"],
            ["10.0.0.1"],
            ["192.168.1.1"],
            ["::1"],
            ["::169.254.169.254"],
            ["64:ff9b::127.0.0.1"],
            ["fec0::1"],
        ]

        from collections.abc import Callable

        def make_resolver(ips: list[str]) -> Callable[[str], list[str]]:
            def resolver(h: str) -> list[str]:
                return ips

            return resolver

        for resolved_ips in rebinding_scenarios:
            sb = SecuritySandbox(policy=policy, dns_resolver=make_resolver(resolved_ips))
            with pytest.raises(PermissionDeniedError) as exc_info:
                sb.validate_network("https://trusted-partner.com/api")
            assert (
                "resolved to prohibited ip" in str(exc_info.value).lower()
                or "prohibited" in str(exc_info.value).lower()
            )

    def test_dns_multi_ip_mixed_public_and_private_blocked(self):
        """If DNS resolves to multiple IPs where at least one is private/metadata, it must be blocked."""
        from collections.abc import Callable

        policy = CapabilityPolicy(
            allow_network=True,
            allowed_domains=["mixed-records.example.com"],
        )

        mixed_dns_cases = [
            ["93.184.216.34", "10.0.0.1"],
            ["93.184.216.34", "169.254.169.254"],
            ["93.184.216.34", "127.0.0.1"],
            ["93.184.216.34", "::1"],
            ["93.184.216.34", "::169.254.169.254"],
            ["93.184.216.34", "64:ff9b::127.0.0.1"],
            ["93.184.216.34", "fec0::1"],
        ]

        def make_resolver(ips: list[str]) -> Callable[[str], list[str]]:
            def resolver(h: str) -> list[str]:
                return ips

            return resolver

        for ips in mixed_dns_cases:
            sb = SecuritySandbox(policy=policy, dns_resolver=make_resolver(ips))
            with pytest.raises(PermissionDeniedError) as exc:
                sb.validate_network("https://mixed-records.example.com/endpoint")
            assert "prohibited ip" in str(exc.value).lower()

    @pytest.mark.parametrize(
        "hex_ipv6_target",
        [
            "http://[::7f00:1]/",  # Hex 127.0.0.1 in ::/96
            "http://[::a9fe:a9fe]/",  # Hex 169.254.169.254 in ::/96
            "http://[::0a00:0001]/",  # Hex 10.0.0.1 in ::/96
            "http://[64:ff9b::7f00:1]/",  # Hex 127.0.0.1 in NAT64
            "http://[64:ff9b::a9fe:a9fe]/",  # Hex 169.254.169.254 in NAT64
            "http://[64:ff9b::0a00:0001]/",  # Hex 10.0.0.1 in NAT64
            "::7f00:1",
            "::a9fe:a9fe",
            "64:ff9b::7f00:1",
            "64:ff9b::a9fe:a9fe",
        ],
    )
    def test_hex_encoded_embedded_ipv4_in_ipv6_blocked(self, hex_ipv6_target: str):
        """Embedded IPv4 addresses represented in hex format within ::/96 and 64:ff9b::/96 must be blocked."""
        policy = CapabilityPolicy(allow_network=True)
        sandbox = SecuritySandbox(policy=policy)

        with pytest.raises(PermissionDeniedError):
            sandbox.validate_network(hex_ipv6_target)

    def test_command_rule_list_vectors(self):
        """CommandRule constraints must be strictly enforced on list argument vectors."""
        rule = CommandRule(
            executable="git",
            allowed_subcommands={"status", "fetch"},
            blocked_flags={"-c", "--upload-pack"},
            allowed_flags={"-v", "--verbose"},
            max_args=3,
        )
        policy = CapabilityPolicy(allowed_commands={"git"}, command_rules=[rule])
        sandbox = SecuritySandbox(policy=policy)

        # Valid list vector
        assert sandbox.validate_command(["git", "status", "-v"])

        # Blocked flag in list vector
        with pytest.raises(PermissionDeniedError) as exc:
            sandbox.validate_command(["git", "-c", "core.fsmonitor=evil", "status"])
        assert "is blocked" in str(exc.value).lower()

        # Unallowed subcommand in list vector
        with pytest.raises(PermissionDeniedError) as exc:
            sandbox.validate_command(["git", "push", "origin"])
        assert "subcommand 'push' is not permitted" in str(exc.value).lower()

        # Max args exceeded in list vector
        with pytest.raises(PermissionDeniedError) as exc:
            sandbox.validate_command(["git", "status", "-v", "extra1", "extra2"])
        assert "exceeds maximum allowed args" in str(exc.value).lower()

