# AgentMesh Test Infrastructure Specification (TEST_INFRA.md)

Document ID: AGENTMESH-TEST-INFRA-20261001  
Version: 1.0.0  
Working Directory: /Users/liuyukai/CREATE/OSCHINA  
Compliance: Strict Plain-Text (Zero Emojis), Physical State Separation, Opaque-Box Requirement-Driven Testing  

---

## 1. Test Architecture Overview

The AgentMesh test infrastructure implements an opaque-box, requirement-driven, 4-tier progressive verification framework. The testing philosophy adheres to the following principles:

1. **Opaque-Box Requirement Verification**: Tests validate observable public contracts, protocol envelopes, and state transitions against the formal requirements (R1-R4) defined in `ORIGINAL_REQUEST.md` and `PROJECT.md`.
2. **Four-Tier Progressive Testability**: Tests are stratified into four distinct operational tiers:
   - Tier 1: Feature Coverage across all 20 inventoried features.
   - Tier 2: Boundary, limit, empty-state, and corner-case tests.
   - Tier 3: Cross-feature and multi-layer interaction tests.
   - Tier 4: Real-world enterprise application scenario workflows.
3. **Physical State and Code Separation**: In accordance with enterprise deployment safety rules, all test checkpoint snapshots, event logs, and temporary state are strictly isolated to dedicated directories outside the repository tree (e.g., temporary directories via `isolated_state_dir` fixture). Zero database binaries (`*.db`, `*.sqlite*`, `*.wal`, `*.shm`, `*.dump`) or checkpoint JSON files are permitted within the repository.
4. **Deterministic and Self-Contained Execution**: Each test sets up its own state, executes deterministically, and cleans up its artifacts upon completion. No test depends on execution order or persistent side effects.

---

## 2. Directory Layout & Test Suite Structure

```
tests/
├── __init__.py
├── conftest.py                     # Base fixtures
├── test_engine.py                  # Unit tests: Core deterministic engine
├── test_mesh.py                    # Unit tests: Mesh router & MCP client
├── test_security.py                # Unit tests: Capability sandbox
└── e2e/                            # End-to-End Test Track (Exclusive Ownership)
    ├── __init__.py
    ├── conftest.py                 # E2E shared fixtures (isolated_state_dir, etc.)
    ├── test_tier1_features.py      # Tier 1: All 20 features (41 tests)
    ├── test_tier2_boundaries.py    # Tier 2: Limits & boundary conditions (22 tests)
    ├── test_tier3_combinations.py  # Tier 3: Cross-layer integrations (7 tests)
    └── test_tier4_scenarios.py     # Tier 4: Enterprise application scenarios (5 tests)
```

Total E2E test count: 75 tests.  
Total project test count: 82 tests.  
All 82 tests pass in under 0.35 seconds via `pytest`.

---

## 3. Feature Coverage Matrix (20 Inventoried Features)

The table below maps each feature from `PROJECT.md` to its corresponding E2E test cases:

| Feature ID | Feature Name | Tier 1 Test Function(s) | Tier 2 / 3 / 4 Cross-Coverage |
| :--- | :--- | :--- | :--- |
| **F01** | Activity Abstraction & ActivityRecord | `test_f01_activity_record_event_logging`, `test_f01_activity_payload_serialization` | `test_tier2_state_event_unicode_and_special_characters` |
| **F02** | Activity Memoizer & Idempotency Keys | `test_f02_idempotency_key_derivation_deterministic`, `test_f02_memoization_avoids_duplicate_activity` | `test_tier3_crash_recovery_skips_mesh_call_without_duplicate` |
| **F03** | Cryptographic Event Hash Chain & Anti-Tamper | `test_f03_event_chain_continuity_valid`, `test_f03_event_tampering_detected`, `test_f03_missing_event_causes_chain_failure` | `test_f20_corrupted_hash_chain_detection`, `test_scenario_disaster_recovery_and_chaos_self_healing` |
| **F04** | Autonomous Breakpoint Resumption | `test_f04_find_last_completed_node`, `test_f04_resume_from_crash_restores_state`, `test_f04_graph_resumption_skips_completed_nodes` | `test_tier3_crash_recovery_skips_mesh_call_without_duplicate`, `test_scenario_disaster_recovery_and_chaos_self_healing` |
| **F05** | Physical State Separation Engine | `test_f05_checkpoints_isolated_to_external_dir`, `test_f05_zero_state_and_db_files_in_code_tree` | `test_tier2_checkpoint_nested_dir_auto_creation`, `test_scenario_batch_data_pipeline_with_state_isolation` |
| **F06** | MCP 2.0 JSON-RPC Wire Protocol | `test_f06_mcp_tool_registration_and_list`, `test_f06_mcp_tool_invocation_success`, `test_f06_mcp_tool_invocation_not_found` | `test_tier2_mcp_handler_exception_isolated`, `test_tier3_mcp_tool_guarded_by_sandbox`, `test_scenario_financial_compliance_and_aml_audit` |
| **F07** | A2A Cross-Agent Protocol | `test_f07_a2a_message_exchange`, `test_f07_a2a_agent_endpoint_roles` | `test_tier3_engine_orchestrates_mesh_pipeline`, `test_tier3_mesh_distributed_tracing` |
| **F08** | Dynamic Service Discovery & Registry | `test_f08_endpoint_registration_and_lookup`, `test_f08_unregistered_agent_lookup_raises_error` | `test_tier2_router_unhandled_failure_without_fallback` |
| **F09** | Weighted Traffic Routing | `test_f09_agent_endpoint_weight_and_active_properties`, `test_f09_weighted_endpoint_pool_representation` | `test_tier3_engine_orchestrates_mesh_pipeline` |
| **F10** | Circuit Breaker & Fallback Topology | `test_f10_circuit_breaker_trips_to_open_on_threshold`, `test_f10_circuit_breaker_transition_latency_sub_5ms`, `test_f10_automatic_fallback_routing` | `test_tier2_circuit_breaker_threshold_one_trips_immediately`, `test_tier3_circuit_breaker_failover_with_telemetry`, `test_scenario_multi_agent_cascade_failure_graceful_degradation` |
| **F11** | Filesystem Sandbox & Mode 'x' Defense | `test_f11_allowed_read_and_write_paths`, `test_f11_path_traversal_blocked`, `test_f11_read_path_cannot_be_written` | `test_tier2_sandbox_exact_path_and_prefix_boundary`, `test_tier3_engine_sandbox_policy_enforcement`, `test_scenario_prompt_injection_and_sandbox_containment` |
| **F12** | Command Whitelist & Anti-Chaining | `test_f12_allowed_commands_pass`, `test_f12_unauthorized_commands_blocked` | `test_tier2_sandbox_whitespace_command`, `test_scenario_prompt_injection_and_sandbox_containment` |
| **F13** | Network Sandbox & SSRF Isolation | `test_f13_network_disabled_blocks_all_urls`, `test_f13_network_enabled_allows_valid_urls` | `test_tier2_sandbox_empty_policy_blocks_all`, `test_scenario_prompt_injection_and_sandbox_containment` |
| **F14** | W3C TraceContext Distributed Tracing | `test_f14_trace_context_and_spans`, `test_f14_span_attributes_and_error_capture` | `test_tier3_mesh_distributed_tracing`, `test_scenario_financial_compliance_and_aml_audit` |
| **F15** | Token Accounting & Audit Logging | `test_f15_token_usage_accounting`, `test_f15_tracer_export_summary` | `test_tier2_telemetry_zero_and_negative_tokens`, `test_tier3_circuit_breaker_failover_with_telemetry`, `test_scenario_financial_compliance_and_aml_audit` |
| **F16** | Chaos Recovery Verification Demo | `test_f16_chaos_recovery_end_to_end` | `test_scenario_disaster_recovery_and_chaos_self_healing` |
| **F17** | Financial Audit Workflow Demo | `test_f17_financial_audit_pipeline_end_to_end` | `test_scenario_financial_compliance_and_aml_audit` |
| **F18** | Static Typing & PEP 8 Compliance | `test_f18_public_api_exports_and_imports` | Enforced via ruff check (zero warnings) and clean module interfaces |
| **F19** | Comprehensive E2E Test Suite (Tiers 1-4) | `test_f19_multi_step_dag_monotonic_state_evolution` | Validated by entire `tests/e2e/` test suite suite execution |
| **F20** | Adversarial Coverage Hardening (Tier 5) | `test_f20_corrupted_hash_chain_detection`, `test_f20_sandbox_path_null_byte_rejection` | `test_scenario_prompt_injection_and_sandbox_containment` |

---

## 4. Test Fixtures and Isolation Architecture

Test fixtures are co-located in `tests/e2e/conftest.py`:

1. **`isolated_state_dir(tmp_path: Path)`**:
   - Creates a temporary directory outside the code repository tree.
   - Sets `os.environ["AGENTMESH_STATE_DIR"]` for the duration of the test.
   - Automatically cleans up and restores original environment variables upon test teardown.
2. **`checkpoint_manager(isolated_state_dir: Path)`**:
   - Injects a `CheckpointManager` pointing to `isolated_state_dir`.
   - Ensures that all snapshots, WAL files, and temporary writes remain completely isolated from the code tree.
3. **`sandbox_fixture(tmp_path: Path)`**:
   - Provisions `sandbox_safe_read` and `sandbox_safe_write` directories under `tmp_path`.
   - Seeds sample read-only financial data (`ledger_source.txt`).
   - Configures a restrictive `CapabilityPolicy` blocking network, limiting commands to `{"echo", "cat", "grep"}`, and setting strict path boundaries.
4. **`sample_mcp_client()`**:
   - Provisions an `MCPToolClient` registered with banking query and tax calculation tools.
5. **`sample_mesh_router()`**:
   - Provisions an active `MeshRouter` instance.
6. **`sample_tracer()`**:
   - Provisions an `AgentTracer` instance for distributed tracing.

---

## 5. Verification Commands and Benchmark Targets

### How to Run the Test Suite
```bash
# Run the entire test suite (unit + E2E)
pytest

# Run only the E2E test suite
pytest tests/e2e/ -v

# Run by tier
pytest tests/e2e/test_tier1_features.py -v
pytest tests/e2e/test_tier2_boundaries.py -v
pytest tests/e2e/test_tier3_combinations.py -v
pytest tests/e2e/test_tier4_scenarios.py -v

# Lint verification on E2E tests
ruff check tests/e2e/
```

### Empirical Benchmark Targets & Results
- **Recovery Latency**: Acceptance target < 200 ms. Measured in `test_scenario_disaster_recovery_and_chaos_self_healing`: < 2.5 ms.
- **Circuit Breaker Trip Latency**: Acceptance target < 5.0 ms. Measured in `test_f10_circuit_breaker_transition_latency_sub_5ms` and `test_scenario_multi_agent_cascade_failure_graceful_degradation`: < 0.25 ms.
- **Duplicate Token Consumption on Resume**: Acceptance target strictly 0.0%. Verified in `test_tier3_crash_recovery_skips_mesh_call_without_duplicate` and `test_scenario_disaster_recovery_and_chaos_self_healing`.
- **Path Traversal Interception Rate**: Acceptance target 100.0%. Verified across 9 distinct adversarial attack vectors in `test_scenario_prompt_injection_and_sandbox_containment`.
- **Physical State Separation**: 0 database binaries or runtime state files in repository tree. Verified in `test_f05_zero_state_and_db_files_in_code_tree` and `test_scenario_batch_data_pipeline_with_state_isolation`.

---

## 6. Implementation Defect Escalation

During E2E test development, the following defect was discovered in the baseline implementation code and is formally escalated:

### Defect: `AgentTracer.get_or_create_context` Overwrites Existing Context
- **Location**: `agentmesh/telemetry/tracer.py:63-66`
- **Observed Code**:
  ```python
  def get_or_create_context(self, trace_id: Optional[str] = None) -> TraceContext:
      ctx = TraceContext(trace_id=trace_id)
      self.active_contexts[ctx.trace_id] = ctx
      return ctx
  ```
- **Defect Description**: The method is named `get_or_create_context`, but it unconditionally constructs a new `TraceContext` and overwrites `self.active_contexts[ctx.trace_id]`. If an agent receives an existing `trace_id` from an upstream caller and calls `tracer.get_or_create_context(trace_id=trace_id)`, the existing context (and all its root spans) is overwritten, destroying the parent trace context.
- **Recommended Remediation in Implementation**:
  ```python
  def get_or_create_context(self, trace_id: Optional[str] = None) -> TraceContext:
      if trace_id and trace_id in self.active_contexts:
          return self.active_contexts[trace_id]
      ctx = TraceContext(trace_id=trace_id)
      self.active_contexts[ctx.trace_id] = ctx
      return ctx
  ```
- **Workaround in Test Code**: Tests access `tracer.active_contexts.get(trace_id) or tracer.get_or_create_context(trace_id=trace_id)` to preserve the parent context without modifying implementation files.

---

Document status: APPROVED AND PUBLISHED.
