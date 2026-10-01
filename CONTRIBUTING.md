# Contributing to AgentMesh

Thank you for your interest in contributing to AgentMesh. We are building a high-reliability open-source agent mesh and deterministic runtime framework for enterprise production workflows.

## Code of Conduct

All contributors and maintainers are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md).

## Development Workflow

1. Fork the repository and create your branch from `main`.
2. Ensure you have Python 3.10+ installed.
3. Install dependencies in your development environment:
   ```bash
   pip install -e ".[dev]"
   ```
4. Run tests to verify the baseline:
   ```bash
   pytest
   ```
5. Implement your feature or bug fix with corresponding unit tests.
6. Verify your changes pass all unit tests and adhere to state separation guidelines.
7. Submit a Pull Request with a clear description of the problem solved.

## Guidelines and Engineering Rules

### 1. State & Code Separation
Mutable state, databases, and checkpoint storage must remain strictly outside the code tree. All storage mechanisms must respect `AGENTMESH_STATE_DIR` and provide default isolated directories. Never hardcode local paths or commit state artifacts.

### 2. Deterministic State Machines
When adding or modifying graph engine operations, ensure that:
* Events are recorded in the event log with immutable IDs and hashes.
* No nondeterministic state operations bypass the event stream.
* Hash chain integrity is preserved across checkpoints.

### 3. Capability Security
New tool integrations or agent capabilities must specify required permissions and pass through `SecuritySandbox` validation.

## Pull Request Checklist

* [ ] Code adheres to PEP 8 standards.
* [ ] Unit tests added under `tests/` covering new logic.
* [ ] All tests pass (`pytest`).
* [ ] Documentation updated if public interfaces changed.
* [ ] No emojis in commit messages, comments, or documentation.
