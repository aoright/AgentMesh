"""Enterprise Agent Service Mesh Router with Circuit Breaker, SWRR, and Failover."""

from __future__ import annotations

import inspect
import logging
import threading
import time
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from agentmesh.mesh.discovery import ServiceRegistry

logger = logging.getLogger("agentmesh.mesh.router")


class CircuitState(str, Enum):
    CLOSED = "CLOSED"          # Normal operation
    OPEN = "OPEN"              # Failing, fast reject
    HALF_OPEN = "HALF_OPEN"    # Testing recovery


class CircuitBreaker:
    """Protects agent services from cascading failures with sub-5ms transition and thread safety."""

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 10.0,
        half_open_success_threshold: int = 2,
    ):
        self.failure_threshold = max(1, failure_threshold)
        self.recovery_timeout = max(0.0, recovery_timeout)
        self.half_open_success_threshold = max(1, half_open_success_threshold)
        self.failure_count = 0
        self.consecutive_successes = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = 0.0
        self._lock = threading.RLock()

    def can_execute(self) -> bool:
        """Determines whether a call can proceed based on current circuit state."""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            if self.state == CircuitState.OPEN:
                now = time.time()
                if now - self.last_failure_time >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.consecutive_successes = 0
                    logger.info("Circuit transition to HALF_OPEN, probing service health.")
                    return True
                return False
            # HALF_OPEN allows probe
            return True

    def record_success(self) -> None:
        """Records a successful call, advancing half-open recovery or resetting failure counters."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.consecutive_successes += 1
                if self.consecutive_successes >= self.half_open_success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.consecutive_successes = 0
                    logger.info("Circuit fully recovered (success threshold reached), transition to CLOSED.")
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0
                self.consecutive_successes = 0

    def record_failure(self) -> None:
        """Records a call failure, executing sub-5ms state transition if threshold is reached."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            self.consecutive_successes = 0
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logger.warning("Circuit probe failed in HALF_OPEN, immediate return to OPEN.")
            elif self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(
                    "Circuit breached failure threshold (%d), transition to OPEN.",
                    self.failure_threshold,
                )

    def reset(self) -> None:
        """Resets the circuit breaker to CLOSED state."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.consecutive_successes = 0
            self.last_failure_time = 0.0


class AgentEndpoint(BaseModel):
    """Represents a registered agent destination in the service mesh."""

    agent_id: str
    role: str
    service_name: str | None = None
    endpoint_url: str = "in-process"
    weight: int = Field(default=100, ge=0)
    is_active: bool = Field(default=True)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __init__(
        self,
        agent_id: str = "",
        role: str = "",
        service_name: str | None = None,
        endpoint_url: str = "in-process",
        weight: int = 100,
        is_active: bool = True,
        metadata: dict[str, Any] | None = None,
        **data: Any,
    ) -> None:
        if "active" in data:
            is_active = data.pop("active")
        super().__init__(
            agent_id=agent_id,
            role=role,
            service_name=service_name,
            endpoint_url=endpoint_url,
            weight=weight,
            is_active=is_active,
            metadata=metadata or {},
            **data,
        )

    @property
    def active(self) -> bool:
        return self.is_active

    @active.setter
    def active(self, value: bool) -> None:
        self.is_active = bool(value)


class SmoothWeightedRoundRobin:
    """Thread-safe implementation of NGINX Smooth Weighted Round-Robin (SWRR) algorithm."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current_weights: dict[str, int] = {}

    def select(self, endpoints: list[AgentEndpoint]) -> AgentEndpoint | None:
        """Selects next endpoint according to smooth weighted round-robin.

        Considers only active endpoints with weight > 0.
        """
        with self._lock:
            candidates = [ep for ep in endpoints if ep.is_active and ep.weight > 0]
            if not candidates:
                return None

            total_weight = sum(ep.weight for ep in candidates)
            best_endpoint: AgentEndpoint | None = None
            max_current_weight: int | None = None

            for ep in candidates:
                curr = self._current_weights.get(ep.agent_id, 0) + ep.weight
                self._current_weights[ep.agent_id] = curr
                if max_current_weight is None or curr > max_current_weight:
                    max_current_weight = curr
                    best_endpoint = ep

            if best_endpoint is not None:
                self._current_weights[best_endpoint.agent_id] -= total_weight

            return best_endpoint

    def reset(self, agent_id: str | None = None) -> None:
        """Resets dynamic current weights."""
        with self._lock:
            if agent_id is not None:
                self._current_weights.pop(agent_id, None)
            else:
                self._current_weights.clear()


class MeshRouter:
    """Routes agent requests across distributed or local agents with SWRR and circuit breaking."""

    def __init__(self, discovery_registry: ServiceRegistry | None = None):
        self.endpoints: dict[str, AgentEndpoint] = {}
        self.handlers: dict[str, Callable[..., Any]] = {}
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.fallbacks: dict[str, str] = {}
        self._role_endpoints: dict[str, list[str]] = {}
        self._service_endpoints: dict[str, list[str]] = {}
        self._swrr_schedulers: dict[str, SmoothWeightedRoundRobin] = {}
        self.discovery_registry = discovery_registry
        self._lock = threading.RLock()

    def register_endpoint(
        self,
        endpoint: AgentEndpoint,
        handler: Callable[..., Any],
        fallback_agent_id: str | None = None,
        failure_threshold: int = 3,
        recovery_timeout: float = 10.0,
        half_open_success_threshold: int = 2,
    ) -> None:
        with self._lock:
            self.endpoints[endpoint.agent_id] = endpoint
            self.handlers[endpoint.agent_id] = handler
            self.circuit_breakers[endpoint.agent_id] = CircuitBreaker(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                half_open_success_threshold=half_open_success_threshold,
            )
            if fallback_agent_id:
                self.fallbacks[endpoint.agent_id] = fallback_agent_id

            role = endpoint.role
            if role not in self._role_endpoints:
                self._role_endpoints[role] = []
            if endpoint.agent_id not in self._role_endpoints[role]:
                self._role_endpoints[role].append(endpoint.agent_id)

            if endpoint.service_name:
                svc = endpoint.service_name
                if svc not in self._service_endpoints:
                    self._service_endpoints[svc] = []
                if endpoint.agent_id not in self._service_endpoints[svc]:
                    self._service_endpoints[svc].append(endpoint.agent_id)

            logger.info("Registered mesh endpoint: %s (%s)", endpoint.agent_id, endpoint.role)

    def get_pool_endpoints(self, pool_key: str) -> list[AgentEndpoint]:
        """Returns all endpoints associated with a role or service name."""
        with self._lock:
            agent_ids = self._role_endpoints.get(pool_key) or self._service_endpoints.get(pool_key) or []
            return [self.endpoints[aid] for aid in agent_ids if aid in self.endpoints]

    def select_endpoint(self, pool_key: str, healthy_only: bool = True) -> AgentEndpoint:
        """Selects an endpoint from a pool using Smooth Weighted Round-Robin."""
        endpoints = self.get_pool_endpoints(pool_key)
        if not endpoints:
            raise ValueError(f"Endpoint pool '{pool_key}' is empty or does not exist.")

        with self._lock:
            if pool_key not in self._swrr_schedulers:
                self._swrr_schedulers[pool_key] = SmoothWeightedRoundRobin()
            scheduler = self._swrr_schedulers[pool_key]

        candidates = endpoints
        if healthy_only:
            healthy = [
                ep for ep in endpoints
                if self.circuit_breakers.get(ep.agent_id, CircuitBreaker()).state != CircuitState.OPEN
            ]
            if healthy:
                candidates = healthy

        selected = scheduler.select(candidates)
        if selected is None:
            raise RuntimeError(f"No active or eligible endpoint available in pool '{pool_key}'.")
        return selected

    async def route_by_capability(self, capability: str, payload: dict[str, Any]) -> Any:
        """Discovers and schedules requests to agents providing the requested capability."""
        if self.discovery_registry is None:
            raise ValueError("No discovery registry configured for capability routing.")
        candidates = self.discovery_registry.find_by_capability(capability)
        if not candidates:
            raise ValueError(f"No healthy agents found for capability '{capability}'.")

        pool_key = f"cap:{capability}"
        with self._lock:
            if pool_key not in self._swrr_schedulers:
                self._swrr_schedulers[pool_key] = SmoothWeightedRoundRobin()
            scheduler = self._swrr_schedulers[pool_key]

        endpoints = [c.to_agent_endpoint() for c in candidates]
        selected = scheduler.select(endpoints)
        chosen_id = selected.agent_id if selected is not None else candidates[0].agent_id
        return await self.route_and_call(chosen_id, payload)

    async def route_and_call(
        self,
        target: str,
        payload: dict[str, Any],
        visited_agents: list[str] | None = None,
    ) -> Any:
        """Invokes target agent or pool with automatic circuit breaking and cycle-protected fallback."""
        visited: list[str] = list(visited_agents) if visited_agents is not None else []

        if target in self.endpoints:
            agent_id = target
        elif target in self._role_endpoints or target in self._service_endpoints:
            selected_ep = self.select_endpoint(target)
            agent_id = selected_ep.agent_id
        elif self.discovery_registry is not None:
            disc_reg = self.discovery_registry.get_agent(target)
            if disc_reg is not None and not disc_reg.is_expired():
                if target not in self.endpoints:
                    self.endpoints[target] = disc_reg.to_agent_endpoint()
                    self.circuit_breakers[target] = CircuitBreaker()
                agent_id = target
            else:
                raise ValueError(f"Agent '{target}' not registered in mesh.")
        else:
            raise ValueError(f"Agent '{target}' not registered in mesh.")

        if agent_id in visited:
            cycle_chain = " -> ".join([*visited, agent_id])
            logger.error("Fallback cycle detected: %s", cycle_chain)
            raise RuntimeError(
                f"Fallback cycle detected for agent '{agent_id}' (chain: {cycle_chain}). "
                "No healthy fallback available."
            )

        visited.append(agent_id)

        cb = self.circuit_breakers.setdefault(agent_id, CircuitBreaker())
        if not cb.can_execute():
            logger.warning("Agent '%s' circuit is OPEN, routing to fallback.", agent_id)
            return await self._handle_fallback(agent_id, payload, reason="circuit_open", visited=visited)

        if agent_id not in self.handlers:
            raise RuntimeError(f"Agent '{agent_id}' has no registered handler.")

        handler = self.handlers[agent_id]
        try:
            if inspect.iscoroutinefunction(handler):
                result = await handler(payload)
            else:
                res = handler(payload)
                if inspect.isawaitable(res):
                    result = await res
                else:
                    result = res

            cb.record_success()
            return result
        except Exception as e:  # noqa: BLE001
            cb.record_failure()
            logger.error("Call to agent '%s' failed: %s", agent_id, e)
            return await self._handle_fallback(agent_id, payload, reason=str(e), visited=visited)

    async def _handle_fallback(
        self,
        agent_id: str,
        payload: dict[str, Any],
        reason: str,
        visited: list[str],
    ) -> Any:
        fallback_id = self.fallbacks.get(agent_id)
        if fallback_id and fallback_id in self.endpoints:
            if fallback_id in visited:
                cycle_chain = " -> ".join([*visited, fallback_id])
                logger.error("Fallback cycle detected: %s", cycle_chain)
                raise RuntimeError(
                    f"Fallback cycle detected for agent '{agent_id}' (chain: {cycle_chain}). "
                    f"No healthy fallback available. Root cause: {reason}"
                )
            logger.info("Executing fallback agent '%s' for '%s' (reason: %s)", fallback_id, agent_id, reason)
            return await self.route_and_call(fallback_id, payload, visited_agents=visited)

        raise RuntimeError(f"Agent '{agent_id}' failed and no healthy fallback available. Root cause: {reason}")
