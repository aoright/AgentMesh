"""Enterprise Agent Service Mesh Router with Circuit Breaker and Failover."""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("agentmesh.mesh.router")


class CircuitState(str, Enum):
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Failing, fast reject
    HALF_OPEN = "HALF_OPEN"  # Testing recovery


class CircuitBreaker:
    """Protects agent services from cascading failures."""

    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 10.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = 0.0

    def can_execute(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit transition to HALF_OPEN, probing service health.")
                return True
            return False
        # HALF_OPEN allows single probe
        return True

    def record_success(self):
        self.failure_count = 0
        if self.state != CircuitState.CLOSED:
            logger.info("Circuit recovered, transition to CLOSED.")
            self.state = CircuitState.CLOSED

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning("Circuit breached failure threshold (%d), transition to OPEN.", self.failure_threshold)


class AgentEndpoint(BaseModel):
    agent_id: str
    role: str
    endpoint_url: str = "in-process"
    weight: int = 100
    is_active: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MeshRouter:
    """Routes agent requests across distributed or local agents with circuit breaking."""

    def __init__(self):
        self.endpoints: Dict[str, AgentEndpoint] = {}
        self.handlers: Dict[str, Callable[..., Any]] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.fallbacks: Dict[str, str] = {}

    def register_endpoint(
        self,
        endpoint: AgentEndpoint,
        handler: Callable[..., Any],
        fallback_agent_id: Optional[str] = None,
        failure_threshold: int = 3,
    ):
        self.endpoints[endpoint.agent_id] = endpoint
        self.handlers[endpoint.agent_id] = handler
        self.circuit_breakers[endpoint.agent_id] = CircuitBreaker(failure_threshold=failure_threshold)
        if fallback_agent_id:
            self.fallbacks[endpoint.agent_id] = fallback_agent_id
        logger.info("Registered mesh endpoint: %s (%s)", endpoint.agent_id, endpoint.role)

    async def route_and_call(self, agent_id: str, payload: Dict[str, Any]) -> Any:
        """Invokes target agent with automatic circuit breaking and fallback routing."""
        if agent_id not in self.endpoints:
            raise ValueError(f"Agent '{agent_id}' not registered in mesh.")

        cb = self.circuit_breakers[agent_id]
        if not cb.can_execute():
            logger.warning("Agent '%s' circuit is OPEN, routing to fallback.", agent_id)
            return await self._handle_fallback(agent_id, payload, reason="circuit_open")

        handler = self.handlers[agent_id]
        try:
            import inspect

            if inspect.iscoroutinefunction(handler):
                result = await handler(payload)
            else:
                result = handler(payload)

            cb.record_success()
            return result
        except Exception as e:
            cb.record_failure()
            logger.error("Call to agent '%s' failed: %s", agent_id, e)
            return await self._handle_fallback(agent_id, payload, reason=str(e))

    async def _handle_fallback(self, agent_id: str, payload: Dict[str, Any], reason: str) -> Any:
        fallback_id = self.fallbacks.get(agent_id)
        if fallback_id and fallback_id in self.endpoints:
            logger.info("Executing fallback agent '%s' for '%s' (reason: %s)", fallback_id, agent_id, reason)
            return await self.route_and_call(fallback_id, payload)
        raise RuntimeError(f"Agent '{agent_id}' failed and no healthy fallback available. Root cause: {reason}")
