"""Dynamic Service Discovery and Agent Registry with Inverted Indexing and Lease TTL."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from agentmesh.mesh.a2a import AgentCard
from agentmesh.mesh.router import AgentEndpoint

if TYPE_CHECKING:
    from agentmesh.mesh.router import MeshRouter

logger = logging.getLogger("agentmesh.mesh.discovery")


class ServiceStatus(str, Enum):
    """Health status classification of a registered agent service."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    EXPIRED = "EXPIRED"


class ServiceRegistration(BaseModel):
    """Runtime instance registration for an agent in the service mesh."""

    agent_id: str = Field(..., description="Unique agent identifier")
    card: AgentCard = Field(..., description="Full self-describing AgentCard")
    endpoint_url: str = Field(default="in-process", description="Transport endpoint URI")
    weight: int = Field(default=100, description="Traffic routing weight")
    ttl_sec: float = Field(default=30.0, description="Heartbeat lease duration in seconds")
    registered_at_ns: int = Field(default_factory=time.time_ns, description="Registration timestamp")
    last_heartbeat_ns: int = Field(default_factory=time.time_ns, description="Latest heartbeat timestamp")
    status: ServiceStatus = Field(default=ServiceStatus.HEALTHY, description="Current health status")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Runtime metadata")

    def is_expired(self, now_ns: int | None = None) -> bool:
        """Determines if elapsed time since last heartbeat exceeds TTL lease."""
        now = now_ns if now_ns is not None else time.time_ns()
        elapsed_sec = (now - self.last_heartbeat_ns) / 1_000_000_000.0
        return elapsed_sec > self.ttl_sec

    def to_agent_endpoint(self) -> AgentEndpoint:
        """Bridges ServiceRegistration to MeshRouter AgentEndpoint."""
        return AgentEndpoint(
            agent_id=self.agent_id,
            role=self.card.name,
            endpoint_url=self.endpoint_url,
            weight=self.weight,
            is_active=(self.status == ServiceStatus.HEALTHY and not self.is_expired()),
            metadata={
                **self.metadata,
                "version": self.card.version,
                "capabilities": [c.name for c in self.card.capabilities],
            },
        )


class ServiceRegistry:
    """Thread-safe dynamic service registry supporting capability inverted indexing and lease eviction."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._instances: dict[str, ServiceRegistration] = {}
        self._capability_index: dict[str, set[str]] = {}
        self._tag_index: dict[str, set[str]] = {}
        self._listeners: list[Callable[[str, ServiceRegistration], None]] = []

    def subscribe(self, listener: Callable[[str, ServiceRegistration], None]) -> None:
        """Registers a callback receiving (event_type, registration) updates."""
        with self._lock:
            self._listeners.append(listener)

    def _notify(self, event_type: str, reg: ServiceRegistration) -> None:
        for listener in self._listeners:
            try:
                listener(event_type, reg)
            except Exception as e:  # noqa: BLE001
                logger.warning("Listener error during '%s' event: %s", event_type, e)

    def register(
        self,
        card: AgentCard,
        endpoint_url: str = "in-process",
        weight: int = 100,
        ttl_sec: float = 30.0,
        metadata: dict[str, Any] | None = None,
    ) -> ServiceRegistration:
        """Registers or updates an agent in the registry and refreshes inverted indexes."""
        with self._lock:
            if card.agent_id in self._instances:
                self._clear_indexes(card.agent_id)

            now_ns = time.time_ns()
            reg = ServiceRegistration(
                agent_id=card.agent_id,
                card=card,
                endpoint_url=endpoint_url,
                weight=weight,
                ttl_sec=ttl_sec,
                registered_at_ns=now_ns,
                last_heartbeat_ns=now_ns,
                status=ServiceStatus.HEALTHY,
                metadata=metadata or {},
            )
            self._instances[card.agent_id] = reg

            for cap in card.capabilities:
                self._capability_index.setdefault(cap.name, set()).add(card.agent_id)
                for tag in cap.tags:
                    self._tag_index.setdefault(tag, set()).add(card.agent_id)

            logger.info("Registered agent '%s' with %d capabilities.", card.agent_id, len(card.capabilities))
            self._notify("REGISTERED", reg)
            return reg

    def _clear_indexes(self, agent_id: str) -> None:
        for agent_set in self._capability_index.values():
            agent_set.discard(agent_id)
        for tag_set in self._tag_index.values():
            tag_set.discard(agent_id)

    def deregister(self, agent_id: str) -> bool:
        """Removes an agent from registry and clears all indexed lookups."""
        with self._lock:
            reg = self._instances.pop(agent_id, None)
            if not reg:
                return False
            self._clear_indexes(agent_id)
            logger.info("Deregistered agent '%s'.", agent_id)
            self._notify("DEREGISTERED", reg)
            return True

    def heartbeat(self, agent_id: str, timestamp_ns: int | None = None) -> bool:
        """Renews heartbeat lease for active agent. Returns False if unknown."""
        with self._lock:
            reg = self._instances.get(agent_id)
            if not reg:
                return False
            reg.last_heartbeat_ns = timestamp_ns if timestamp_ns is not None else time.time_ns()
            reg.status = ServiceStatus.HEALTHY
            self._notify("HEARTBEAT", reg)
            return True

    def get_agent(self, agent_id: str, include_expired: bool = False) -> ServiceRegistration | None:
        """Retrieves registration record by agent_id, checking expiration."""
        with self._lock:
            reg = self._instances.get(agent_id)
            if not reg:
                return None
            if not include_expired and reg.is_expired():
                reg.status = ServiceStatus.EXPIRED
                return None
            return reg

    def find_by_capability(
        self,
        capability_name: str,
        min_version: str | None = None,
        include_expired: bool = False,
    ) -> list[ServiceRegistration]:
        """Fast inverted index query returning agents providing the requested capability."""
        with self._lock:
            agent_ids = self._capability_index.get(capability_name, set())
            matches: list[ServiceRegistration] = []
            for aid in agent_ids:
                reg = self._instances.get(aid)
                if not reg:
                    continue
                if not include_expired and reg.is_expired():
                    reg.status = ServiceStatus.EXPIRED
                    continue
                if min_version and not reg.card.has_capability(capability_name, min_version):
                    continue
                matches.append(reg)
            return matches

    def find_by_tag(self, tag: str, include_expired: bool = False) -> list[ServiceRegistration]:
        """Queries agents tagged with the specified label."""
        with self._lock:
            agent_ids = self._tag_index.get(tag, set())
            matches: list[ServiceRegistration] = []
            for aid in agent_ids:
                reg = self._instances.get(aid)
                if not reg:
                    continue
                if not include_expired and reg.is_expired():
                    reg.status = ServiceStatus.EXPIRED
                    continue
                matches.append(reg)
            return matches

    def list_active_agents(self, include_expired: bool = False) -> list[ServiceRegistration]:
        """Returns snapshot of active registered agents."""
        with self._lock:
            if include_expired:
                return list(self._instances.values())
            active: list[ServiceRegistration] = []
            for reg in self._instances.values():
                if reg.is_expired():
                    reg.status = ServiceStatus.EXPIRED
                else:
                    active.append(reg)
            return active

    def evict_expired(self, now_ns: int | None = None) -> list[str]:
        """Prunes agents whose leases exceeded TTL. Returns list of evicted IDs."""
        with self._lock:
            now = now_ns if now_ns is not None else time.time_ns()
            evicted: list[str] = []
            for aid, reg in list(self._instances.items()):
                if reg.is_expired(now):
                    reg.status = ServiceStatus.EXPIRED
                    self._clear_indexes(aid)
                    del self._instances[aid]
                    evicted.append(aid)
                    logger.warning("Evicted expired agent '%s' (TTL=%ss).", aid, reg.ttl_sec)
                    self._notify("EXPIRED", reg)
            return evicted

    def sync_to_router(
        self,
        router: MeshRouter,
        handler_map: dict[str, Callable[..., Any]] | None = None,
    ) -> None:
        """Synchronizes healthy discovered endpoints to a MeshRouter instance."""
        with self._lock:
            handler_map = handler_map or {}
            for reg in self.list_active_agents():
                endpoint = reg.to_agent_endpoint()
                handler = handler_map.get(reg.agent_id, router.handlers.get(reg.agent_id))
                if handler:
                    router.register_endpoint(endpoint, handler=handler)
                else:
                    router.endpoints[reg.agent_id] = endpoint
