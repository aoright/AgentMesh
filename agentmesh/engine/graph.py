"""Deterministic DAG Workflow Engine for AgentMesh."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Set
from agentmesh.engine.state import State, EventType
from agentmesh.engine.checkpoint import CheckpointManager

logger = logging.getLogger("agentmesh.engine.graph")


class Node:
    """Represents a computational unit or Agent task inside the graph."""

    def __init__(self, name: str, func: Callable[[State], Any]):
        self.name = name
        self.func = func

    async def execute(self, state: State) -> Any:
        state.append_event(EventType.NODE_START, node_id=self.name, payload={"state_version": state.version})
        try:
            if inspect.iscoroutinefunction(self.func):
                result = await self.func(state)
            else:
                result = self.func(state)

            payload = {"result_keys": list(result.keys()) if isinstance(result, dict) else str(type(result))}
            state.append_event(EventType.NODE_COMPLETE, node_id=self.name, payload=payload)
            return result
        except Exception as e:
            state.append_event(EventType.NODE_FAILED, node_id=self.name, payload={"error": str(e)})
            raise


class Edge:
    """Directed connection between two nodes with optional conditional logic."""

    def __init__(self, source: str, target: str, condition: Optional[Callable[[State], bool]] = None):
        self.source = source
        self.target = target
        self.condition = condition

    def evaluate(self, state: State) -> bool:
        if self.condition is None:
            return True
        return bool(self.condition(state))


class Graph:
    """Deterministic Directed Acyclic Graph (DAG) runtime engine."""

    def __init__(self, name: str = "default_graph"):
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.entry_point: Optional[str] = None
        self.finish_points: Set[str] = set()

    def add_node(self, name: str, func: Callable[[State], Any]) -> Node:
        node = Node(name, func)
        self.nodes[name] = node
        return node

    def add_edge(self, source: str, target: str, condition: Optional[Callable[[State], bool]] = None):
        if source not in self.nodes:
            raise ValueError(f"Source node '{source}' not registered in graph.")
        if target not in self.nodes:
            raise ValueError(f"Target node '{target}' not registered in graph.")
        self.edges.append(Edge(source, target, condition))

    def set_entry_point(self, name: str):
        if name not in self.nodes:
            raise ValueError(f"Entry node '{name}' not found.")
        self.entry_point = name

    def set_finish_point(self, name: str):
        if name not in self.nodes:
            raise ValueError(f"Finish node '{name}' not found.")
        self.finish_points.add(name)

    async def run(
        self,
        state: State,
        checkpoint_manager: Optional[CheckpointManager] = None,
        max_steps: int = 100,
        start_from_node: Optional[str] = None,
    ) -> State:
        """Executes the graph deterministically with state checkpointing after each node."""
        if not self.entry_point:
            raise RuntimeError("Graph entry point is not defined.")

        current_node_name = start_from_node or self.entry_point
        step_count = 0

        state.append_event(
            EventType.WORKFLOW_START,
            payload={"graph": self.name, "start_node": current_node_name, "resumed": bool(start_from_node)},
        )

        while current_node_name and step_count < max_steps:
            step_count += 1
            node = self.nodes.get(current_node_name)
            if not node:
                raise RuntimeError(f"Node '{current_node_name}' not found during execution.")

            logger.info("Executing node: %s (step: %d)", current_node_name, step_count)
            await node.execute(state)

            # Checkpoint automatically after successful node execution
            if checkpoint_manager:
                checkpoint_manager.save_checkpoint(state, label=current_node_name)

            if current_node_name in self.finish_points:
                logger.info("Reached finish point: %s", current_node_name)
                break

            # Find next outgoing edge
            next_node = None
            for edge in self.edges:
                if edge.source == current_node_name and edge.evaluate(state):
                    next_node = edge.target
                    break

            current_node_name = next_node

        state.append_event(
            EventType.WORKFLOW_COMPLETE,
            payload={"total_steps": step_count, "final_version": state.version},
        )

        if checkpoint_manager:
            checkpoint_manager.save_checkpoint(state, label="finished")

        return state
