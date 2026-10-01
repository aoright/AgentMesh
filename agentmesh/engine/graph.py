"""Deterministic DAG Workflow Engine for AgentMesh with Activity Isolation."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import Any

from agentmesh.engine.activity import (
    ActivityExecutionContext,
    current_activity_context,
)
from agentmesh.engine.checkpoint import CheckpointManager
from agentmesh.engine.memoizer import ActivityMemoizer
from agentmesh.engine.state import EventType, State

logger = logging.getLogger("agentmesh.engine.graph")


class Node:
    """Represents a computational unit or Agent task inside the graph."""

    def __init__(self, name: str, func: Callable[[State], Any]):
        self.name = name
        self.func = func

    async def execute(
        self,
        state: State,
        memoizer: ActivityMemoizer | None = None,
        tracer: Any = None,
    ) -> Any:
        state.append_event(
            EventType.NODE_START,
            node_id=self.name,
            payload={"state_version": state.version},
        )

        ctx = ActivityExecutionContext(
            workflow_id=state.workflow_id,
            node_id=self.name,
            memoizer=memoizer,
            state=state,
            tracer=tracer,
        )
        token = current_activity_context.set(ctx)

        try:
            if inspect.iscoroutinefunction(self.func):
                result = await self.func(state)
            else:
                result = self.func(state)

            payload: dict[str, Any] = {}
            if isinstance(result, dict):
                payload["result"] = result
                payload["result_keys"] = list(result.keys())
            elif result is not None:
                payload["result_repr"] = str(result)

            state.append_event(EventType.NODE_COMPLETE, node_id=self.name, payload=payload)
            return result
        except Exception as e:
            state.append_event(
                EventType.NODE_FAILED,
                node_id=self.name,
                payload={"error": str(e)},
            )
            raise
        finally:
            current_activity_context.reset(token)


class Edge:
    """Directed connection between two nodes with optional conditional logic."""

    def __init__(
        self,
        source: str,
        target: str,
        condition: Callable[[State], bool] | None = None,
    ):
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
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.entry_point: str | None = None
        self.finish_points: set[str] = set()

    def add_node(self, name: str, func: Callable[[State], Any]) -> Node:
        node = Node(name, func)
        self.nodes[name] = node
        return node

    def add_edge(
        self,
        source: str,
        target: str,
        condition: Callable[[State], bool] | None = None,
    ) -> None:
        if source not in self.nodes:
            raise ValueError(f"Source node '{source}' not registered in graph.")
        if target not in self.nodes:
            raise ValueError(f"Target node '{target}' not registered in graph.")
        self.edges.append(Edge(source, target, condition))

    def set_entry_point(self, name: str) -> None:
        if name not in self.nodes:
            raise ValueError(f"Entry node '{name}' not found.")
        self.entry_point = name

    def set_finish_point(self, name: str) -> None:
        if name not in self.nodes:
            raise ValueError(f"Finish node '{name}' not found.")
        self.finish_points.add(name)

    async def run(
        self,
        state: State,
        checkpoint_manager: CheckpointManager | None = None,
        memoizer: ActivityMemoizer | None = None,
        tracer: Any = None,
        max_steps: int = 100,
        start_from_node: str | None = None,
    ) -> State:
        """Executes the graph deterministically with state checkpointing and activity memoization."""
        if not self.entry_point:
            raise RuntimeError("Graph entry point is not defined.")

        active_memoizer = memoizer or getattr(state, "memoizer", None) or ActivityMemoizer()
        if state.events:
            active_memoizer.populate_from_events(state.events)
        state.memoizer = active_memoizer

        is_resumed = False
        current_node_name: str | None = None

        if start_from_node is not None:
            current_node_name = start_from_node
            is_resumed = True
        elif len(state.events) > 0:
            from agentmesh.engine.replay import ResumptionPlanner

            plan = ResumptionPlanner.plan_resumption(self, state)
            logger.info("Autonomous Resumption Plan: %s", plan.reason)
            if plan.is_completed:
                logger.info(
                    "Workflow '%s' is already completed. Resumption is a no-op.",
                    state.workflow_id,
                )
                return state
            current_node_name = plan.resume_node or self.entry_point
            is_resumed = True
        else:
            current_node_name = self.entry_point

        step_count = 0

        state.append_event(
            EventType.WORKFLOW_START,
            payload={
                "graph": self.name,
                "start_node": current_node_name,
                "resumed": is_resumed,
            },
        )

        while current_node_name and step_count < max_steps:
            step_count += 1
            node = self.nodes.get(current_node_name)
            if not node:
                raise RuntimeError(f"Node '{current_node_name}' not found during execution.")

            logger.info("Executing node: %s (step: %d)", current_node_name, step_count)
            await node.execute(state, memoizer=active_memoizer, tracer=tracer)

            # Checkpoint automatically after successful node execution
            if checkpoint_manager:
                checkpoint_manager.save_checkpoint(state, label=current_node_name)
                active_memoizer.save_to_disk(state.workflow_id)

            if current_node_name in self.finish_points:
                logger.info("Reached finish point: %s", current_node_name)
                break

            # Find next outgoing edge
            next_node: str | None = None
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
            active_memoizer.save_to_disk(state.workflow_id)

        return state
