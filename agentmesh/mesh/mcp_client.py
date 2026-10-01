"""Model Context Protocol (MCP) tool integration and schema serialization."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

logger = logging.getLogger("agentmesh.mesh.mcp")


class MCPToolDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class MCPToolCallRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class MCPToolCallResponse(BaseModel):
    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None


class MCPToolClient:
    """Manages MCP tools and facilitates standardized tool invocations for Agents."""

    def __init__(self, server_name: str = "local_mcp"):
        self.server_name = server_name
        self.registry: Dict[str, Tuple[MCPToolDefinition, Callable]] = {}

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable[..., Any],
    ):
        definition = MCPToolDefinition(name=name, description=description, parameters=parameters)
        self.registry[name] = (definition, handler)
        logger.info("Registered MCP tool '%s' on server '%s'", name, self.server_name)

    def list_tools(self) -> List[MCPToolDefinition]:
        return [defn for defn, _ in self.registry.values()]

    async def invoke_tool(self, request: MCPToolCallRequest) -> MCPToolCallResponse:
        if request.tool_name not in self.registry:
            return MCPToolCallResponse(
                tool_name=request.tool_name,
                success=False,
                error=f"Tool '{request.tool_name}' not found on MCP server '{self.server_name}'",
            )

        _, handler = self.registry[request.tool_name]
        try:
            import inspect

            if inspect.iscoroutinefunction(handler):
                result = await handler(**request.arguments)
            else:
                result = handler(**request.arguments)

            return MCPToolCallResponse(tool_name=request.tool_name, success=True, result=result)
        except Exception as e:
            logger.exception("Error executing MCP tool '%s'", request.tool_name)
            return MCPToolCallResponse(tool_name=request.tool_name, success=False, error=str(e))
