"""Model Context Protocol (MCP) 2.0 Client and Transport Implementations.

Provides transport abstractions (InMemoryTransport, StdioTransport, HttpSseTransport),
a high-level modern MCPClient, and 100% backward-compatible MCPToolClient.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from types import TracebackType
from typing import Any, Self, cast

from pydantic import BaseModel, Field

from agentmesh.mesh.mcp_protocol import (
    JSONRPCError,
    JSONRPCErrorCode,
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
    MCPMethod,
    MCPResource,
    MCPTextContent,
    MCPTimeoutError,
    MCPTool,
    MCPToolInputSchema,
    MCPTransportError,
    ResourceContent,
    ResourcesListResult,
    ResourcesReadResult,
    ToolsCallResult,
    ToolsListResult,
    parse_jsonrpc,
)

logger = logging.getLogger("agentmesh.mesh.mcp")


# --- Backward Compatible Data Models ---

class MCPToolDefinition(BaseModel):
    """Legacy tool definition model for backward compatibility."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class MCPToolCallRequest(BaseModel):
    """Legacy tool call request model for backward compatibility."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class MCPToolCallResponse(BaseModel):
    """Legacy tool call response model for backward compatibility."""

    tool_name: str
    success: bool
    result: Any = None
    error: str | None = None


# --- Transport Abstraction Layer ---

class BaseTransport(ABC):
    """Abstract Base Class for MCP 2.0 communication transports."""

    @abstractmethod
    async def connect(self) -> None:
        """Establishes transport connection."""

    @abstractmethod
    async def send_request(self, request: JSONRPCRequest) -> JSONRPCResponse:
        """Sends a JSON-RPC 2.0 request and returns the validated response."""

    @abstractmethod
    async def send_notification(self, notification: JSONRPCNotification) -> None:
        """Sends a JSON-RPC 2.0 notification."""

    @abstractmethod
    async def close(self) -> None:
        """Closes transport and releases underlying resources."""

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()


# --- In-Memory Transport ---

class InMemoryTransport(BaseTransport):
    """Direct in-memory callable transport with full JSON-RPC 2.0 wire protocol validation."""

    def __init__(self, server_name: str = "local_mcp"):
        self.server_name = server_name
        self.tools: dict[str, tuple[MCPTool, Callable[..., Any]]] = {}
        self.resources: dict[str, tuple[MCPResource, Callable[..., Any]]] = {}
        self.custom_handlers: dict[str, Callable[..., Any]] = {}
        self._connected = True

    def register_tool_handler(self, tool: MCPTool, handler: Callable[..., Any]) -> None:
        self.tools[tool.name] = (tool, handler)

    def register_resource_handler(self, resource: MCPResource, handler: Callable[..., Any]) -> None:
        self.resources[resource.uri] = (resource, handler)

    async def connect(self) -> None:
        self._connected = True

    async def close(self) -> None:
        self._connected = False

    async def send_notification(self, notification: JSONRPCNotification) -> None:
        # Fire-and-forget notification
        pass

    async def send_request(self, request: JSONRPCRequest) -> JSONRPCResponse:
        if not self._connected:
            raise MCPTransportError("InMemoryTransport is closed")

        # Simulate wire serialization fidelity
        raw = request.model_dump_json()
        deserialized = parse_jsonrpc(raw)
        if not isinstance(deserialized, JSONRPCRequest):
            return JSONRPCResponse(
                id=request.id,
                error=JSONRPCError.invalid_request("Deserialized object is not a JSONRPCRequest"),
            )

        method = deserialized.method
        params = deserialized.params or {}

        if method == MCPMethod.TOOLS_LIST:
            tools = [tool for tool, _ in self.tools.values()]
            tools_result = ToolsListResult(tools=tools)
            return JSONRPCResponse(id=request.id, result=tools_result.model_dump())

        if method == MCPMethod.TOOLS_CALL:
            if not isinstance(params, dict) or "name" not in params:
                return JSONRPCResponse(
                    id=request.id,
                    error=JSONRPCError.invalid_params("Parameter 'name' is required for tools/call"),
                )

            tool_name = params["name"]
            arguments = params.get("arguments", {})

            if tool_name not in self.tools:
                return JSONRPCResponse(
                    id=request.id,
                    error=JSONRPCError(
                        code=JSONRPCErrorCode.METHOD_NOT_FOUND,
                        message=f"Tool '{tool_name}' not found on MCP server '{self.server_name}'",
                    ),
                )

            _, handler = self.tools[tool_name]
            try:
                if inspect.iscoroutinefunction(handler):
                    exec_result = await handler(**arguments)
                else:
                    exec_result = handler(**arguments)

                # Return structured result
                text_repr = json.dumps(exec_result) if not isinstance(exec_result, str) else exec_result
                content_item = MCPTextContent(text=text_repr)
                call_result = ToolsCallResult(
                    content=[content_item],
                    isError=False,
                    structured_result=exec_result,
                )
                return JSONRPCResponse(id=request.id, result=call_result.model_dump())
            except Exception as e:
                logger.exception("Error executing MCP tool '%s'", tool_name)
                # MCP specification recommends isError: true on tool failures
                content_item = MCPTextContent(text=str(e))
                call_result = ToolsCallResult(
                    content=[content_item],
                    isError=True,
                    structured_result=None,
                )
                return JSONRPCResponse(id=request.id, result=call_result.model_dump())

        if method == MCPMethod.RESOURCES_LIST:
            resources = [r for r, _ in self.resources.values()]
            resources_result = ResourcesListResult(resources=resources)
            return JSONRPCResponse(id=request.id, result=resources_result.model_dump())

        if method == MCPMethod.RESOURCES_READ:
            if not isinstance(params, dict) or "uri" not in params:
                return JSONRPCResponse(
                    id=request.id,
                    error=JSONRPCError.invalid_params("Parameter 'uri' is required for resources/read"),
                )
            uri = params["uri"]
            if uri not in self.resources:
                return JSONRPCResponse(id=request.id, error=JSONRPCError.resource_not_found(uri))

            _, handler = self.resources[uri]
            try:
                if inspect.iscoroutinefunction(handler):
                    content = await handler(uri)
                else:
                    content = handler(uri)

                if isinstance(content, ResourceContent):
                    items = [content]
                elif isinstance(content, str):
                    items = [ResourceContent(uri=uri, text=content)]
                else:
                    items = [ResourceContent(uri=uri, text=str(content))]

                read_result = ResourcesReadResult(contents=items)
                return JSONRPCResponse(id=request.id, result=read_result.model_dump())
            except Exception as e:  # noqa: BLE001
                return JSONRPCResponse(id=request.id, error=JSONRPCError.internal_error(str(e)))

        if method == MCPMethod.PING:
            return JSONRPCResponse(id=request.id, result={})

        if method == MCPMethod.INITIALIZE:
            return JSONRPCResponse(
                id=request.id,
                result={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}, "resources": {}},
                    "serverInfo": {"name": self.server_name, "version": "1.0.0"},
                },
            )

        if method in self.custom_handlers:
            handler = self.custom_handlers[method]
            try:
                res = await handler(params) if inspect.iscoroutinefunction(handler) else handler(params)
                return JSONRPCResponse(id=request.id, result=res)
            except Exception as e:  # noqa: BLE001
                return JSONRPCResponse(id=request.id, error=JSONRPCError.internal_error(str(e)))

        return JSONRPCResponse(id=request.id, error=JSONRPCError.method_not_found(method))


# --- Stdio Stream Adapter Transport ---

class StdioTransport(BaseTransport):
    """Transport communicating with external MCP subprocesses via standard I/O (NDJSON)."""

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: float = 30.0,
    ):
        self.command = command
        self.args = args or []
        self.env = env
        self.cwd = cwd
        self.timeout = timeout
        self.process: asyncio.subprocess.Process | None = None
        self._pending_requests: dict[str | int, asyncio.Future[JSONRPCResponse]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    async def connect(self) -> None:
        if self.process is not None and self.process.returncode is None:
            return

        self.process = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.env,
            cwd=self.cwd,
        )
        self._closed = False
        self._reader_task = asyncio.create_task(self._stdout_reader_loop())
        logger.info("Spawned MCP stdio process pid=%d", self.process.pid)

    async def _stdout_reader_loop(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        while not self._closed:
            try:
                line = await self.process.stdout.readline()
                if not line:
                    break
                decoded = line.decode("utf-8").strip()
                if not decoded:
                    continue

                msg = parse_jsonrpc(decoded)
                if isinstance(msg, JSONRPCResponse):
                    req_id = msg.id
                    if req_id in self._pending_requests:
                        future = self._pending_requests[req_id]
                        if not future.done():
                            future.set_result(msg)
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001
                logger.error("StdioTransport reader error: %s", e)

    async def send_request(self, request: JSONRPCRequest) -> JSONRPCResponse:
        if self._closed or self.process is None or self.process.stdin is None:
            raise MCPTransportError("StdioTransport is not connected")

        loop = asyncio.get_running_loop()
        future: asyncio.Future[JSONRPCResponse] = loop.create_future()
        self._pending_requests[request.id] = future

        wire_payload = request.model_dump_json() + "\n"
        async with self._lock:
            self.process.stdin.write(wire_payload.encode("utf-8"))
            await self.process.stdin.drain()

        try:
            return await asyncio.wait_for(future, timeout=self.timeout)
        except asyncio.TimeoutError as e:
            raise MCPTimeoutError(f"Request {request.id} timed out after {self.timeout}s") from e
        finally:
            self._pending_requests.pop(request.id, None)

    async def send_notification(self, notification: JSONRPCNotification) -> None:
        if self._closed or self.process is None or self.process.stdin is None:
            raise MCPTransportError("StdioTransport is not connected")

        wire_payload = notification.model_dump_json() + "\n"
        async with self._lock:
            self.process.stdin.write(wire_payload.encode("utf-8"))
            await self.process.stdin.drain()

    async def close(self) -> None:
        self._closed = True
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()

        if self.process:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                self.process.terminate()
                await self.process.wait()
            except ProcessLookupError:
                pass


# --- HTTP with Server-Sent Events (SSE) Transport ---

class HttpSseTransport(BaseTransport):
    """Transport communicating over HTTP with Server-Sent Events (SSE) payload framing."""

    def __init__(
        self,
        endpoint_url: str,
        http_handler: Callable[[str, dict[str, Any]], Any] | None = None,
        timeout: float = 30.0,
    ):
        self.endpoint_url = endpoint_url
        self.http_handler = http_handler
        self.timeout = timeout
        self._connected = True

    @staticmethod
    def encode_sse_event(event: str, data: str, event_id: str | None = None) -> str:
        """Formats data into standard SSE framing."""
        lines = []
        if event_id:
            lines.append(f"id: {event_id}")
        lines.append(f"event: {event}")
        for chunk in data.splitlines():
            lines.append(f"data: {chunk}")
        return "\n".join(lines) + "\n\n"

    @staticmethod
    def parse_sse_events(raw_stream: str) -> list[tuple[str, str]]:
        """Parses a multi-event SSE stream into a list of (event, data) tuples."""
        events: list[tuple[str, str]] = []
        blocks = raw_stream.strip().split("\n\n")
        for block in blocks:
            if not block.strip():
                continue
            cur_event = "message"
            data_lines = []
            for line in block.splitlines():
                if line.startswith("event:"):
                    cur_event = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[len("data:"):].strip())
            events.append((cur_event, "\n".join(data_lines)))
        return events

    async def connect(self) -> None:
        self._connected = True

    async def close(self) -> None:
        self._connected = False

    async def send_notification(self, notification: JSONRPCNotification) -> None:
        await self.send_request(
            JSONRPCRequest(
                id=str(uuid.uuid4()),
                method=notification.method,
                params=notification.params,
            )
        )

    async def send_request(self, request: JSONRPCRequest) -> JSONRPCResponse:
        if not self._connected:
            raise MCPTransportError("HttpSseTransport is closed")

        if self.http_handler is not None:
            payload = json.loads(request.model_dump_json())
            try:
                res = await self.http_handler(self.endpoint_url, payload)
                if isinstance(res, str):
                    parsed = parse_jsonrpc(res)
                    if isinstance(parsed, JSONRPCResponse):
                        return parsed
                    raise MCPTransportError("Parsed response is not a JSONRPCResponse")
                if isinstance(res, dict):
                    return JSONRPCResponse.model_validate(res)
                raise MCPTransportError(f"Unexpected response format from HTTP handler: {type(res)}")
            except Exception as e:
                raise MCPTransportError(f"HTTP dispatch error: {e}") from e

        raise NotImplementedError("Direct network HTTP/SSE requires an http_handler adapter in hermetic environments")


# --- High-Level MCP Client ---

class MCPClient:
    """High-level, strongly-typed client for MCP 2.0 communication."""

    def __init__(self, transport: BaseTransport):
        self.transport = transport

    async def initialize(self, client_name: str = "agentmesh_client", version: str = "1.0.0") -> dict[str, Any]:
        req = JSONRPCRequest(
            id=str(uuid.uuid4()),
            method=MCPMethod.INITIALIZE,
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": client_name, "version": version},
            },
        )
        resp = await self.transport.send_request(req)
        return cast(dict[str, Any], resp.unwrap())

    async def list_tools(self, cursor: str | None = None) -> ToolsListResult:
        req = JSONRPCRequest(
            id=str(uuid.uuid4()),
            method=MCPMethod.TOOLS_LIST,
            params={"cursor": cursor} if cursor else {},
        )
        resp = await self.transport.send_request(req)
        return ToolsListResult.model_validate(resp.unwrap())

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> ToolsCallResult:
        req = JSONRPCRequest(
            id=str(uuid.uuid4()),
            method=MCPMethod.TOOLS_CALL,
            params={"name": name, "arguments": arguments or {}},
        )
        resp = await self.transport.send_request(req)
        return ToolsCallResult.model_validate(resp.unwrap())

    async def list_resources(self, cursor: str | None = None) -> ResourcesListResult:
        req = JSONRPCRequest(
            id=str(uuid.uuid4()),
            method=MCPMethod.RESOURCES_LIST,
            params={"cursor": cursor} if cursor else {},
        )
        resp = await self.transport.send_request(req)
        return ResourcesListResult.model_validate(resp.unwrap())

    async def read_resource(self, uri: str) -> ResourcesReadResult:
        req = JSONRPCRequest(
            id=str(uuid.uuid4()),
            method=MCPMethod.RESOURCES_READ,
            params={"uri": uri},
        )
        resp = await self.transport.send_request(req)
        return ResourcesReadResult.model_validate(resp.unwrap())

    async def ping(self) -> bool:
        req = JSONRPCRequest(id=str(uuid.uuid4()), method=MCPMethod.PING, params={})
        resp = await self.transport.send_request(req)
        return resp.is_success


# --- Upgraded Backward-Compatible MCPToolClient ---

class MCPToolClient:
    """Manages MCP tools and facilitates standardized tool invocations.

    Fully backwards-compatible with AgentMesh legacy interfaces while powered
    internally by the full MCP 2.0 JSON-RPC wire protocol.
    """

    def __init__(
        self,
        server_name: str = "local_mcp",
        transport: BaseTransport | None = None,
    ):
        self.server_name = server_name
        self.in_memory_transport: InMemoryTransport | None
        if transport is None:
            self.in_memory_transport = InMemoryTransport(server_name=server_name)
            self.transport: BaseTransport = self.in_memory_transport
        else:
            self.in_memory_transport = transport if isinstance(transport, InMemoryTransport) else None
            self.transport = transport

        self.registry: dict[str, tuple[MCPToolDefinition, Callable[..., Any]]] = {}
        self.mcp_client = MCPClient(self.transport)

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        """Registers a tool with both legacy definition and standard MCP 2.0 schema."""
        legacy_definition = MCPToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
        )
        self.registry[name] = (legacy_definition, handler)

        # Convert simple parameters to MCPToolInputSchema
        if self.in_memory_transport is not None:
            properties: dict[str, Any] = {}
            for param_name, param_type in parameters.items():
                if isinstance(param_type, str):
                    p_type = "number" if param_type in ("int", "float", "number") else "string"
                    properties[param_name] = {"type": p_type}
                elif isinstance(param_type, dict):
                    properties[param_name] = param_type
                else:
                    properties[param_name] = {"type": "string"}

            input_schema = MCPToolInputSchema(
                type="object",
                properties=properties,
                required=list(parameters.keys()),
            )
            mcp_tool = MCPTool(name=name, description=description, inputSchema=input_schema)
            self.in_memory_transport.register_tool_handler(mcp_tool, handler)

        logger.info("Registered MCP tool '%s' on server '%s'", name, self.server_name)

    def register_resource(
        self,
        uri: str,
        name: str,
        handler: Callable[..., Any],
        description: str | None = None,
        mime_type: str | None = None,
    ) -> None:
        """Registers a resource on the internal in-memory transport."""
        if self.in_memory_transport is not None:
            resource = MCPResource(uri=uri, name=name, description=description, mimeType=mime_type)
            self.in_memory_transport.register_resource_handler(resource, handler)

    def list_tools(self) -> list[MCPToolDefinition]:
        """Returns the list of registered tools using the legacy model."""
        return [defn for defn, _ in self.registry.values()]

    async def list_tools_mcp(self, cursor: str | None = None) -> ToolsListResult:
        """Returns the list of registered tools using standard MCP 2.0 ToolsListResult."""
        return await self.mcp_client.list_tools(cursor=cursor)

    async def invoke_tool(self, request: MCPToolCallRequest) -> MCPToolCallResponse:
        """Invokes a tool through the MCP 2.0 JSON-RPC 2.0 wire protocol envelope."""
        rpc_req = JSONRPCRequest(
            id=str(uuid.uuid4()),
            method=MCPMethod.TOOLS_CALL,
            params={"name": request.tool_name, "arguments": request.arguments},
        )

        try:
            rpc_resp = await self.transport.send_request(rpc_req)

            if rpc_resp.error is not None:
                # Protocol-level error (e.g. tool not found, invalid params)
                return MCPToolCallResponse(
                    tool_name=request.tool_name,
                    success=False,
                    result=None,
                    error=rpc_resp.error.message,
                )

            # Parse tool execution result
            result_dict = rpc_resp.result if isinstance(rpc_resp.result, dict) else {}
            is_error = result_dict.get("isError", False)

            if is_error:
                content = result_dict.get("content", [])
                error_msg = ""
                if content and isinstance(content[0], dict):
                    error_msg = content[0].get("text", "Tool execution failed")
                else:
                    error_msg = "Tool execution failed"

                return MCPToolCallResponse(
                    tool_name=request.tool_name,
                    success=False,
                    result=None,
                    error=error_msg,
                )

            # Success: extract structured or primitive result
            if "structured_result" in result_dict and result_dict["structured_result"] is not None:
                actual_result = result_dict["structured_result"]
            else:
                content = result_dict.get("content", [])
                if content and isinstance(content[0], dict) and "text" in content[0]:
                    text = content[0]["text"]
                    try:
                        actual_result = json.loads(text)
                    except (json.JSONDecodeError, TypeError):
                        actual_result = text
                else:
                    actual_result = result_dict

            return MCPToolCallResponse(
                tool_name=request.tool_name,
                success=True,
                result=actual_result,
                error=None,
            )

        except Exception as e:
            logger.exception("Error executing MCP tool '%s'", request.tool_name)
            return MCPToolCallResponse(
                tool_name=request.tool_name,
                success=False,
                result=None,
                error=str(e),
            )
