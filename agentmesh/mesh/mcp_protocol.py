"""Model Context Protocol (MCP) 2.0 JSON-RPC 2.0 Wire Protocol Specification.

Defines standard JSON-RPC 2.0 request/response/error envelopes, standard error codes,
and MCP 2.0 method schemas for tools, resources, and protocol lifecycle.
"""

from __future__ import annotations

import json
from enum import IntEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# --- Standard JSON-RPC 2.0 Error Codes ---

class JSONRPCErrorCode(IntEnum):
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # Implementation-defined Server Error Range (-32000 to -32099)
    SERVER_ERROR = -32000
    TOOL_EXECUTION_ERROR = -32001
    RESOURCE_NOT_FOUND = -32002
    TRANSPORT_ERROR = -32003
    TIMEOUT_ERROR = -32004


# --- Core JSON-RPC 2.0 Data Models ---

class JSONRPCError(BaseModel):
    """JSON-RPC 2.0 error object."""

    code: int
    message: str
    data: Any | None = None

    @classmethod
    def parse_error(cls, data: Any | None = None) -> JSONRPCError:
        return cls(code=JSONRPCErrorCode.PARSE_ERROR, message="Parse error", data=data)

    @classmethod
    def invalid_request(cls, data: Any | None = None) -> JSONRPCError:
        return cls(code=JSONRPCErrorCode.INVALID_REQUEST, message="Invalid Request", data=data)

    @classmethod
    def method_not_found(cls, method: str) -> JSONRPCError:
        return cls(
            code=JSONRPCErrorCode.METHOD_NOT_FOUND,
            message=f"Method not found: '{method}'",
        )

    @classmethod
    def invalid_params(cls, message: str = "Invalid params", data: Any | None = None) -> JSONRPCError:
        return cls(code=JSONRPCErrorCode.INVALID_PARAMS, message=message, data=data)

    @classmethod
    def internal_error(cls, message: str = "Internal error", data: Any | None = None) -> JSONRPCError:
        return cls(code=JSONRPCErrorCode.INTERNAL_ERROR, message=message, data=data)

    @classmethod
    def tool_execution_error(cls, message: str, data: Any | None = None) -> JSONRPCError:
        return cls(code=JSONRPCErrorCode.TOOL_EXECUTION_ERROR, message=message, data=data)

    @classmethod
    def resource_not_found(cls, uri: str) -> JSONRPCError:
        return cls(code=JSONRPCErrorCode.RESOURCE_NOT_FOUND, message=f"Resource not found: '{uri}'")


class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 request envelope requiring an ID."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int
    method: str
    params: dict[str, Any] | list[Any] | None = None


class JSONRPCNotification(BaseModel):
    """JSON-RPC 2.0 notification envelope with no ID."""

    jsonrpc: Literal["2.0"] = "2.0"
    method: str
    params: dict[str, Any] | list[Any] | None = None


class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 response envelope with mutually exclusive result and error."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int | None = None
    result: Any | None = None
    error: JSONRPCError | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_result_xor_error(cls, data: Any) -> Any:
        if isinstance(data, dict):
            has_result = "result" in data
            has_error = "error" in data
            if has_result and has_error and data["result"] is not None and data["error"] is not None:
                raise ValueError("JSON-RPC 2.0 response cannot contain both 'result' and 'error'")
            if not has_result and not has_error:
                raise ValueError("JSON-RPC 2.0 response must contain either 'result' or 'error'")
        return data

    @property
    def is_success(self) -> bool:
        return self.error is None

    def unwrap(self) -> Any:
        if self.error is not None:
            raise MCPProtocolError(self.error.code, self.error.message, self.error.data)
        return self.result


# --- MCP 2.0 Protocol Method Constants ---

class MCPMethod:
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"
    RESOURCES_LIST = "resources/list"
    RESOURCES_READ = "resources/read"
    INITIALIZE = "initialize"
    PING = "ping"


# --- MCP 2.0 Tool Schemas ---

class MCPToolInputSchema(BaseModel):
    """JSON Schema definition for tool arguments."""

    model_config = ConfigDict(extra="allow")

    type: str = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class MCPTool(BaseModel):
    """MCP tool definition."""

    name: str
    description: str = ""
    inputSchema: MCPToolInputSchema = Field(default_factory=MCPToolInputSchema)


class ToolsListParams(BaseModel):
    cursor: str | None = None


class ToolsListResult(BaseModel):
    tools: list[MCPTool] = Field(default_factory=list)
    nextCursor: str | None = None


class ToolsCallParams(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class MCPTextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class MCPImageContent(BaseModel):
    type: Literal["image"] = "image"
    data: str
    mimeType: str


class MCPResourceContentItem(BaseModel):
    type: Literal["resource"] = "resource"
    resource: dict[str, Any]


MCPContent = MCPTextContent | MCPImageContent | MCPResourceContentItem | dict[str, Any]


class ToolsCallResult(BaseModel):
    """Standard result returned by tools/call."""

    content: list[Any] = Field(default_factory=list)
    isError: bool = False
    structured_result: Any | None = None

    def get_text(self) -> str:
        texts = []
        for item in self.content:
            if isinstance(item, MCPTextContent):
                texts.append(item.text)
            elif isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
            elif isinstance(item, str):
                texts.append(item)
        return "\n".join(texts)


# --- MCP 2.0 Resource Schemas ---

class MCPResource(BaseModel):
    """MCP resource metadata."""

    uri: str
    name: str
    description: str | None = None
    mimeType: str | None = None


class ResourcesListParams(BaseModel):
    cursor: str | None = None


class ResourcesListResult(BaseModel):
    resources: list[MCPResource] = Field(default_factory=list)
    nextCursor: str | None = None


class ResourcesReadParams(BaseModel):
    uri: str


class ResourceContent(BaseModel):
    uri: str
    mimeType: str | None = None
    text: str | None = None
    blob: str | None = None


class ResourcesReadResult(BaseModel):
    contents: list[ResourceContent] = Field(default_factory=list)


# --- Protocol Exceptions ---

class MCPError(Exception):
    """Base exception for all MCP related failures."""


class MCPProtocolError(MCPError):
    """Represents a JSON-RPC 2.0 error returned over the wire."""

    def __init__(self, code: int, message: str, data: Any | None = None):
        super().__init__(f"MCP Error [{code}]: {message}")
        self.code = code
        self.message = message
        self.data = data


class MCPTransportError(MCPError):
    """Represents a transport layer failure."""


class MCPTimeoutError(MCPTransportError):
    """Raised when an MCP request times out."""


# --- Message Serialization Helpers ---

def serialize_jsonrpc(msg: JSONRPCRequest | JSONRPCNotification | JSONRPCResponse) -> str:
    """Serializes a JSON-RPC message into standard JSON string."""
    return msg.model_dump_json(exclude_none=True)


def parse_jsonrpc(raw_json: str | bytes | dict[str, Any]) -> JSONRPCRequest | JSONRPCNotification | JSONRPCResponse:
    """Parses raw JSON into the appropriate JSON-RPC 2.0 model."""
    if isinstance(raw_json, (str, bytes)):
        try:
            data = json.loads(raw_json)
        except Exception as e:
            raise MCPProtocolError(JSONRPCErrorCode.PARSE_ERROR, f"Parse error: {e}") from e
    else:
        data = raw_json

    if not isinstance(data, dict):
        raise MCPProtocolError(JSONRPCErrorCode.INVALID_REQUEST, "Invalid JSON-RPC payload: must be a dict")

    if data.get("jsonrpc") != "2.0":
        raise MCPProtocolError(JSONRPCErrorCode.INVALID_REQUEST, "Missing or invalid 'jsonrpc' field")

    # Discriminate between Request, Notification, and Response
    if "result" in data or "error" in data:
        return JSONRPCResponse.model_validate(data)
    elif "method" in data:
        if "id" in data:
            return JSONRPCRequest.model_validate(data)
        return JSONRPCNotification.model_validate(data)
    else:
        raise MCPProtocolError(JSONRPCErrorCode.INVALID_REQUEST, "Cannot identify message type")
