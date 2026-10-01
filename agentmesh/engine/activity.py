"""Activity abstraction and isolation models for AgentMesh durable execution."""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import time
from collections.abc import Callable
from contextvars import ContextVar
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("agentmesh.engine.activity")


class ActivityStatus(str, Enum):
    """Lifecycle status of an external activity."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"


class ActivityRecord(BaseModel):
    """Immutable record of an executed activity, including output and token accounting."""

    idempotency_key: str
    activity_name: str
    node_id: str
    workflow_id: str = ""
    run_id: str | None = None
    status: ActivityStatus = ActivityStatus.COMPLETED
    result: Any = None
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    duration_ms: float = 0.0
    timestamp_ns: int = Field(default_factory=time.time_ns)
    completed_at_ns: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, context: Any) -> None:
        if self.completed_at_ns == 0:
            self.completed_at_ns = self.timestamp_ns


class RetryPolicy(BaseModel):
    """Configurable retry policy with exponential backoff for transient activity failures."""

    max_attempts: int = 3
    initial_backoff_ms: float = 100.0
    backoff_multiplier: float = 2.0
    max_backoff_ms: float = 5000.0
    retryable_exceptions: list[str] = Field(
        default_factory=lambda: ["TimeoutError", "ConnectionError", "RuntimeError"]
    )

    def should_retry(self, attempt: int, exception: Exception) -> bool:
        if attempt >= self.max_attempts:
            return False
        exc_type = type(exception).__name__
        if exc_type in self.retryable_exceptions:
            return True
        for cls_name in self.retryable_exceptions:
            builtin_cls = globals().get(cls_name) or getattr(__builtins__, cls_name, None)
            if builtin_cls and isinstance(exception, builtin_cls):
                return True
        return False

    def get_delay_seconds(self, attempt: int) -> float:
        delay_ms = self.initial_backoff_ms * (self.backoff_multiplier ** (attempt - 1))
        return min(delay_ms, self.max_backoff_ms) / 1000.0


class ActivityDefinition:
    """Descriptor for a declared activity."""

    def __init__(
        self,
        name: str,
        func: Callable[..., Any],
        timeout_seconds: float | None = None,
        retry_policy: RetryPolicy | None = None,
    ):
        self.name = name
        self.func = func
        self.timeout_seconds = timeout_seconds
        self.retry_policy = retry_policy or RetryPolicy()


class ActivityExecutionContext:
    """Thread/Task-local execution context providing activity runtime dependencies."""

    def __init__(
        self,
        workflow_id: str,
        node_id: str,
        memoizer: Any,
        state: Any = None,
        tracer: Any = None,
    ):
        self.workflow_id = workflow_id
        self.node_id = node_id
        self.memoizer = memoizer
        self.state = state
        self.tracer = tracer
        self.invocation_counter: dict[str, int] = {}

    def next_invocation_index(self, activity_name: str) -> int:
        idx = self.invocation_counter.get(activity_name, 0)
        self.invocation_counter[activity_name] = idx + 1
        return idx


# Global context variable for active node/activity scope
current_activity_context: ContextVar[ActivityExecutionContext | None] = ContextVar(
    "current_activity_context", default=None
)


async def execute_activity(
    activity_name: str,
    fn: Callable[..., Any],
    args: tuple[Any, ...] | list[Any] = (),
    kwargs: dict[str, Any] | None = None,
    *,
    idempotency_key: str | None = None,
    timeout_seconds: float | None = None,
    retry_policy: RetryPolicy | None = None,
    context: ActivityExecutionContext | None = None,
) -> Any:
    """
    Executes an external activity with memoizer lookup, retry protection, and token tracking.

    If the activity was previously completed, returns the cached result immediately
    with 0 additional tokens billed.
    """
    kwargs = kwargs or {}
    ctx = context or current_activity_context.get()
    if ctx is None:
        raise RuntimeError(
            "execute_activity must be called within an active ActivityExecutionContext "
            "or with an explicit context parameter."
        )

    memoizer = ctx.memoizer
    invocation_index = ctx.next_invocation_index(activity_name)

    # 1. Compute deterministic idempotency key
    computed_key = memoizer.compute_key(
        workflow_id=ctx.workflow_id,
        node_id=ctx.node_id,
        activity_name=activity_name,
        args=args,
        kwargs=kwargs,
        invocation_index=invocation_index,
        custom_key=idempotency_key,
    )

    # 2. Check Memoizer Cache (Zero duplicate execution)
    cached_record = memoizer.get(computed_key)
    if cached_record is not None:
        if cached_record.status == ActivityStatus.COMPLETED:
            logger.info(
                "Memoizer HIT for activity '%s' (key: %s). Bypassing execution (0 duplicate tokens).",
                activity_name,
                computed_key[:12],
            )
            if ctx.state:
                from agentmesh.engine.state import EventType

                ctx.state.append_event(
                    EventType.ACTIVITY_MEMOIZED_HIT,
                    node_id=ctx.node_id,
                    payload={
                        "activity_name": activity_name,
                        "idempotency_key": computed_key,
                        "cached_total_tokens": cached_record.total_tokens,
                        "billed_tokens": 0,
                    },
                )
            return cached_record.result
        elif cached_record.status == ActivityStatus.FAILED:
            logger.warning(
                "Memoizer found prior failed execution for activity '%s' (key: %s): %s",
                activity_name,
                computed_key[:12],
                cached_record.error,
            )

    # 3. Cache Miss: Execute physical activity
    logger.info(
        "Memoizer MISS for activity '%s' (key: %s). Invoking physical callable.",
        activity_name,
        computed_key[:12],
    )
    if ctx.state:
        from agentmesh.engine.state import EventType

        ctx.state.append_event(
            EventType.ACTIVITY_STARTED,
            node_id=ctx.node_id,
            payload={"activity_name": activity_name, "idempotency_key": computed_key},
        )

    policy = retry_policy or RetryPolicy()
    attempt = 0
    t_start = time.perf_counter()

    call_args = tuple(args) if isinstance(args, (list, tuple)) else (args,)

    while True:
        attempt += 1
        try:
            if inspect.iscoroutinefunction(fn):
                if timeout_seconds:
                    res = await asyncio.wait_for(fn(*call_args, **kwargs), timeout=timeout_seconds)
                else:
                    res = await fn(*call_args, **kwargs)
            else:
                if timeout_seconds:
                    res = await asyncio.wait_for(
                        asyncio.to_thread(fn, *call_args, **kwargs),
                        timeout=timeout_seconds,
                    )
                else:
                    res = fn(*call_args, **kwargs)

            duration_ms = (time.perf_counter() - t_start) * 1000.0

            # Extract token metrics if present in result
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            if isinstance(res, dict):
                usage = res.get("usage", {})
                if isinstance(usage, dict):
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
                if total_tokens == 0:
                    total_tokens = res.get("tokens_spent", 0) or res.get("total_tokens", 0)
                    prompt_tokens = res.get("prompt_tokens", 0)
                    completion_tokens = res.get("completion_tokens", 0)
                    if prompt_tokens == 0 and completion_tokens == 0 and total_tokens > 0:
                        prompt_tokens = total_tokens
            if total_tokens == 0:
                total_tokens = prompt_tokens + completion_tokens

            # Record token metrics in active tracer
            if ctx.tracer and total_tokens > 0 and hasattr(ctx.tracer, "record_token_usage"):
                ctx.tracer.record_token_usage(prompt_tokens, completion_tokens)

            # Store completed record in memoizer
            record = ActivityRecord(
                idempotency_key=computed_key,
                activity_name=activity_name,
                node_id=ctx.node_id,
                workflow_id=ctx.workflow_id,
                status=ActivityStatus.COMPLETED,
                result=res,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                duration_ms=duration_ms,
            )
            memoizer.put(record)

            if ctx.state:
                from agentmesh.engine.state import EventType

                ctx.state.append_event(
                    EventType.ACTIVITY_COMPLETED,
                    node_id=ctx.node_id,
                    payload={
                        "activity_name": activity_name,
                        "idempotency_key": computed_key,
                        "duration_ms": duration_ms,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "result": res,
                    },
                )

            return res

        except Exception as e:
            if policy.should_retry(attempt, e):
                delay = policy.get_delay_seconds(attempt)
                logger.warning(
                    "Activity '%s' failed on attempt %d: %s. Retrying in %.3fs...",
                    activity_name,
                    attempt,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)
                continue

            duration_ms = (time.perf_counter() - t_start) * 1000.0
            fail_record = ActivityRecord(
                idempotency_key=computed_key,
                activity_name=activity_name,
                node_id=ctx.node_id,
                workflow_id=ctx.workflow_id,
                status=ActivityStatus.FAILED,
                error=str(e),
                duration_ms=duration_ms,
            )
            memoizer.put(fail_record)

            if ctx.state:
                from agentmesh.engine.state import EventType

                ctx.state.append_event(
                    EventType.ACTIVITY_FAILED,
                    node_id=ctx.node_id,
                    payload={
                        "activity_name": activity_name,
                        "idempotency_key": computed_key,
                        "error": str(e),
                        "attempts": attempt,
                    },
                )
            raise


def activity(
    name: str | None = None,
    timeout_seconds: float | None = None,
    retry_policy: RetryPolicy | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to declare a function as a managed Activity."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        act_name = name or fn.__name__

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await execute_activity(
                activity_name=act_name,
                fn=fn,
                args=args,
                kwargs=kwargs,
                timeout_seconds=timeout_seconds,
                retry_policy=retry_policy,
            )

        wrapper.__activity_name__ = act_name  # type: ignore[attr-defined]
        return wrapper

    return decorator
