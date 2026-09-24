# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark-neutral entry point for RPent planners and user-owned MCP tools."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import queue
import threading
import uuid
from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic_ai import BinaryContent

from rpent.context import ContextDocument, assemble_context, load_skill
from rpent.dashboard.events import NullDashboardEventSink
from rpent.llm import LLMConfig, LLMUsage
from rpent.planner.base import PlannerResult, build_planner
from rpent.runtime import RuntimeConfig
from rpent.session import EnvState
from rpent.tools.toolkit import ToolResult
from rpent.utils.logging import get_logger

logger = get_logger("embodied_agent")


@dataclass(frozen=True, slots=True)
class McpServer:
    """Connection to a user-owned Streamable HTTP or stdio MCP server.

    Exactly one of ``url`` and ``command`` must be set. A stdio server is
    started for each :meth:`EmbodiedAgent.run` or
    :meth:`EmbodiedAgent.start_episode` and stopped when it ends.
    """

    name: str
    url: str | None = None
    command: str | None = None
    args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None
    cwd: str | None = None
    headers: Mapping[str, str] | None = None
    expose_unprefixed: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError("MCP server name must contain only letters, digits, or _")
        if bool(self.url) == bool(self.command):
            raise ValueError("set exactly one of url and command")
        if self.url and not self.url.startswith(("http://", "https://")):
            raise ValueError("MCP server URL must use http:// or https://")


@dataclass(slots=True)
class _RemoteTool:
    server: str
    name: str
    spec: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PendingToolCall:
    """A model tool call awaiting execution by the benchmark owner."""

    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class _PendingResponse:
    call: PendingToolCall
    ready: threading.Event = field(default_factory=threading.Event)
    result: ToolResult | None = None


class _ToolBridge:
    """Hand selected MCP calls to the thread that owns the environment."""

    def __init__(self) -> None:
        self.calls: queue.Queue[PendingToolCall | None] = queue.Queue()
        self._pending: dict[str, _PendingResponse] = {}
        self._lock = threading.Lock()
        self._closed = False

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        call = PendingToolCall(uuid.uuid4().hex, name, dict(arguments))
        pending = _PendingResponse(call)
        with self._lock:
            if self._closed:
                raise RuntimeError("embodied episode is closed")
            self._pending[call.call_id] = pending
        self.calls.put(call)
        pending.ready.wait()
        with self._lock:
            self._pending.pop(call.call_id, None)
        if pending.result is None:
            raise RuntimeError("embodied episode closed before tool completion")
        return pending.result

    def respond(self, call: PendingToolCall, result: ToolResult) -> None:
        with self._lock:
            pending = self._pending.get(call.call_id)
            if pending is None or pending.call != call or pending.ready.is_set():
                raise ValueError("unknown or already completed embodied tool call")
            if result.name != call.name:
                raise ValueError("tool result name does not match pending call")
            pending.result = result
            pending.ready.set()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for pending in self._pending.values():
                pending.ready.set()
        self.calls.put(None)


class EmbodiedEpisode:
    """Run an agent while the benchmark executes selected tool calls."""

    def __init__(self, bridge: _ToolBridge, worker: threading.Thread) -> None:
        self._bridge = bridge
        self._worker = worker
        self._result: PlannerResult | None = None
        self._error: BaseException | None = None
        self._done = threading.Event()

    def next_call(self, timeout: float | None = None) -> PendingToolCall | None:
        """Wait for a tool call, or return ``None`` after the episode ends."""
        try:
            return self._bridge.calls.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("no embodied tool call before timeout") from None

    def complete(self, call: PendingToolCall, result: ToolResult) -> None:
        """Return an executed tool result, including any camera content blocks."""
        self._bridge.respond(call, result)

    def wait(self, timeout: float | None = None) -> PlannerResult:
        """Wait for the planner result, propagating startup failures."""
        if not self._done.wait(timeout):
            raise TimeoutError("embodied episode did not finish before timeout")
        if self._error is not None:
            raise RuntimeError("embodied episode failed") from self._error
        assert self._result is not None
        return self._result

    def close(self) -> None:
        """Release a blocked tool call and wait for the agent to clean up."""
        self._bridge.close()
        self._worker.join(timeout=5)
        if self._worker.is_alive():
            logger.warning("embodied episode worker is still stopping")

    def __enter__(self) -> EmbodiedEpisode:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class _McpToolkit:
    """Adapt MCP discovery and calls to the planner's toolkit contract."""

    def __init__(
        self,
        servers: Sequence[McpServer],
        output_dir: Path,
        *,
        bridge: _ToolBridge | None = None,
        deferred_tools: Sequence[str] = (),
        include_finish: bool = True,
    ):
        self.state = EnvState(output_dir)
        self._servers = tuple(servers)
        self._sessions: dict[str, Any] = {}
        self._tools: dict[str, _RemoteTool] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop: asyncio.Event | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._call_lock = threading.Lock()
        self._bridge = bridge
        self._deferred_tools = frozenset(deferred_tools)
        self._include_finish = include_finish

    async def _serve(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from mcp.client.streamable_http import streamable_http_client

        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        try:
            async with AsyncExitStack() as stack:
                for server in self._servers:
                    if server.url:
                        # MCP 1.23 accepted headers on the transport; newer
                        # 1.x releases configure them on an HTTP client.
                        transport_kwargs: dict[str, Any]
                        if (
                            "http_client"
                            in inspect.signature(streamable_http_client).parameters
                        ):
                            import httpx

                            http_client = await stack.enter_async_context(
                                httpx.AsyncClient(
                                    headers=dict(server.headers or {}),
                                    timeout=httpx.Timeout(30, read=300),
                                )
                            )
                            transport_kwargs = {"http_client": http_client}
                        else:
                            transport_kwargs = {"headers": dict(server.headers or {})}
                        read, write, _ = await stack.enter_async_context(
                            streamable_http_client(server.url, **transport_kwargs)
                        )
                    else:
                        params = StdioServerParameters(
                            command=server.command,
                            args=list(server.args),
                            env={**os.environ, **(server.env or {})},
                            cwd=server.cwd,
                        )
                        read, write = await stack.enter_async_context(
                            stdio_client(params)
                        )
                    session = await stack.enter_async_context(
                        ClientSession(read, write)
                    )
                    await session.initialize()
                    self._sessions[server.name] = session
                    for tool in (await session.list_tools()).tools:
                        public_name = (
                            tool.name
                            if server.expose_unprefixed
                            else f"{server.name}__{tool.name}"
                        )
                        if public_name in self._tools or public_name == "finish":
                            raise ValueError(f"duplicate MCP tool: {public_name}")
                        self._tools[public_name] = _RemoteTool(
                            server=server.name,
                            name=tool.name,
                            spec={
                                "name": public_name,
                                "description": tool.description or "",
                                "input_schema": tool.inputSchema,
                            },
                        )
                self._ready.set()
                await self._stop.wait()
        except Exception as exc:
            self._startup_error = exc
            logger.error("MCP connection failed: %s", exc)
            self._ready.set()
        finally:
            self._loop = None

    def start(self) -> None:
        """Connect to every MCP server and discover its tools."""
        if self._thread is not None:
            raise RuntimeError("MCP toolkit is already started")
        self._thread = threading.Thread(
            target=lambda: asyncio.run(self._serve()), name="embodied-mcp", daemon=True
        )
        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            self.close()
            raise RuntimeError("failed to start MCP tools") from self._startup_error
        unknown = self._deferred_tools.difference(self._tools)
        if unknown:
            self.close()
            raise ValueError(f"unknown deferred MCP tools: {sorted(unknown)}")

    def get_tools_spec(self) -> list[dict[str, Any]]:
        """Return remote tool schemas and a local completion tool."""
        tools = [tool.spec for tool in self._tools.values()]
        if not self._include_finish:
            return tools
        return tools + [
            {
                "name": "finish",
                "description": "Report your conclusion. Benchmark success is judged by the environment.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["success", "failure", "stuck"],
                        },
                        "summary": {"type": "string"},
                    },
                    "required": ["status", "summary"],
                    "additionalProperties": False,
                },
            }
        ]

    async def _call(self, tool: _RemoteTool, arguments: dict[str, Any]) -> Any:
        return await self._sessions[tool.server].call_tool(tool.name, arguments)

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        """Call one remote tool while preserving its MCP text and image blocks."""
        if name == "finish" and self._include_finish:
            status = input_dict.get("status")
            summary = input_dict.get("summary")
            if status not in {"success", "failure", "stuck"} or not isinstance(
                summary, str
            ):
                return ToolResult(
                    name=name, result={"error": "invalid finish arguments"}
                )
            return ToolResult(
                name=name,
                result={"_finish": True, "status": status, "summary": summary},
            )
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(name=name, result={"error": f"unknown tool: {name}"})
        if name in self._deferred_tools:
            assert self._bridge is not None
            return self._bridge.execute(name, input_dict)
        with self._call_lock:
            if self._loop is None:
                raise RuntimeError("MCP toolkit is closed")
            future = asyncio.run_coroutine_threadsafe(
                self._call(tool, input_dict), self._loop
            )
            try:
                response = future.result()
            except Exception as exc:
                return ToolResult(name=name, result={"error": str(exc)})
        blocks: list[dict[str, Any]] = []
        for item in response.content:
            if item.type == "text":
                blocks.append({"type": "text", "text": item.text})
            elif item.type == "image":
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": item.mimeType,
                            "data": item.data,
                        },
                    }
                )
            else:
                blocks.append(
                    {
                        "type": "text",
                        "text": item.model_dump_json(by_alias=True),
                    }
                )
        structured = getattr(response, "structuredContent", None)
        if structured is None:
            text_blocks = [block["text"] for block in blocks if block["type"] == "text"]
            if len(text_blocks) == 1:
                try:
                    structured = json.loads(text_blocks[0])
                except json.JSONDecodeError:
                    structured = {"text": text_blocks[0]}
            else:
                structured = {"text": "\n".join(text_blocks)}
        result_dict = (
            structured if isinstance(structured, dict) else {"value": structured}
        )
        if response.isError:
            result_dict.setdefault("error", "MCP tool failed")
        result = ToolResult(
            name=name,
            result=result_dict,
        )
        result.is_finish = False
        result.content_blocks = blocks or result.content_blocks
        return result

    def cancel_active_and_wait(self) -> None:
        """Wait for an active physical action before releasing its transport."""
        with self._call_lock:
            return None

    def close(self) -> None:
        """Close owned MCP sessions after outstanding calls finish."""
        if self._bridge is not None:
            self._bridge.close()
        with self._call_lock:
            if self._loop is not None and self._stop is not None:
                self._loop.call_soon_threadsafe(self._stop.set)
            if self._thread is not None:
                self._thread.join()
            self._thread = None
            self._loop = None


@dataclass(slots=True)
class EmbodiedAgent:
    """Run RPent with benchmark-provided instructions and MCP robot tools.

    Construct once per evaluation configuration, then call :meth:`run` for
    each episode. The benchmark owns reset, success checks, and scoring.
    ``runtime`` optionally configures isolated delegates for the API planner;
    it requires the ``runtime`` installation extra.
    """

    mcp_servers: Sequence[McpServer]
    output_dir: str | Path
    planner: str = "api"
    model: str | None = None
    base_url: str | None = None
    llm: LLMConfig | None = None
    max_turns: int = 100
    max_tokens: int = 8192
    planner_timeout_s: int | None = None
    reasoning_effort: str = "none"
    runtime: RuntimeConfig | None = None
    _run_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def run(
        self,
        task: str,
        *,
        system_prompt: str,
        skills: Sequence[str | Path] = (),
        memory: Sequence[ContextDocument] = (),
        initial_context: Sequence[str | BinaryContent] = (),
        output_dir: str | Path | None = None,
    ) -> PlannerResult:
        """Run one episode, loading each supplied ``SKILL.md`` into context.

        Args:
            task: Episode instruction or benchmark task description.
            system_prompt: Benchmark and robot-specific rules.
            skills: Paths to skill Markdown files. Each file is read fresh for
                this episode, so task-specific files can change between runs.
            memory: Selected, authorized memory excerpts with optional source
                identifiers. Appended to the task as reference context, not
                system instructions. No memory files are read automatically.
            initial_context: Text and images placed after the task in the first
                user message. A stable prefix can be reused by prompt caching.
            output_dir: Optional per-episode artifact directory. Defaults to
                the directory given to the agent constructor.

        Returns:
            RPent's planner result. Its ``finish_result`` is self-reported;
            callers must use the benchmark's own success signal for scoring.
        """
        return self._run(
            task,
            system_prompt=system_prompt,
            skills=skills,
            memory=memory,
            initial_context=initial_context,
            output_dir=output_dir,
        )

    def start_episode(
        self,
        task: str,
        *,
        system_prompt: str,
        deferred_tools: Sequence[str],
        skills: Sequence[str | Path] = (),
        memory: Sequence[ContextDocument] = (),
        initial_context: Sequence[str | BinaryContent] = (),
        output_dir: str | Path | None = None,
        include_finish: bool = False,
    ) -> EmbodiedEpisode:
        """Start one persistent run with benchmark-executed MCP tools."""
        if not deferred_tools:
            raise ValueError("start_episode requires at least one deferred tool")
        if self.planner != "api":
            raise ValueError("deferred tool episodes require the api planner")
        bridge = _ToolBridge()
        episode: EmbodiedEpisode

        def run_worker() -> None:
            try:
                episode._result = self._run(
                    task,
                    system_prompt=system_prompt,
                    skills=skills,
                    memory=memory,
                    initial_context=initial_context,
                    output_dir=output_dir,
                    bridge=bridge,
                    deferred_tools=deferred_tools,
                    include_finish=include_finish,
                )
            except BaseException as exc:
                episode._error = exc
            finally:
                bridge.close()
                episode._done.set()

        worker = threading.Thread(
            target=run_worker, name="embodied-episode", daemon=True
        )
        episode = EmbodiedEpisode(bridge, worker)
        worker.start()
        return episode

    def _run(
        self,
        task: str,
        *,
        system_prompt: str,
        skills: Sequence[str | Path],
        initial_context: Sequence[str | BinaryContent],
        output_dir: str | Path | None,
        memory: Sequence[ContextDocument] = (),
        bridge: _ToolBridge | None = None,
        deferred_tools: Sequence[str] = (),
        include_finish: bool = True,
    ) -> PlannerResult:
        if self.planner not in {"api", "claude_code", "codex"}:
            raise ValueError(f"unsupported embodied planner: {self.planner}")
        if self.runtime is not None and self.planner != "api":
            raise ValueError("runtime is supported only by the api planner")
        if self.llm is not None and self.planner != "api":
            raise ValueError("llm is supported only by the api planner")
        if initial_context and self.planner != "api":
            raise ValueError("initial_context requires the api planner")
        if self.llm is not None and (
            self.model is not None or self.base_url is not None
        ):
            raise ValueError("pass either llm or model/base_url")
        if not self.mcp_servers:
            raise ValueError("at least one MCP server is required")
        names = [server.name for server in self.mcp_servers]
        if len(names) != len(set(names)):
            raise ValueError("MCP server names must be unique")
        if not task.strip() or not system_prompt.strip():
            raise ValueError("task and system_prompt must be non-empty")
        if self.max_turns < 1:
            raise ValueError("max_turns must be positive")
        context = assemble_context(
            prompt=system_prompt,
            query=task,
            memory=memory,
            skills=[load_skill(path) for path in skills],
            initial_context=initial_context,
        )
        output = Path(output_dir if output_dir is not None else self.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("EmbodiedAgent already has an active episode")
        toolkit = _McpToolkit(
            self.mcp_servers,
            output,
            bridge=bridge,
            deferred_tools=deferred_tools,
            include_finish=include_finish,
        )
        try:
            toolkit.start()
            planner = build_planner(
                self.planner,
                output_dir=output,
                recipe_tag="embodied_agent",
                robot_name="embodied_agent",
                base_url=self.base_url,
                model=self.model,
                llm_config=self.llm,
                max_tokens=self.max_tokens,
                planner_timeout_s=self.planner_timeout_s,
                reasoning_effort=self.reasoning_effort,
                dashboard_events=NullDashboardEventSink(),
                runtime=self.runtime,
                include_image_reader=False,
                require_tool_call=bridge is not None,
            )
            result = planner.solve(
                system_prompt=context.system_prompt,
                user_message=context.user_message,
                toolkit=toolkit,
                max_turns=self.max_turns,
            )
            result.stats["llm_usage"] = LLMUsage.from_planner_stats(
                result.stats
            ).as_dict()
            return result
        finally:
            toolkit.close()
            self._run_lock.release()
