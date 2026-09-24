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

"""Assemble PydanticAI agents from RPent's existing models, context and tools."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import Model
from pydantic_ai.toolsets import FunctionToolset, ToolsetTool
from pydantic_ai.toolsets.wrapper import WrapperToolset

from rpent.context import assemble_context, load_skill
from rpent.llm.client import build_model_settings
from rpent.llm.retry import RetryLoggingModel
from rpent.runtime.config import RuntimeConfig

if TYPE_CHECKING:
    from pydantic_ai.capabilities import AgentCapability, WrapRunHandler
    from pydantic_ai.run import AgentRunResult
    from pydantic_ai.usage import UsageLimits

_CHILD_READERS = frozenset({"read_image", "read_text_file", "list_dir"})


class _InheritRequestLimit(AbstractCapability):
    """Forward the planner's request limit alongside the SDK's shared usage."""

    def __init__(self, limits: ContextVar[UsageLimits | None]) -> None:
        super().__init__()
        self._limits = limits

    async def wrap_run(
        self, ctx: RunContext, *, handler: WrapRunHandler
    ) -> AgentRunResult:
        parent_limits = self._limits.get()
        if parent_limits is not None:
            # SubAgents 0.34 shares usage but starts children with default limits.
            # Mutate the run-owned limit object also used by the SDK request loop.
            ctx.usage_limits.request_limit = parent_limits.request_limit
        token = self._limits.set(ctx.usage_limits)
        try:
            return await handler()
        finally:
            self._limits.reset(token)


class _SharedToolset(WrapperToolset):
    """Keep existing tool calls serial across otherwise concurrent agent runs."""

    def __init__(self, tools: Sequence[Tool], lock: asyncio.Lock) -> None:
        super().__init__(FunctionToolset(tools=tools))
        self._lock = lock

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext,
        tool: ToolsetTool,
    ) -> Any:
        async with self._lock:
            return await self.wrapped.call_tool(name, tool_args, ctx, tool)


def build_runtime_agent(
    *,
    model: Model,
    system_prompt: str,
    tools: Sequence[Tool],
    max_tokens: int,
    capabilities: Sequence[AgentCapability],
    runtime: RuntimeConfig | None = None,
) -> Agent:
    """Build the existing parent agent with optional explicitly configured delegates.

    Child skill and model errors propagate before any model request. Child runs
    use the SDK's isolated histories, shared usage, and cancellation lifecycle.

    Args:
        model: Existing parent model, including request retry configuration.
        system_prompt: Already assembled parent instructions.
        tools: Existing planner tool wrappers.
        max_tokens: Output-token cap inherited by all configured agents.
        capabilities: Existing thinking and history-processing capabilities.
        runtime: Explicit delegates; omitted or empty retains a single agent.

    Returns:
        The parent Agent driven by the existing planner loop.

    Raises:
        ValueError: If a child requests a forbidden or unavailable tool, or the
            toolkit already contains the reserved delegation tool name.
        ImportError: If delegation is configured without the runtime extra.
    """
    if runtime is None or not runtime.subagents:
        return Agent(
            model,
            instructions=system_prompt or None,
            tools=tools,
            model_settings=build_model_settings(model, max_tokens),
            capabilities=capabilities,
        )

    try:
        from pydantic_ai_harness import SubAgent, SubAgents
    except ImportError as exc:
        raise ImportError(
            "Configured sub-agents require the runtime extra: pip install 'rpent[runtime]'"
        ) from exc

    catalog = {tool.name: tool for tool in tools}
    if "delegate_task" in catalog:
        raise ValueError("delegate_task is reserved for configured sub-agents")
    lock = asyncio.Lock()
    limits: ContextVar[UsageLimits | None] = ContextVar(
        "rpent_request_limits", default=None
    )
    run_capabilities = [*capabilities, _InheritRequestLimit(limits)]
    children = []
    for name, config in runtime.subagents.items():
        for tool_name in config.tools:
            if tool_name not in _CHILD_READERS:
                raise ValueError(
                    f"sub-agent {name!r} cannot use tool {tool_name!r}; only artifact and text readers are supported"
                )
            if tool_name not in catalog:
                raise ValueError(
                    f"sub-agent {name!r} requests unavailable tool {tool_name!r}"
                )
        child_model = model
        if config.model is not None:
            from rpent.planner.base import build_api_model

            child_model = RetryLoggingModel(
                build_api_model(config.model),
                policy=model.policy if isinstance(model, RetryLoggingModel) else None,
                log_path=model.log_path
                if isinstance(model, RetryLoggingModel)
                else None,
            )
        context = assemble_context(
            prompt=config.instructions,
            query="",
            skills=[load_skill(path) for path in config.skills],
        )
        child = Agent(
            child_model,
            name=name,
            description=config.description,
            instructions=context.system_prompt,
            model_settings=build_model_settings(child_model, max_tokens),
            toolsets=[_SharedToolset([catalog[key] for key in config.tools], lock)],
            capabilities=run_capabilities,
        )
        children.append(SubAgent(child))

    return Agent(
        model,
        instructions=system_prompt or None,
        model_settings=build_model_settings(model, max_tokens),
        toolsets=[_SharedToolset(tools, lock)],
        capabilities=[
            *run_capabilities,
            SubAgents(agents=children, agent_folders=None, inherit_tools=False),
        ],
    )
