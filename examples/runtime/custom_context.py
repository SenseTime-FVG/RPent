# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example async compression policy; install its factory, not a shared instance."""

from __future__ import annotations

from pydantic_ai import RunContext
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart

from rpent.runtime.context_engine import ContextEngine, recent_turns, summarize_history


def make_context_engine() -> ContextEngine:
    """Create one independent compressor for each root or child invocation.

    Compression preserves the first task, latest user input and complete recent
    tool exchanges. Its summary requests consume the same usage/request budget
    as the parent run, and appear with purpose=context_compression in the trace.
    """
    summary = ""

    async def compress(
        ctx: RunContext, messages: list[ModelMessage]
    ) -> list[ModelMessage]:
        nonlocal summary
        if len(messages) < 7:
            return messages
        previous_summary = "Earlier history summary (reference):\n" + summary
        current = [
            message
            for message in messages
            if not (
                summary
                and isinstance(message, ModelRequest)
                and len(message.parts) == 1
                and isinstance(message.parts[0], UserPromptPart)
                and message.parts[0].content == previous_summary
            )
        ]
        kept = recent_turns(ctx, current, keep_turns=2)
        # recent_turns returns the original message objects from this callback's
        # private history copy, so identity identifies precisely what was dropped.
        retained = {id(message) for message in kept}
        dropped = [message for message in current if id(message) not in retained]
        if not dropped:
            return messages
        # Include a previous summary when its message leaves the working window.
        # The helper returns ordinary text and does not recursively use this hook.
        if summary:
            dropped.insert(0, ModelRequest(parts=[UserPromptPart(previous_summary)]))
        summary = await summarize_history(ctx, dropped)
        memory = ModelRequest(
            parts=[UserPromptPart("Earlier history summary (reference):\n" + summary)]
        )
        # Put summary before the current task/history so the latest real input
        # remains unchanged. Fixed agent instructions and tool permissions remain
        # outside this hook's message list.
        return [memory, *kept]

    return compress
