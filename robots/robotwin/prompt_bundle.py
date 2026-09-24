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

"""RoboTwin prompt bundle assembly."""

from __future__ import annotations

from collections.abc import Mapping

from robots.robotwin.prompts import evaluate as evaluate_parts
from robots.robotwin.prompts import explore as explore_parts
from robots.robotwin.prompts import user as user_parts
from rpent.prompt.utils import PromptNode


def system_prompt(
    variables: Mapping[str, object] | None = None,
) -> PromptNode:
    """Return the RoboTwin system prompt for the selected run mode."""
    if not (variables or {}).get("enable_vla", True):
        explore = (variables or {}).get("mode", "eval") == "explore"
        parts = explore_parts if explore else evaluate_parts
        return {
            "ROLE": parts.ROLE,
            "READ ORDER": parts.READ_ORDER,
            "CONTROL": (
                "VLA is disabled. Use only the exposed scripted primitives. "
                "Ground every target in current perception, specify the selected "
                "arm, and inspect the returned state after each bounded action. "
                "Stop if the task needs an unavailable learned capability; do "
                "not call tools absent from the schema or start model services. "
                "Treat memory and guides as technique references subject to "
                "this run's available tools."
            ),
            "PERCEPTION": evaluate_parts.PERCEPTION,
            "RUNTIME": parts.RUNTIME,
            "BUDGET AND SUCCESS": parts.BUDGET_AND_SUCCESS,
            "MODE": parts.USER_MODE,
            **({"MEMORY": parts.MEMORY, "OUTPUT": parts.OUTPUT} if explore else {}),
        }
    if (variables or {}).get("mode", "eval") == "explore":
        return explore_parts.system_prompt()
    return evaluate_parts.system_prompt(variables)


def user_prompt(
    variables: Mapping[str, object] | None = None,
) -> PromptNode:
    prompt = {
        "CELL": user_parts.CELL,
        "BEGIN": user_parts.BEGIN,
    }
    if (variables or {}).get("mode", "eval") == "explore":
        prompt["EXPLORE MODE"] = explore_parts.USER_MODE
    return prompt


__all__ = ["system_prompt", "user_prompt"]
