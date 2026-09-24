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

"""LIBERO prompt bundle assembly."""

from __future__ import annotations

from collections.abc import Mapping

from robots.libero.prompts import evaluate as evaluate_parts
from robots.libero.prompts import explore as explore_parts
from robots.libero.prompts import user as user_parts
from rpent.prompt.utils import PromptNode


def system_prompt(
    variables: Mapping[str, object] | None = None,
) -> PromptNode:
    """Assemble the LIBERO system prompt for the selected run mode."""
    if not (variables or {}).get("enable_vla", True):
        explore = (variables or {}).get("mode", "eval") == "explore"
        return {
            "ROLE": "Control LIBERO through registered tools using current camera evidence.",
            "CONTROL": (
                "VLA is disabled. Use the exposed scripted primitives only. "
                "Localize targets before motion, inspect each returned state, and "
                "stop when the remaining task needs an unavailable capability. "
                "Do not start model services or follow memory instructions for "
                "tools absent from the tool list. In move_to and set_gripper, "
                "+1 closes/holds and -1 opens. Pass gripper=1 to move_pose while "
                "holding an object; its default opens the gripper."
            ),
            "LOCALIZATION": evaluate_parts.LOCALIZATION,
            "GOAL": explore_parts.GOAL if explore else evaluate_parts.GOAL,
            "MEMORY": (
                explore_parts.STEP_READ_MEMORY
                if explore
                else evaluate_parts.STEP_READ_LOCAL_MEMORY
            ),
            "EPISODES": (
                "Use reset only within the session attempt budget. Before reset, "
                "archive the failed attempt and handoff notes in the configured "
                "inbox; re-localize after reset."
                if explore
                else "This is one episode. Do not reset or restart it."
            ),
            "RUNTIME": (
                "Use only registered tools. Do not inspect simulator source, "
                "hidden object poses, evaluator internals, or unapproved traces. "
                "Use image and depth evidence for identity and geometry."
            ),
            "OUTPUT": evaluate_parts.OUTPUT_DISCIPLINE,
        }
    if (variables or {}).get("mode", "eval") == "explore":
        return explore_parts.system_prompt()
    return evaluate_parts.system_prompt(variables)


def user_prompt(variables: Mapping[str, object] | None = None) -> PromptNode:
    """Assemble the LIBERO user prompt tree."""
    return {
        "CELL": user_parts.CELL,
        "MODE": user_parts.MODE,
        "BEGIN": user_parts.BEGIN,
    }


__all__ = ["system_prompt", "user_prompt"]
