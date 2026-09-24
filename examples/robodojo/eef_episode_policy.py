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

"""User-owned RoboProbe EEF logic driven by RPent's persistent EmbodiedAgent.

Install this file as ``RoboDojo_EmbodiedAgent_EEF/policy.py``. RoboDojo keeps
ownership of Isaac, CuRobo planning, episode termination, and native scoring.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from pydantic_ai import BinaryContent
from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.types import ActionChunk, Observation
from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect_EEF.policy import EefAgentPolicy

from rpent.embodied_agent import (
    EmbodiedAgent,
    EmbodiedEpisode,
    McpServer,
    PendingToolCall,
)
from rpent.evaluation.result import write_json_atomic
from rpent.llm import LLMConfig
from rpent.tools.toolkit import ToolResult


def _message_parts(message: dict[str, Any]) -> list[str | BinaryContent]:
    content = message["content"]
    if isinstance(content, str):
        return [content]
    parts: list[str | BinaryContent] = []
    for item in content:
        if item["type"] == "text":
            parts.append(str(item["text"]))
        elif item["type"] == "image_url":
            data_uri = item["image_url"]["url"]
            if not data_uri.startswith("data:image/jpeg;base64,"):
                raise ValueError("RoboDojo model images must be JPEG data URLs")
            parts.append(
                BinaryContent(
                    data=base64.b64decode(data_uri.split(",", 1)[1]),
                    media_type="image/jpeg",
                )
            )
    return parts


def _observation_result(
    call: PendingToolCall,
    observation: dict[str, Any],
    *,
    feedback: str,
    executed_waypoints: int,
    episode_done: bool = False,
) -> ToolResult:
    parts = _message_parts(observation)
    text = "\n".join(item for item in parts if isinstance(item, str))
    payload = {
        "status": "executed",
        "feedback": feedback,
        "executed_waypoints": executed_waypoints,
        "observation": text,
        "_finish": episode_done,
    }
    result = ToolResult(call.name, payload)
    result.content_blocks = [{"type": "text", "text": feedback}]
    for part in parts:
        if isinstance(part, str):
            result.content_blocks.append({"type": "text", "text": part})
        else:
            result.content_blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.b64encode(part.data).decode("ascii"),
                    },
                }
            )
    return result


class _UnusedClient:
    def complete(self, *_: Any, **__: Any) -> None:
        raise AssertionError("RPent owns the LLM loop")


class EmbodiedEefPolicy(EefAgentPolicy):
    """Keep RoboProbe's EEF planner while RPent owns each model/tool turn."""

    def __init__(self, **kwargs: Any) -> None:
        self._episode: EmbodiedEpisode | None = None
        self._pending: PendingToolCall | None = None
        self._pending_feedback = ""
        self._pending_request_id: str | None = None
        self._played = 0
        self._started_at: float | None = None
        self._artifact_dir = Path(kwargs["env"]["L3_INSPECT_TRACE_DIR"]) / "rpent"
        super().__init__(client=_UnusedClient(), **kwargs)

    def _start(self, observation: Observation) -> None:
        self._started_at = time.monotonic()
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        specs = [item["function"] for item in self._tools]
        schema_path = self._artifact_dir / "eef_tools.json"
        schema_path.write_text(json.dumps(specs), encoding="utf-8")
        context: list[str | BinaryContent] = []
        # The goal is passed separately as the task. Only demonstration
        # images are protected; the first live observation ages out.
        demonstration = self._messages[2:]
        demonstration_image_count = 0
        for index, message in enumerate(
            [*demonstration, self._observation_message(observation)]
        ):
            parts = _message_parts(message)
            if index < len(demonstration):
                demonstration_image_count += sum(
                    isinstance(part, BinaryContent) for part in parts
                )
            context.extend(parts)
        model = self._env.get("L3_INSPECT_MODEL", "gpt-6-astra/azure_L/qwb")
        agent = EmbodiedAgent(
            mcp_servers=[
                McpServer(
                    name="dojo",
                    expose_unprefixed=True,
                    command=sys.executable,
                    args=(
                        "-m",
                        "XPolicyLab.policy.RoboDojo_EmbodiedAgent_EEF.mcp_schema",
                        str(schema_path),
                    ),
                    env={"PYTHONPATH": os.environ.get("PYTHONPATH", "")},
                )
            ],
            output_dir=self._artifact_dir,
            llm=LLMConfig(
                provider="openai",
                model=model,
                base_url=self._env.get("L3_INSPECT_BASE_URL"),
                openai_format="responses",
                prompt_cache_key=self._env.get(
                    "L3_INSPECT_PROMPT_CACHE_KEY", "rpent-robodojo-eef-v1"
                ),
                prompt_cache_mode="explicit",
                image_history_groups=2,
                preserve_initial_image_count=demonstration_image_count,
                parallel_tool_calls=False,
            ),
            max_turns=self._max_llm_calls,
            max_tokens=int(self._env.get("L3_INSPECT_MAX_TOKENS", "8000")),
            planner_timeout_s=10800,
            reasoning_effort=self._env.get("L3_INSPECT_REASONING_EFFORT", "medium"),
        )
        self._episode = agent.start_episode(
            self._goal_text or f"Goal: {observation.instruction or ''}",
            system_prompt=self._system_message(),
            initial_context=context,
            deferred_tools=[item["name"] for item in specs],
        )

    def act(self, observation: Observation) -> ActionChunk:
        self.prepare(observation)
        if self._episode is None:
            deferred = self._deferred_llm_chunk(observation)
            if deferred is not None:
                return deferred
            self._start(observation)
        elif self._pending is not None:
            self._episode.complete(
                self._pending,
                _observation_result(
                    self._pending,
                    self._observation_message(observation),
                    feedback=self._pending_feedback,
                    executed_waypoints=self._played,
                ),
            )
            self._pending = None
            if self._pending_request_id is not None:
                for record in self._transcript:
                    if record.get("response_id") == self._pending_request_id:
                        record["tool_result"] = "executed with fresh RGB and state"
                        record["executed_waypoints"] = self._played
                        break
                self._pending_request_id = None

        repair_attempts = 0
        while True:
            call = self._episode.next_call(timeout=300)
            if call is None:
                result = self._episode.wait(timeout=5)
                return self._give_up_chunk(
                    result.error or "model ended without another tool call", observation
                )
            self._sync_requests()
            name = call.name
            if self._transcript:
                self._transcript[-1].update(
                    {
                        "policy_step": observation.step,
                        "tool": name,
                        "arguments": call.arguments,
                    }
                )
            if name == "move_eef":
                outcome = self._handle_motion(name, call.arguments, observation)
                if outcome.chunk is not None:
                    self._pending = call
                    self._pending_feedback = outcome.tool_result
                    self._pending_request_id = (
                        self._transcript[-1]["response_id"]
                        if self._transcript
                        else None
                    )
                    self._played = 0
                    return outcome.chunk
                if self._transcript:
                    self._transcript[-1]["validation_error"] = outcome.tool_result
                if outcome.repairable:
                    repair_attempts += 1
                    if repair_attempts >= 3:
                        self._episode.complete(
                            call,
                            ToolResult(
                                call.name,
                                {"_finish": True, "status": "invalid_action_budget"},
                            ),
                        )
                        return self._give_up_chunk(
                            "three invalid move_eef calls", observation
                        )
                self._episode.complete(
                    call,
                    ToolResult(
                        call.name, {"status": "rejected", "reason": outcome.tool_result}
                    ),
                )
                continue
            if name in {"give_up", "done"}:
                self._episode.complete(
                    call, ToolResult(call.name, {"_finish": True, "status": name})
                )
                return self._stop_chunk(name, call.arguments, observation)
            self._episode.complete(
                call, ToolResult(call.name, {"error": f"unsupported tool {name}"})
            )

    def confirm_executed(self, played: int) -> None:
        self._played = played

    def complete_post_action(
        self, observation: Observation, *, episode_done: bool
    ) -> None:
        """Return the actual final waypoint observation to the pending MCP call."""
        if self._pending is None or self._episode is None:
            return
        self._episode.complete(
            self._pending,
            _observation_result(
                self._pending,
                self._observation_message(
                    replace(observation, step=observation.step + 1)
                ),
                feedback=self._pending_feedback,
                executed_waypoints=self._played,
                episode_done=episode_done,
            ),
        )
        if self._pending_request_id is not None:
            for record in self._transcript:
                if record.get("response_id") == self._pending_request_id:
                    record["tool_result"] = "executed with fresh RGB and state"
                    record["executed_waypoints"] = self._played
                    break
        self._pending = None
        self._pending_request_id = None

    def _sync_requests(self) -> None:
        path = self._artifact_dir / "llm_requests.jsonl"
        if not path.exists():
            return
        records = [json.loads(line) for line in path.read_text().splitlines() if line]
        successes = [record for record in records if record.get("outcome") == "success"]
        self._calls = len(successes)
        previous = {item["response_id"]: item for item in self._transcript}
        self._transcript = [
            {
                **previous.get(record["request_id"], {}),
                "response_id": record["request_id"],
                "usage": record.get("usage"),
            }
            for record in successes
        ]

    def transcript(self) -> list[dict[str, Any]]:
        self._sync_requests()
        return super().transcript() or []

    @property
    def calls(self) -> int:
        self._sync_requests()
        return self._calls

    def audit_config(self) -> dict[str, Any]:
        config = super().audit_config()
        config["adapter"] = "rpent-embodied-agent-roboprobe-eef"
        config["rpent_artifacts"] = str(self._artifact_dir)
        return config

    def close(self) -> None:
        if self._episode is None:
            return
        if self._pending is not None:
            self._episode.complete(
                self._pending,
                ToolResult(
                    self._pending.name,
                    {"_finish": True, "status": "benchmark_episode_end"},
                ),
            )
            self._pending = None
        result = None
        error = None
        try:
            result = self._episode.wait(timeout=20)
        except (TimeoutError, RuntimeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            self._episode.close()
            self._episode = None
        self._sync_requests()
        usage = (result.stats.get("llm_usage") or {}) if result else {}
        input_tokens = int(usage.get("input_tokens") or 0)
        cache_read = int(usage.get("cache_read_tokens") or 0)
        write_json_atomic(
            self._artifact_dir / "episode.json",
            {
                "task": self._task_name,
                "model": self._env.get("L3_INSPECT_MODEL"),
                "requests": self._calls,
                "input_tokens": input_tokens,
                "output_tokens": int(usage.get("output_tokens") or 0),
                "cache_read_tokens": cache_read,
                "cache_write_tokens": int(usage.get("cache_write_tokens") or 0),
                "cache_hit_rate": cache_read / input_tokens if input_tokens else None,
                "agent_elapsed_s": (
                    time.monotonic() - self._started_at
                    if self._started_at is not None
                    else None
                ),
                "agent_error": error or (result.error if result else None),
            },
        )
