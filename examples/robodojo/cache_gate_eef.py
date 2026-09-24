#!/usr/bin/env python3
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

"""Measure provider cache reuse through RPent's real multimodal tool loop.

This replays recorded RoboDojo RGB frames. It makes model requests but does
not move a robot or produce a benchmark score.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

from pydantic_ai import BinaryContent

from rpent.embodied_agent import EmbodiedAgent, McpServer
from rpent.evaluation.result import write_json_atomic
from rpent.llm import LLMConfig
from rpent.tools.toolkit import ToolResult

CAMERAS = ("head", "left_wrist", "right_wrist")
TASKS = ("general_pickup", "align_blocks", "stack_blocks")


def _source_trace(source: Path, task: str) -> Path:
    status = json.loads((source / "results/once-status.json").read_text())
    row = next(item for item in status["results"] if item["task"] == task)
    traces = list(
        (source / "runtime/traces" / task).glob(
            f"**/{row['run_id']}/layout-0/l3_inspect_transcript.json"
        )
    )
    if len(traces) != 1:
        raise ValueError(f"expected one scored trace for {task}, found {len(traces)}")
    return traces[0]


def _feedback(name: str, frames: dict[str, bytes], turn: int, turns: int) -> ToolResult:
    payload = {
        "status": "executed",
        "observation": f"Recorded RGB replay after action {turn}; choose another move_eef.",
        "_finish": turn == turns,
    }
    result = ToolResult(name, payload)
    result.content_blocks = [
        {"type": "text", "text": json.dumps(payload)},
        *(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(frames[camera]).decode("ascii"),
                },
            }
            for camera in CAMERAS
        ),
    ]
    return result


def run_task(source: Path, output: Path, key: str, task: str, turns: int) -> dict:
    trace_path = _source_trace(source, task)
    trace = json.loads(trace_path.read_text())
    prompt = trace["policy_config"]["prompt"]
    move = next(
        item["function"]
        for item in prompt["tools"]
        if item["function"]["name"] == "move_eef"
    )
    output.mkdir(parents=True)
    schema = output / "tools.json"
    schema.write_text(json.dumps([move]), encoding="utf-8")
    frames = {
        camera: next((trace_path.parent / "frames" / camera).glob("*.jpg")).read_bytes()
        for camera in CAMERAS
    }
    agent = EmbodiedAgent(
        mcp_servers=[
            McpServer(
                name="dojo",
                expose_unprefixed=True,
                command=sys.executable,
                args=(str(Path(__file__).with_name("eef_episode_mcp.py")), str(schema)),
                env={"PYTHONPATH": os.environ.get("PYTHONPATH", "")},
            )
        ],
        output_dir=output,
        llm=LLMConfig(
            provider="openai",
            model="gpt-6-astra/azure_L/qwb",
            api_key=key,
            base_url="https://tokenhub.sensetime.com/v1",
            openai_format="responses",
            prompt_cache_key="rpent-robodojo-eef-v1",
            prompt_cache_mode="explicit",
            image_history_groups=2,
            parallel_tool_calls=False,
        ),
        max_turns=turns,
        max_tokens=8000,
        planner_timeout_s=1200,
        reasoning_effort="medium",
    )
    initial: list[str | BinaryContent] = ["Initial recorded robot observation:"]
    for camera in CAMERAS:
        initial.extend(
            [
                f"camera '{camera}':",
                BinaryContent(data=frames[camera], media_type="image/jpeg"),
            ]
        )
    with agent.start_episode(
        prompt["goal"] + f" For this cache preflight, call move_eef {turns} times.",
        system_prompt=prompt["system"]
        + f"\nFor this cache preflight, use move_eef for {turns} turns before stopping.",
        initial_context=initial,
        deferred_tools=["move_eef"],
    ) as episode:
        for turn in range(1, turns + 1):
            call = episode.next_call(timeout=240)
            if call is None:
                raise RuntimeError(f"{task}: model ended before action {turn}")
            if call.name != "move_eef":
                raise RuntimeError(f"{task}: unexpected tool {call.name}")
            episode.complete(call, _feedback(call.name, frames, turn, turns))
        result = episode.wait(timeout=120)
    usage = result.stats["llm_usage"]
    if result.error or int(usage["requests"]) < turns:
        raise RuntimeError(f"{task}: incomplete RPent tool loop: {result.error}")
    return {
        "task": task,
        "requests": int(usage["requests"]),
        "input_tokens": int(usage["input_tokens"]),
        "output_tokens": int(usage["output_tokens"]),
        "cache_read_tokens": int(usage["cache_read_tokens"]),
        "cache_write_tokens": int(usage["cache_write_tokens"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-experiment", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--turns", type=int, default=12)
    args = parser.parse_args()
    if args.turns < 4:
        parser.error("--turns must be at least 4")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    key = (args.experiment / "runtime/openai_api_key").read_text().strip()
    rows = [
        run_task(args.source_experiment, output / task, key, task, args.turns)
        for task in args.tasks
    ]
    total = sum(row["input_tokens"] for row in rows)
    cached = sum(row["cache_read_tokens"] for row in rows)
    rate = cached / total if total else 0.0
    report = {"tasks": rows, "cache_hit_rate": rate, "gate_passed": rate > 0.6}
    write_json_atomic(output / "cache-gate.json", report)
    print(f"RPent cache gate: {cached}/{total} = {rate:.1%}")
    return 0 if rate > 0.6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
