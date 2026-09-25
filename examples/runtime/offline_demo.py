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

"""Exercise the runtime with scripted model output and explicitly synthetic media."""

from __future__ import annotations

import argparse
import asyncio
from io import BytesIO
from pathlib import Path

from pydantic_ai import BinaryContent, Tool, ToolReturn
from pydantic_ai.messages import (
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage, UsageLimits

from rpent.cli.trajectory import export_trajectory
from rpent.data_convert import convert_planner_input
from rpent.llm.retry import RetryLoggingModel
from rpent.runtime import ContextPolicy, RuntimeConfig, SubAgentConfig
from rpent.runtime.factory import build_runtime_agent
from rpent.runtime.trace import RuntimeTraceCapability, TraceRecorder, exception_status
from rpent.utils.logging import get_logger, init_output_dir

logger = get_logger("runtime.demo")
HERE = Path(__file__).resolve().parent


async def run_demo(output: Path, *, compress: bool = False) -> Path:
    """Record a deterministic run; all token counts and visual data are synthetic."""
    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg  # noqa: F401 - check the encoder before starting
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Install demo dependencies: pip install -e '.[test,runtime]' imageio imageio-ffmpeg"
        ) from exc
    if output.exists() and any(output.iterdir()):
        raise ValueError("demo output directory must be new or empty")
    output = init_output_dir(output).resolve()
    frames = []
    for step in range(12):
        frame = np.full((128, 192, 3), 225, dtype=np.uint8)
        frame[48:80, 10 + step * 12 : 42 + step * 12] = (40, 90, 220)
        frames.append(frame)
    buffer = BytesIO()
    imageio.imwrite(buffer, frames[0], format="png")
    initial_image = BinaryContent(data=buffer.getvalue(), media_type="image/png")
    position = 0
    root_turn = 0

    def model(messages, info):
        nonlocal root_turn
        # Deliberate fixture usage, not measured tokenizer or provider counts.
        usage = RequestUsage(
            input_tokens=100,
            output_tokens=20,
            cache_read_tokens=40,
            cache_write_tokens=0,
        )
        if (info.instructions or "").startswith("Summarize the task"):
            return ModelResponse(
                parts=[
                    TextPart(
                        "A synthetic observation was read. The reviewer checks visible evidence before movement."
                    )
                ],
                usage=usage,
            )
        if (info.instructions or "").startswith("REVIEWER"):
            returned = [
                part
                for message in messages
                for part in message.parts
                if isinstance(part, ToolReturnPart)
            ]
            parts = (
                [
                    TextPart(
                        "The provided scene is synthetic. Movement may be demonstrated; physical success is unverified."
                    )
                ]
                if returned
                else [ToolCallPart("read_skill", {"name": "review"}, "review-skill")]
            )
            return ModelResponse(parts=parts, usage=usage)
        root_turn += 1
        calls = [
            ("read_skill", {"name": "operation"}),
            ("read_skill", {"name": "operation", "resource": "resources/checklist.md"}),
            ("observe", {}),
            (
                "delegate_task",
                {
                    "agent_name": "reviewer",
                    "task": "Review this synthetic observation: a blue marker is at the left of a grey frame. No robot measurements are available.",
                },
            ),
            ("move_marker", {}),
            ("observe", {}),
        ]
        parts = (
            [ToolCallPart(*calls[root_turn - 1], tool_call_id=f"demo-{root_turn}")]
            if root_turn <= len(calls)
            else [
                TextPart(
                    "Offline demonstration complete: read both skill resources, delegated review, moved and observed the synthetic marker."
                )
            ]
        )
        return ModelResponse(parts=parts, usage=usage)

    trace = TraceRecorder(output)

    def observe() -> ToolReturn:
        """Return one synthetic camera frame and its marker position."""
        image = BytesIO()
        imageio.imwrite(image, frames[position], format="png")
        return ToolReturn(
            return_value={"synthetic": True, "frame": position},
            content=[BinaryContent(data=image.getvalue(), media_type="image/png")],
        )

    async def move_marker() -> dict:
        """Move the synthetic marker and record the exact generated frame range."""
        nonlocal position
        clip = output / "synthetic-action.mp4"
        await asyncio.to_thread(imageio.mimwrite, clip, frames, fps=12, codec="libx264")
        position = len(frames) - 1
        mapping = {
            "artifact_ref": clip.name,
            "video_ref": clip.name,
            "media_type": "video/mp4",
            "frame_start": 0,
            "frame_end": len(frames),
            "fps": 12,
            "synthetic": True,
            "step_idx": 1,
        }
        trace.emit("artifact_written", mapping)
        return {"synthetic": True, "marker_frame": position, "media": mapping}

    context = {"context": ContextPolicy(strategy="recent_turns", keep_turns=2)}
    if compress:
        from examples.runtime.custom_context import make_context_engine

        context = {"context_engine_factory": make_context_engine}
    runtime = RuntimeConfig(
        **context,
        skill_paths=[HERE / "skills/operation"],
        subagents={
            "reviewer": SubAgentConfig(
                instructions="REVIEWER: Inspect explicitly supplied evidence.",
                skill_paths=[HERE / "skills/review"],
                tools=["read_skill"],
            )
        },
    )
    runtime.validate_resources()
    agent = build_runtime_agent(
        model=RetryLoggingModel(
            FunctionModel(model), log_path=output / "llm_errors.jsonl"
        ),
        system_prompt="Demonstrate the runtime using synthetic observations. Report physical task success as unverified.",
        tools=[Tool(observe), Tool(move_marker)],
        max_tokens=512,
        capabilities=[RuntimeTraceCapability()],
        runtime=runtime,
    )
    inputs = convert_planner_input(
        prompt="",
        query="Read the operation skill, inspect, request a review, move the synthetic marker and inspect again.",
        initial_context=[initial_image],
    )
    failure = None
    with trace.activate():
        trace.emit(
            "run_metadata",
            {"synthetic_model": True, "synthetic_usage": True, "synthetic_media": True},
        )
        try:
            result = await agent.run(
                inputs.user_message, usage_limits=UsageLimits(request_limit=30)
            )
            logger.info("%s", result.output)
            logger.info("Synthetic SDK usage: %s", result.usage)
        except BaseException as exc:
            failure = exc
            raise
        finally:
            trace.close(
                status=exception_status(failure) if failure else "completed",
                error=failure,
            )
    report = export_trajectory(output, output / "report")
    logger.info("Offline report: %s", report)
    return report


def main() -> None:
    """Run the offline example in a fresh output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Use the async summary policy instead of recent_turns.",
    )
    args = parser.parse_args()
    asyncio.run(run_demo(args.output, compress=args.compress))


if __name__ == "__main__":
    main()
