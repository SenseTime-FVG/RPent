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

"""Real offline runtime traces convert to portable training calls."""

import hashlib
import json
from pathlib import Path

import pytest
from pydantic_ai import Agent, BinaryContent, CachePoint, ImageUrl, ToolReturn
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from examples.robodojo.export_training import export_experiment
from rpent.llm.retry import RetryLoggingModel, RetryPolicy
from rpent.runtime.trace import RuntimeTraceCapability, TraceConfig, TraceRecorder
from rpent.training_data import load_training_episode
from rpent.training_dataset import TrainingDataset
from rpent.training_trace import convert_rpent_trace


def test_full_trace_exports_actual_multimodal_requests_and_tool_results(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    provider_attempts = 0

    def model(messages, info):
        nonlocal provider_attempts
        provider_attempts += 1
        if provider_attempts == 1:
            raise ModelHTTPError(
                503, "offline", body={"error": {"code": "unavailable"}}
            )
        if any(
            isinstance(part, ToolReturnPart)
            for message in messages
            for part in message.parts
        ):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart("observe", {}, "tool-1")])

    def observe() -> ToolReturn:
        return ToolReturn(
            return_value={"state": "visible"},
            content=[BinaryContent(b"camera", media_type="image/png")],
        )

    agent = Agent(
        RetryLoggingModel(
            FunctionModel(model),
            policy=RetryPolicy(max_retries=1, initial_delay_s=0, max_delay_s=0),
            log_path=run / "llm_errors.jsonl",
        ),
        instructions="Inspect before moving.",
        tools=[observe],
        capabilities=[RuntimeTraceCapability()],
    )
    recorder = TraceRecorder(run)
    with recorder.activate():
        assert (
            agent.run_sync(
                [
                    "Find the block",
                    CachePoint(),
                    BinaryContent(b"initial", media_type="image/png"),
                ]
            ).output
            == "done"
        )
    recorder.close()

    path = convert_rpent_trace(
        run,
        {
            "benchmark": "robodojo",
            "task_id": "find-block",
            "instruction": "Find the block",
            "seed": 0,
        },
        output_dir=tmp_path / "training",
        outcome={"success": True, "score": 1.0},
    )
    episode = load_training_episode(path)
    assert len(episode["calls"]) == 2
    first, second = episode["calls"]
    assert provider_attempts == 3
    assert [attempt["status"] for attempt in first["attempts"]] == [
        "error",
        "completed",
    ]
    assert first["request"]["system_prompt"] == "Inspect before moving."
    assert any(
        part == {"type": "cache_point", "ttl": "5m"}
        for message in first["request"]["messages"]
        for part in message["content"]
    )
    assert first["request"]["tools"][0]["name"] == "observe"
    assert first["response"]["message"]["tool_calls"][0]["id"] == "tool-1"
    assert first["tool_results"][0]["tool_call_id"] == "tool-1"
    assert second["request"]["messages"][-2]["role"] == "tool"
    assert second["response"]["message"]["content"][0]["text"] == "done"
    assert len(list((path.parent / "assets").iterdir())) == 2


def test_trace_export_requires_full_capture(tmp_path: Path) -> None:
    run = tmp_path / "metadata-run"
    run.mkdir()
    recorder = TraceRecorder(run, TraceConfig(mode="metadata"))
    recorder.close()
    with pytest.raises(ValueError, match="full RPent trace"):
        convert_rpent_trace(
            run,
            {"benchmark": "robodojo", "task_id": "one", "instruction": "Do it"},
            output_dir=tmp_path / "training",
        )


def test_trace_export_rejects_remote_images_without_local_bytes(tmp_path: Path) -> None:
    run = tmp_path / "remote-image"
    run.mkdir()
    recorder = TraceRecorder(run)
    agent = Agent(
        RetryLoggingModel(
            FunctionModel(
                lambda messages, info: ModelResponse(parts=[TextPart("done")])
            )
        ),
        capabilities=[RuntimeTraceCapability()],
    )
    with recorder.activate():
        agent.run_sync(["Find the block", ImageUrl("https://example.org/camera.png")])
    recorder.close()
    with pytest.raises(ValueError, match="remote image URL"):
        convert_rpent_trace(
            run,
            {"benchmark": "robodojo", "task_id": "one", "instruction": "Do it"},
            output_dir=tmp_path / "training",
        )


def test_robodojo_export_joins_native_score_with_matching_trace(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    run = experiment / "runtime/traces/stack-block/run-1/rpent"
    run.mkdir(parents=True)
    recorder = TraceRecorder(run)
    agent = Agent(
        RetryLoggingModel(
            FunctionModel(
                lambda messages, info: ModelResponse(parts=[TextPart("done")])
            )
        ),
        capabilities=[RuntimeTraceCapability()],
    )
    with recorder.activate():
        agent.run_sync("Stack the block")
    recorder.close()
    (run / "episode.json").write_text(json.dumps({"instruction": "Stack the block"}))
    results = experiment / "results"
    results.mkdir()
    (results / "once-status.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "task": "stack-block",
                        "run_id": "run-1",
                        "success": True,
                        "score": 0.8,
                        "path": "native/result.json",
                    }
                ]
            }
        )
    )
    exported = export_experiment(
        experiment,
        tmp_path / "training",
        dataset_id="robodojo-eef",
        dataset_version="2026-09.v1",
        split="train",
        seed=3,
        benchmark_version="2026-09",
        versions={"agent_commit": "test-sha", "simulator": "isaac-test"},
    )
    assert len(exported) == 1
    record = load_training_episode(exported[0])
    assert record["task"]["benchmark"] == "robodojo"
    assert record["task"]["seed"] == 3
    assert record["outcome"]["score"] == 0.8
    assert record["outcome"]["metadata"]["native_score_source"] == "native/result.json"
    assert record["provenance"]["source_kind"] == "rpent_trace"
    assert record["provenance"]["versions"]["agent_commit"] == "test-sha"
    assert (
        record["provenance"]["source_manifest_sha256"]
        == hashlib.sha256((run / "trace/manifest.json").read_bytes()).hexdigest()
    )
    dataset = TrainingDataset.open(tmp_path / "training/robodojo-eef/2026-09.v1")
    assert dataset.verify()["episodes"][0]["split"] == "train"
