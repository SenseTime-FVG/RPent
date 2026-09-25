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

"""Cross-benchmark training records preserve calls, feedback and visual input."""

import hashlib
import json
from pathlib import Path

import pytest

from rpent.training_data import convert_episode, load_training_episode


def _task() -> dict:
    return {
        "benchmark": "robodojo",
        "benchmark_version": "2026-09",
        "task_id": "put-block",
        "instruction": "Put the red block in the bowl.",
        "seed": 7,
    }


def test_convert_episode_keeps_each_llm_call_tool_feedback_and_images(
    tmp_path: Path,
) -> None:
    camera = b"test-camera-pixels"
    calls = [
        {
            "call_id": "llm-1",
            "request": {
                "model": "vision-model",
                "provider": "openai",
                "system_prompt": "Use robot tools.",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            "Move the block.",
                            {
                                "type": "image",
                                "media_type": "image/png",
                                "data": camera,
                            },
                        ],
                    }
                ],
                "tools": [
                    {
                        "name": "move_eef",
                        "input_schema": {
                            "type": "object",
                            "properties": {"x": {"type": "number"}},
                        },
                    }
                ],
            },
            "response": {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "action-1", "name": "move_eef", "arguments": {"x": 0.2}}
                    ],
                }
            },
            "tool_results": [
                {
                    "role": "tool",
                    "name": "move_eef",
                    "tool_call_id": "action-1",
                    "content": [
                        '{"state":"arrived"}',
                        {"type": "image", "media_type": "image/png", "data": camera},
                    ],
                }
            ],
            "attempts": [
                {"number": 1, "status": "error", "error_type": "ModelHTTPError"},
                {"number": 2, "status": "completed", "usage": {"input_tokens": 24}},
            ],
        },
        {
            "request": {
                "system_prompt": "Use robot tools.",
                "messages": [
                    {"role": "tool", "tool_call_id": "action-1", "content": "arrived"}
                ],
            },
            "response": "The block is in the bowl.",
        },
    ]
    path = convert_episode(
        _task(),
        calls,
        output_dir=tmp_path / "episode-1",
        outcome={"success": True, "score": 1.0},
    )
    record = load_training_episode(path)
    assert record["schema_version"] == 1
    assert record["task"]["seed"] == 7
    assert record["outcome"]["status"] == "success"
    assert [call["call_id"] for call in record["calls"]] == ["llm-1", "call-000002"]
    assert (
        record["calls"][0]["response"]["message"]["tool_calls"][0]["id"] == "action-1"
    )
    assert (
        record["calls"][0]["tool_results"][0]["content"][0]["text"]
        == '{"state":"arrived"}'
    )
    assert len(record["calls"][0]["attempts"]) == 2
    images = list((path.parent / "assets").iterdir())
    assert len(images) == 1
    assert images[0].read_bytes() == camera
    assert images[0].stem == hashlib.sha256(camera).hexdigest()


def test_convert_episode_rejects_incomplete_calls_and_unsafe_assets(
    tmp_path: Path,
) -> None:
    request = {"messages": [{"role": "user", "content": "task"}]}
    with pytest.raises(ValueError, match="needs a response"):
        convert_episode(
            _task(), [{"request": request}], output_dir=tmp_path / "missing"
        )
    assert not (tmp_path / "missing").exists()

    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"image")
    with pytest.raises(ValueError, match="escapes source_dir"):
        convert_episode(
            _task(),
            [
                {
                    "request": {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "media_type": "image/png",
                                        "asset_ref": "../outside.png",
                                    }
                                ],
                            }
                        ]
                    },
                    "response": "done",
                }
            ],
            output_dir=tmp_path / "unsafe",
            source_dir=source,
        )
    assert not (tmp_path / "unsafe").exists()


def test_load_episode_detects_modified_media(tmp_path: Path) -> None:
    path = convert_episode(
        _task(),
        [
            {
                "request": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "media_type": "image/png",
                                    "data": b"a",
                                }
                            ],
                        }
                    ]
                },
                "response": "done",
            }
        ],
        output_dir=tmp_path / "tamper",
    )
    next((path.parent / "assets").iterdir()).write_bytes(b"b")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_training_episode(path)


def test_version_one_episode_without_new_provenance_still_loads(tmp_path: Path) -> None:
    path = convert_episode(
        _task(),
        [{"request": {"messages": []}, "response": "done"}],
        output_dir=tmp_path / "legacy",
    )
    old_record = json.loads(path.read_text())
    old_record.pop("provenance")
    path.write_text(json.dumps(old_record))
    assert load_training_episode(path)["schema_version"] == 1
