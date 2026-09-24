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

"""Independent video capture and frame mappings for simulator toolkits."""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from rpent.dashboard.events import NullDashboardEventSink
from rpent.session import EnvState


@pytest.mark.parametrize("robot", ["libero", "robocasa", "robotwin"])
@pytest.mark.parametrize("capture_video", [False, True])
@pytest.mark.parametrize("encode_failure", [False, True])
def test_action_clip_without_dashboard(
    robot, capture_video, encode_failure, monkeypatch, tmp_path
):
    module = importlib.import_module(f"robots.{robot}.toolkit")
    cls = getattr(
        module,
        {
            "libero": "LiberoToolkit",
            "robocasa": "RoboCasaToolkit",
            "robotwin": "RoboTwinToolkit",
        }[robot],
    )
    toolkit = cls.__new__(cls)
    frames = [np.full((2, 2, 3), index, dtype=np.uint8) for index in range(5)]
    toolkit._primitives = SimpleNamespace(
        recorded_frame_count=lambda: len(frames),
        frame_slice=lambda start: frames[start:],
        status=lambda: {},
    )
    toolkit._state = EnvState(tmp_path)
    toolkit._dashboard_events = NullDashboardEventSink()
    if capture_video:
        toolkit.capture_video = True
    toolkit._action_frame_cursor = 2
    toolkit._solved = False
    tools = getattr(module, f"{robot}_tools", None) or module.tools

    def dump(*args, **kwargs):
        with toolkit._state.record_step(state={}) as step:
            pass
        return toolkit._state.get(step)

    if robot == "robotwin":
        monkeypatch.setattr(toolkit, "_capture_full_observation", lambda: {})
        monkeypatch.setattr(tools, "dump_observation", dump)
    else:
        monkeypatch.setattr(tools, "dump_state", dump)
    monkeypatch.setattr(tools, "view_env_state", lambda *a, **k: {})
    encoded = []

    def encode(path, video_frames, **kwargs):
        encoded.append(video_frames)
        if encode_failure:
            raise RuntimeError("encoder unavailable")
        path.write_bytes(b"offline-video")

    monkeypatch.setattr("rpent.session.base.imageio.mimwrite", encode)
    toolkit.get_env_state(command={"action": "move_to"}, result={}, elapsed_s=1.0)
    state = toolkit._state
    assert state.exists("action_move_to.mp4", step=0) is (
        capture_video and not encode_failure
    )
    manifest = state.artifact_path("episode.mp4", step=None).parent / "states.json"
    if not capture_video:
        assert encoded == []
        assert "videos" not in state.get(0).extras
        return
    assert len(encoded[0]) == 3
    if encode_failure:
        assert "videos" not in state.get(0).extras
        return
    assert state.get(0).extras["videos"] == [
        {
            "video_ref": "action_move_to.mp4/00.mp4",
            "frame_start": 0,
            "frame_end": 3,
            "fps": 20,
        },
        {
            "video_ref": "episode.mp4",
            "frame_start": 2,
            "frame_end": 5,
            "fps": 20,
        },
    ]
    assert (
        json.loads(manifest.read_text())["steps"][0]["extras"]["videos"]
        == (state.get(0).extras["videos"])
    )
