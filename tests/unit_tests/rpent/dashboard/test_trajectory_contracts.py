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

"""Live Dashboard trajectories use the same projection as offline reports."""

import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rpent.dashboard.events import TraceUpdatedEvent, UsageEvent
from rpent.dashboard.server import DashboardServer
from rpent.dashboard.state import DashboardState
from rpent.evaluation.trajectory import TrajectoryReader
from rpent.runtime.trace import TraceRecorder

SPEC = {
    "task": {
        "command": "/rpent-task",
        "usage": "/rpent-task <seed>",
        "fields": ({"name": "seed", "kind": "integer", "minimum": 0},),
        "display": "seed {seed}",
        "output_slug": "s{seed}",
    },
    "runtime_components": (),
    "primitives": (),
}


def _state(root: Path) -> DashboardState:
    return DashboardState(output_dir=root, dashboard_spec=SPEC)


def _record(root: Path, state: DashboardState) -> TraceRecorder:
    class Sink:
        def emit(self, event):
            state.emit(TraceUpdatedEvent(root, event.run_id, event.seq))

    recorder = TraceRecorder(root, sinks=[Sink()])
    recorder.emit("agent_start", {"name": "root"}, agent_id="agent-1")
    recorder.emit("turn_start", {}, agent_id="agent-1", turn_id="turn-1")
    request_ref = recorder.snapshot(
        "requests", "request-1", {"messages": ["<script>untrusted input</script>"]}
    )
    recorder.emit(
        "model_request_start",
        {"request_ref": request_ref, "model": "offline"},
        agent_id="agent-1",
        turn_id="turn-1",
        request_id="request-1",
    )
    recorder.emit(
        "model_attempt_end",
        {
            "status": "completed",
            "usage": {"input_tokens": 10, "output_tokens": 2, "cache_read_tokens": 3},
        },
        agent_id="agent-1",
        turn_id="turn-1",
        request_id="request-1",
        attempt=1,
    )
    return recorder


def test_trace_event_wakes_existing_sse_snapshot_without_changing_usage(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    state.emit(UsageEvent(inp=7, out=2, tool_calls=1))
    version, _ = state.wait_for_snapshot(-1, timeout=0)
    recorder = _record(tmp_path, state)
    new_version, snapshot = state.wait_for_snapshot(version, timeout=0)
    assert new_version > version
    assert snapshot["trace"] == {"run_id": recorder.run_id, "last_seq": 5}
    assert snapshot["usage"] == {"in": 7, "out": 2, "tool_calls": 1}
    assert "messages" not in snapshot["trace"]
    recorder.close()


def test_index_is_incremental_details_lazy_and_media_served(tmp_path: Path) -> None:
    state = _state(tmp_path)
    recorder = _record(tmp_path, state)
    image = tmp_path / "frame.png"
    image.write_bytes(b"recorded image")
    recorder.record_artifact(image)
    client = TestClient(DashboardServer(state=state)._app)
    response = client.get("/api/session/trajectory")
    assert response.status_code == 200
    data = response.json()
    assert data["summary"] == TrajectoryReader(tmp_path).index()["summary"]
    assert data["task_generation"] == 0
    assert "untrusted input" not in response.text
    assert (
        client.get(
            "/api/session/trajectory", params={"after_seq": data["summary"]["last_seq"]}
        ).json()["turns"]
        == []
    )
    identity = {"run_id": recorder.run_id, "generation": 0}
    turn = client.get("/api/session/trajectory/turn/turn-1", params=identity)
    assert turn.status_code == 200
    assert "<script>untrusted input</script>" in turn.text
    media = client.get(
        "/api/session/trajectory/artifact", params={**identity, "ref": "frame.png"}
    )
    assert media.status_code == 200
    assert media.headers["content-type"] == "image/png"
    assert media.content == b"recorded image"
    recorder.close()


def test_task_switch_clears_trace_and_rejects_old_detail_and_media(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    state.shared_services_ready()
    state.request_task({"seed": 1})
    first = state.wait_for_task(timeout=0)
    old = _record(first.output_dir, state)
    old.close()
    client = TestClient(DashboardServer(state=state)._app)
    assert (
        client.get("/api/session/trajectory").json()["summary"]["run_id"] == old.run_id
    )
    state.complete_task(state="succeeded")
    state.request_task({"seed": 2})
    second = state.wait_for_task(timeout=0)
    assert state.snapshot()["trace"] is None
    new = _record(second.output_dir / "sessions/session_001", state)
    # An old writer's late event must never select its old root again.
    state.emit(TraceUpdatedEvent(first.output_dir, old.run_id, 999))
    assert state.snapshot()["trace"]["run_id"] == new.run_id
    identity = {"run_id": old.run_id, "generation": 1}
    assert (
        client.get("/api/session/trajectory/turn/turn-1", params=identity).status_code
        == 409
    )
    assert (
        client.get(
            "/api/session/trajectory/artifact", params={**identity, "ref": "frame.png"}
        ).status_code
        == 409
    )
    fresh = client.get("/api/session/trajectory").json()
    assert fresh["task_generation"] == 2
    assert fresh["summary"]["run_id"] == new.run_id
    new.close()


def test_trajectory_artifacts_reject_traversal_symlinks_and_absent_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    state = _state(root)
    recorder = _record(root, state)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    (root / "link.png").symlink_to(outside)
    client = TestClient(DashboardServer(state=state)._app)
    for ref in ("../outside.png", str(outside), "link.png"):
        assert (
            client.get(
                "/api/session/trajectory/artifact", params={"ref": ref}
            ).status_code
            == 403
        )
    assert (
        client.get(
            "/api/session/trajectory/artifact", params={"ref": "absent.png"}
        ).status_code
        == 404
    )
    assert client.get("/api/session/trajectory/turn/absent").status_code == 404
    assert (
        client.get("/api/session/trajectory", params={"after_seq": -1}).status_code
        == 422
    )
    recorder.close()


def test_task_switch_during_index_read_rejects_stale_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path)
    state.shared_services_ready()
    state.request_task({"seed": 1})
    first = state.wait_for_task(timeout=0)
    recorder = _record(first.output_dir, state)
    recorder.close()
    entered = threading.Event()
    release = threading.Event()
    original = TrajectoryReader.index

    def blocking(reader, after_seq=0):
        entered.set()
        assert release.wait(5)
        return original(reader, after_seq)

    monkeypatch.setattr(TrajectoryReader, "index", blocking)
    client = TestClient(DashboardServer(state=state)._app)
    responses = []
    worker = threading.Thread(
        target=lambda: responses.append(client.get("/api/session/trajectory"))
    )
    worker.start()
    assert entered.wait(5)
    state.complete_task(state="succeeded")
    state.request_task({"seed": 2})
    state.wait_for_task(timeout=0)
    release.set()
    worker.join(5)
    assert responses[0].status_code == 409


@pytest.mark.parametrize("language", ["en", "zh-cn"])
def test_live_page_loads_shared_renderer_and_adapter(
    tmp_path: Path, language: str
) -> None:
    client = TestClient(DashboardServer(state=_state(tmp_path), language=language)._app)
    page = client.get("/trajectory")
    assert page.status_code == 200
    assert f'lang="{language}"' in page.text
    assert "/static/trajectory.js" in page.text
    assert "/static/trajectory_live.js" in page.text
    assert "/static/trajectory.css" in page.text
    assert 'href="/trajectory"' in client.get("/").text
