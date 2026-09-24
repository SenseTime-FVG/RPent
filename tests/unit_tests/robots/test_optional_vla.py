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

"""Optional VLA contracts without simulators, model services, or hardware."""

from __future__ import annotations

import importlib
import re
from types import SimpleNamespace

import pytest

from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.robots import RunConfig

ROBOTS = ["libero", "robocasa", "robotwin", "franka", "dual_franka"]
TOOLKITS = {
    "libero": ("LiberoToolkit", {"pi0_pick", "pi0_doubled"}, "model"),
    "robocasa": ("RoboCasaToolkit", {"rldx_skill", "rldx_arm"}, "vla_client"),
    "robotwin": ("RoboTwinToolkit", {"lingbot_act"}, "model"),
    "franka": ("FrankaToolkit", {"vla_grasp"}, "model"),
    "dual_franka": (
        "DualFrankaToolkit",
        {"vla_right_grasp", "vla_handoff", "vla_left_place"},
        "model",
    ),
}


@pytest.mark.parametrize("robot", ROBOTS)
@pytest.mark.parametrize("components", [None, {"vla"}])
def test_disabled_vla_never_spawns_or_connects(
    robot, components, monkeypatch, tmp_path
):
    module = importlib.import_module(f"robots.{robot}.robot_spec")
    started = []

    def spawn(_owned, _events, component, _starter):
        started.append(component)
        return None, object()

    monkeypatch.setattr(module, "try_spawn_server", spawn)
    monkeypatch.setattr(module, "try_wait_server", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        module, "_spawn_vla_server", lambda *args: pytest.fail("VLA started")
    )
    args = SimpleNamespace(
        enable_vla=False,
        vla_endpoint="localhost:9999",
        sam3_endpoint=None,
        task_id=None,
        planner="api",
        collect_flywheel_data=False,
    )
    owned, kwargs = module._init_runtime(
        args, tmp_path, NullDashboardEventSink(), components
    )
    assert "vla" not in started
    assert kwargs[TOOLKITS[robot][2]] is None
    assert owned == []
    if components is not None:
        assert components == {"vla"}, "component selection must not mutate caller data"


@pytest.mark.parametrize("robot", ROBOTS)
def test_factory_passes_explicit_vla_flag(robot, monkeypatch, tmp_path):
    module = importlib.import_module(f"robots.{robot}.robot_spec")
    toolkit_module = importlib.import_module(f"robots.{robot}.toolkit")
    monkeypatch.setattr(
        toolkit_module, TOOLKITS[robot][0], lambda **kwargs: SimpleNamespace(**kwargs)
    )
    config = RunConfig("offline", tmp_path, {"memory_dir": str(tmp_path)}, {})
    result = module.get_toolkit(
        runtime_kwargs={},
        dashboard_events=NullDashboardEventSink(),
        config=config,
        enable_vla=False,
    )
    assert result.enable_vla is False


class FakePrimitives:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.env = SimpleNamespace(last_reset_info={})

    def __getattr__(self, name):
        if name == "recorded_frame_count":
            return lambda: 0
        return lambda *args, **kwargs: {"ok": True}


@pytest.mark.parametrize("robot", ROBOTS)
@pytest.mark.parametrize("availability", ["enabled", "disabled", "no_client"])
def test_tool_availability_matches_schema_and_dispatch(
    robot, availability, monkeypatch, tmp_path
):
    from rpent.utils import templates

    monkeypatch.setattr(
        templates, "default_variables", lambda: {"output_dir": str(tmp_path)}
    )
    module = importlib.import_module(f"robots.{robot}.toolkit")
    cls_name, vla_tools, model_key = TOOLKITS[robot]
    cls = getattr(module, cls_name)
    record = SimpleNamespace(step_idx=0, terminated=False, extras={})
    if robot == "libero":
        monkeypatch.setattr(module.libero_tools, "LiberoPrimitives", FakePrimitives)
        monkeypatch.setattr(module.libero_tools, "dump_state", lambda *a, **k: record)
    elif robot == "robocasa":
        primitives = importlib.import_module("robots.robocasa.primitives")
        monkeypatch.setattr(primitives, "RoboCasaPrimitives", FakePrimitives)
        monkeypatch.setattr(module.robocasa_tools, "dump_state", lambda *a, **k: record)
    elif robot == "robotwin":
        monkeypatch.setattr(module, "RoboTwinPrimitives", FakePrimitives)
        monkeypatch.setattr(cls, "get_env_state", lambda *a, **k: {})
    else:
        monkeypatch.setattr(cls, "_primitives_cls", FakePrimitives)
        monkeypatch.setattr(cls._tools_module, "dump_state", lambda *a, **k: record)

    client = None if availability == "no_client" else object()
    kwargs = {model_key: client}
    toolkit = cls(
        runtime_kwargs=kwargs,
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        state_output_dir=tmp_path / "state",
        **({"enable_vla": False} if availability == "disabled" else {}),
    )
    names = {spec["name"] for spec in toolkit.get_tools_spec()}
    assert cls.VLA_TOOLS == vla_tools
    assert {"finish", "view_env_state"} <= names
    assert kwargs[model_key] is client, "must not mutate caller runtime data"
    if availability == "enabled":
        assert vla_tools <= names
        assert toolkit._primitives.kwargs[model_key] is client
        return
    assert not names.intersection(vla_tools)
    assert toolkit._primitives.kwargs[model_key] is None
    for name in vla_tools:
        result = toolkit.execute_tool(name, {})
        assert result.result == {"error": f"unknown tool: {name}"}


@pytest.mark.parametrize("robot", ROBOTS)
@pytest.mark.parametrize("mode", ["eval", "explore"])
def test_disabled_prompt_does_not_instruct_missing_tools(robot, mode):
    from rpent.prompt.utils import format_prompt

    module = importlib.import_module(f"robots.{robot}.prompt_bundle")
    node = module.system_prompt({"enable_vla": False, "mode": mode})
    variables = dict.fromkeys(re.findall(r"\{\{(\w+)\}\}", str(node)), "offline")
    prompt = format_prompt(node, variables=variables)
    assert "VLA is disabled" in prompt
    for tool in TOOLKITS[robot][1]:
        assert tool not in prompt


def test_robocasa_reset_without_vla(tmp_path):
    import numpy as np

    from robots.robocasa.primitives import RoboCasaPrimitives

    env = SimpleNamespace(reset=lambda: None, eef_pos=np.zeros(3))
    primitives = RoboCasaPrimitives(
        env_client=env, workdir=str(tmp_path), hi_res=None, allow_reset=True
    )
    assert primitives.reset()["ok"] is True
    with pytest.raises(RuntimeError, match="VLA"):
        primitives.rldx_skill()


def test_libero_scripted_gripper_without_vla():
    import numpy as np

    from robots.libero.tools import LiberoPrimitives

    actions = []
    observation = {"states": np.zeros(8), "task_descriptions": "open gripper"}
    env = SimpleNamespace(
        reset=lambda: (observation, {}),
        step=lambda action: (
            actions.append(action) or observation,
            0.0,
            False,
            False,
            {},
        ),
        terminated=False,
        truncated=False,
    )
    primitives = LiberoPrimitives(
        env=env, model=None, sam3_client=None, check_cancelled=lambda: None
    )
    primitives.reset()
    assert primitives.set_gripper(gripper=-1, steps=1)["steps"] == 1
    assert actions[0][-1] == -1
    with pytest.raises(RuntimeError, match="VLA"):
        primitives._vlm_chunk("pick")


@pytest.mark.parametrize("robot", ["franka", "dual_franka"])
def test_physical_robot_direct_control_without_vla(robot):
    module = importlib.import_module(f"robots.{robot}.tools")
    cls = module.FrankaPrimitives if robot == "franka" else module.DualFrankaPrimitives
    moves = []
    env = SimpleNamespace(
        reset=lambda: {"ok": True},
        move_delta=lambda *args: moves.append(args) or {"ok": True},
    )
    primitives = cls(env=env, task_description="inspect", check_cancelled=lambda: None)
    assert primitives.reset()["ok"]
    params = {"delta_xyz": [0.01, 0, 0]}
    if robot == "dual_franka":
        params["arm"] = "left"
    assert primitives.move_delta(**params)["ok"]
    assert len(moves) == 1


def test_robotwin_reset_and_finish_without_vla():
    from robots.robotwin.primitives import RoboTwinPrimitives

    env = SimpleNamespace(
        reset=lambda: ({}, {"actual_seed": 7}),
        last_info={"episode_status": {"eval_success": False}},
    )
    primitives = RoboTwinPrimitives(env=env, seed=7, check_cancelled=lambda: None)
    assert primitives.reset()["success"]
    assert primitives.finish(status="failure", summary="No learned action")["_finish"]
    with pytest.raises(RuntimeError, match="VLA"):
        primitives.lingbot_act()


def test_flash_rejects_disabled_vla_before_spawning(monkeypatch, tmp_path):
    from robots.libero import robot_spec

    monkeypatch.setattr(
        robot_spec, "try_spawn_server", lambda *args: pytest.fail("resource started")
    )
    args = SimpleNamespace(enable_vla=False, planner="flash")
    with pytest.raises(ValueError, match="flash requires VLA"):
        robot_spec._init_runtime(args, tmp_path, NullDashboardEventSink(), None)
